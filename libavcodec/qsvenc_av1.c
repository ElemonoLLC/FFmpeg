/*
 * Intel MediaSDK QSV based AV1 encoder
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */


#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include "libavutil/common.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "bsf.h"
#include "qsv.h"
#include "qsvenc.h"

/* -maxrate / -bufsize in CQP mode: a VBV leaky bucket. The bucket fills at
 * maxrate / fps per submitted frame and drains by the real size of each output
 * packet. The frame QP is the scheduled (or base) QP, raised only as far as
 * needed for the bucket to stay non-negative. Output lags submission, so each
 * submitted frame's predicted size is held as in flight until its packet
 * arrives. Predicted size is C * 2^(-QP / VBV_QP_SCALE) per frame type. */
#define VBV_QP_SCALE   22.0   /* QP per halving of bits, rough fit on this clip */
#define VBV_PENDING    256    /* submitted frames not yet matched to a packet */
#define VBV_MAX_QP     255

typedef struct VBVPending {
    uint64_t ts;
    double   pred;
    int      qp;
    int      intra;
} VBVPending;

typedef struct QSVAV1EncContext {
    AVClass *class;
    AVBSFContext *extra_data_bsf;
    QSVEncContext qsv;

    int maxrate_feedback;

    /* VBV state, in bits */
    double vbv_cap;
    double vbv_refill;        /* per submitted frame */
    double vbv_fill;          /* bucket level from real output so far */
    double vbv_inflight;      /* predicted bits of submitted, unmatched frames */
    double cplx[2];           /* [0] P/B, [1] I: bits * 2^(QP / VBV_QP_SCALE) */
    int    cplx_init[2];
    VBVPending pending[VBV_PENDING];
    int    pend_head, pend_count;
    uint64_t submit_idx;      /* frames submitted so far, for GOP position */
    int    vbv_raised;        /* frames coded above their scheduled QP */
    int    vbv_capped;        /* frames where even QP 255 would not fit */
    int    vbv_underflow;     /* output packets that left the bucket negative */
    int    vbv_overflow;      /* submitted frames that hit the bucket ceiling */
} QSVAV1EncContext;

/* EXPERIMENTAL: per-frame QP test hook. If QSV_AV1_QP_SCHEDULE names a file
 * with one QP per line (in input order), each frame's mfxEncodeCtrl.QP is set
 * from it. Only effective in CQP mode. Not the final side-data design. */
static int *qp_sched;
static int qp_sched_n, qp_sched_idx, qp_sched_loaded;

static void qp_sched_load(AVCodecContext *avctx)
{
    const char *path = getenv("QSV_AV1_QP_SCHEDULE");
    FILE *f;
    int v;

    qp_sched_loaded = 1;
    if (!path)
        return;
    f = fopen(path, "r");
    if (!f) {
        av_log(avctx, AV_LOG_WARNING, "QP schedule %s not readable\n", path);
        return;
    }
    while (fscanf(f, "%d", &v) == 1) {
        int *n = realloc(qp_sched, (qp_sched_n + 1) * sizeof(*qp_sched));
        if (!n)
            break;
        qp_sched = n;
        qp_sched[qp_sched_n++] = v;
    }
    fclose(f);
    av_log(avctx, AV_LOG_INFO, "Loaded %d per-frame QPs from %s\n", qp_sched_n, path);
}

static double vbv_pred_bits(const QSVAV1EncContext *q, int intra, int qp)
{
    return q->cplx[intra] * exp2(-(double)qp / VBV_QP_SCALE);
}

/* Smallest QP at or above base whose predicted size fits in the bucket after
 * the in-flight frames, then record this frame as in flight. */
static int vbv_choose_qp(QSVAV1EncContext *q, uint64_t ts, int intra, int base)
{
    double room, pred;
    int qp = base, n;

    if (!q->cplx_init[intra]) {
        /* No history yet: the first frames are submitted before any packet
         * comes back, so start from a typical cost at QP 30 (about 3 Mbit for
         * P and 6 Mbit for I at 720p on this clip). Packets replace this. */
        q->cplx[intra] = (intra ? 6e6 : 3e6) * exp2(30.0 / VBV_QP_SCALE);
        q->cplx_init[intra] = 1;
    }

    /* This frame adds one refill before it is coded. */
    q->vbv_fill = FFMIN(q->vbv_fill + q->vbv_refill, q->vbv_cap);
    if (q->vbv_fill == q->vbv_cap)
        q->vbv_overflow++;
    room = q->vbv_fill - q->vbv_inflight;

    for (; qp < VBV_MAX_QP; qp++)
        if (vbv_pred_bits(q, intra, qp) <= room)
            break;
    pred = vbv_pred_bits(q, intra, qp);
    if (qp > base)
        q->vbv_raised++;
    if (pred > room)
        q->vbv_capped++;

    if (q->pend_count == VBV_PENDING) {
        /* Packets are not arriving for the oldest entry: drop its prediction. */
        q->vbv_inflight -= q->pending[q->pend_head].pred;
        q->pend_head = (q->pend_head + 1) % VBV_PENDING;
        q->pend_count--;
    }
    n = (q->pend_head + q->pend_count) % VBV_PENDING;
    q->pending[n].ts    = ts;
    q->pending[n].pred  = pred;
    q->pending[n].qp    = qp;
    q->pending[n].intra = intra;
    q->pend_count++;
    q->vbv_inflight += pred;
    return qp;
}

static void qsv_av1_packet_done(AVCodecContext *avctx, uint64_t timestamp_90k,
                                uint32_t data_len)
{
    QSVAV1EncContext *q = avctx->priv_data;
    double actual = (double)data_len * 8;
    VBVPending *p = NULL;
    int i, idx = -1;

    if (!q->maxrate_feedback)
        return;

    /* The bucket is drained by real output regardless of matching. */
    q->vbv_fill -= actual;
    if (q->vbv_fill < 0)
        q->vbv_underflow++;

    /* Match the packet to its submitted frame by timestamp, else the oldest. */
    for (i = 0; i < q->pend_count; i++) {
        int k = (q->pend_head + i) % VBV_PENDING;
        if (q->pending[k].ts == timestamp_90k) {
            idx = k;
            break;
        }
    }
    if (idx < 0 && q->pend_count)
        idx = q->pend_head;
    if (idx >= 0) {
        p = &q->pending[idx];
        q->vbv_inflight -= p->pred;
        /* Remove the entry by moving the head past it when it is the head. */
        if (idx == q->pend_head) {
            q->pend_head = (q->pend_head + 1) % VBV_PENDING;
            q->pend_count--;
        } else {
            /* Out-of-order match: mark it as consumed with zero prediction. */
            p->pred = 0;
            p->ts   = (uint64_t)-1;
        }
        {
            double sample = actual * exp2((double)p->qp / VBV_QP_SCALE);
            q->cplx[p->intra] = 0.75 * q->cplx[p->intra] + 0.25 * sample;
        }
    }
}

static int qsv_av1_set_encode_ctrl(AVCodecContext *avctx,
                                   const AVFrame *frame, mfxEncodeCtrl *enc_ctrl)
{
    QSVAV1EncContext *q = avctx->priv_data;
    AVFrameSideData *sd;

    if (!frame)
        return 0;

    if (!qp_sched_loaded)
        qp_sched_load(avctx);

    {
        int qp = 0, qp_set = 0;

        if (qp_sched_idx < qp_sched_n) {
            qp = qp_sched[qp_sched_idx];
            /* QSV_AV1_QP_TYPE_OFFSETS=1: keep the encoder's own I-frame offset
             * (QPI - QPP from -q:v) on top of the schedule value for keyframes. */
            if (getenv("QSV_AV1_QP_TYPE_OFFSETS") && q->qsv.param.mfx.GopPicSize > 0 &&
                qp_sched_idx % q->qsv.param.mfx.GopPicSize == 0)
                qp += (int)q->qsv.param.mfx.QPI - (int)q->qsv.param.mfx.QPP;
            qp_sched_idx++;
            qp_set = 1;
        }

        if (q->maxrate_feedback) {
            int gop = q->qsv.param.mfx.GopPicSize;
            int intra = frame->pict_type == AV_PICTURE_TYPE_I ||
                        (gop > 0 && q->submit_idx % gop == 0);
            q->submit_idx++;
            if (!qp_set) {
                qp = intra ? (int)q->qsv.param.mfx.QPI : (int)q->qsv.param.mfx.QPP;
                qp_set = 1;
            }
            qp = vbv_choose_qp(q, frame->pts >= 0 ? (uint64_t)av_rescale_q(frame->pts, avctx->time_base, (AVRational){1, 90000}) : 0,
                               intra, av_clip(qp, 1, VBV_MAX_QP));
        }

        if (qp_set)
            enc_ctrl->QP = (mfxU16)av_clip(qp, 1, 255);
    }

    if (!QSV_RUNTIME_VERSION_ATLEAST(q->qsv.ver, 2, 11))
        return 0;

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd) {
        AVMasteringDisplayMetadata *mdm = (AVMasteringDisplayMetadata *)sd->data;
        if (mdm->has_primaries && mdm->has_luminance) {
            const int chroma_den   = 1 << 16;
            const int max_luma_den = 1 << 8;
            const int min_luma_den = 1 << 14;
            mfxExtMasteringDisplayColourVolume *mdcv = av_mallocz(sizeof(*mdcv));
            if (!mdcv)
                return AVERROR(ENOMEM);

            mdcv->Header.BufferId = MFX_EXTBUFF_MASTERING_DISPLAY_COLOUR_VOLUME;
            mdcv->Header.BufferSz = sizeof(*mdcv);

            for (int i = 0; i < 3; i++) {
                mdcv->DisplayPrimariesX[i] =
                    av_rescale(mdm->display_primaries[i][0].num, chroma_den,
                               mdm->display_primaries[i][0].den);
                mdcv->DisplayPrimariesY[i] =
                    av_rescale(mdm->display_primaries[i][1].num, chroma_den,
                               mdm->display_primaries[i][1].den);
            }

            mdcv->WhitePointX =
                av_rescale(mdm->white_point[0].num, chroma_den,
                           mdm->white_point[0].den);
            mdcv->WhitePointY =
                av_rescale(mdm->white_point[1].num, chroma_den,
                           mdm->white_point[1].den);

            mdcv->MaxDisplayMasteringLuminance =
                av_rescale(mdm->max_luminance.num, max_luma_den,
                           mdm->max_luminance.den);
            mdcv->MinDisplayMasteringLuminance =
                av_rescale(mdm->min_luminance.num, min_luma_den,
                           mdm->min_luminance.den);

            enc_ctrl->ExtParam[enc_ctrl->NumExtParam++] = (mfxExtBuffer *)mdcv;
        }
    }

    sd = av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (sd) {
        AVContentLightMetadata *clm = (AVContentLightMetadata *)sd->data;
        mfxExtContentLightLevelInfo *clli = av_mallocz(sizeof(*clli));
        if (!clli)
            return AVERROR(ENOMEM);

        clli->Header.BufferId = MFX_EXTBUFF_CONTENT_LIGHT_LEVEL_INFO;
        clli->Header.BufferSz = sizeof(*clli);

        clli->MaxContentLightLevel          = clm->MaxCLL;
        clli->MaxPicAverageLightLevel       = clm->MaxFALL;

        enc_ctrl->ExtParam[enc_ctrl->NumExtParam++] = (mfxExtBuffer *)clli;
    }

    return 0;
}

static av_cold int qsv_enc_init(AVCodecContext *avctx)
{
    QSVAV1EncContext *q = avctx->priv_data;
    int ret;

    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        const AVBitStreamFilter *filter = av_bsf_get_by_name("extract_extradata");
        if (!filter) {
            av_log(avctx, AV_LOG_ERROR, "Cannot get extract_extradata bitstream filter\n");
            return AVERROR_BUG;
        }
        ret = av_bsf_alloc(filter, &q->extra_data_bsf);
        if (ret < 0)
            return ret;
        ret = avcodec_parameters_from_context(q->extra_data_bsf->par_in, avctx);
        if (ret < 0)
           return ret;
        ret = av_bsf_init(q->extra_data_bsf);
        if (ret < 0)
           return ret;
    }

    q->qsv.set_encode_ctrl_cb = qsv_av1_set_encode_ctrl;
    q->qsv.packet_done_cb     = qsv_av1_packet_done;

    ret = ff_qsv_enc_init(avctx, &q->qsv);
    if (ret < 0)
        return ret;

    /* select_rc_mode() picks CQP over -maxrate, so the cap is applied here by
     * feedback. QSV_AV1_MAXRATE_FEEDBACK=0 disables it for A/B comparison. */
    if (avctx->rc_max_rate > 0 &&
        q->qsv.param.mfx.RateControlMethod == MFX_RATECONTROL_CQP) {
        const char *env = getenv("QSV_AV1_MAXRATE_FEEDBACK");
        q->maxrate_feedback = !(env && env[0] == '0');
        if (q->maxrate_feedback) {
            AVRational fr = avctx->framerate.num > 0 ? avctx->framerate
                                                     : av_inv_q(avctx->time_base);
            double fps = fr.den > 0 ? av_q2d(fr) : 25.0;
            q->vbv_cap    = avctx->rc_buffer_size > 0 ? avctx->rc_buffer_size
                                                      : avctx->rc_max_rate;
            q->vbv_refill = avctx->rc_max_rate / fps;
            q->vbv_fill   = q->vbv_cap;
            av_log(avctx, AV_LOG_VERBOSE, "CQP with -maxrate %"PRId64" bit/s and "
                   "-bufsize %"PRId64" bit: VBV bucket, QP raised above the "
                   "schedule only where needed\n",
                   (int64_t)avctx->rc_max_rate, (int64_t)q->vbv_cap);
        } else if (avctx->rc_max_rate > 0) {
            av_log(avctx, AV_LOG_WARNING, "CQP with -maxrate: feedback disabled "
                   "by QSV_AV1_MAXRATE_FEEDBACK=0, rate cap not enforced\n");
        }
    }

    return 0;
}

static int qsv_enc_frame(AVCodecContext *avctx, AVPacket *pkt,
                         const AVFrame *frame, int *got_packet)
{
    QSVAV1EncContext *q = avctx->priv_data;
    int ret;

    ret = ff_qsv_encode(avctx, &q->qsv, pkt, frame, got_packet);
    if (ret < 0)
        return ret;

    if (*got_packet && avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        ret = av_bsf_send_packet(q->extra_data_bsf, pkt);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "extract_extradata filter "
                "failed to send input packet\n");
            return ret;
        }

        ret = av_bsf_receive_packet(q->extra_data_bsf, pkt);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "extract_extradata filter "
                "failed to receive output packet\n");
            return ret;
        }
    }

    return ret;
}

static av_cold int qsv_enc_close(AVCodecContext *avctx)
{
    QSVAV1EncContext *q = avctx->priv_data;

    av_bsf_free(&q->extra_data_bsf);

    if (q->maxrate_feedback)
        av_log(avctx, AV_LOG_INFO, "VBV: %d frames raised above the schedule, "
               "%d frames capped at QP 255, %d underflow and %d overflow events\n",
               q->vbv_raised, q->vbv_capped, q->vbv_underflow, q->vbv_overflow);

    return ff_qsv_enc_close(avctx, &q->qsv);
}

#define OFFSET(x) offsetof(QSVAV1EncContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    QSV_COMMON_OPTS
    QSV_OPTION_B_STRATEGY
    QSV_OPTION_ADAPTIVE_I
    QSV_OPTION_ADAPTIVE_B
    QSV_OPTION_EXTBRC
    QSV_OPTION_LOW_DELAY_BRC
    QSV_OPTION_MAX_FRAME_SIZE
    { "profile", NULL, OFFSET(qsv.profile), AV_OPT_TYPE_INT, { .i64 = MFX_PROFILE_UNKNOWN }, 0, INT_MAX, VE, .unit = "profile" },
        { "unknown" , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = MFX_PROFILE_UNKNOWN      }, INT_MIN, INT_MAX,     VE, .unit = "profile" },
        { "main"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = MFX_PROFILE_AV1_MAIN     }, INT_MIN, INT_MAX,     VE, .unit = "profile" },
    { "tile_cols",  "Number of columns for tiled encoding",   OFFSET(qsv.tile_cols),    AV_OPT_TYPE_INT, { .i64 = 0 }, 0, UINT16_MAX, VE },
    { "tile_rows",  "Number of rows for tiled encoding",      OFFSET(qsv.tile_rows),    AV_OPT_TYPE_INT, { .i64 = 0 }, 0, UINT16_MAX, VE },
    { "look_ahead_depth", "Depth of look ahead in number frames, available when extbrc option is enabled", OFFSET(qsv.look_ahead_depth), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 100, VE },
    { NULL },
};

static const AVClass class = {
    .class_name = "av1_qsv encoder",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const FFCodecDefault qsv_enc_defaults[] = {
    { "b",         "0"    },
    { "g",         "-1"   },
    { "bf",        "-1"   },
    { "refs",      "0"    },
    { NULL },
};

FFCodec ff_av1_qsv_encoder = {
    .p.name           = "av1_qsv",
    .p.long_name      = NULL_IF_CONFIG_SMALL("AV1 (Intel Quick Sync Video acceleration)"),
    .priv_data_size = sizeof(QSVAV1EncContext),
    .p.type           = AVMEDIA_TYPE_VIDEO,
    .p.id             = AV_CODEC_ID_AV1,
    .init             = qsv_enc_init,
    FF_CODEC_ENCODE_CB(qsv_enc_frame),
    .close            = qsv_enc_close,
    .p.capabilities   = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_HYBRID,
    CODEC_PIXFMTS(AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_QSV),
    .color_ranges     = AVCOL_RANGE_MPEG | AVCOL_RANGE_JPEG,
    .p.priv_class     = &class,
    .defaults       = qsv_enc_defaults,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .p.wrapper_name   = "qsv",
    .hw_configs     = ff_qsv_enc_hw_configs,
};
