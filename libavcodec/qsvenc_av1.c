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

/* -maxrate in CQP mode: the SDK has no rate cap for CQP, so the output rate is
 * measured over a sliding window of output timestamps and a non-negative QP
 * bias is added to the base QP while the window exceeds avctx->rc_max_rate. */
#define MAXRATE_WIN_TICKS 90000   /* 1 s on the 90 kHz output timestamp clock */
#define MAXRATE_WIN_CAP   1024    /* max packets tracked in the window */
#define QP_BIAS_MAX       255
#define BIAS_FP           16      /* fixed-point units per QP step */
#define MAXRATE_GAIN_DIV  1       /* QP step per packet = error/max / this (error is in units of max) */

typedef struct QSVAV1EncContext {
    AVClass *class;
    AVBSFContext *extra_data_bsf;
    QSVEncContext qsv;

    int maxrate_feedback;
    int64_t qp_bias_fp;       /* BIAS_FP units per QP step */
    int qp_bias;
    struct {
        uint64_t ts;
        uint64_t bits;
    } win[MAXRATE_WIN_CAP];
    int win_head, win_count;
    uint64_t win_bits;
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

static void maxrate_win_push(QSVAV1EncContext *q, uint64_t ts, uint64_t bits)
{
    int idx;

    if (q->win_count == MAXRATE_WIN_CAP) {
        q->win_bits -= q->win[q->win_head].bits;
        q->win_head = (q->win_head + 1) % MAXRATE_WIN_CAP;
        q->win_count--;
    }
    idx = (q->win_head + q->win_count) % MAXRATE_WIN_CAP;
    q->win[idx].ts   = ts;
    q->win[idx].bits = bits;
    q->win_count++;
    q->win_bits += bits;

    /* Drop packets more than one window older than the newest one. The
     * unsigned difference is cast so out-of-order timestamps are not dropped. */
    while (q->win_count > 1 &&
           (int64_t)(ts - q->win[q->win_head].ts) > MAXRATE_WIN_TICKS) {
        q->win_bits -= q->win[q->win_head].bits;
        q->win_head = (q->win_head + 1) % MAXRATE_WIN_CAP;
        q->win_count--;
    }
}

/* Returns the windowed output rate in bit/s, or 0 when there is not yet
 * enough history for a meaningful estimate. */
static int64_t maxrate_win_bps(const QSVAV1EncContext *q)
{
    uint64_t newest, oldest, span;

    if (q->win_count < 2)
        return 0;
    newest = q->win[(q->win_head + q->win_count - 1) % MAXRATE_WIN_CAP].ts;
    oldest = q->win[q->win_head].ts;
    span   = newest - oldest;
    if (span < MAXRATE_WIN_TICKS / 4)
        return 0;
    return (int64_t)(q->win_bits * 90000 / span);
}

/* Integral controller, updated once per output packet. The window rate lags
 * the QP change by about one window, so updating per submitted frame would
 * keep integrating a stale error and oscillate. */
static void maxrate_update_bias(AVCodecContext *avctx, QSVAV1EncContext *q)
{
    int64_t rate = maxrate_win_bps(q);
    int64_t max  = avctx->rc_max_rate;
    int64_t err;

    if (rate <= 0)
        return;
    /* Error is clamped to [-max, +2*max], so one packet moves the bias by at
     * most 2 * BIAS_FP / MAXRATE_GAIN_DIV units upward. */
    err = av_clip64(rate - max, -max, 2 * max);
    q->qp_bias_fp = av_clip64(q->qp_bias_fp +
                              err * BIAS_FP / (max * MAXRATE_GAIN_DIV),
                              0, QP_BIAS_MAX * BIAS_FP);
    q->qp_bias = (int)(q->qp_bias_fp / BIAS_FP);
}

static void qsv_av1_packet_done(AVCodecContext *avctx, uint64_t timestamp_90k,
                                uint32_t data_len)
{
    QSVAV1EncContext *q = avctx->priv_data;

    if (!q->maxrate_feedback)
        return;
    maxrate_win_push(q, timestamp_90k, (uint64_t)data_len * 8);
    maxrate_update_bias(avctx, q);
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
            if (q->qp_bias > 0) {
                /* Without a schedule the SDK would pick QPI/QPP/QPB itself, so
                 * the bias needs an explicit base: QPI for I frames, else QPP.
                 * B frames therefore lose their QPB offset while biased. */
                if (!qp_set) {
                    qp = frame->pict_type == AV_PICTURE_TYPE_I ?
                         (int)q->qsv.param.mfx.QPI : (int)q->qsv.param.mfx.QPP;
                    qp_set = 1;
                }
                qp += q->qp_bias;
            }
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
        if (q->maxrate_feedback)
            av_log(avctx, AV_LOG_VERBOSE, "CQP with -maxrate %"PRId64" bit/s: "
                   "enforcing the rate cap by per-frame QP feedback\n",
                   (int64_t)avctx->rc_max_rate);
        else
            av_log(avctx, AV_LOG_WARNING, "CQP with -maxrate: feedback disabled "
                   "by QSV_AV1_MAXRATE_FEEDBACK=0, rate cap not enforced\n");
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
