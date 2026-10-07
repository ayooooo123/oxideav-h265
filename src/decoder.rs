//! Registry decoder — the [`oxideav_core::Decoder`] contract over the
//! whole-bitstream [`crate::sequence`] driver.
//!
//! [`make_decoder`] is the direct factory endpoint (the crate's
//! historical direct-API convention); [`crate::register`] wires the
//! same factory into the [`oxideav_core`] codec registry under the
//! `"h265"` / `"hevc"` ids and the common container tags.
//!
//! Packets carry either Annex B byte-stream chunks (start-code
//! delimited NAL units) or, when `CodecParameters::extradata` is an
//! `hvcC` / `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15
//! §8.3.3.1), length-prefixed NAL runs as ISO-BMFF samples carry them.
//! Extradata in either form is fed ahead of the first packet so
//! out-of-band parameter sets activate. Output frames come in output
//! (PicOrderCntVal) order, with packet PTS values re-attached in
//! ascending order; an empty packet flushes the reorder queue.

use std::collections::BinaryHeap;
use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
    VideoPlane,
};

use crate::hvcc::{
    extradata_is_hvcc, extradata_is_lhvc, parse_hvcc_with_len, parse_lhvc, split_length_prefixed,
};
use crate::picture::{sub_wh_c, Picture, Plane};
use crate::sequence::{CropWindow, DecodedFrame, LayerTarget, SequenceDecoder};

/// The default reorder depth when no SPS has been activated yet (the
/// §7.4.3.2.1 `sps_max_num_reorder_pics` bound once one has).
const DEFAULT_REORDER: usize = 8;

/// H.265 / HEVC Annex B streaming decoder.
pub struct H265Decoder {
    codec_id: CodecId,
    seq: SequenceDecoder,
    /// Decoded pictures not yet emitted, sorted on demand by
    /// `(cvs_index, poc)`.
    reorder: Vec<DecodedFrame>,
    /// Frames ready to hand out, each with its [`Layout`].
    ready: VecDeque<(Frame, Layout)>,
    /// The [`Layout`] of the frame `receive_frame` last returned.
    last_output: Option<Layout>,
    /// Min-heap of packet PTS values, re-attached in output order.
    pts_queue: BinaryHeap<std::cmp::Reverse<i64>>,
    /// `Some(n)` when the extradata was an `hvcC` record: packets are
    /// length-prefixed NAL runs with `n`-byte big-endian sizes
    /// (ISO/IEC 14496-15 §8.3.3.1.3 `lengthSizeMinusOne + 1`).
    /// `None` for Annex B packets.
    nal_length_size: Option<usize>,
    flushed: bool,
}

impl std::fmt::Debug for H265Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H265Decoder")
            .field("codec_id", &self.codec_id)
            .field("reorder", &self.reorder.len())
            .field("ready", &self.ready.len())
            .finish()
    }
}

/// Direct factory endpoint: construct the software H.265 decoder.
///
/// The `extradata` form selects the packet framing: an `hvcC`
/// (`HEVCDecoderConfigurationRecord`) extradata activates its carried
/// parameter sets and switches packets to length-prefixed NAL runs;
/// Annex B extradata (or none) keeps start-code framing. An `hvcC`
/// record may be followed by an `lhvC` (`LHEVCDecoderConfigurationRecord`)
/// carrying the non-base layers' parameter sets — the extradata a
/// layered HEIF item (`lhv1`, e.g. a stereo / spatial photo) resolves
/// to; the packets then carry every layer's NAL units and the decoder
/// runs the Annex F/G/H multi-layer processes.
///
/// Codec options (multi-layer streams only; single-layer streams
/// ignore them): `layer=<nuh_layer_id>` decodes that layer (plus its
/// reference layers) and outputs it alone, `view=<ViewId>` selects the
/// layer carrying that Annex G view, `ols=<idx>` selects an output
/// layer set. Without any, the highest output layer set is decoded and
/// every one of its output layers is emitted — the frames of one access
/// unit come out consecutively in increasing `nuh_layer_id` order (the
/// base view first, then the second view).
///
/// # Errors
/// [`Error::InvalidData`] when the `extradata` or an option fails to
/// parse.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut seq = SequenceDecoder::new();
    let mut nal_length_size = None;
    let parse_u32 = |key: &str| -> Result<Option<u32>> {
        match params.options.get(key) {
            None => Ok(None),
            Some(v) => v.parse::<u32>().map(Some).map_err(|_| {
                Error::InvalidData(format!("h265 decode: {key} must be an integer, got {v:?}"))
            }),
        }
    };
    let (layer, view, ols) = (parse_u32("layer")?, parse_u32("view")?, parse_u32("ols")?);
    let target = match (layer, view, ols) {
        (Some(l), _, _) if l < 64 => LayerTarget::Layer(l as u8),
        (Some(l), _, _) => {
            return Err(Error::InvalidData(format!(
                "h265 decode: layer must be 0..=63, got {l}"
            )))
        }
        (None, Some(v), _) if v <= u32::from(u16::MAX) => LayerTarget::View(v as u16),
        (None, Some(v), _) => {
            return Err(Error::InvalidData(format!(
                "h265 decode: view must be 0..=65535, got {v}"
            )))
        }
        (None, None, Some(o)) => LayerTarget::Ols(o as usize),
        (None, None, None) => LayerTarget::HighestOls,
    };
    seq.set_layer_target(target);
    if !params.extradata.is_empty() {
        if extradata_is_hvcc(&params.extradata) {
            // hvcC record: out-of-band VPS/SPS/PPS (+ SEI) arrays,
            // optionally followed by an lhvC record with the non-base
            // layers' parameter sets.
            let (rec, end) = parse_hvcc_with_len(&params.extradata)
                .map_err(|e| Error::InvalidData(format!("h265 hvcC extradata: {e}")))?;
            for unit in rec.nal_units {
                seq.push_nal_unit(unit)
                    .map_err(|e| Error::InvalidData(format!("h265 hvcC extradata: {e}")))?;
            }
            nal_length_size = Some(rec.length_size);
            let rest = &params.extradata[end..];
            if !rest.is_empty() {
                if !extradata_is_lhvc(rest) {
                    return Err(Error::InvalidData(
                        "h265 hvcC extradata: trailing bytes are not an lhvC record".into(),
                    ));
                }
                let (lrec, _) = parse_lhvc(rest)
                    .map_err(|e| Error::InvalidData(format!("h265 lhvC extradata: {e}")))?;
                for unit in lrec.nal_units {
                    seq.push_nal_unit(unit)
                        .map_err(|e| Error::InvalidData(format!("h265 lhvC extradata: {e}")))?;
                }
            }
        } else {
            // Out-of-band parameter sets in Annex B form.
            seq.push_annexb(&params.extradata)
                .map_err(|e| Error::InvalidData(format!("h265 extradata: {e}")))?;
        }
    }
    Ok(Box::new(H265Decoder {
        codec_id: params.codec_id.clone(),
        seq,
        reorder: Vec::new(),
        ready: VecDeque::new(),
        last_output: None,
        pts_queue: BinaryHeap::new(),
        nal_length_size,
        flushed: false,
    }))
}

impl H265Decoder {
    /// Move decoded pictures into the reorder buffer and emit every
    /// frame that is guaranteed next in output order.
    fn drain(&mut self, flush: bool) {
        self.reorder.extend(self.seq.take_decoded());
        // Output order: POC within a CVS, the layers of one access unit
        // consecutively by increasing nuh_layer_id.
        self.reorder
            .sort_by_key(|f| (f.cvs_index, f.poc, f.layer_id));
        // The reorder bound counts access units; every output layer
        // of a multi-layer stream adds one frame per access unit.
        let layers_out = self.seq.layer_plan().1.len().max(1);
        let depth = if flush {
            0
        } else {
            self.seq
                .max_num_reorder_pics()
                .map_or(DEFAULT_REORDER, |n| n as usize)
                * layers_out
        };
        // Released planes are kept for the next pictures while decoding
        // continues; a flush ends the stream, so they are freed as soon
        // as they are packed.
        let recycle = !flush;
        while self.reorder.len() > depth {
            let f = self.reorder.remove(0);
            if !f.output {
                if recycle {
                    for plane in f.picture.into_shared_planes() {
                        self.seq.recycle_plane(plane);
                    }
                }
                continue;
            }
            let pts = self.pts_queue.pop().map(|r| r.0);
            let layout = frame_layout(&f.picture, &f.crop);
            // §7.4.3.2.1 — output the conformance-cropped picture,
            // packed straight from the decoded planes.
            let frame = video_frame_owned(f.picture, &f.crop, pts, |plane| {
                if recycle {
                    self.seq.recycle_plane(plane);
                }
            });
            self.ready.push_back((Frame::Video(frame), layout));
        }
    }

    /// The [`Layout`] `output_*` report: the frame last returned, or before
    /// the first one the next frame in output order (never parameter sets
    /// no decoded picture uses yet).
    fn reported_layout(&self) -> Option<Layout> {
        self.last_output.or_else(|| {
            self.ready.front().map(|(_, layout)| *layout).or_else(|| {
                self.reorder
                    .iter()
                    .find(|f| f.output)
                    .map(|f| frame_layout(&f.picture, &f.crop))
            })
        })
    }
}

impl Decoder for H265Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            // An empty packet is treated as a flush signal too.
            return self.flush();
        }
        if let Some(pts) = packet.pts {
            self.pts_queue.push(std::cmp::Reverse(pts));
        }
        if let Some(length_size) = self.nal_length_size {
            let units = split_length_prefixed(&packet.data, length_size)
                .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
            for unit in units {
                self.seq
                    .push_nal_unit(unit)
                    .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
            }
        } else {
            self.seq
                .push_annexb(&packet.data)
                .map_err(|e| Error::InvalidData(format!("h265 decode: {e}")))?;
        }
        self.drain(false);
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some((f, layout)) = self.ready.pop_front() {
            self.last_output = Some(layout);
            return Ok(f);
        }
        if self.flushed {
            if !self.reorder.is_empty() {
                self.drain(true);
                if let Some((f, layout)) = self.ready.pop_front() {
                    self.last_output = Some(layout);
                    return Ok(f);
                }
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn output_video_dimensions(&self) -> Option<(u32, u32)> {
        self.reported_layout().and_then(|(dims, _)| dims)
    }

    fn output_pixel_format(&self) -> Option<PixelFormat> {
        self.reported_layout().and_then(|(_, format)| format)
    }

    fn set_execution_context(&mut self, ctx: &oxideav_core::ExecutionContext) {
        // Bounded through the core contract: at most `threads` workers
        // for the per-picture wavefront / row-parallel filters.
        self.seq.set_threads(ctx.threads);
    }

    fn flush(&mut self) -> Result<()> {
        // Decode any pending picture and release the reorder queue.
        self.seq
            .flush()
            .map_err(|e| Error::InvalidData(format!("h265 flush: {e}")))?;
        // End of stream: the reference pictures go, so the frames
        // drained below own their planes outright.
        self.seq.release_references();
        self.flushed = true;
        self.drain(true);
        Ok(())
    }
}

/// The visible size and pixel layout of an output frame: `None` dimensions
/// for an empty cropped window, `None` layout where no [`PixelFormat`]
/// describes the packed planes.
type Layout = (Option<(u32, u32)>, Option<PixelFormat>);

/// The §7.4.3.2.1 conformance window `(x0, y0, width, height)` of `pic`,
/// clamped to the coded picture as [`Picture::cropped`] does.
fn visible_window(pic: &Picture, crop: &CropWindow) -> (usize, usize, usize, usize) {
    let (lw, lh) = pic.plane_dims(Plane::Luma);
    let (x0, y0) = (crop.x0.min(lw), crop.y0.min(lh));
    (x0, y0, crop.width.min(lw - x0), crop.height.min(lh - y0))
}

/// The [`Layout`] of the frame [`video_frame_owned`] packs from `pic`.
fn frame_layout(pic: &Picture, crop: &CropWindow) -> Layout {
    let (_, _, w, h) = visible_window(pic, crop);
    let dims = (w > 0 && h > 0)
        .then(|| Some((u32::try_from(w).ok()?, u32::try_from(h).ok()?)))
        .flatten();
    (dims, pixel_format(pic))
}

/// The [`PixelFormat`] of [`video_frame_owned`]'s planes: one byte per
/// 8-bit sample, else little-endian 16-bit words. `None` for depths no
/// format names (9, 11, 13–15 bits) and for luma and chroma of different
/// depths.
fn pixel_format(pic: &Picture) -> Option<PixelFormat> {
    use PixelFormat::*;
    let cat = pic.chroma_array_type();
    let depth = pic.bit_depth(Plane::Luma);
    if cat != 0 && pic.bit_depth(Plane::Cb) != depth {
        return None;
    }
    Some(match (cat, depth) {
        (0, 8) => Gray8,
        (0, 10) => Gray10Le,
        (0, 12) => Gray12Le,
        (0, 16) => Gray16Le,
        (1, 8) => Yuv420P,
        (1, 10) => Yuv420P10Le,
        (1, 12) => Yuv420P12Le,
        (1, 16) => Yuv420P16Le,
        (2, 8) => Yuv422P,
        (2, 10) => Yuv422P10Le,
        (2, 12) => Yuv422P12Le,
        (2, 16) => Yuv422P16Le,
        (3, 8) => Yuv444P,
        (3, 10) => Yuv444P10Le,
        (3, 12) => Yuv444P12Le,
        (3, 16) => Yuv444P16Le,
        _ => return None,
    })
}

/// Pack the §7.4.3.2.1 conformance-cropped window of a reconstructed
/// [`Picture`] into a [`VideoFrame`] (8-bit planes as one byte per
/// sample; higher bit depths as little-endian 16-bit, the planar
/// `p010le`-family layout). The window is clamped to the coded picture
/// as [`Picture::cropped`] does and read straight from the decoded
/// planes. Each plane buffer is handed to `release` as soon as it is
/// packed.
fn video_frame_owned(
    pic: Picture,
    crop: &CropWindow,
    pts: Option<i64>,
    mut release: impl FnMut(std::sync::Arc<Vec<u16>>),
) -> VideoFrame {
    let chroma = pic.chroma_array_type() != 0;
    let wide = pic.bit_depth(Plane::Luma) > 8 || (chroma && pic.bit_depth(Plane::Cb) > 8);
    let (lw, _) = pic.plane_dims(Plane::Luma);
    let (x0, y0, w, h) = visible_window(&pic, crop);
    // (x, y, width, height, plane stride) per packed plane.
    let mut windows = vec![(x0, y0, w, h, lw)];
    if chroma {
        let (sw, sh) = sub_wh_c(pic.chroma_array_type());
        let window = (
            x0 / sw,
            y0 / sh,
            w / sw,
            h / sh,
            pic.plane_dims(Plane::Cb).0,
        );
        windows.extend([window, window]);
    }
    let mut out = Vec::with_capacity(windows.len());
    for (buf, (x, y, w, h, stride)) in pic.into_shared_planes().into_iter().zip(windows) {
        let rows = h.min((buf.len() / stride.max(1)).saturating_sub(y));
        let row = |r: usize| &buf[(y + r) * stride + x..(y + r) * stride + x + w];
        let (stride, data) = if w == 0 {
            (0, Vec::new())
        } else if wide {
            // Filled in place of a zeroed buffer: every byte is written once.
            let mut data = Vec::with_capacity(w * rows * 2);
            for r in 0..rows {
                let mut src8 = row(r).chunks_exact(8);
                for s in &mut src8 {
                    let s: &[u16; 8] = s.try_into().unwrap();
                    let mut bytes = [0u8; 16];
                    for (pair, &v) in bytes.chunks_exact_mut(2).zip(s) {
                        pair.copy_from_slice(&v.to_le_bytes());
                    }
                    data.extend_from_slice(&bytes);
                }
                for &v in src8.remainder() {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
            (w * 2, data)
        } else {
            let mut data = Vec::with_capacity(w * rows);
            for r in 0..rows {
                data.extend(row(r).iter().map(|&v| v as u8));
            }
            (w, data)
        };
        release(buf);
        out.push(VideoPlane { stride, data });
    }
    VideoFrame { pts, planes: out }
}
