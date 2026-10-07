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
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result, VideoFrame, VideoPlane,
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
    /// Frames ready to hand out.
    ready: VecDeque<Frame>,
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
            // §7.4.3.2.1 — output the conformance-cropped picture,
            // packed straight from the decoded planes.
            let frame = video_frame_owned(f.picture, &f.crop, pts, |plane| {
                if recycle {
                    self.seq.recycle_plane(plane);
                }
            });
            self.ready.push_back(Frame::Video(frame));
        }
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
        if let Some(f) = self.ready.pop_front() {
            return Ok(f);
        }
        if self.flushed {
            if !self.reorder.is_empty() {
                self.drain(true);
                if let Some(f) = self.ready.pop_front() {
                    return Ok(f);
                }
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
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
    let (lw, lh) = pic.plane_dims(Plane::Luma);
    let (x0, y0) = (crop.x0.min(lw), crop.y0.min(lh));
    let (w, h) = (crop.width.min(lw - x0), crop.height.min(lh - y0));
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
            let mut data = vec![0u8; w * rows * 2];
            for (r, dst) in data.chunks_exact_mut(w * 2).enumerate() {
                let src = row(r);
                let mut dst8 = dst.chunks_exact_mut(16);
                let mut src8 = src.chunks_exact(8);
                for (d, s) in (&mut dst8).zip(&mut src8) {
                    let d: &mut [u8; 16] = d.try_into().unwrap();
                    let s: &[u16; 8] = s.try_into().unwrap();
                    for (pair, &v) in d.chunks_exact_mut(2).zip(s) {
                        pair.copy_from_slice(&v.to_le_bytes());
                    }
                }
                for (d, &v) in dst8
                    .into_remainder()
                    .chunks_exact_mut(2)
                    .zip(src8.remainder())
                {
                    d.copy_from_slice(&v.to_le_bytes());
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
