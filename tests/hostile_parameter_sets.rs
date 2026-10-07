//! Parameter sets that contradict the pictures a stream still references,
//! or exceed their bit-depth bounds, are decode errors with either kernel
//! set — never panics or silently wrong samples — while the same streams
//! with valid parameter sets decode exactly as the unmodified ones (whose
//! complete output `availability_recovery.rs` checks against FFmpeg).

mod fixture_bytes;

use fixture_bytes::{I_THEN_P_HEVC, MAIN10_HEVC, QP_HIGH_HEVC};
use oxideav_h265::encoder::bitwriter::BitWriter;
use oxideav_h265::encoder::nal::{annexb, nal_unit};
use oxideav_h265::nal::{collect_nal_units, NalUnit};
use oxideav_h265::pps::PicParameterSet;
use oxideav_h265::SequenceDecoder;
use std::sync::Mutex;

const VPS: u8 = 32;
const SPS: u8 = 33;
const PPS: u8 = 34;
const TRAIL_R: u8 = 1;
const IDR_W_RADL: u8 = 19;
const IDR_N_LP: u8 = 20;

/// Serializes the decodes that flip the process-wide kernel switch.
static SWITCH: Mutex<()> = Mutex::new(());

/// Every output picture of `stream` (as little-endian 16-bit planes), or
/// its first error, decoded with the portable kernels and then with the
/// architecture ones.
fn decode_both(stream: &[u8]) -> [Result<Vec<Vec<u8>>, String>; 2] {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            oxideav_h265::simd::set_enabled(true);
        }
    }
    let _guard = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = Restore;
    [false, true].map(|arch| {
        oxideav_h265::simd::set_enabled(arch);
        let mut decoder = SequenceDecoder::new();
        decoder
            .push_annexb(stream)
            .and_then(|()| decoder.finish())
            .map(|frames| {
                frames
                    .iter()
                    .map(|f| f.output_picture().to_planar_le16())
                    .collect()
            })
            .map_err(|e| e.to_string())
    })
}

fn units(stream: &[u8]) -> Vec<NalUnit> {
    collect_nal_units(stream).expect("fixture NAL units")
}

/// The coded NAL unit `u` with its RBSP replaced by `rbsp`.
fn coded(u: &NalUnit, rbsp: &[u8]) -> Vec<u8> {
    nal_unit(
        u.header.nal_unit_type,
        u.header.nuh_layer_id,
        u.header.temporal_id,
        rbsp,
    )
}

/// The first NAL unit of `units` of each type in `types`, in that order.
fn pick(units: &[NalUnit], types: &[u8]) -> Vec<Vec<u8>> {
    types
        .iter()
        .map(|&t| {
            let u = units
                .iter()
                .find(|u| u.header.nal_unit_type == t)
                .unwrap_or_else(|| panic!("fixture NAL unit type {t}"));
            coded(u, &u.rbsp)
        })
        .collect()
}

/// The IDR of `first`, then the SPS, PPS and P slice of the 64×64 8-bit
/// `I_THEN_P_HEVC`: its SPS (same id) follows `first`'s, and the P slice's
/// RPS names POC 0, the picture coded under `first`'s SPS.
fn retained_across_an_sps_change(first: &[u8]) -> Vec<u8> {
    let first = units(first);
    let idr = first
        .iter()
        .find(|u| matches!(u.header.nal_unit_type, IDR_W_RADL | IDR_N_LP))
        .map(|u| u.header.nal_unit_type)
        .expect("fixture IDR");
    let mut stream = pick(&first, &[VPS, SPS, PPS, idr]);
    stream.extend(pick(&units(I_THEN_P_HEVC), &[SPS, PPS, TRAIL_R]));
    annexb(&stream)
}

/// As in FFmpeg, activating a changed SPS clears the layer's references
/// (hevcdec.c `hevc_frame_start`, `ff_hevc_clear_refs`), so the P picture
/// whose RPS names the old picture finds no reference and fails (refs.c
/// `add_candidate_ref`: "Could not find ref with POC 0"; FFmpeg 9.0.2
/// outputs the IDR and drops the P picture of both streams). Neither
/// reaches prediction from the stale picture.
fn assert_reference_missing(stream: &[u8], what: &str) {
    for result in decode_both(stream) {
        let error = result.expect_err(what);
        assert!(
            error.contains("reference picture is missing"),
            "{what}: {error}"
        );
    }
}

/// A 10-bit picture kept across an 8-bit SPS: predicting from it read its
/// samples through 8-bit lanes before the format guard.
#[test]
fn a_reference_retained_across_an_sps_of_another_bit_depth_is_a_decode_error() {
    assert_reference_missing(
        &retained_across_an_sps_change(MAIN10_HEVC),
        "an 8-bit P picture naming a 10-bit picture",
    );
}

/// A 32×32 picture of the same format kept across a 64×64 SPS: the format
/// guard admits it, and predicting from it read a picture of another size.
#[test]
fn a_reference_retained_across_a_changed_sps_of_another_size_is_a_decode_error() {
    assert_reference_missing(
        &retained_across_an_sps_change(QP_HIGH_HEVC),
        "a 64x64 P picture naming a 32x32 picture",
    );
}

/// Re-sending the active SPS and PPS unchanged before a P slice keeps the
/// references (FFmpeg keeps a byte-identical SPS, ps.c `compare_sps`):
/// the stream decodes exactly as the fixture.
#[test]
fn an_identical_sps_resend_keeps_the_references() {
    let [expected, _] = decode_both(I_THEN_P_HEVC);
    let expected = expected.expect("the unmodified fixture decodes");
    assert_eq!(expected.len(), 2);
    for result in decode_both(&retained_across_an_sps_change(I_THEN_P_HEVC)) {
        assert_eq!(result.expect("a re-sent, unchanged SPS"), expected);
    }
}

/// `stream` with each PPS extended by a range extension that signals the
/// SAO offset scales `(luma, chroma)` and changes nothing else (the
/// fixture PPS ends with `pps_extension_present_flag == 0`).
fn with_sao_offset_scales(stream: &[u8], (luma, chroma): (u32, u32)) -> Vec<u8> {
    let rebuilt: Vec<_> = units(stream)
        .iter()
        .map(|u| {
            if u.header.nal_unit_type != PPS {
                return coded(u, &u.rbsp);
            }
            let rbsp = &u.rbsp;
            let pps = PicParameterSet::parse(rbsp).expect("fixture PPS");
            assert!(pps.pps_range_extension.is_none());
            let last = rbsp
                .iter()
                .rposition(|&b| b != 0)
                .expect("rbsp_stop_one_bit");
            let stop = last * 8 + 7 - rbsp[last].trailing_zeros() as usize;
            let bit = |i: usize| (rbsp[i / 8] >> (7 - i % 8)) & 1;
            assert_eq!(bit(stop - 1), 0, "pps_extension_present_flag");
            let mut w = BitWriter::new();
            for i in 0..stop - 1 {
                w.put_bit(bit(i));
            }
            w.put_bit(1); // pps_extension_present_flag
            w.put_bit(1); // pps_range_extension_flag
            w.put_bits(0, 3); // pps_multilayer / 3d / scc_extension_flag
            w.put_bits(0, 4); // pps_extension_4bits
            if pps.transform_skip_enabled_flag {
                w.ue(0); // log2_max_transform_skip_block_size_minus2
            }
            w.put_bit(0); // cross_component_prediction_enabled_flag
            w.put_bit(0); // chroma_qp_offset_list_enabled_flag
            w.ue(luma); // log2_sao_offset_scale_luma
            w.ue(chroma); // log2_sao_offset_scale_chroma
            w.rbsp_trailing_bits();
            let extended = w.finish();
            let range = PicParameterSet::parse(&extended)
                .expect("extended PPS")
                .pps_range_extension
                .expect("range extension");
            assert_eq!(
                (
                    range.log2_sao_offset_scale_luma,
                    range.log2_sao_offset_scale_chroma
                ),
                (luma, chroma)
            );
            coded(u, &extended)
        })
        .collect();
    annexb(&rebuilt)
}

/// §7.4.3.3.2 bounds `log2_sao_offset_scale_luma` / `_chroma` by `Max( 0,
/// BitDepth − 10 )`, 0 for the 8-bit fixture (which signals SAO). A scale
/// of 31 would make an offset of 1 `i32::MIN`.
#[test]
fn sao_offset_scales_beyond_the_bit_depth_bound_are_decode_errors() {
    for scales in [(1, 0), (0, 1), (31, 0), (0, 31)] {
        for result in decode_both(&with_sao_offset_scales(I_THEN_P_HEVC, scales)) {
            let error = result.expect_err("an out-of-range SAO offset scale");
            assert!(
                error.contains("log2_sao_offset_scale"),
                "{scales:?}: {error}"
            );
        }
    }
    let [expected, _] = decode_both(I_THEN_P_HEVC);
    let expected = expected.expect("the unmodified fixture decodes");
    for result in decode_both(&with_sao_offset_scales(I_THEN_P_HEVC, (0, 0))) {
        assert_eq!(result.expect("in-range SAO offset scales"), expected);
    }
}
