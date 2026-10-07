//! Architecture kernels behind portable ones.
//!
//! Every kernel here mirrors a portable form elsewhere in the crate, which
//! stays the reference. On aarch64 the NEON forms run by default and
//! [`set_enabled`]`(false)` selects the portable ones at run time; the two
//! are compared bit for bit by `tests/simd_differential.rs` and this
//! module's tests. Off aarch64 only the portable forms exist. All of the
//! crate's `unsafe` code is confined to the `aarch64` submodule.

use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod aarch64;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Run the architecture kernels (`true`, the default) or the portable
/// ones. No effect off aarch64.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// `true` when the architecture kernels run.
#[must_use]
#[inline]
pub fn enabled() -> bool {
    cfg!(target_arch = "aarch64") && ENABLED.load(Ordering::Relaxed)
}

/// 16-bit filter inputs: samples of at most 15 bits (`u16`, read as the
/// same `i16` values) or `i16` rows.
pub(crate) trait In16: Copy {}

impl In16 for u16 {}

impl In16 for i16 {}

/// The output lanes of [`interp_pass`].
#[cfg(target_arch = "aarch64")]
pub(crate) trait PassOut: aarch64::NeonOut {}

/// The output lanes of [`interp_pass`].
#[cfg(not(target_arch = "aarch64"))]
pub(crate) trait PassOut: Copy {}

impl PassOut for i16 {}

impl PassOut for i32 {}

/// One separable interpolation filter pass (the portable form is
/// `inter_pred::filter_block` over 16-bit inputs): `dst[ y·w + x ] = put(
/// ( Σ_t k[ t ] · src[ y·src_stride + t·step + x ] ) >> shift )` for every
/// `x < w` and row `y < dst.len() / w`, where `put` truncates to the
/// output lanes. `acc16` accumulates in `i16` lanes, exact only when every
/// partial tap sum fits them (8-bit samples, `shift == 0`); otherwise the
/// sums are `i32`. Returns `false`, writing nothing, when the
/// architecture kernel does not run (disabled, another target, a width
/// that is not a multiple of 4, or an input past the end of `src`); the
/// caller then runs the portable pass.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn interp_pass<S: In16, D: PassOut, const TAPS: usize>(
    src: &[S],
    src_stride: usize,
    step: usize,
    k: &[i16; TAPS],
    acc16: bool,
    shift: u32,
    w: usize,
    dst: &mut [D],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    if enabled() {
        return aarch64::filter16(src, src_stride, step, k, acc16, shift, w, dst);
    }
    let _ = (src, src_stride, step, k, acc16, shift, w, dst);
    false
}

/// The architecture form of the `i16`-lane SAO edge-offset span (samples
/// of at most 14 bits, offsets of magnitude at most 1023; the portable
/// form is `sao::edge_span`): `span[ i ] = Clip3( 0, max, cur[ i ] +
/// by_edge[ 2 + Sign( cur[ i ] − a[ i ] ) + Sign( cur[ i ] − b[ i ] ) ] )`.
/// Returns how many leading samples it wrote: all of `span`, or none when
/// the kernel does not run (spans shorter than 8 included); the caller
/// finishes the rest.
#[inline]
pub(crate) fn sao_edge(
    span: &mut [u16],
    cur: &[u16],
    a: &[u16],
    b: &[u16],
    by_edge: [i32; 5],
    max: i32,
) -> usize {
    #[cfg(target_arch = "aarch64")]
    if enabled() {
        return aarch64::sao_edge(span, cur, a, b, by_edge, max);
    }
    let _ = (span, cur, a, b, by_edge, max);
    0
}

/// The architecture form of the `i16`-lane SAO band-offset span (the
/// portable form is `sao::band_span`): `span[ i ] = Clip3( 0, max, cur[ i ]
/// + offset )` with the offset of band `cur[ i ] >> band_shift` — `off[ k
/// + 1 ]` for band `bands[ k ]`, else `off[ 0 ]`. Returns how many leading
/// samples it wrote, as [`sao_edge`].
#[inline]
pub(crate) fn sao_band(
    span: &mut [u16],
    cur: &[u16],
    band_shift: u32,
    bands: [u32; 4],
    off: &[i32; 5],
    max: i32,
) -> usize {
    #[cfg(target_arch = "aarch64")]
    if enabled() {
        return aarch64::sao_band(span, cur, band_shift, bands, off, max);
    }
    let _ = (span, cur, band_shift, bands, off, max);
    0
}

/// The architecture form of one row of the §8.5.3.3.4.2 default weighted
/// sample prediction (the portable form is `PlanePrediction::row`): `out[
/// x ] = Clip3( 0, max, ( p0[ x ] + p1[ x ] + offset ) >> shift )`, or
/// without `p1` for one list. Returns `false`, writing nothing, when the
/// kernel does not run (disabled, another target, or fewer than 4
/// samples); the caller then runs the portable form.
#[inline]
pub(crate) fn pred_default_row(
    out: &mut [u16],
    p0: &[i32],
    p1: Option<&[i32]>,
    shift: i32,
    offset: i32,
    max: i32,
) -> bool {
    #[cfg(target_arch = "aarch64")]
    if enabled() {
        return aarch64::pred_default_row(out, p0, p1, shift, offset, max);
    }
    let _ = (out, p0, p1, shift, offset, max);
    false
}

/// The architecture form of the §8.7.2.5.7 luma filtering of one 4-line
/// deblocking edge segment (the portable form is `deblock`'s
/// `filter_luma_sample` over the four lines) with its §8.7.2.5.3 decision
/// `(dE, dEp, dEq)` (`dE` 1 or 2) and `tc`, no sample suppressed. `q00` is
/// the index of `q0,0` in `samples` (row stride `stride`); the lines run
/// down the rows of a vertical edge (`p_i,k` at `q00 + k·stride − 1 − i`,
/// `q_i,k` at `q00 + k·stride + i`) and along the columns of a horizontal
/// one (`p_i,k` at `q00 + k − ( 1 + i )·stride`). `max` is `( 1 <<
/// BitDepthY ) − 1`. Returns `false`, writing nothing, when the kernel
/// does not run; the caller then runs the portable filter.
#[inline]
pub(crate) fn luma_edge(
    samples: &mut [u16],
    q00: usize,
    stride: usize,
    vertical: bool,
    decision: (u8, u8, u8),
    tc: i32,
    max: i32,
) -> bool {
    #[cfg(target_arch = "aarch64")]
    if enabled() {
        return aarch64::luma_edge(samples, q00, stride, vertical, decision, tc, max);
    }
    let _ = (samples, q00, stride, vertical, decision, tc, max);
    false
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::inter_pred::{interp_chroma_block, interp_luma_block, RefPlane};
    use std::sync::Mutex;

    /// Serializes the tests that flip the process-wide switch.
    static SWITCH: Mutex<()> = Mutex::new(());

    /// `f` with the portable kernels, then with the architecture ones.
    fn both<R>(f: impl Fn() -> R) -> (R, R) {
        let _guard = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
        set_enabled(false);
        let portable = f();
        set_enabled(true);
        let arch = f();
        (portable, arch)
    }

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    /// Every phase pair, bit depths 8..=16, widths on and off the four
    /// lanes, and windows inside and across the plane edges.
    #[test]
    fn interpolation_matches_portable_on_random_blocks() {
        let mut rng = Lcg(0x4856_4343);
        for case in 0..4000 {
            let bit_depth = [8u8, 8, 10, 10, 12, 9, 11, 16][rng.below(8) as usize];
            let (pw, ph) = (8 + rng.below(72) as usize, 8 + rng.below(56) as usize);
            let samples: Vec<u16> = (0..pw * ph)
                .map(|_| rng.below(1 << bit_depth) as u16)
                .collect();
            let plane = RefPlane::new(&samples, pw, ph).unwrap();
            let w = [2usize, 4, 6, 8, 12, 16, 24, 32, 48, 64][rng.below(10) as usize];
            let h = [2usize, 4, 8, 12, 16, 32, 64][rng.below(7) as usize];
            let x = rng.below(pw as u64 + 24) as i32 - 12;
            let y = rng.below(ph as u64 + 24) as i32 - 12;
            let luma = rng.below(2) == 0;
            let max_frac = if luma { 4 } else { 8 };
            let (xf, yf) = (rng.below(max_frac) as i32, rng.below(max_frac) as i32);
            let (portable, arch) = both(|| {
                if luma {
                    interp_luma_block(&plane, x, y, xf, yf, w, h, bit_depth)
                } else {
                    interp_chroma_block(&plane, x, y, xf, yf, w, h, bit_depth)
                }
                .unwrap()
            });
            assert_eq!(
                portable, arch,
                "case {case}: luma {luma} {w}x{h} at ({x},{y}) phase ({xf},{yf}) \
                 {bit_depth}-bit on {pw}x{ph}"
            );
        }
    }

    /// Whole-picture SAO with every type, EO class and band position,
    /// bit depths 8..=16 (the `i16` and `i32` lanes) and offsets within
    /// and beyond the `i16` range, on picture sizes that leave partial
    /// CTBs and spans of every length.
    #[test]
    fn sao_matches_portable_on_random_pictures() {
        use crate::picture::{Picture, Plane};
        use crate::sao::{apply_sao_picture_in_place, ResolvedSao, ResolvedSaoComponent};
        let mut rng = Lcg(0x5341_4f21);
        for case in 0..300 {
            let bit_depth = [8u8, 10, 12, 14, 16, 9][rng.below(6) as usize];
            let cat = [1u8, 2, 3][rng.below(3) as usize];
            let ctb_log2 = 4 + rng.below(3) as u32;
            let (w, h) = (
                8 * (1 + rng.below(24) as usize),
                8 * (1 + rng.below(16) as usize),
            );
            let mut pic = Picture::new(w, h, cat, bit_depth, bit_depth);
            for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
                let (pw, ph) = pic.plane_dims(plane);
                for y in 0..ph {
                    for x in 0..pw {
                        let v = rng.below(1 << bit_depth) as i32;
                        pic.set_sample(plane, x, y, v);
                    }
                }
            }
            let ctbs = w.div_ceil(1 << ctb_log2) * h.div_ceil(1 << ctb_log2);
            let wide_offsets = rng.below(4) == 0;
            let ctb_sao: Vec<ResolvedSao> = (0..ctbs)
                .map(|_| ResolvedSao {
                    components: core::array::from_fn(|_| {
                        let reach = if wide_offsets { 4000 } else { 1024 };
                        let mut offset_val = [0i32; 5];
                        for o in &mut offset_val[1..] {
                            *o = rng.below(2 * reach - 1) as i32 - (reach as i32 - 1);
                        }
                        ResolvedSaoComponent {
                            sao_type_idx: rng.below(3) as u8,
                            offset_val,
                            band_position: rng.below(32) as u8,
                            eo_class: rng.below(4) as u8,
                        }
                    }),
                })
                .collect();
            let chroma = rng.below(2) == 0;
            let (portable, arch) = both(|| {
                let mut p = pic.clone();
                apply_sao_picture_in_place(
                    &mut p, &ctb_sao, ctb_log2, cat, true, chroma, None, None,
                );
                [Plane::Luma, Plane::Cb, Plane::Cr].map(|plane| p.plane(plane).to_vec())
            });
            assert!(
                portable == arch,
                "case {case}: {w}x{h} cat {cat} ctb {ctb_log2} {bit_depth}-bit"
            );
        }
    }

    /// Motion-compensated prediction units written into the picture (the
    /// default combine, uni and bi, through the interpolation kernels):
    /// random PU geometry, motion and samples at every bit depth and
    /// chroma format.
    #[test]
    fn inter_prediction_matches_portable_on_random_units() {
        use crate::picture::{sub_wh_c, Picture, Plane};
        use crate::recon::{reconstruct_inter_pu_weighted, ReconParams, ResolvedList};
        let mut rng = Lcg(0x5055_2121);
        for case in 0..1500 {
            let bit_depth = [8u8, 8, 10, 10, 12, 9, 16][rng.below(7) as usize];
            let cat = [0u8, 1, 1, 2, 3][rng.below(5) as usize];
            let (pw, ph) = (64usize, 64usize);
            let random_picture = |rng: &mut Lcg| {
                let mut p = Picture::new(pw, ph, cat, bit_depth, bit_depth);
                for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
                    if plane != Plane::Luma && cat == 0 {
                        continue;
                    }
                    let (w, h) = p.plane_dims(plane);
                    for y in 0..h {
                        for x in 0..w {
                            p.set_sample(plane, x, y, rng.below(1 << bit_depth) as i32);
                        }
                    }
                }
                p
            };
            let refs = [random_picture(&mut rng), random_picture(&mut rng)];
            let (w, h) = (
                [4usize, 8, 12, 16, 24, 32, 48, 64][rng.below(8) as usize],
                [4usize, 8, 12, 16, 24, 32, 64][rng.below(7) as usize],
            );
            let (x_pb, y_pb) = (
                rng.below((pw - w) as u64 / 4 + 1) as usize * 4,
                rng.below((ph - h) as u64 / 4 + 1) as usize * 4,
            );
            let (sw, sh) = if cat == 0 { (1, 1) } else { sub_wh_c(cat) };
            fn list<'p>(
                rng: &mut Lcg,
                pic: &'p Picture,
                used: bool,
                sub: (usize, usize),
            ) -> ResolvedList<'p> {
                let mv = [rng.below(161) as i32 - 80, rng.below(161) as i32 - 80];
                ResolvedList {
                    pred_flag: used,
                    mv_l: mv,
                    mv_c: crate::motion::derive_chroma_mv(mv, sub.0 as i32, sub.1 as i32),
                    ref_pic: pic,
                }
            }
            let kind = rng.below(3);
            let l0 = list(&mut rng, &refs[0], kind != 1, (sw, sh));
            let l1 = list(&mut rng, &refs[1], kind != 0, (sw, sh));
            let params = ReconParams {
                chroma_array_type: cat,
                bit_depth_luma: bit_depth,
                bit_depth_chroma: bit_depth,
                intra_smoothing_disabled: false,
                strong_intra_smoothing_enabled: false,
                slice_qp_y: 26,
                cb_qp_offset: 0,
                cr_qp_offset: 0,
                act_y_qp_offset: 0,
                act_cb_qp_offset: 0,
                act_cr_qp_offset: 0,
                transform_skip_rotation_enabled: false,
                implicit_rdpcm_enabled: false,
                intra_boundary_filtering_disabled: false,
                extended_precision: false,
                scaling: None,
                chroma_qp_offset_list: Vec::new(),
                cu_qp_offset_c: core::cell::Cell::new((0, 0)),
            };
            let (portable, arch) = both(|| {
                let mut pic = Picture::new(pw, ph, cat, bit_depth, bit_depth);
                reconstruct_inter_pu_weighted(
                    &mut pic, &params, x_pb, y_pb, w, h, l0, l1, None, None, None, None,
                )
                .unwrap();
                [Plane::Luma, Plane::Cb, Plane::Cr].map(|plane| {
                    if plane != Plane::Luma && cat == 0 {
                        Vec::new()
                    } else {
                        pic.plane(plane).to_vec()
                    }
                })
            });
            assert!(
                portable == arch,
                "case {case}: {w}x{h} at ({x_pb},{y_pb}) cat {cat} {bit_depth}-bit kind {kind}"
            );
        }
    }

    /// Luma deblocking of random edge segments, vertical and horizontal:
    /// smooth sides with a step at the edge and noise of every size reach
    /// the strong, weak and unfiltered decisions at every bit depth.
    #[test]
    fn luma_deblocking_matches_portable_on_random_segments() {
        use crate::deblock::{
            filter_luma_block_edge_gated, EdgePos, EdgeQp, EdgeType, SamplePlane,
        };
        let mut rng = Lcg(0x4442_4c4b);
        let mut decisions = [0usize; 3];
        for case in 0..20000 {
            let bit_depth = [8u8, 8, 10, 10, 12, 9, 16][rng.below(7) as usize];
            let max = (1i64 << bit_depth) - 1;
            let (w, h) = (24usize, 24usize);
            let stride = w + rng.below(3) as usize * 4;
            let base = rng.below(max as u64 + 1) as i64;
            let (step_shift, noise_shift) = (
                [2u32, 4, 6, 8][rng.below(4) as usize],
                [4u32, 6, 8, 12][rng.below(4) as usize],
            );
            let step = rng.below(1 + (max as u64 >> step_shift)) as i64;
            let noise = rng.below(1 + (max as u64 >> noise_shift).max(1)) as i64;
            let vertical = rng.below(2) == 0;
            let (ex, ey) = (8usize, 8usize);
            let mut samples = vec![0u16; stride * h];
            for y in 0..h {
                for x in 0..w {
                    let q_side = if vertical { x >= ex } else { y >= ey };
                    let v =
                        base + if q_side { step } else { 0 } + rng.below(noise as u64 + 1) as i64;
                    samples[y * stride + x] = v.clamp(0, max) as u16;
                }
            }
            let qp = EdgeQp {
                qp_q: rng.below(52) as i32,
                qp_p: rng.below(52) as i32,
                beta_offset_div2: rng.below(13) as i32 - 6,
                tc_offset_div2: rng.below(13) as i32 - 6,
                bit_depth,
            };
            let bs = 1 + rng.below(2) as u8;
            let pos = EdgePos {
                ex,
                ey,
                edge: if vertical {
                    EdgeType::Vertical
                } else {
                    EdgeType::Horizontal
                },
            };
            let (portable, arch) = both(|| {
                let mut buf = samples.clone();
                let mut plane = SamplePlane {
                    samples: &mut buf,
                    width: w,
                    stride,
                    y_origin: 0,
                };
                let dec = filter_luma_block_edge_gated(&mut plane, pos, bs, qp, None);
                (dec, buf)
            });
            decisions[usize::from(portable.0.de)] += 1;
            assert!(
                portable == arch,
                "case {case}: {bit_depth}-bit vertical {vertical} bs {bs} {qp:?} decision {:?}",
                portable.0
            );
        }
        // Every decision is exercised.
        assert!(decisions.iter().all(|&n| n > 500), "{decisions:?}");
    }
}
