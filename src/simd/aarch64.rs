//! NEON forms of the portable kernels ([`super`]). Each public function
//! checks every slice bound its loads and stores need before touching a
//! pointer, so the accesses inside each `unsafe` block stay within those
//! checked ranges.

use super::In16;
use core::arch::aarch64::{
    int16x4_t, int16x8_t, int32x4_t, uint16x4_t, uint16x8_t, uint8x16_t, uint8x16x4_t, vabsq_s32,
    vaddq_s16, vaddq_s32, vaddq_u16, vbslq_s32, vcgtq_s16, vcltq_s16, vcltq_s32, vcombine_s16,
    vcombine_u16, vdup_n_s16, vdup_n_u16, vdupq_n_s16, vdupq_n_s32, vdupq_n_u16, vget_high_u16,
    vget_low_s16, vget_low_u16, vld1_s16, vld1_u16, vld1q_s16, vld1q_s32, vld1q_u16, vld1q_u8,
    vmaxq_s16, vmaxq_s32, vmin_u16, vminq_s16, vminq_s32, vmla_n_s16, vmlal_high_n_s16,
    vmlal_n_s16, vmlaq_n_s16, vmovl_high_s16, vmovl_s16, vmovl_u16, vmovn_s32, vmovn_u32,
    vmulq_n_s32, vorrq_u16, vqmovun_s32, vqtbl1q_u8, vqtbl4q_u8, vreinterpretq_s16_u16,
    vreinterpretq_s16_u8, vreinterpretq_s32_u32, vreinterpretq_u16_s16, vreinterpretq_u16_u32,
    vreinterpretq_u32_s32, vreinterpretq_u32_u16, vreinterpretq_u8_u16, vshlq_n_u16, vshlq_s32,
    vshlq_u16, vshrq_n_s32, vst1_s16, vst1_u16, vst1q_s16, vst1q_s32, vst1q_u16, vsubq_s16,
    vsubq_s32, vtrn1q_u16, vtrn1q_u32, vtrn2q_u16, vtrn2q_u32,
};

/// Storing [`filter16`] results into `i16` (truncated) or `i32` lanes.
pub(crate) trait NeonOut: Copy {
    /// Store eight `i32` results, `lo` then `hi`.
    ///
    /// # Safety
    /// `dst .. dst + 8` must be writable.
    unsafe fn put8(dst: *mut Self, lo: int32x4_t, hi: int32x4_t);
    /// Store four `i32` results.
    ///
    /// # Safety
    /// `dst .. dst + 4` must be writable.
    unsafe fn put4(dst: *mut Self, v: int32x4_t);
    /// Store eight `i16` results.
    ///
    /// # Safety
    /// `dst .. dst + 8` must be writable.
    unsafe fn put8_16(dst: *mut Self, v: int16x8_t);
    /// Store four `i16` results.
    ///
    /// # Safety
    /// `dst .. dst + 4` must be writable.
    unsafe fn put4_16(dst: *mut Self, v: int16x4_t);
}

impl NeonOut for i16 {
    #[inline(always)]
    unsafe fn put8(dst: *mut i16, lo: int32x4_t, hi: int32x4_t) {
        // SAFETY: the caller guarantees eight writable lanes.
        unsafe { vst1q_s16(dst, vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))) }
    }
    #[inline(always)]
    unsafe fn put4(dst: *mut i16, v: int32x4_t) {
        // SAFETY: the caller guarantees four writable lanes.
        unsafe { vst1_s16(dst, vmovn_s32(v)) }
    }
    #[inline(always)]
    unsafe fn put8_16(dst: *mut i16, v: int16x8_t) {
        // SAFETY: the caller guarantees eight writable lanes.
        unsafe { vst1q_s16(dst, v) }
    }
    #[inline(always)]
    unsafe fn put4_16(dst: *mut i16, v: int16x4_t) {
        // SAFETY: the caller guarantees four writable lanes.
        unsafe { vst1_s16(dst, v) }
    }
}

impl NeonOut for i32 {
    #[inline(always)]
    unsafe fn put8(dst: *mut i32, lo: int32x4_t, hi: int32x4_t) {
        // SAFETY: the caller guarantees eight writable lanes.
        unsafe {
            vst1q_s32(dst, lo);
            vst1q_s32(dst.add(4), hi);
        }
    }
    #[inline(always)]
    unsafe fn put4(dst: *mut i32, v: int32x4_t) {
        // SAFETY: the caller guarantees four writable lanes.
        unsafe { vst1q_s32(dst, v) }
    }
    #[inline(always)]
    unsafe fn put8_16(dst: *mut i32, v: int16x8_t) {
        // SAFETY: the caller guarantees eight writable lanes.
        unsafe {
            vst1q_s32(dst, vmovl_s16(vget_low_s16(v)));
            vst1q_s32(dst.add(4), vmovl_high_s16(v));
        }
    }
    #[inline(always)]
    unsafe fn put4_16(dst: *mut i32, v: int16x4_t) {
        // SAFETY: the caller guarantees four writable lanes.
        unsafe { vst1q_s32(dst, vmovl_s16(v)) }
    }
}

/// [`super::interp_pass`] in NEON: eight outputs at a time, then four.
#[allow(clippy::too_many_arguments)]
pub(super) fn filter16<S: In16, D: NeonOut, const TAPS: usize>(
    src: &[S],
    src_stride: usize,
    step: usize,
    k: &[i16; TAPS],
    acc16: bool,
    shift: u32,
    w: usize,
    dst: &mut [D],
) -> bool {
    debug_assert!(!acc16 || shift == 0);
    const {
        assert!(core::mem::size_of::<S>() == 2 && core::mem::align_of::<S>() == 2);
    }
    if w == 0 || w % 4 != 0 || dst.len() % w != 0 || TAPS == 0 {
        return false;
    }
    let h = dst.len() / w;
    if h == 0 {
        return true;
    }
    // The farthest input any output reads; past the end (or beyond
    // `usize`), the portable pass reports the out-of-range read as before.
    let last = (h - 1)
        .checked_mul(src_stride)
        .and_then(|rows| {
            (TAPS - 1)
                .checked_mul(step)
                .and_then(|taps| rows.checked_add(taps))
        })
        .and_then(|base| base.checked_add(w - 1));
    if last.map_or(true, |last| last >= src.len()) {
        return false;
    }
    // `S` is `u16` or `i16`: two-byte lanes read as `i16`.
    let src = src.as_ptr().cast::<i16>();
    let dst = dst.as_mut_ptr();
    // SAFETY: NEON is part of the AArch64 baseline; no memory is touched.
    let shift = unsafe { vdupq_n_s32(-(shift as i32)) };
    for y in 0..h {
        let row = y * src_stride;
        let out = y * w;
        let mut x = 0;
        while x + 8 <= w {
            // SAFETY: tap `t` reads `src[ row + t·step + x .. + 8 ]`, whose
            // last index is at most `last < src.len()` (`x + 8 <= w`, `t <
            // TAPS`, `y < h`); `u16` and `i16` share size and alignment.
            // The store writes `dst[ out + x .. + 8 ]` within `h · w ==
            // dst.len()`.
            unsafe {
                let base = src.add(row + x);
                if acc16 {
                    let mut acc = vdupq_n_s16(0);
                    for (t, &c) in k.iter().enumerate() {
                        acc = vmlaq_n_s16(acc, vld1q_s16(base.add(t * step)), c);
                    }
                    D::put8_16(dst.add(out + x), acc);
                } else {
                    let (mut lo, mut hi) = (vdupq_n_s32(0), vdupq_n_s32(0));
                    for (t, &c) in k.iter().enumerate() {
                        let v = vld1q_s16(base.add(t * step));
                        lo = vmlal_n_s16(lo, vget_low_s16(v), c);
                        hi = vmlal_high_n_s16(hi, v, c);
                    }
                    D::put8(dst.add(out + x), vshlq_s32(lo, shift), vshlq_s32(hi, shift));
                }
            }
            x += 8;
        }
        if x < w {
            // SAFETY: as above for the final four columns (`w` is a
            // multiple of 4, so `x + 4 == w`).
            unsafe {
                let base = src.add(row + x);
                if acc16 {
                    let mut acc = vdup_n_s16(0);
                    for (t, &c) in k.iter().enumerate() {
                        acc = vmla_n_s16(acc, vld1_s16(base.add(t * step)), c);
                    }
                    D::put4_16(dst.add(out + x), acc);
                } else {
                    let mut acc = vdupq_n_s32(0);
                    for (t, &c) in k.iter().enumerate() {
                        acc = vmlal_n_s16(acc, vld1_s16(base.add(t * step)), c);
                    }
                    D::put4(dst.add(out + x), vshlq_s32(acc, shift));
                }
            }
        }
    }
    true
}

/// `Sign( c − n )` per lane: comparison lanes are −1 when true, so `[ c <
/// n ] · −1 − [ c > n ] · −1`.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn sign(c: int16x8_t, n: int16x8_t) -> int16x8_t {
    // SAFETY: register-only NEON operations.
    unsafe {
        vsubq_s16(
            vreinterpretq_s16_u16(vcltq_s16(c, n)),
            vreinterpretq_s16_u16(vcgtq_s16(c, n)),
        )
    }
}

/// The byte lanes picking `i16` entry `idx` of a little-endian byte
/// table: bytes `2 · idx` and `2 · idx + 1`.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn entry_bytes(idx: uint16x8_t) -> uint8x16_t {
    // SAFETY: register-only NEON operations.
    unsafe {
        let low = vshlq_n_u16::<1>(idx);
        vreinterpretq_u8_u16(vorrq_u16(
            low,
            vshlq_n_u16::<8>(vaddq_u16(low, vdupq_n_u16(1))),
        ))
    }
}

/// [`super::sao_edge`] in NEON: eight samples at a time, the last group
/// ending at the span's end; spans shorter than eight stay portable.
pub(super) fn sao_edge(
    span: &mut [u16],
    cur: &[u16],
    a: &[u16],
    b: &[u16],
    by_edge: [i32; 5],
    max: i32,
) -> usize {
    let n = span.len();
    if n < 8 || cur.len() < n || a.len() < n || b.len() < n {
        return 0;
    }
    let mut table = [0u8; 16];
    for (pair, &o) in table.chunks_exact_mut(2).zip(&by_edge) {
        pair.copy_from_slice(&(o as i16).to_le_bytes());
    }
    let (out, cur, a, b) = (span.as_mut_ptr(), cur.as_ptr(), a.as_ptr(), b.as_ptr());
    // SAFETY: every group loads `cur` / `a` / `b[ i .. i + 8 ]` and stores
    // `span[ i .. i + 8 ]` with `i + 8 <= n`, within every length; the
    // table load reads the 16-byte local. The final group may start inside
    // the previous one: it recomputes those samples from the unchanged
    // inputs, which the exclusive `span` cannot overlap, so it rewrites
    // the same values.
    unsafe {
        let table = vld1q_u8(table.as_ptr());
        let (zero, top, two) = (vdupq_n_s16(0), vdupq_n_s16(max as i16), vdupq_n_s16(2));
        let group = |i: usize| {
            let c = vreinterpretq_s16_u16(vld1q_u16(cur.add(i)));
            let sa = sign(c, vreinterpretq_s16_u16(vld1q_u16(a.add(i))));
            let sb = sign(c, vreinterpretq_s16_u16(vld1q_u16(b.add(i))));
            let idx = vreinterpretq_u16_s16(vaddq_s16(vaddq_s16(sa, sb), two));
            let off = vreinterpretq_s16_u8(vqtbl1q_u8(table, entry_bytes(idx)));
            let r = vminq_s16(vmaxq_s16(vaddq_s16(c, off), zero), top);
            vst1q_u16(out.add(i), vreinterpretq_u16_s16(r));
        };
        let mut i = 0;
        while i + 8 <= n {
            group(i);
            i += 8;
        }
        if i < n {
            group(n - 8);
        }
    }
    n
}

/// [`super::sao_band`] in NEON: eight samples at a time, the last group
/// ending at the span's end; spans shorter than eight stay portable.
pub(super) fn sao_band(
    span: &mut [u16],
    cur: &[u16],
    band_shift: u32,
    bands: [u32; 4],
    off: &[i32; 5],
    max: i32,
) -> usize {
    let n = span.len();
    // An out-of-range band (a sample at or past 1 << BitDepth) looks up
    // zero, which is SaoOffsetVal[ 0 ] only when that is zero.
    if n < 8 || cur.len() < n || off[0] != 0 || band_shift > 15 || bands.iter().any(|&b| b > 31) {
        return 0;
    }
    // bandTable: SaoOffsetVal[ k + 1 ] for band `bands[ k ]`, else [ 0 ];
    // the first matching `k` wins, as in the portable chain.
    let mut entries = [0i16; 32];
    for (k, &band) in bands.iter().enumerate().rev() {
        entries[band as usize] = off[k + 1] as i16;
    }
    let mut table = [0u8; 64];
    for (pair, &o) in table.chunks_exact_mut(2).zip(&entries) {
        pair.copy_from_slice(&o.to_le_bytes());
    }
    let (out, cur) = (span.as_mut_ptr(), cur.as_ptr());
    // SAFETY: every group loads `cur[ i .. i + 8 ]` and stores `span[ i ..
    // i + 8 ]` with `i + 8 <= n`, within both lengths; the table loads read
    // the 64-byte local. An overlapping final group rewrites the same
    // values, as in [`sao_edge`].
    unsafe {
        let p = table.as_ptr();
        let table = uint8x16x4_t(
            vld1q_u8(p),
            vld1q_u8(p.add(16)),
            vld1q_u8(p.add(32)),
            vld1q_u8(p.add(48)),
        );
        let shift = vdupq_n_s16(-(band_shift as i16));
        let (zero, top) = (vdupq_n_s16(0), vdupq_n_s16(max as i16));
        let group = |i: usize| {
            let raw = vld1q_u16(cur.add(i));
            let o = vreinterpretq_s16_u8(vqtbl4q_u8(table, entry_bytes(vshlq_u16(raw, shift))));
            let c = vreinterpretq_s16_u16(raw);
            let r = vminq_s16(vmaxq_s16(vaddq_s16(c, o), zero), top);
            vst1q_u16(out.add(i), vreinterpretq_u16_s16(r));
        };
        let mut i = 0;
        while i + 8 <= n {
            group(i);
            i += 8;
        }
        if i < n {
            group(n - 8);
        }
    }
    n
}

/// [`super::pred_default_row`] in NEON: eight samples at a time, then
/// four, the last group ending at the row's end.
pub(super) fn pred_default_row(
    out: &mut [u16],
    p0: &[i32],
    p1: Option<&[i32]>,
    shift: i32,
    offset: i32,
    max: i32,
) -> bool {
    let n = out.len();
    if n < 4
        || p0.len() < n
        || p1.is_some_and(|p| p.len() < n)
        || !(0..=31).contains(&shift)
        || !(0..=0xffff).contains(&max)
    {
        return false;
    }
    let (dst, a, b) = (out.as_mut_ptr(), p0.as_ptr(), p1.map(<[i32]>::as_ptr));
    // SAFETY: every group of four loads `p0` (and `p1`)`[ i .. i + 4 ]`
    // and stores `out[ i .. i + 4 ]` with `i + 4 <= n`, within every
    // length. A final group overlapping the previous one recomputes those
    // samples from the unchanged inputs (the exclusive `out` cannot alias
    // them), so it rewrites the same values.
    unsafe {
        let (off, down, top) = (
            vdupq_n_s32(offset),
            vdupq_n_s32(-shift),
            vdup_n_u16(max as u16),
        );
        // ( v + offset ) >> shift (`vshlq` by a negative count is the
        // truncating arithmetic right shift), saturated to `u16`, then
        // capped at `max`: Clip3( 0, max, … ).
        let four = |i: usize| -> uint16x4_t {
            let mut v = vld1q_s32(a.add(i));
            if let Some(b) = b {
                v = vaddq_s32(v, vld1q_s32(b.add(i)));
            }
            vmin_u16(vqmovun_s32(vshlq_s32(vaddq_s32(v, off), down)), top)
        };
        let mut i = 0;
        while i + 8 <= n {
            vst1q_u16(dst.add(i), vcombine_u16(four(i), four(i + 4)));
            i += 8;
        }
        if i + 4 <= n {
            vst1_u16(dst.add(i), four(i));
            i += 4;
        }
        if i < n {
            vst1_u16(dst.add(n - 4), four(n - 4));
        }
    }
    true
}

/// The eight positions `p3 p2 p1 p0 q0 q1 q2 q3` of four 8-sample rows
/// (`r[ k ][ j ]` is position `j` of line `k`), each as one vector over
/// the four lines, and back: the transposition is its own inverse.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn transpose_4x8(r: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
    // SAFETY: register-only NEON operations.
    unsafe {
        let t0 = vreinterpretq_u32_u16(vtrn1q_u16(r[0], r[1]));
        let t1 = vreinterpretq_u32_u16(vtrn2q_u16(r[0], r[1]));
        let t2 = vreinterpretq_u32_u16(vtrn1q_u16(r[2], r[3]));
        let t3 = vreinterpretq_u32_u16(vtrn2q_u16(r[2], r[3]));
        // Positions 0 | 4, 1 | 5, 2 | 6, 3 | 7 (low | high halves).
        [
            vreinterpretq_u16_u32(vtrn1q_u32(t0, t2)),
            vreinterpretq_u16_u32(vtrn1q_u32(t1, t3)),
            vreinterpretq_u16_u32(vtrn2q_u32(t0, t2)),
            vreinterpretq_u16_u32(vtrn2q_u32(t1, t3)),
        ]
    }
}

/// [`transpose_4x8`]'s inverse layout: four vectors holding positions `j`
/// (low) and `j + 4` (high) back to the four 8-sample rows.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn untranspose_4x8(u: [uint16x8_t; 4]) -> [uint16x8_t; 4] {
    // SAFETY: register-only NEON operations.
    unsafe {
        let (u0, u1, u2, u3) = (
            vreinterpretq_u32_u16(u[0]),
            vreinterpretq_u32_u16(u[1]),
            vreinterpretq_u32_u16(u[2]),
            vreinterpretq_u32_u16(u[3]),
        );
        let t0 = vreinterpretq_u16_u32(vtrn1q_u32(u0, u2));
        let t2 = vreinterpretq_u16_u32(vtrn2q_u32(u0, u2));
        let t1 = vreinterpretq_u16_u32(vtrn1q_u32(u1, u3));
        let t3 = vreinterpretq_u16_u32(vtrn2q_u32(u1, u3));
        [
            vtrn1q_u16(t0, t1),
            vtrn2q_u16(t0, t1),
            vtrn1q_u16(t2, t3),
            vtrn2q_u16(t2, t3),
        ]
    }
}

/// §8.7.2.5.7 strong luma filter (eqs. 8-389 .. 8-394) on the positions
/// `p3 .. q3` of four lines, each clamped to ±2·tC of its input.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn luma_strong(v: [int32x4_t; 8], tc: i32) -> [int32x4_t; 8] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
    // SAFETY: register-only NEON operations.
    unsafe {
        let tc2 = vdupq_n_s32(2 * tc);
        // ( p − 2·tC ).max( ( p + 2·tC ).min( x ) ), as the portable form.
        let near = |x: int32x4_t, p: int32x4_t| {
            vmaxq_s32(vminq_s32(x, vaddq_s32(p, tc2)), vsubq_s32(p, tc2))
        };
        let sum = |terms: &[int32x4_t]| terms.iter().fold(vdupq_n_s32(0), |a, &t| vaddq_s32(a, t));
        let (four, two) = (vdupq_n_s32(4), vdupq_n_s32(2));
        let p0n = vshrq_n_s32::<3>(sum(&[p2, p1, p1, p0, p0, q0, q0, q1, four]));
        let p1n = vshrq_n_s32::<2>(sum(&[p2, p1, p0, q0, two]));
        let p2n = vshrq_n_s32::<3>(sum(&[p3, p3, p2, p2, p2, p1, p0, q0, four]));
        let q0n = vshrq_n_s32::<3>(sum(&[p1, p0, p0, q0, q0, q1, q1, q2, four]));
        let q1n = vshrq_n_s32::<2>(sum(&[p0, q0, q1, q2, two]));
        let q2n = vshrq_n_s32::<3>(sum(&[p0, q0, q1, q2, q2, q2, q3, q3, four]));
        [
            p3,
            near(p2n, p2),
            near(p1n, p1),
            near(p0n, p0),
            near(q0n, q0),
            near(q1n, q1),
            near(q2n, q2),
            q3,
        ]
    }
}

/// §8.7.2.5.7 weak luma filter (eqs. 8-395 .. 8-402) on the positions
/// `p3 .. q3` of four lines: lines with `|Δ| >= 10·tC` keep every sample.
///
/// # Safety
/// None beyond NEON, part of the AArch64 baseline; no memory is touched.
#[inline(always)]
unsafe fn luma_weak(v: [int32x4_t; 8], (dep, deq): (u8, u8), tc: i32, max: i32) -> [int32x4_t; 8] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
    // SAFETY: register-only NEON operations.
    unsafe {
        let clamp = |x: int32x4_t, lo: i32, hi: i32| {
            vmaxq_s32(vminq_s32(x, vdupq_n_s32(hi)), vdupq_n_s32(lo))
        };
        let one = vdupq_n_s32(1);
        // eq. 8-395: Δ = ( 9·( q0 − p0 ) − 3·( q1 − p1 ) + 8 ) >> 4.
        let delta = vshrq_n_s32::<4>(vaddq_s32(
            vsubq_s32(
                vmulq_n_s32(vsubq_s32(q0, p0), 9),
                vmulq_n_s32(vsubq_s32(q1, p1), 3),
            ),
            vdupq_n_s32(8),
        ));
        let on = vcltq_s32(vabsq_s32(delta), vdupq_n_s32(tc * 10));
        let d = clamp(delta, -tc, tc); // eq. 8-396
        let p0n = clamp(vaddq_s32(p0, d), 0, max); // eq. 8-397
        let q0n = clamp(vsubq_s32(q0, d), 0, max); // eq. 8-398
        let p1n = if dep == 1 {
            // eqs. 8-399 / 8-400.
            let mid = vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(p2, p0), one));
            let dp = vshrq_n_s32::<1>(vaddq_s32(vsubq_s32(mid, p1), d));
            clamp(vaddq_s32(p1, clamp(dp, -(tc >> 1), tc >> 1)), 0, max)
        } else {
            p1
        };
        let q1n = if deq == 1 {
            // eqs. 8-401 / 8-402.
            let mid = vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(q2, q0), one));
            let dq = vshrq_n_s32::<1>(vsubq_s32(vsubq_s32(mid, q1), d));
            clamp(vaddq_s32(q1, clamp(dq, -(tc >> 1), tc >> 1)), 0, max)
        } else {
            q1
        };
        let pick = |new: int32x4_t, old: int32x4_t| vbslq_s32(on, new, old);
        [
            p3,
            p2,
            pick(p1n, p1),
            pick(p0n, p0),
            pick(q0n, q0),
            pick(q1n, q1),
            q2,
            q3,
        ]
    }
}

/// [`super::luma_edge`] in NEON: the four lines as the lanes of `i32`
/// vectors, one per position across the edge.
pub(super) fn luma_edge(
    samples: &mut [u16],
    q00: usize,
    stride: usize,
    vertical: bool,
    (de, dep, deq): (u8, u8, u8),
    tc: i32,
    max: i32,
) -> bool {
    if !(1..=2).contains(&de) || !(0..=0xffff).contains(&max) || !(0..=1 << 24).contains(&tc) {
        return false;
    }
    // The segment's first sample (`p3,0`) and last (`q3,3`).
    let first = if vertical {
        q00.checked_sub(4)
    } else {
        stride.checked_mul(4).and_then(|up| q00.checked_sub(up))
    };
    let last = stride
        .checked_mul(3)
        .and_then(|down| q00.checked_add(down))
        .and_then(|at| at.checked_add(3));
    let (Some(first), Some(last)) = (first, last) else {
        return false;
    };
    if last >= samples.len() {
        return false;
    }
    let base = samples.as_mut_ptr();
    // SAFETY: a vertical segment reads and writes the eight samples from
    // `first + k·stride` (k < 4), a horizontal one the four from `first +
    // j·stride` (j < 8); both ranges end at `q00 + 3·stride + 3 == last <
    // samples.len()` and start at `first`. The rest is register-only NEON.
    unsafe {
        let wide = |x: uint16x4_t| vreinterpretq_s32_u32(vmovl_u16(x));
        let narrow = |x: int32x4_t| vmovn_u32(vreinterpretq_u32_s32(x));
        let v: [int32x4_t; 8] = if vertical {
            let u = transpose_4x8(core::array::from_fn(|k| {
                vld1q_u16(base.add(first + k * stride))
            }));
            [
                wide(vget_low_u16(u[0])),
                wide(vget_low_u16(u[1])),
                wide(vget_low_u16(u[2])),
                wide(vget_low_u16(u[3])),
                wide(vget_high_u16(u[0])),
                wide(vget_high_u16(u[1])),
                wide(vget_high_u16(u[2])),
                wide(vget_high_u16(u[3])),
            ]
        } else {
            core::array::from_fn(|j| wide(vld1_u16(base.add(first + j * stride))))
        };
        let out = if de == 2 {
            luma_strong(v, tc)
        } else {
            luma_weak(v, (dep, deq), tc, max)
        };
        if vertical {
            let rows = untranspose_4x8(core::array::from_fn(|j| {
                vcombine_u16(narrow(out[j]), narrow(out[j + 4]))
            }));
            for (k, &row) in rows.iter().enumerate() {
                vst1q_u16(base.add(first + k * stride), row);
            }
        } else {
            for (j, &v) in out.iter().enumerate() {
                vst1_u16(base.add(first + j * stride), narrow(v));
            }
        }
    }
    true
}
