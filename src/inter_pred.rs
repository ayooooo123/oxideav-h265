//! §8.5.3.3.3 — fractional sample interpolation, plus the §8.5.3.3.4
//! weighted sample prediction combines (default and explicit).
//!
//! This module turns a reference-picture sample plane and a motion vector
//! into an `(nPbW)x(nPbH)` array of inter-predicted samples. Four ITU-T
//! H.265 subclauses are implemented, in the order §8.5.3.3.3.1 invokes
//! them:
//!
//! * §8.5.3.3.3.2 **luma sample interpolation** ([`interp_luma_block`]) —
//!   the separable 8-tap quarter-pel filter of equations 8-224..8-238,
//!   with the Table 8-8 phase selection. `shift1 = Min(4, BitDepthY − 8)`,
//!   `shift2 = 6`, `shift3 = Max(2, 14 − BitDepthY)`; the full-pel case is
//!   `A << shift3`.
//! * §8.5.3.3.3.3 **chroma sample interpolation** ([`interp_chroma_block`])
//!   — the separable 4-tap eighth-pel filter of equations 8-241..8-261,
//!   with the Table 8-9 phase selection. `shift1 = Min(4, BitDepthC − 8)`,
//!   `shift2 = 6`, `shift3 = Max(2, 14 − BitDepthC)`.
//! * §8.5.3.3.4.2 **default weighted sample prediction**
//!   ([`default_weighted_pred`]) — the uni- / bi-predictive combine of
//!   equations 8-262..8-264 (`weightedPredFlag == 0`), with
//!   `shift1 = Max(2, 14 − bitDepth)`, `shift2 = Max(3, 15 − bitDepth)`.
//! * §8.5.3.3.4.3 **explicit weighted sample prediction**
//!   ([`explicit_weighted_pred`]) — the per-reference weight / offset
//!   combine of equations 8-265..8-277 (`weightedPredFlag == 1`, i.e.
//!   `weighted_pred_flag` for P slices / `weighted_bipred_flag` for B
//!   slices), with `log2Wd = log2WeightDenom + shift1`.
//!
//! The interpolation processes emit *intermediate* sample values at the
//! `14 − BitDepth`-bit internal precision the spec carries between
//! §8.5.3.3.3 and §8.5.3.3.4 (i.e. the `>> shift1` / `>> shift2` outputs,
//! `A << shift3` for full-pel — they are **not** yet clipped to the
//! sample range). The weighted combines consume those intermediate
//! arrays and produce the final `[0, (1 << bitDepth) − 1]` prediction
//! samples.
//!
//! ## Scope
//!
//! The numerics are self-contained. The §8.5.3.2 merge / §8.5.3.1 MV
//! derivation that produces `mvLX`, the §8.5.3.3.1 driver that splits a
//! motion vector into its integer / fractional parts and walks the
//! prediction block, and the §8.6.5 picture-construction step that adds
//! the residual are the caller's responsibility — this module starts at
//! a `(xInt, yInt, xFrac, yFrac)` location and a reference plane, and
//! stops at the prediction sample arrays.

use crate::simd;
use std::cell::RefCell;

/// A reference-picture luma / chroma sample plane with the §8.5.3.3.3
/// `Clip3( 0, dim − 1, … )` edge-extension border (equations 8-222 /
/// 8-223 for luma, 8-239 / 8-240 for chroma).
///
/// The interpolation filters read samples at negative and past-the-edge
/// coordinates; this type clamps every access into the valid plane so the
/// callers can index with the raw `xInt + i` / `yInt + j` offsets the
/// equations use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPlane<'a> {
    /// Row-major samples, `width * height` of them. `sample[ y * width + x ]`
    /// is the plane sample at full-sample location `( x, y )`.
    samples: &'a [u16],
    /// Plane width in samples (`pic_width_in_luma_samples` for luma, or
    /// `pic_width_in_luma_samples / SubWidthC` for chroma).
    width: usize,
    /// Plane height in samples.
    height: usize,
}

impl<'a> RefPlane<'a> {
    /// Wraps a row-major `width * height` sample plane.
    ///
    /// # Errors
    ///
    /// [`InterPredError::PlaneLengthMismatch`] if `samples.len()` is not
    /// exactly `width * height`, or [`InterPredError::EmptyPlane`] if
    /// either dimension is zero.
    pub fn new(samples: &'a [u16], width: usize, height: usize) -> Result<Self, InterPredError> {
        if width == 0 || height == 0 {
            return Err(InterPredError::EmptyPlane);
        }
        let expected = width
            .checked_mul(height)
            .ok_or(InterPredError::EmptyPlane)?;
        if samples.len() != expected {
            return Err(InterPredError::PlaneLengthMismatch {
                expected,
                got: samples.len(),
            });
        }
        Ok(Self {
            samples,
            width,
            height,
        })
    }

    /// The plane width in samples.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The plane height in samples.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Sample at full-sample location `( x, y )` with the §8.5.3.3.3
    /// `Clip3( 0, dim − 1, … )` edge extension (equations 8-222 / 8-223 /
    /// 8-239 / 8-240). `x` and `y` may be negative or past the edge.
    #[inline]
    #[must_use]
    pub fn at(&self, x: i32, y: i32) -> i32 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        i32::from(self.samples[yc * self.width + xc])
    }
}

/// Errors from the §8.5.3.3 inter-prediction processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterPredError {
    /// A reference plane had a zero width or height.
    EmptyPlane,
    /// A reference plane's sample count did not equal `width * height`.
    PlaneLengthMismatch {
        /// The `width * height` count the plane requires.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
    /// A prediction-block dimension (`nPbW` or `nPbH`) was zero.
    EmptyBlock,
    /// `xFracL` / `yFracL` was outside the `0..=3` quarter-pel range, or
    /// `xFracC` / `yFracC` was outside the `0..=7` eighth-pel range.
    InvalidFraction(i32),
    /// `bitDepth` was outside the 8..=16 range the equations are
    /// dimensioned for.
    InvalidBitDepth(u8),
    /// The two `predSamplesLX` arrays handed to the weighted combine did
    /// not have matching `nPbW * nPbH` lengths.
    ArrayLengthMismatch {
        /// The `nPbW * nPbH` count both arrays require.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
}

impl core::fmt::Display for InterPredError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPlane => write!(f, "reference plane has zero width or height"),
            Self::PlaneLengthMismatch { expected, got } => {
                write!(
                    f,
                    "reference plane length {got} != width*height = {expected}"
                )
            }
            Self::EmptyBlock => write!(f, "prediction block dimension nPbW/nPbH is zero"),
            Self::InvalidFraction(v) => {
                write!(
                    f,
                    "invalid fractional offset {v} (luma 0..=3, chroma 0..=7)"
                )
            }
            Self::InvalidBitDepth(b) => write!(f, "invalid bitDepth {b} (expected 8..=16)"),
            Self::ArrayLengthMismatch { expected, got } => {
                write!(f, "prediction array length {got} != nPbW*nPbH = {expected}")
            }
        }
    }
}

impl std::error::Error for InterPredError {}

/// `shift1` for luma / chroma interpolation: `Min( 4, BitDepth − 8 )`.
#[inline]
fn interp_shift1(bit_depth: u8) -> i32 {
    core::cmp::min(4, bit_depth as i32 - 8)
}

/// `shift3` for luma / chroma interpolation: `Max( 2, 14 − BitDepth )`.
#[inline]
fn interp_shift3(bit_depth: u8) -> i32 {
    core::cmp::max(2, 14 - bit_depth as i32)
}

// ---------------------------------------------------------------------------
// §8.5.3.3.3.2 — luma sample interpolation
// ---------------------------------------------------------------------------

/// The §8.5.3.3.3.2 horizontal 8-tap luma filters, indexed by `xFracL`.
///
/// Row 0 (`xFracL == 0`) is the identity (the spec leaves the integer
/// sample untouched on the horizontal pass); rows 1/2/3 are the `a`/`b`/`c`
/// kernels of equations 8-224 / 8-225 / 8-226, each over the eight taps
/// `A[−3..4]`.
const LUMA_FILTER: [[i16; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Validates one block interpolation's inputs in the order the
/// public interpolation functions report them: block dimensions,
/// horizontal / vertical phase (`0..=max_frac`), then bit depth.
fn check_interp(
    w: usize,
    h: usize,
    x_frac: i32,
    y_frac: i32,
    max_frac: i32,
    bit_depth: u8,
) -> Result<(), InterPredError> {
    if w == 0 || h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(0..=max_frac).contains(&x_frac) {
        return Err(InterPredError::InvalidFraction(x_frac));
    }
    if !(0..=max_frac).contains(&y_frac) {
        return Err(InterPredError::InvalidFraction(y_frac));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    Ok(())
}

/// §8.5.3.3.3.2 — fill an `(nPbW)x(nPbH)` luma prediction block.
///
/// `( x_int, y_int )` is the integer part of the motion-compensated
/// top-left location (`xPb + ( mvLX[0] >> 2 )`, `yPb + ( mvLX[1] >> 2 )`
/// per equations 8-214 / 8-215) and `( x_frac, y_frac )` the
/// quarter-pel remainder (`mvLX[..] & 3`, equations 8-216 / 8-217). The
/// output is row-major, `predSamples[ y * nPbW + x ]`, holding the
/// intermediate-precision values §8.5.3.3.4 consumes: `A << shift3` for
/// the full-sample phase, `>> shift1` for the one-dimensional phases,
/// and `>> shift1` horizontally then `>> 6` vertically for the
/// two-dimensional ones.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidFraction`] for a fraction outside `0..=3`, and
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`.
// The §8.5.3.3.3.2 location / fraction / dimension / bit-depth inputs are
// each distinct spec quantities; bundling them would obscure the mapping.
#[allow(clippy::too_many_arguments)]
pub fn interp_luma_block(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    check_interp(n_pb_w, n_pb_h, x_frac, y_frac, 3, bit_depth)?;
    let mut out = vec![0i32; n_pb_w * n_pb_h];
    with_scratch(|s| {
        let kernel = |f: i32| (f != 0).then(|| &LUMA_FILTER[f as usize]);
        interp_into(
            plane,
            x_int,
            y_int,
            kernel(x_frac),
            kernel(y_frac),
            n_pb_w,
            bit_depth,
            &mut s.work.interp,
            &mut out,
        );
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// §8.5.3.3.3.3 — chroma sample interpolation
// ---------------------------------------------------------------------------

/// The §8.5.3.3.3.3 4-tap chroma filters, indexed by the eighth-pel phase.
///
/// Row 0 (phase 0) is the identity; rows 1..7 are the `ab`/`ac`/`ad`/`ae`/
/// `af`/`ag`/`ah` kernels of equations 8-241..8-247, each over the four
/// taps `B[−1..2]`.
const CHROMA_FILTER: [[i16; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// §8.5.3.3.3.3 — fill an `(nPbW / SubWidthC)x(nPbH / SubHeightC)` chroma
/// prediction block.
///
/// `( x_int, y_int )` is the integer chroma location and
/// `( x_frac, y_frac )` the eighth-pel remainder (`mvCLX[..] & 7`,
/// equations 8-220 / 8-221). `block_w` / `block_h` are the chroma block
/// dimensions. Output is row-major intermediate-precision values, with
/// the same per-phase shifts as [`interp_luma_block`].
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidFraction`] for a fraction outside `0..=7`, and
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`.
// The §8.5.3.3.3.3 location / fraction / dimension / bit-depth inputs are
// each distinct spec quantities; bundling them would obscure the mapping.
#[allow(clippy::too_many_arguments)]
pub fn interp_chroma_block(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    block_w: usize,
    block_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    check_interp(block_w, block_h, x_frac, y_frac, 7, bit_depth)?;
    let mut out = vec![0i32; block_w * block_h];
    with_scratch(|s| {
        let kernel = |f: i32| (f != 0).then(|| &CHROMA_FILTER[f as usize]);
        interp_into(
            plane,
            x_int,
            y_int,
            kernel(x_frac),
            kernel(y_frac),
            block_w,
            bit_depth,
            &mut s.work.interp,
            &mut out,
        );
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Separable block filtering shared by §8.5.3.3.3.2 and §8.5.3.3.3.3
// ---------------------------------------------------------------------------

/// Reusable buffers for one block interpolation: the edge-extended
/// source window and the horizontally filtered rows of the
/// two-dimensional phases (`i16` when they fit, else `i32`).
#[derive(Default)]
struct InterpScratch {
    window: Vec<u16>,
    rows: Vec<i32>,
    rows16: Vec<i16>,
}

/// Copies the `ww × wh` source window whose top-left plane location is
/// `( x0, y0 )` into `window`, row-major, applying the §8.5.3.3.3
/// `Clip3( 0, dim − 1, … )` edge extension to each coordinate that
/// leaves the plane (the values [`RefPlane::at`] returns).
fn gather_window(
    plane: &RefPlane<'_>,
    x0: i64,
    y0: i64,
    ww: usize,
    wh: usize,
    window: &mut Vec<u16>,
) {
    window.clear();
    let (pw, ph) = (plane.width as i64, plane.height as i64);
    let inside = x0 >= 0 && x0 + ww as i64 <= pw;
    for r in 0..wh as i64 {
        let y = (y0 + r).clamp(0, ph - 1) as usize;
        let row = &plane.samples[y * plane.width..(y + 1) * plane.width];
        if inside {
            let x = x0 as usize;
            window.extend_from_slice(&row[x..x + ww]);
        } else {
            window.extend((0..ww as i64).map(|k| row[(x0 + k).clamp(0, pw - 1) as usize]));
        }
    }
}

/// One separable filter pass over a block: `dst[ y·w + x ] = put( ( Σ_t
/// k[ t ] · get( src[ y·src_stride + t·step + x ] ) ) >> shift )` for
/// every `x < w` and row `y < dst.len() / w`. The horizontal pass is
/// `step == 1` with `src` at the first tap sample; the vertical pass is
/// `step == src_stride` with `src` at the first tap row.
///
/// Each tap's `w` inputs of a row are sliced once; the row is then
/// computed eight outputs at a time with every tap accumulated in
/// registers, then four, then one at a time.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn filter_block<S: Copy, D, A: FilterAcc, const TAPS: usize>(
    src: &[S],
    src_stride: usize,
    step: usize,
    k: &[i16; TAPS],
    shift: u32,
    w: usize,
    dst: &mut [D],
    get: impl Fn(S) -> A + Copy,
    put: impl Fn(A) -> D + Copy,
) {
    let c = k.map(A::from);
    for (y, out) in dst.chunks_exact_mut(w).enumerate() {
        let taps: [&[S]; TAPS] = core::array::from_fn(|t| {
            let o = y * src_stride + t * step;
            &src[o..o + w]
        });
        let mut x = 0;
        while x + 8 <= w {
            filter_group::<S, D, A, TAPS, 8>(&taps, x, &c, shift, &mut out[x..x + 8], get, put);
            x += 8;
        }
        if x + 4 <= w {
            filter_group::<S, D, A, TAPS, 4>(&taps, x, &c, shift, &mut out[x..x + 4], get, put);
            x += 4;
        }
        while x < w {
            filter_group::<S, D, A, TAPS, 1>(&taps, x, &c, shift, &mut out[x..x + 1], get, put);
            x += 1;
        }
    }
}

/// [`filter_block`]'s `L` outputs `out` from column `x` of each tap's
/// row inputs `taps`.
#[inline(always)]
fn filter_group<S: Copy, D, A: FilterAcc, const TAPS: usize, const L: usize>(
    taps: &[&[S]; TAPS],
    x: usize,
    c: &[A; TAPS],
    shift: u32,
    out: &mut [D],
    get: impl Fn(S) -> A,
    put: impl Fn(A) -> D,
) {
    let mut acc = [A::default(); L];
    for (tap, &c) in taps.iter().zip(c) {
        let s: &[S; L] = tap[x..x + L].try_into().unwrap();
        for (a, &v) in acc.iter_mut().zip(s) {
            *a = *a + c * get(v);
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        *o = put(a >> shift);
    }
}

/// The accumulator lanes of [`filter_block`]: `i32`, or `i16` where
/// every partial tap sum provably fits it.
trait FilterAcc:
    Copy
    + Default
    + From<i16>
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Shr<u32, Output = Self>
{
}

impl FilterAcc for i16 {}

impl FilterAcc for i32 {}

/// Separable §8.5.3.3.3.2 / §8.5.3.3.3.3 interpolation of a `w`-wide
/// block into `out` (row-major; the height is `out.len() / w`).
///
/// `TAPS` is 8 for luma (taps `−3..4`) or 4 for chroma (taps `−1..2`);
/// `hk` / `vk` are the kernels of a non-zero horizontal / vertical
/// phase, `None` for phase 0. Every output equals the per-sample
/// equation: `A << shift3` at the full-sample phase, the one-dimensional
/// tap sum `>> shift1`, or the vertical sum of the `>> shift1`
/// horizontal results `>> 6`. The two-dimensional case computes each
/// horizontal result once per source row instead of once per output
/// sample; integer sums are unchanged by that sharing. A source window
/// inside the plane is read in place; one that leaves it is gathered
/// with the edge extension first.
///
/// Up to 12-bit samples every multiplicand is exact in 16 bits: the
/// samples, the taps, and each horizontal result `>> shift1` (whose
/// magnitude stays below 88 · 2^8). Those cases multiply 16-bit values
/// into 32-bit sums and keep the rows as `i16`.
#[allow(clippy::too_many_arguments)]
fn interp_into<const TAPS: usize>(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    hk: Option<&[i16; TAPS]>,
    vk: Option<&[i16; TAPS]>,
    w: usize,
    bit_depth: u8,
    scratch: &mut InterpScratch,
    out: &mut [i32],
) {
    let h = out.len() / w;
    let shift1 = interp_shift1(bit_depth);
    let before = (TAPS / 2 - 1) as i64;
    let (x0, ww) = match hk {
        Some(_) => (i64::from(x_int) - before, w + TAPS - 1),
        None => (i64::from(x_int), w),
    };
    let (y0, wh) = match vk {
        Some(_) => (i64::from(y_int) - before, h + TAPS - 1),
        None => (i64::from(y_int), h),
    };
    let (pw, ph) = (plane.width as i64, plane.height as i64);
    let (src, stride): (&[u16], usize) =
        if x0 >= 0 && y0 >= 0 && x0 + ww as i64 <= pw && y0 + wh as i64 <= ph {
            let start = y0 as usize * plane.width + x0 as usize;
            (&plane.samples[start..], plane.width)
        } else {
            gather_window(plane, x0, y0, ww, wh, &mut scratch.window);
            (&scratch.window, ww)
        };
    // A sample of at most 15 bits is the same value as `i16`: up to
    // 12-bit samples every multiplicand is an exact `i16` (the samples,
    // the taps, each horizontal result `>> shift1`), multiplied into
    // `i32` sums. 8-bit samples (`shift1 == 0`): with taps summing to 64
    // from negative parts of at most 24, every partial horizontal or
    // one-dimensional tap sum lies in [ −24 · 255, 88 · 255 ], exact in
    // `i16` lanes.
    let sample16 = |s: u16| i32::from(s as i16);
    let sample32 = |s: u16| i32::from(s);
    let byte16 = |s: u16| s as i16;
    let same = |v: i32| v;
    let same16 = |v: i16| v;
    let shift1 = shift1 as u32;
    match (hk, vk) {
        (None, None) => {
            let shift3 = interp_shift3(bit_depth);
            for (y, dst) in out.chunks_exact_mut(w).enumerate() {
                for (o, &a) in dst.iter_mut().zip(&src[y * stride..y * stride + w]) {
                    *o = i32::from(a) << shift3;
                }
            }
        }
        (Some(k), None) | (None, Some(k)) => {
            let step = if hk.is_some() { 1 } else { stride };
            let byte = bit_depth == 8;
            if bit_depth <= 12 && simd::interp_pass(src, stride, step, k, byte, shift1, w, out) {
                return;
            }
            if byte {
                filter_block(src, stride, step, k, 0, w, out, byte16, i32::from);
            } else if bit_depth <= 12 {
                filter_block(src, stride, step, k, shift1, w, out, sample16, same);
            } else {
                filter_block(src, stride, step, k, shift1, w, out, sample32, same);
            }
        }
        (Some(hk), Some(vk)) => {
            let len = wh * w;
            if bit_depth <= 12 {
                if scratch.rows16.len() < len {
                    scratch.rows16.resize(len, 0);
                }
                let rows = &mut scratch.rows16[..len];
                let byte = bit_depth == 8;
                if !simd::interp_pass(src, stride, 1, hk, byte, shift1, w, rows) {
                    if byte {
                        filter_block(src, stride, 1, hk, 0, w, rows, byte16, same16);
                    } else {
                        filter_block(src, stride, 1, hk, shift1, w, rows, sample16, |v| v as i16);
                    }
                }
                if !simd::interp_pass(&*rows, w, w, vk, false, 6, w, out) {
                    filter_block(rows, w, w, vk, 6, w, out, i32::from, same);
                }
            } else {
                if scratch.rows.len() < len {
                    scratch.rows.resize(len, 0);
                }
                let rows = &mut scratch.rows[..len];
                filter_block(src, stride, 1, hk, shift1, w, rows, sample32, same);
                filter_block(rows, w, w, vk, 6, w, out, same, same);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// §8.5.3.3.4.2 — default weighted sample prediction
// ---------------------------------------------------------------------------

/// §8.5.3.3.4.2 — combine the L0 / L1 intermediate prediction arrays into
/// the final `(nPbW)x(nPbH)` prediction samples (the
/// `weighted_pred_flag == 0` path, equations 8-262..8-264).
///
/// `pred_l0` / `pred_l1` are the intermediate-precision arrays produced by
/// [`interp_luma_block`] / [`interp_chroma_block`]; `pred_flag_l0` /
/// `pred_flag_l1` are the §8.5.3.2.1 prediction-list utilisation flags. At
/// least one flag must be set (the spec only invokes this process for a
/// predicted block). Output is the clipped `[0, (1 << bitDepth) − 1]`
/// sample array.
///
/// Unused arrays may be empty when their `pred_flag` is `false`; only the
/// array(s) whose flag is set are read and length-checked.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`,
/// [`InterPredError::EmptyPlane`] (re-used as "no list selected") when
/// both flags are `false`, and [`InterPredError::ArrayLengthMismatch`]
/// when a selected array is not `nPbW * nPbH` long.
pub fn default_weighted_pred(
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    check_combine(
        pred_l0,
        pred_l1,
        pred_flag_l0,
        pred_flag_l1,
        n_pb_w,
        n_pb_h,
        bit_depth,
    )?;
    let pred = PlanePrediction {
        p0: pred_flag_l0.then_some(pred_l0),
        p1: pred_flag_l1.then_some(pred_l1),
        width: n_pb_w,
        height: n_pb_h,
        combine: Combine::new(bit_depth, None),
    };
    Ok(pred.to_vec())
}

/// The [`default_weighted_pred`] / [`explicit_weighted_pred`] input
/// checks, in their documented order.
fn check_combine(
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<(), InterPredError> {
    if n_pb_w == 0 || n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    let count = n_pb_w * n_pb_h;
    for (flag, pred) in [(pred_flag_l0, pred_l0), (pred_flag_l1, pred_l1)] {
        if flag && pred.len() != count {
            return Err(InterPredError::ArrayLengthMismatch {
                expected: count,
                got: pred.len(),
            });
        }
    }
    if !pred_flag_l0 && !pred_flag_l1 {
        return Err(InterPredError::EmptyPlane);
    }
    Ok(())
}

/// The §8.5.3.3.4 weighted sample prediction constants of one plane.
#[derive(Debug, Clone, Copy)]
enum Combine {
    /// §8.5.3.3.4.2: `shift1` / `offset1` for one list (equations 8-262 /
    /// 8-263), `shift2` / `offset2` for both (equation 8-264).
    Default {
        shift1: i32,
        offset1: i32,
        shift2: i32,
        offset2: i32,
        max: i32,
    },
    /// §8.5.3.3.4.3 with `log2Wd` (equations 8-265 / 8-270) and the
    /// already-scaled offsets.
    Explicit {
        log2_wd: i64,
        w0: i64,
        o0: i64,
        w1: i64,
        o1: i64,
        max: i64,
    },
}

impl Combine {
    /// The default combine (`weights == None`) or the explicit one with
    /// `(log2WeightDenom, w0, o0, w1, o1)`, at `bit_depth`.
    fn new(bit_depth: u8, weights: Option<(u8, i32, i32, i32, i32)>) -> Self {
        match weights {
            None => {
                let shift1 = core::cmp::max(2, 14 - i32::from(bit_depth));
                let shift2 = core::cmp::max(3, 15 - i32::from(bit_depth));
                Self::Default {
                    shift1,
                    offset1: 1 << (shift1 - 1),
                    shift2,
                    offset2: 1 << (shift2 - 1),
                    max: (1 << bit_depth) - 1,
                }
            }
            Some((log2_weight_denom, w0, o0, w1, o1)) => Self::Explicit {
                log2_wd: i64::from(log2_weight_denom)
                    + core::cmp::max(2, 14 - i64::from(bit_depth)),
                w0: i64::from(w0),
                o0: i64::from(o0),
                w1: i64::from(w1),
                o1: i64::from(o1),
                max: (1i64 << bit_depth) - 1,
            },
        }
    }
}

/// One plane of a PU's §8.5.3.3.4 weighted sample prediction: the used
/// lists' `width × height` intermediate arrays (`None` for an unused
/// list, at least one present) and the combine.
pub(crate) struct PlanePrediction<'a> {
    p0: Option<&'a [i32]>,
    p1: Option<&'a [i32]>,
    width: usize,
    height: usize,
    combine: Combine,
}

impl PlanePrediction<'_> {
    /// The block's `(width, height)`.
    pub(crate) fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Row `y` of the final `[0, (1 << bitDepth) − 1]` prediction samples
    /// into `out` (`width` long).
    pub(crate) fn row<T: PredSample>(&self, y: usize, out: &mut [T]) {
        let span = y * self.width..(y + 1) * self.width;
        let p0 = self.p0.map(|p| &p[span.clone()]);
        let p1 = self.p1.map(|p| &p[span]);
        match self.combine {
            Combine::Default {
                shift1,
                offset1,
                shift2,
                offset2,
                max,
            } => match (p0, p1) {
                // Uni-predictive from L0 / L1 (equations 8-262 / 8-263).
                (Some(p), None) | (None, Some(p)) => {
                    if T::default_row(out, p, None, shift1, offset1, max) {
                        return;
                    }
                    for (o, &p) in out.iter_mut().zip(p) {
                        *o = T::clipped(((p + offset1) >> shift1).clamp(0, max));
                    }
                }
                // Bi-predictive (equation 8-264).
                (Some(a), Some(b)) => {
                    if T::default_row(out, a, Some(b), shift2, offset2, max) {
                        return;
                    }
                    for ((o, &p0), &p1) in out.iter_mut().zip(a).zip(b) {
                        *o = T::clipped(((p0 + p1 + offset2) >> shift2).clamp(0, max));
                    }
                }
                (None, None) => {}
            },
            Combine::Explicit {
                log2_wd,
                w0,
                o0,
                w1,
                o1,
                max,
            } => {
                // Uni-predictive (equations 8-275 / 8-276).
                let uni = |out: &mut [T], p: &[i32], w: i64, o: i64| {
                    let round = 1i64 << (log2_wd - 1);
                    for (d, &p) in out.iter_mut().zip(p) {
                        *d = T::clipped(
                            (((i64::from(p) * w + round) >> log2_wd) + o).clamp(0, max) as i32,
                        );
                    }
                };
                match (p0, p1) {
                    (Some(p), None) => uni(out, p, w0, o0),
                    (None, Some(p)) => uni(out, p, w1, o1),
                    // Bi-predictive (equation 8-277).
                    (Some(a), Some(b)) => {
                        let round = (o0 + o1 + 1) << log2_wd;
                        for ((o, &p0), &p1) in out.iter_mut().zip(a).zip(b) {
                            *o = T::clipped(
                                ((i64::from(p0) * w0 + i64::from(p1) * w1 + round) >> (log2_wd + 1))
                                    .clamp(0, max) as i32,
                            );
                        }
                    }
                    (None, None) => {}
                }
            }
        }
    }

    /// Every row, row-major.
    fn to_vec(&self) -> Vec<i32> {
        let mut out = vec![0i32; self.width * self.height];
        for (y, row) in out.chunks_exact_mut(self.width).enumerate() {
            self.row(y, row);
        }
        out
    }
}

/// A final prediction sample: [`PlanePrediction::row`] values lie in
/// `[0, (1 << bitDepth) − 1]` with `bitDepth <= 16`, so either type holds
/// them exactly.
pub(crate) trait PredSample: Copy {
    /// The sample for an already clipped value.
    fn clipped(v: i32) -> Self;

    /// The architecture form of a default-combine row (`out[ x ] = Clip3(
    /// 0, max, ( p0[ x ] (+ p1[ x ]) + offset ) >> shift )`), `false`
    /// when it does not run.
    #[inline(always)]
    fn default_row(
        out: &mut [Self],
        p0: &[i32],
        p1: Option<&[i32]>,
        shift: i32,
        offset: i32,
        max: i32,
    ) -> bool {
        let _ = (out, p0, p1, shift, offset, max);
        false
    }
}

impl PredSample for i32 {
    #[inline(always)]
    fn clipped(v: i32) -> Self {
        v
    }
}

impl PredSample for u16 {
    #[inline(always)]
    fn clipped(v: i32) -> Self {
        v as u16
    }

    #[inline(always)]
    fn default_row(
        out: &mut [u16],
        p0: &[i32],
        p1: Option<&[i32]>,
        shift: i32,
        offset: i32,
        max: i32,
    ) -> bool {
        simd::pred_default_row(out, p0, p1, shift, offset, max)
    }
}

/// The first `n` elements of `buf`, growing it (never shrinking) to fit;
/// callers overwrite every one of them.
fn prefix(buf: &mut Vec<i32>, n: usize) -> &mut [i32] {
    if buf.len() < n {
        buf.resize(n, 0);
    }
    &mut buf[..n]
}

// ---------------------------------------------------------------------------
// §8.5.3.3.4.3 — explicit weighted sample prediction
// ---------------------------------------------------------------------------

/// §8.5.3.3.4.3 — combine the L0 / L1 intermediate prediction arrays with
/// per-reference explicit weights and offsets (the `weightedPredFlag == 1`
/// path, equations 8-265..8-277).
///
/// `log2_weight_denom` is the component's raw denominator
/// (`luma_log2_weight_denom` for luma, `ChromaLog2WeightDenom` for
/// chroma); the process adds `shift1 = Max(2, 14 − bitDepth)` internally
/// (equations 8-265 / 8-270). `w0` / `w1` are `LumaWeightLX[refIdxLX]` or
/// `ChromaWeightLX[refIdxLX][cIdx−1]`; `o0` / `o1` are the offsets
/// **already scaled** by `WpOffsetBdShiftY` / `WpOffsetBdShiftC`
/// (equations 8-268 / 8-269 / 8-273 / 8-274). An unused list's weight /
/// offset values are ignored.
///
/// # Errors
/// Same contract as [`default_weighted_pred`].
#[allow(clippy::too_many_arguments)]
pub fn explicit_weighted_pred(
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    log2_weight_denom: u8,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    check_combine(
        pred_l0,
        pred_l1,
        pred_flag_l0,
        pred_flag_l1,
        n_pb_w,
        n_pb_h,
        bit_depth,
    )?;
    let pred = PlanePrediction {
        p0: pred_flag_l0.then_some(pred_l0),
        p1: pred_flag_l1.then_some(pred_l1),
        width: n_pb_w,
        height: n_pb_h,
        combine: Combine::new(bit_depth, Some((log2_weight_denom, w0, o0, w1, o1))),
    };
    Ok(pred.to_vec())
}

/// One reference list's §8.5.3.3.4.3 weights / offsets for one PU,
/// resolved for its `refIdxLX`: `w` is `LumaWeightLX[refIdx]` /
/// `ChromaWeightLX[refIdx][j]`; `o` is the corresponding offset already
/// scaled by `WpOffsetBdShiftY` / `WpOffsetBdShiftC` (equations
/// 8-268 / 8-269 / 8-273 / 8-274).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WpListWeights {
    /// `LumaWeightLX[refIdxLX]` (equations 8-266 / 8-267).
    pub w_luma: i32,
    /// `luma_offset_lX[refIdxLX] << WpOffsetBdShiftY`.
    pub o_luma: i32,
    /// `ChromaWeightLX[refIdxLX][0]` (Cb).
    pub w_cb: i32,
    /// `ChromaOffsetLX[refIdxLX][0] << WpOffsetBdShiftC` (Cb).
    pub o_cb: i32,
    /// `ChromaWeightLX[refIdxLX][1]` (Cr).
    pub w_cr: i32,
    /// `ChromaOffsetLX[refIdxLX][1] << WpOffsetBdShiftC` (Cr).
    pub o_cr: i32,
}

/// The complete §8.5.3.3.4.3 inputs for one PU's weighted combine: the
/// two log2 denominators plus each list's per-component weights, already
/// resolved for the PU's reference indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PuWeights {
    /// `luma_log2_weight_denom` (§7.4.7.3).
    pub luma_log2_weight_denom: u8,
    /// `ChromaLog2WeightDenom` (§7.4.7.3).
    pub chroma_log2_weight_denom: u8,
    /// L0 weights (ignored when `predFlagL0 == 0`).
    pub l0: WpListWeights,
    /// L1 weights (ignored when `predFlagL1 == 0`).
    pub l1: WpListWeights,
}

// ---------------------------------------------------------------------------
// §8.5.3.3.1 — inter prediction sample block-walk driver
// ---------------------------------------------------------------------------

/// A `[mvLX[0], mvLX[1]]` motion vector in quarter-luma-sample units
/// (the §8.5.3 luma MV) — the integer / fractional split of equations
/// 8-214..8-217 is performed by the driver.
pub type MotionVector = [i32; 2];

/// One reference list's prediction inputs for a prediction unit: the
/// reference picture planes selected by §8.5.3.3.2 plus the luma motion
/// vector mvLX (quarter-pel) and chroma motion vector mvCLX (eighth-pel,
/// already derived per §8.5.3.2.10). `pred_flag == false` means the list
/// is not used and the planes / vectors are ignored.
#[derive(Debug, Clone, Copy)]
pub struct ListPrediction<'a> {
    /// `predFlagLX` — whether reference list X contributes to this PU.
    pub pred_flag: bool,
    /// `refPicLXL` — the §8.5.3.3.2 luma reference plane.
    pub luma: RefPlane<'a>,
    /// `refPicLXCb` — the §8.5.3.3.2 Cb reference plane (ignored when
    /// `chroma_array_type == 0`).
    pub cb: Option<RefPlane<'a>>,
    /// `refPicLXCr` — the §8.5.3.3.2 Cr reference plane.
    pub cr: Option<RefPlane<'a>>,
    /// `mvLX` in quarter-luma-sample units (equations 8-214..8-217).
    pub mv_l: MotionVector,
    /// `mvCLX` in eighth-chroma-sample units (equations 8-218..8-221,
    /// derived from `mvLX` by §8.5.3.2.10).
    pub mv_c: MotionVector,
}

/// The reconstructed prediction-sample planes for one inter prediction
/// block, produced by [`predict_inter_pu`]. Each plane is row-major and
/// holds the final clipped `[0, (1 << bitDepth) − 1]` prediction samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterPrediction {
    /// `predSamplesL` — `nPbW * nPbH` luma prediction samples.
    pub luma: Vec<i32>,
    /// `predSamplesCb` — `(nPbW / SubWidthC) * (nPbH / SubHeightC)` Cb
    /// prediction samples, empty when `chroma_array_type == 0`.
    pub cb: Vec<i32>,
    /// `predSamplesCr` — Cr prediction samples, empty when monochrome.
    pub cr: Vec<i32>,
}

/// §8.5.3.3.1 — geometry / format inputs constant for one PU prediction.
#[derive(Debug, Clone, Copy)]
pub struct InterPredGeometry {
    /// `xPb = xCb + xBl` (equation 8-212) — the PU's luma top-left x.
    pub x_pb: i32,
    /// `yPb = yCb + yBl` (equation 8-213) — the PU's luma top-left y.
    pub y_pb: i32,
    /// `nPbW` — luma prediction-block width.
    pub n_pb_w: usize,
    /// `nPbH` — luma prediction-block height.
    pub n_pb_h: usize,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
}

/// `(SubWidthC, SubHeightC)` from Table 6-1 (mirrors
/// [`crate::picture::sub_wh_c`] without the cross-module dependency).
#[inline]
fn sub_wh_c_local(chroma_array_type: u8) -> (i32, i32) {
    match chroma_array_type {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (2, 2),
    }
}

/// The per-list working buffers of one PU prediction: the interpolation
/// window / rows and the two lists' intermediate prediction arrays.
#[derive(Default)]
struct PredWork {
    interp: InterpScratch,
    l0: Vec<i32>,
    l1: Vec<i32>,
}

/// Reusable per-thread buffers of the interpolation entry points and the
/// PU driver.
#[derive(Default)]
struct PredScratch {
    work: PredWork,
}

thread_local! {
    static PRED_SCRATCH: RefCell<PredScratch> = RefCell::new(PredScratch::default());
}

/// Runs `f` with this thread's [`PredScratch`], or with fresh buffers
/// when they are already borrowed further up the stack.
fn with_scratch<R>(f: impl FnOnce(&mut PredScratch) -> R) -> R {
    PRED_SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => f(&mut scratch),
        Err(_) => f(&mut PredScratch::default()),
    })
}

/// Fill one list's intermediate luma prediction array for a PU into the
/// first `nPbW * nPbH` elements of `out` (§8.5.3.3.3.1 equations
/// 8-214..8-217 + the §8.5.3.3.3.2 interpolation); returns that count.
/// `xPb`/`yPb` are added inside the integer split.
fn list_luma_into(
    list: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    scratch: &mut InterpScratch,
    out: &mut Vec<i32>,
) -> Result<usize, InterPredError> {
    // §8.5.3.3.3.1: xIntL = xPb + (mvLX[0] >> 2), xFracL = mvLX[0] & 3.
    let x_int = geom.x_pb + (list.mv_l[0] >> 2);
    let y_int = geom.y_pb + (list.mv_l[1] >> 2);
    let x_frac = list.mv_l[0] & 3;
    let y_frac = list.mv_l[1] & 3;
    let (w, h) = (geom.n_pb_w, geom.n_pb_h);
    check_interp(w, h, x_frac, y_frac, 3, geom.bit_depth_luma)?;
    let kernel = |f: i32| (f != 0).then(|| &LUMA_FILTER[f as usize]);
    interp_into(
        &list.luma,
        x_int,
        y_int,
        kernel(x_frac),
        kernel(y_frac),
        w,
        geom.bit_depth_luma,
        scratch,
        prefix(out, w * h),
    );
    Ok(w * h)
}

/// Fill one list's intermediate chroma prediction array for a PU into the
/// first `(nPbW / SubWidthC) * (nPbH / SubHeightC)` elements of `out`
/// (§8.5.3.3.3.1 equations 8-218..8-221 + §8.5.3.3.3.3 interpolation);
/// returns that count.
#[allow(clippy::too_many_arguments)]
fn list_chroma_into(
    plane: &RefPlane<'_>,
    list: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    sub_w: i32,
    sub_h: i32,
    scratch: &mut InterpScratch,
    out: &mut Vec<i32>,
) -> Result<usize, InterPredError> {
    // §8.5.3.3.3.1: xIntC = (xPb / SubWidthC) + (mvCLX[0] >> 3),
    //               xFracC = mvCLX[0] & 7.
    let x_int = geom.x_pb / sub_w + (list.mv_c[0] >> 3);
    let y_int = geom.y_pb / sub_h + (list.mv_c[1] >> 3);
    let x_frac = list.mv_c[0] & 7;
    let y_frac = list.mv_c[1] & 7;
    let (w, h) = (geom.n_pb_w / sub_w as usize, geom.n_pb_h / sub_h as usize);
    check_interp(w, h, x_frac, y_frac, 7, geom.bit_depth_chroma)?;
    let kernel = |f: i32| (f != 0).then(|| &CHROMA_FILTER[f as usize]);
    interp_into(
        plane,
        x_int,
        y_int,
        kernel(x_frac),
        kernel(y_frac),
        w,
        geom.bit_depth_chroma,
        scratch,
        prefix(out, w * h),
    );
    Ok(w * h)
}

/// §8.5.3.3.1 — drive the inter-prediction sample process for one
/// prediction block: split each used list's motion vector into its
/// integer / fractional parts, run the §8.5.3.3.3 fractional-sample
/// interpolation over the whole block for luma and (when chroma is
/// present) Cb / Cr, then combine the L0 / L1 intermediate arrays with
/// the §8.5.3.3.4.2 default weighted sample prediction.
///
/// This is the `weightedPredFlag == 0` path of §8.5.3.3.4.1; see
/// [`predict_inter_pu_weighted`] for the full dispatch including the
/// §8.5.3.3.4.3 explicit-weighting path.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero PU dimension,
/// [`InterPredError::EmptyPlane`] when neither list is used, and the
/// interpolation / combine errors propagated from the primitives.
pub fn predict_inter_pu(
    l0: &ListPrediction<'_>,
    l1: &ListPrediction<'_>,
    geom: &InterPredGeometry,
) -> Result<InterPrediction, InterPredError> {
    predict_inter_pu_weighted(l0, l1, geom, None)
}

/// §8.5.3.3.1 + §8.5.3.3.4.1 — as [`predict_inter_pu`], with the
/// weighted-sample-prediction dispatch: `weights == None` is the
/// `weightedPredFlag == 0` default combine (§8.5.3.3.4.2);
/// `weights == Some(..)` is the explicit per-reference combine
/// (§8.5.3.3.4.3), with the PU's `refIdxLX`-resolved weights carried in
/// the [`PuWeights`].
///
/// # Errors
/// Same contract as [`predict_inter_pu`].
pub fn predict_inter_pu_weighted(
    l0: &ListPrediction<'_>,
    l1: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    weights: Option<&PuWeights>,
) -> Result<InterPrediction, InterPredError> {
    let mut planes: [Vec<i32>; 3] = Default::default();
    for_each_plane_prediction(l0, l1, geom, weights, |c_idx, pred| {
        planes[c_idx] = pred.to_vec();
    })?;
    let [luma, cb, cr] = planes;
    Ok(InterPrediction { luma, cb, cr })
}

/// [`predict_inter_pu_weighted`] one plane at a time without allocating:
/// `sink` receives each component index (`0` luma, `1` Cb, `2` Cr; no
/// chroma when monochrome) with its [`PlanePrediction`], which borrows
/// this thread's reusable buffers.
///
/// # Errors
/// Same contract as [`predict_inter_pu`].
pub(crate) fn for_each_plane_prediction(
    l0: &ListPrediction<'_>,
    l1: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    weights: Option<&PuWeights>,
    mut sink: impl FnMut(usize, &PlanePrediction<'_>),
) -> Result<(), InterPredError> {
    if geom.n_pb_w == 0 || geom.n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !l0.pred_flag && !l1.pred_flag {
        return Err(InterPredError::EmptyPlane);
    }

    with_scratch(|scratch| {
        let work = &mut scratch.work;
        // Luma: interpolate each used list, then combine per §8.5.3.3.4.1.
        let (mut n0, mut n1) = (0, 0);
        if l0.pred_flag {
            n0 = list_luma_into(l0, geom, &mut work.interp, &mut work.l0)?;
        }
        if l1.pred_flag {
            n1 = list_luma_into(l1, geom, &mut work.interp, &mut work.l1)?;
        }
        let luma_weights = weights.map(|wp| {
            (
                wp.luma_log2_weight_denom,
                wp.l0.w_luma,
                wp.l0.o_luma,
                wp.l1.w_luma,
                wp.l1.o_luma,
            )
        });
        sink(
            0,
            &PlanePrediction {
                p0: l0.pred_flag.then(|| &work.l0[..n0]),
                p1: l1.pred_flag.then(|| &work.l1[..n1]),
                width: geom.n_pb_w,
                height: geom.n_pb_h,
                combine: Combine::new(geom.bit_depth_luma, luma_weights),
            },
        );

        if geom.chroma_array_type != 0 {
            let sub = sub_wh_c_local(geom.chroma_array_type);
            let wp_cb = weights.map(|wp| {
                (
                    wp.chroma_log2_weight_denom,
                    wp.l0.w_cb,
                    wp.l0.o_cb,
                    wp.l1.w_cb,
                    wp.l1.o_cb,
                )
            });
            let wp_cr = weights.map(|wp| {
                (
                    wp.chroma_log2_weight_denom,
                    wp.l0.w_cr,
                    wp.l0.o_cr,
                    wp.l1.w_cr,
                    wp.l1.o_cr,
                )
            });
            sink(
                1,
                &chroma_prediction(l0, l1, geom, sub, wp_cb, |lp| lp.cb, work)?,
            );
            sink(
                2,
                &chroma_prediction(l0, l1, geom, sub, wp_cr, |lp| lp.cr, work)?,
            );
        }
        Ok(())
    })
}

/// Interpolate one chroma component of a PU into `work` and describe its
/// §8.5.3.3.4 combine. `select` picks the Cb or Cr reference plane from a
/// [`ListPrediction`]; `wp` is `Some((ChromaLog2WeightDenom, w0, o0, w1,
/// o1))` for the §8.5.3.3.4.3 explicit path, `None` for the §8.5.3.3.4.2
/// default.
fn chroma_prediction<'w, 'a>(
    l0: &ListPrediction<'a>,
    l1: &ListPrediction<'a>,
    geom: &InterPredGeometry,
    (sub_w, sub_h): (i32, i32),
    wp: Option<(u8, i32, i32, i32, i32)>,
    select: impl Fn(&ListPrediction<'a>) -> Option<RefPlane<'a>>,
    work: &'w mut PredWork,
) -> Result<PlanePrediction<'w>, InterPredError> {
    let (mut n0, mut n1) = (0, 0);
    if l0.pred_flag {
        let plane = select(l0).ok_or(InterPredError::EmptyPlane)?;
        n0 = list_chroma_into(
            &plane,
            l0,
            geom,
            sub_w,
            sub_h,
            &mut work.interp,
            &mut work.l0,
        )?;
    }
    if l1.pred_flag {
        let plane = select(l1).ok_or(InterPredError::EmptyPlane)?;
        n1 = list_chroma_into(
            &plane,
            l1,
            geom,
            sub_w,
            sub_h,
            &mut work.interp,
            &mut work.l1,
        )?;
    }
    Ok(PlanePrediction {
        p0: l0.pred_flag.then(|| &work.l0[..n0]),
        p1: l1.pred_flag.then(|| &work.l1[..n1]),
        width: geom.n_pb_w / sub_w as usize,
        height: geom.n_pb_h / sub_h as usize,
        combine: Combine::new(geom.bit_depth_chroma, wp),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat plane interpolates to the constant sample value scaled to
    /// the internal precision: every tap kernel sums to 64, so
    /// `64 * v >> shift1` at 8-bit (`shift1 == 0`) is `64 * v`, matching
    /// the `v << shift3` (`shift3 == 6`) full-pel value.
    #[test]
    fn luma_flat_plane_constant() {
        let plane_samples = vec![100u16; 16 * 16];
        let plane = RefPlane::new(&plane_samples, 16, 16).unwrap();
        for xf in 0..=3 {
            for yf in 0..=3 {
                let blk = interp_luma_block(&plane, 5, 5, xf, yf, 4, 4, 8).unwrap();
                for &s in &blk {
                    assert_eq!(s, 100 << 6, "xf={xf} yf={yf}");
                }
            }
        }
    }

    /// Full-pel luma is `A << shift3`; at 8-bit `shift3 == 6`.
    #[test]
    fn luma_full_pel_shift3() {
        let mut s = vec![0u16; 8 * 8];
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as u16;
        }
        let plane = RefPlane::new(&s, 8, 8).unwrap();
        let blk = interp_luma_block(&plane, 2, 3, 0, 0, 2, 2, 8).unwrap();
        // predSamples[0][0] = A(2,3) << 6 = (3*8 + 2) << 6 = 26 << 6.
        assert_eq!(blk[0], (3 * 8 + 2) << 6);
        // predSamples[1][1] = A(3,4) << 6 = (4*8 + 3) << 6 = 35 << 6.
        assert_eq!(blk[3], (4 * 8 + 3) << 6);
    }

    /// The luma `a` kernel (xFracL == 1) on a known column reproduces
    /// equation 8-224 hand-computed.
    #[test]
    fn luma_a_kernel_hand_value() {
        // A row of samples; pick a center so the 8 taps land inside.
        // Coords x = −3..4 around x_int = 5 -> indices 2..9.
        let mut s = vec![0u16; 16];
        let vals = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        s.copy_from_slice(&vals);
        let plane = RefPlane::new(&s, 16, 1).unwrap();
        let blk = interp_luma_block(&plane, 5, 0, 1, 0, 1, 1, 8).unwrap();
        // a = −A−3 + 4A−2 − 10A−1 + 58A0 + 17A1 − 5A2 + A3, >> shift1(=0).
        // A−3..A3 = x=2..8 = 30,40,50,60,70,80,90.
        let expected = -30 + 4 * 40 - 10 * 50 + 58 * 60 + 17 * 70 - 5 * 80 + 90;
        assert_eq!(blk[0], expected);
    }

    /// Chroma flat plane interpolates to the constant value (each 4-tap
    /// kernel sums to 64) for all 8x8 eighth-pel phases.
    #[test]
    fn chroma_flat_plane_constant() {
        let plane_samples = vec![77u16; 12 * 12];
        let plane = RefPlane::new(&plane_samples, 12, 12).unwrap();
        for xf in 0..=7 {
            for yf in 0..=7 {
                let blk = interp_chroma_block(&plane, 4, 4, xf, yf, 3, 3, 8).unwrap();
                for &s in &blk {
                    assert_eq!(s, 77 << 6, "xf={xf} yf={yf}");
                }
            }
        }
    }

    /// Chroma `ab` kernel (xFracC == 1) reproduces equation 8-241.
    #[test]
    fn chroma_ab_kernel_hand_value() {
        let s = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let plane = RefPlane::new(&s, 8, 1).unwrap();
        // x_int = 3, taps x = −1..2 -> x=2,3,4,5 = 30,40,50,60.
        let blk = interp_chroma_block(&plane, 3, 0, 1, 0, 1, 1, 8).unwrap();
        // ab = −2B−1 + 58B0 + 10B1 − 2B2, >> shift1(=0).
        let expected = -2 * 30 + 58 * 40 + 10 * 50 - 2 * 60;
        assert_eq!(blk[0], expected);
    }

    /// Edge extension clamps negative / past-edge coordinates.
    #[test]
    fn ref_plane_edge_extension() {
        let s = vec![1, 2, 3, 4, 5, 6]; // 3x2
        let plane = RefPlane::new(&s, 3, 2).unwrap();
        assert_eq!(plane.at(-5, -5), 1); // top-left corner
        assert_eq!(plane.at(99, 99), 6); // bottom-right corner
        assert_eq!(plane.at(1, -1), 2); // clamp y to row 0
        assert_eq!(plane.at(99, 1), 6); // clamp x to col 2, row 1
    }

    /// Uni-predictive L0 default weight: (p + offset1) >> shift1, clipped.
    #[test]
    fn weighted_uni_l0() {
        // 8-bit: shift1 = Max(2, 6) = 6, offset1 = 32.
        let p0 = vec![100 << 6, 0, 200 << 6, 255 << 6];
        let out = default_weighted_pred(&p0, &[], true, false, 2, 2, 8).unwrap();
        assert_eq!(out[0], ((100 << 6) + 32) >> 6); // = 100
        assert_eq!(out[1], 32 >> 6); // = 0
        assert_eq!(out[2], ((200 << 6) + 32) >> 6); // = 200
        assert_eq!(out[3], ((255 << 6) + 32) >> 6); // = 255
    }

    /// Bi-predictive default weight: (p0 + p1 + offset2) >> shift2.
    #[test]
    fn weighted_bi() {
        // 8-bit: shift2 = Max(3, 7) = 7, offset2 = 64.
        let p0 = vec![100 << 6];
        let p1 = vec![140 << 6];
        let out = default_weighted_pred(&p0, &p1, true, true, 1, 1, 8).unwrap();
        // ((100<<6) + (140<<6) + 64) >> 7 = (6400 + 8960 + 64) >> 7 = 15424>>7 = 120.
        assert_eq!(out[0], ((100 << 6) + (140 << 6) + 64) >> 7);
        assert_eq!(out[0], 120);
    }

    /// Default weight clips to the sample range.
    #[test]
    fn weighted_clips() {
        let p0 = vec![-50 << 6, 1000 << 6];
        let out = default_weighted_pred(&p0, &[], true, false, 2, 1, 8).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 255);
    }

    /// Explicit uni-predictive L0 weight (equation 8-275): 8-bit,
    /// denom 3 → log2Wd = 9, w0 = 7, o0 = 2:
    /// `((100·64·7 + 256) >> 9) + 2 = 88 + 2`.
    #[test]
    fn explicit_weighted_uni_l0() {
        let p0 = vec![100 << 6];
        let out = explicit_weighted_pred(&p0, &[], true, false, 1, 1, 3, 7, 2, 0, 0, 8).unwrap();
        assert_eq!(out[0], 90);
    }

    /// Explicit uni-predictive L1 weight (equation 8-276) mirrors L0.
    #[test]
    fn explicit_weighted_uni_l1() {
        let p1 = vec![100 << 6];
        let out = explicit_weighted_pred(&[], &p1, false, true, 1, 1, 3, 0, 0, 7, 2, 8).unwrap();
        assert_eq!(out[0], 90);
    }

    /// Explicit bi-predictive combine (equation 8-277): denom 0 →
    /// log2Wd = 6, w0 = 1, w1 = 3, zero offsets:
    /// `(50·64 + 100·64·3 + 64) >> 7 = 175`.
    #[test]
    fn explicit_weighted_bi() {
        let p0 = vec![50 << 6];
        let p1 = vec![100 << 6];
        let out = explicit_weighted_pred(&p0, &p1, true, true, 1, 1, 0, 1, 0, 3, 0, 8).unwrap();
        assert_eq!(out[0], 175);
        // Bi offsets enter as ((o0 + o1 + 1) << log2Wd) >> (log2Wd + 1)
        // ≈ (o0 + o1 + 1) / 2 added to the weighted mean.
        let out = explicit_weighted_pred(&p0, &p1, true, true, 1, 1, 0, 1, 10, 3, 9, 8).unwrap();
        assert_eq!(out[0], 175 + 10);
    }

    /// The explicit combine clips to the sample range on both sides.
    #[test]
    fn explicit_weighted_clips() {
        let p0 = vec![100 << 6, 200 << 6];
        // Large negative offset floors at 0; weight 127 saturates at 255.
        let out = explicit_weighted_pred(&p0, &[], true, false, 2, 1, 0, 1, -128, 0, 0, 8).unwrap();
        assert_eq!(out[0], 0);
        let out = explicit_weighted_pred(&p0, &[], true, false, 2, 1, 0, 127, 0, 0, 0, 8).unwrap();
        assert_eq!(out[1], 255);
    }

    /// With weight `1 << denom` and zero offset, the explicit combine
    /// degenerates to the default uni combine (§7.4.7.3 inferred values).
    #[test]
    fn explicit_default_weights_match_default_combine() {
        let p0: Vec<i32> = (0..16).map(|v| v << 6).collect();
        for denom in 0..=7u8 {
            let explicit =
                explicit_weighted_pred(&p0, &[], true, false, 4, 4, denom, 1 << denom, 0, 0, 0, 8)
                    .unwrap();
            let default = default_weighted_pred(&p0, &[], true, false, 4, 4, 8).unwrap();
            assert_eq!(explicit, default, "denom={denom}");
        }
    }

    /// Explicit-combine argument validation matches the default combine.
    #[test]
    fn explicit_weighted_errors() {
        assert_eq!(
            explicit_weighted_pred(&[], &[], false, false, 1, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::EmptyPlane)
        );
        assert_eq!(
            explicit_weighted_pred(&[1, 2], &[], true, false, 1, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::ArrayLengthMismatch {
                expected: 1,
                got: 2
            })
        );
        assert_eq!(
            explicit_weighted_pred(&[1], &[], true, false, 0, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::EmptyBlock)
        );
        assert_eq!(
            explicit_weighted_pred(&[1], &[], true, false, 1, 1, 0, 1, 0, 1, 0, 7),
            Err(InterPredError::InvalidBitDepth(7))
        );
    }

    /// 10-bit full-pel luma uses shift3 = Max(2, 4) = 4.
    #[test]
    fn luma_full_pel_10bit() {
        let s = vec![500u16; 8 * 8];
        let plane = RefPlane::new(&s, 8, 8).unwrap();
        let blk = interp_luma_block(&plane, 2, 2, 0, 0, 1, 1, 10).unwrap();
        assert_eq!(blk[0], 500 << 4);
    }

    /// Error surface: zero block, bad fraction, bad bit depth, bad plane.
    #[test]
    fn errors() {
        let s = vec![0u16; 4];
        let plane = RefPlane::new(&s, 2, 2).unwrap();
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 0, 0, 0, 1, 8),
            Err(InterPredError::EmptyBlock)
        );
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 4, 0, 1, 1, 8),
            Err(InterPredError::InvalidFraction(4))
        );
        assert_eq!(
            interp_chroma_block(&plane, 0, 0, 8, 0, 1, 1, 8),
            Err(InterPredError::InvalidFraction(8))
        );
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 0, 0, 1, 1, 7),
            Err(InterPredError::InvalidBitDepth(7))
        );
        assert!(matches!(
            RefPlane::new(&[0, 1, 2], 2, 2),
            Err(InterPredError::PlaneLengthMismatch { .. })
        ));
        assert_eq!(RefPlane::new(&[], 0, 2), Err(InterPredError::EmptyPlane));
        assert_eq!(
            default_weighted_pred(&[], &[], false, false, 1, 1, 8),
            Err(InterPredError::EmptyPlane)
        );
        assert!(matches!(
            default_weighted_pred(&[1, 2], &[], true, false, 1, 1, 8),
            Err(InterPredError::ArrayLengthMismatch { .. })
        ));
    }

    /// End-to-end: interpolate two reference blocks and bi-combine.
    #[test]
    fn pipeline_luma_bi() {
        let a = vec![80u16; 16 * 16];
        let b = vec![120u16; 16 * 16];
        let pa = RefPlane::new(&a, 16, 16).unwrap();
        let pb = RefPlane::new(&b, 16, 16).unwrap();
        let l0 = interp_luma_block(&pa, 4, 4, 2, 2, 4, 4, 8).unwrap();
        let l1 = interp_luma_block(&pb, 4, 4, 1, 3, 4, 4, 8).unwrap();
        let out = default_weighted_pred(&l0, &l1, true, true, 4, 4, 8).unwrap();
        // Flat planes: l0 == 80<<6 everywhere, l1 == 120<<6; bi-combine
        // = ((80+120)<<6 + 64) >> 7 = (12800 + 64) >> 7 = 100.
        for &s in &out {
            assert_eq!(s, 100);
        }
    }

    // -- §8.5.3.3.1 driver tests -------------------------------------------

    /// A full-pel uni-L0 PU on a flat luma plane reproduces the reference
    /// sample value (full-pel: `A << shift3`, then default-weight
    /// `(p + offset1) >> shift1` recovers `A`).
    #[test]
    fn driver_uni_l0_full_pel_flat() {
        let luma = vec![130u16; 32 * 32];
        let cb = vec![70u16; 16 * 16];
        let cr = vec![200u16; 16 * 16];
        let lp = RefPlane::new(&luma, 32, 32).unwrap();
        let cbp = RefPlane::new(&cb, 16, 16).unwrap();
        let crp = RefPlane::new(&cr, 16, 16).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: lp,
            cb: Some(cbp),
            cr: Some(crp),
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        // Unused L1: a dummy (1x1) plane that is never read.
        let dummy = vec![0u16; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let l1 = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 4,
            y_pb: 4,
            n_pb_w: 8,
            n_pb_h: 8,
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        assert_eq!(pred.luma.len(), 64);
        assert_eq!(pred.cb.len(), 16);
        assert_eq!(pred.cr.len(), 16);
        assert!(pred.luma.iter().all(|&v| v == 130));
        assert!(pred.cb.iter().all(|&v| v == 70));
        assert!(pred.cr.iter().all(|&v| v == 200));
    }

    /// A full-pel motion vector shifts the reference window: a ramp plane
    /// predicted with `mvL = [4, 0]` (one full luma sample right) reads
    /// the column one to the right of `xPb`.
    #[test]
    fn driver_full_pel_mv_shifts_window() {
        // 16-wide luma ramp where sample(x,y) == x.
        let mut luma = vec![0u16; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                luma[y * 16 + x] = x as u16;
            }
        }
        let lp = RefPlane::new(&luma, 16, 16).unwrap();
        let dummy = vec![0u16; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: lp,
            cb: None,
            cr: None,
            mv_l: [4, 0], // +1 full luma sample horizontally.
            mv_c: [0, 0],
        };
        let l1 = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 2,
            y_pb: 2,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        // predSamples[xL] reads ref column xPb + 1 + xL = 3 + xL.
        for yl in 0..4 {
            for xl in 0..4 {
                assert_eq!(pred.luma[yl * 4 + xl], 3 + xl as i32, "xl={xl}");
            }
        }
        assert!(pred.cb.is_empty(), "monochrome PU has no chroma");
    }

    /// Bi-prediction on two flat planes averages the two reference values.
    #[test]
    fn driver_bi_averages() {
        let a = vec![60u16; 16 * 16];
        let b = vec![100u16; 16 * 16];
        let pa = RefPlane::new(&a, 16, 16).unwrap();
        let pb = RefPlane::new(&b, 16, 16).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: pa,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let l1 = ListPrediction {
            pred_flag: true,
            luma: pb,
            cb: None,
            cr: None,
            mv_l: [2, 1], // quarter-pel; flat plane is unaffected.
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 4,
            y_pb: 4,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        // ((60 + 100) >> 1) == 80.
        assert!(pred.luma.iter().all(|&v| v == 80));
    }

    /// The driver rejects a PU with no list selected and a zero block.
    #[test]
    fn driver_errors() {
        let dummy = vec![0u16; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let none = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 0,
            y_pb: 0,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        assert_eq!(
            predict_inter_pu(&none, &none, &geom),
            Err(InterPredError::EmptyPlane)
        );
        let l0 = ListPrediction {
            pred_flag: true,
            ..none
        };
        let zero = InterPredGeometry { n_pb_w: 0, ..geom };
        assert_eq!(
            predict_inter_pu(&l0, &none, &zero),
            Err(InterPredError::EmptyBlock)
        );
    }
}
