//! The architecture kernels (`oxideav_h265::simd`) against the portable
//! ones on complete decodes, both run in this one binary: every output
//! frame (planes and strides) and every reported error must match bit for
//! bit and in order. The corpus is every embedded and `tests/fixture_bytes`
//! stream, every original FATE HEVC conformance stream (`FATE_SUITE`,
//! default `~/projects/fate-suite`; required), and the seeded VCL mutation
//! corpus of `availability_recovery.rs`, each serial and two-thread.
//! Elsewhere than aarch64 only the portable kernels exist.
#![cfg(target_arch = "aarch64")]

mod fixture_bytes;

use fixture_bytes::*;
use oxideav_core::{CodecParameters, Decoder, Error, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h265::simd;
use std::path::{Path, PathBuf};

/// One step of a decode: an output frame's `(stride, data)` planes, or a
/// reported error.
#[derive(PartialEq)]
enum Event {
    Frame(Vec<(usize, Vec<u8>)>),
    Error(String),
}

fn receive(decoder: &mut dyn Decoder, sink: &mut impl FnMut(Event)) {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(v)) => {
                sink(Event::Frame(
                    v.planes.into_iter().map(|p| (p.stride, p.data)).collect(),
                ));
            }
            Ok(_) => {}
            Err(Error::NeedMore | Error::Eof) => break,
            Err(e) => {
                sink(Event::Error(format!("receive: {e}")));
                break;
            }
        }
    }
}

/// Decode `data` NAL unit by NAL unit through the registry decoder with a
/// `threads` budget, reporting every frame and error to `sink` in order.
fn decode(data: &[u8], threads: usize, mut sink: impl FnMut(Event)) {
    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("registry decoder");
    decoder.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut starts: Vec<_> = data
        .windows(3)
        .enumerate()
        .filter_map(|(i, bytes)| (bytes == [0, 0, 1]).then_some(i))
        .collect();
    starts.push(data.len());
    for (k, unit) in starts.windows(2).enumerate() {
        let packet = Packet::new(0, TimeBase::new(1, 1), data[unit[0]..unit[1]].to_vec());
        if let Err(e) = decoder.send_packet(&packet) {
            sink(Event::Error(format!("send {k}: {e}")));
        }
        receive(decoder.as_mut(), &mut sink);
    }
    if let Err(e) = decoder.flush() {
        sink(Event::Error(format!("flush: {e}")));
    }
    receive(decoder.as_mut(), &mut sink);
}

/// The portable decode of `data`, then the architecture decode checked
/// against it event by event; returns the number of output frames.
fn compare(label: &str, data: &[u8], threads: usize) -> usize {
    simd::set_enabled(false);
    let mut reference = Vec::new();
    decode(data, threads, |e| reference.push(e));
    simd::set_enabled(true);
    let mut at = 0;
    decode(data, threads, |e| {
        assert!(
            reference.get(at) == Some(&e),
            "{label} ({threads} threads): event {at} differs"
        );
        at += 1;
    });
    assert_eq!(
        at,
        reference.len(),
        "{label} ({threads} threads): event count"
    );
    reference
        .iter()
        .filter(|e| matches!(e, Event::Frame(_)))
        .count()
}

/// Every file under `dir` (recursively) whose name ends in `suffix`, or
/// every file when `suffix` is empty.
fn files(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files(&path, suffix, out);
        } else if path.to_string_lossy().ends_with(suffix) {
            out.push(path);
        }
    }
}

#[test]
fn architecture_kernels_match_portable_on_complete_decodes() {
    let embedded: [(&str, &[u8]); 29] = [
        ("TINY_I", TINY_I_HEVC),
        ("QP_HIGH", QP_HIGH_HEVC),
        ("QP_LOW", QP_LOW_HEVC),
        ("MAIN_STILL", MAIN_STILL_HEVC),
        ("SAO_ON", SAO_ON_HEVC),
        ("ALLINTRA", ALLINTRA_HEVC),
        ("I_THEN_P", I_THEN_P_HEVC),
        ("MAIN10", MAIN10_HEVC),
        ("BIPRED", BIPRED_HEVC),
        ("WPP", WPP_HEVC),
        ("MULTI_SLICE", MULTI_SLICE_HEVC),
        ("TILE_COLS", TILE_COLS_HEVC),
        ("WEIGHTED_P", WEIGHTED_P_HEVC),
        ("WEIGHTED_B", WEIGHTED_B_HEVC),
        ("PERSLICE_LF", PERSLICE_LF_HEVC),
        ("TRUE_TILES", TRUE_TILES_HEVC),
        ("BPYR", r410::BPYR_HEVC),
        ("SCALING", r410::SCALING_HEVC),
        ("STRONG", r410::STRONG_HEVC),
        ("RECTAMP", r410::RECTAMP_HEVC),
        ("CI", r410::CI_HEVC),
        ("TSKIP", r410::TSKIP_HEVC),
        ("WPPSLICES", r410::WPPSLICES_HEVC),
        ("OPENGOP", r410::OPENGOP_HEVC),
        ("M422", r410::M422_HEVC),
        ("LL444", r413::LL444_HEVC),
        ("CCP", r416::CCP_HEVC),
        ("ACT", r416::ACT_HEVC),
        ("IBC", r416::IBC_HEVC),
    ];
    let mut frames = 0;
    for (name, stream) in embedded {
        for threads in [1, 2] {
            frames += compare(name, stream, threads);
        }
    }

    let mut paths = Vec::new();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixture_bytes");
    files(&fixtures, ".hevc", &mut paths);
    let fixture_files = paths.len();
    let fate = std::env::var_os("FATE_SUITE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join("projects/fate-suite")
        })
        .join("hevc-conformance");
    files(&fate, "", &mut paths);
    let fate_files = paths.len() - fixture_files;
    assert!(
        fate_files > 0,
        "no FATE HEVC conformance streams under {}",
        fate.display()
    );
    for path in &paths {
        let data = std::fs::read(path).unwrap();
        let label = path.display().to_string();
        for threads in [1, 2] {
            frames += compare(&label, &data, threads);
        }
    }

    let mut seed = MUTATION_SEED;
    let mut mutations = 0;
    for (name, stream, _) in RECOVERY_CASES {
        let domain = vcl_payload_domain(stream);
        for trial in 0..128 {
            let damaged = vcl_mutation(stream, &domain, &mut seed, trial);
            let threads = if trial % 2 == 0 { 1 } else { 2 };
            frames += compare(&format!("{name} mutation {trial}"), &damaged, threads);
            mutations += 1;
        }
    }
    eprintln!(
        "{} embedded, {fixture_files} fixture-file and {fate_files} FATE streams (serial + two \
         threads) and {mutations} mutations: {frames} frames and every error identical",
        embedded.len()
    );
}
