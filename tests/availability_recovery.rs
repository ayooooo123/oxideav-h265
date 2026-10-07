//! Complete FFmpeg output checks across the availability boundaries, plus
//! deterministic VCL corruption followed by decoder recreation and full recovery.
//! FFmpeg must be installed; no missing-oracle or missing-fixture skips.

mod fixture_bytes;

use fixture_bytes::*;
use oxideav_h265::SequenceDecoder;
use std::process::Command;

fn decoded(stream: &[u8], threads: usize, wide: bool) -> (usize, Vec<u8>) {
    let mut decoder = SequenceDecoder::new();
    decoder.set_threads(threads);
    decoder.push_annexb(stream).expect("complete stream");
    let frames = decoder.finish().expect("complete flush");
    let mut bytes = Vec::new();
    for frame in &frames {
        let picture = frame.output_picture();
        if wide {
            bytes.extend(picture.to_planar_le16());
        } else {
            bytes.extend(picture.to_planar_u8().expect("8-bit output"));
        }
    }
    (frames.len(), bytes)
}

#[test]
fn complete_ffmpeg_oracles_and_seeded_mutation_recovery() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-availability-oracles");
    std::fs::create_dir_all(&dir).unwrap();
    let mut seed = MUTATION_SEED;
    for (name, stream, wide) in RECOVERY_CASES {
        let input = dir.join(format!("{name}.hevc"));
        std::fs::write(&input, stream).unwrap();
        let reference = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-threads", "1", "-i"])
            .arg(&input)
            .args([
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                if wide { "yuv420p10le" } else { "yuv420p" },
                "-f",
                "rawvideo",
                "-",
            ])
            .output()
            .expect("FFmpeg oracle required");
        assert!(
            reference.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let expected = reference.stdout;
        std::fs::write(dir.join(format!("{name}.yuv")), &expected).unwrap();
        let (count, serial) = decoded(stream, 1, wide);
        assert!(
            serial == expected,
            "{name}: complete serial output differs from FFmpeg"
        );
        let (parallel_count, parallel) = decoded(stream, 2, wide);
        assert_eq!(parallel_count, count, "{name}: band frame count");
        assert!(
            parallel == expected,
            "{name}: complete band output differs from FFmpeg"
        );

        // Change VCL payload only: retain the original VPS/SPS/PPS geometry
        // and transport headers, while reaching the actual CABAC/recon path.
        let payload = vcl_payload_domain(stream);
        assert!(!payload.is_empty(), "{name}: VCL mutation domain");
        for trial in 0..128 {
            let damaged = vcl_mutation(stream, &payload, &mut seed, trial);
            let mut decoder = SequenceDecoder::new();
            decoder.set_threads(if trial % 2 == 0 { 1 } else { 2 });
            let _ = decoder.push_annexb(&damaged);
            let _ = decoder.finish();
            // SequenceDecoder explicitly leaves state unspecified on error;
            // recreation is its recovery contract, not an invented reset API.
            let (recovered_count, recovered) = decoded(stream, 1, wide);
            assert_eq!(
                recovered_count, count,
                "{name}: recovery count, trial {trial}"
            );
            assert!(
                recovered == expected,
                "{name}: complete recovery, trial {trial}"
            );
        }
        eprintln!("{name}: {count} frames, {} bytes, FFmpeg MD5 {}, serial/bands + 128 mutations/recoveries exact",
            expected.len(), md5::hex(&expected));
    }
}
