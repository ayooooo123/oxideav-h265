//! Original FATE conformance streams, never renamed, capped, or rewritten.
//! FATE_SUITE defaults to ~/projects/fate-suite. FFmpeg/FFprobe are required.

mod fixture_bytes;

use oxideav_core::{CodecParameters, Decoder, Error, ExecutionContext, Frame, Packet, TimeBase};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn receive(decoder: &mut dyn Decoder, hashes: &mut Vec<String>) -> Result<(), String> {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let mut bytes = Vec::new();
                for plane in frame.planes {
                    bytes.extend(plane.data);
                }
                hashes.push(fixture_bytes::md5::hex(&bytes));
            }
            Ok(_) => return Err("non-video frame".into()),
            Err(Error::NeedMore | Error::Eof) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn decode(data: &[u8], threads: usize) -> Result<Vec<String>, String> {
    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .map_err(|e| e.to_string())?;
    decoder.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut starts: Vec<_> = data
        .windows(3)
        .enumerate()
        .filter_map(|(i, bytes)| (bytes == [0, 0, 1]).then_some(i))
        .collect();
    starts.push(data.len());
    let mut hashes = Vec::new();
    for offsets in starts.windows(2) {
        decoder
            .send_packet(&Packet::new(
                0,
                TimeBase::new(1, 25),
                data[offsets[0]..offsets[1]].to_vec(),
            ))
            .map_err(|e| e.to_string())?;
        receive(decoder.as_mut(), &mut hashes)?;
    }
    decoder.flush().map_err(|e| e.to_string())?;
    receive(decoder.as_mut(), &mut hashes)?;
    Ok(hashes)
}

fn check(path: &Path, dir: &Path) -> Result<(), String> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    if !probe.status.success() {
        return Err(String::from_utf8_lossy(&probe.stderr).into_owned());
    }
    let pixel_format = String::from_utf8(probe.stdout).map_err(|e| e.to_string())?;
    let oracle = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-threads", "1", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            pixel_format.trim(),
            "-f",
            "framemd5",
            "-",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !oracle.status.success() {
        return Err(String::from_utf8_lossy(&oracle.stderr).into_owned());
    }
    let name = path.file_name().unwrap().to_str().unwrap();
    fs::write(dir.join(format!("{name}.ffmpeg.framemd5")), &oracle.stdout).unwrap();
    let text = String::from_utf8(oracle.stdout).map_err(|e| e.to_string())?;
    let expected: Vec<_> = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.rsplit(',').next().unwrap().trim().to_owned())
        .collect();
    if expected.is_empty() {
        return Err("FFmpeg produced no frames".into());
    }
    let data = fs::read(path).map_err(|e| e.to_string())?;
    for threads in [1, 2] {
        let actual = decode(&data, threads)?;
        fs::write(
            dir.join(format!("{name}.threads{threads}.md5")),
            actual.join("\n"),
        )
        .unwrap();
        if actual != expected {
            let matched = actual.iter().zip(&expected).filter(|(a, b)| a == b).count();
            return Err(format!(
                "{threads} threads: {matched}/{} hashes match; decoded {}",
                expected.len(),
                actual.len()
            ));
        }
    }
    eprintln!(
        "{name}: {0}/{0} complete {1} frames exact, serial + two threads",
        expected.len(),
        pixel_format.trim()
    );
    Ok(())
}

#[test]
fn original_fate_main_main10_and_rext_complete_output() {
    let fate = std::env::var_os("FATE_SUITE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join("projects/fate-suite")
        });
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-fate-oracles");
    fs::create_dir_all(&dir).unwrap();
    let mut failed = Vec::new();
    for name in [
        "WPP_HIGH_TP_444_8BIT_RExt_Apple_2.bit",
        "PERSIST_RPARAM_A_RExt_Sony_3.bit",
        "PERSIST_RPARAM_A_RExt_Sony_1.bit",
        "ADJUST_IPRED_ANGLE_A_RExt_Mitsubishi_1.bit",
        "IPCM_A_RExt_NEC.bit",
        "IPCM_B_RExt_NEC.bit",
        "Main_422_10_A_RExt_Sony_1.bin",
        "Main_422_10_B_RExt_Sony_1.bin",
        "QMATRIX_A_RExt_Sony_1.bit",
        "SAO_A_RExt_MediaTek_1.bit",
        "AMP_A_Samsung_6.bit",
        "AMP_B_Samsung_6.bit",
        "AMP_D_Hisilicon.bit",
        "AMP_E_Hisilicon.bit",
        "AMP_F_Hisilicon_3.bit",
        "AMVP_A_MTK_4.bit",
        "AMVP_B_MTK_4.bit",
        "AMVP_C_Samsung_6.bit",
        "MERGE_A_TI_3.bit",
        "MERGE_B_TI_3.bit",
        "MERGE_C_TI_3.bit",
        "MERGE_D_TI_3.bit",
        "MERGE_E_TI_3.bit",
        "MERGE_F_MTK_4.bit",
        "MERGE_G_HHI_4.bit",
        "PMERGE_A_TI_3.bit",
        "PMERGE_B_TI_3.bit",
        "PMERGE_C_TI_3.bit",
        "PMERGE_D_TI_3.bit",
        "PMERGE_E_TI_3.bit",
        "TILES_A_Cisco_2.bit",
        "TILES_B_Cisco_1.bit",
        "SLICES_A_Rovi_3.bit",
        "DSLICE_A_HHI_5.bit",
        "DSLICE_B_HHI_5.bit",
        "DSLICE_C_HHI_5.bit",
        "WPP_A_ericsson_MAIN_2.bit",
        "WPP_A_ericsson_MAIN10_2.bit",
        "WPP_B_ericsson_MAIN_2.bit",
        "WPP_B_ericsson_MAIN10_2.bit",
        "WPP_C_ericsson_MAIN_2.bit",
        "WPP_C_ericsson_MAIN10_2.bit",
        "WPP_D_ericsson_MAIN_2.bit",
        "WPP_D_ericsson_MAIN10_2.bit",
        "WPP_E_ericsson_MAIN_2.bit",
        "WPP_E_ericsson_MAIN10_2.bit",
        "WPP_F_ericsson_MAIN_2.bit",
        "WPP_F_ericsson_MAIN10_2.bit",
        "DBLK_A_MAIN10_VIXS_3.bit",
        "TSUNEQBD_A_MAIN10_Technicolor_2.bit",
        "WP_A_MAIN10_Toshiba_3.bit",
        "WP_MAIN10_B_Toshiba_3.bit",
        "RAP_A_docomo_4.bit",
        "RAP_B_Bossen_1.bit",
    ] {
        if let Err(error) = check(&fate.join("hevc-conformance").join(name), &dir) {
            eprintln!("FAIL {name}: {error}");
            failed.push(format!("{name}: {error}"));
        }
    }
    assert!(
        failed.is_empty(),
        "original FATE conformance failures: {failed:#?}"
    );
}

#[test]
fn decode_from_midstream_cra_after_reset_matches_ffmpeg() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-cra-repro");
    fs::create_dir_all(&dir).unwrap();
    let stream_path = dir.join("open_gop_repro.hevc");
    let status = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25",
            "-t",
            "4",
            "-c:v",
            "libx265",
            "-x265-params",
            "keyint=25:min-keyint=25:open-gop=1:bframes=3",
            "-f",
            "hevc",
        ])
        .arg(&stream_path)
        .status()
        .expect("ffmpeg with libx265 required");
    assert!(status.success(), "ffmpeg generated open-GOP stream");

    let data = fs::read(&stream_path).unwrap();

    // Split stream into Annex B NAL chunks and find the second CRA access unit.
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < data.len().saturating_sub(2) {
        if data[i..].starts_with(&[0, 0, 1]) {
            starts.push(i);
            i += 3;
        } else if data[i..].starts_with(&[0, 0, 0, 1]) {
            starts.push(i);
            i += 4;
        } else {
            i += 1;
        }
    }
    starts.push(data.len());

    // Locate the second CRA (nal_type == 21) and its preceding parameter sets.
    let mut cra_count = 0;
    let mut cra_offset = None;
    for &offset in &starts[..starts.len() - 1] {
        let sc = if data[offset..].starts_with(&[0, 0, 0, 1]) { 4 } else { 3 };
        let nal_type = (data[offset + sc] >> 1) & 0x3f;
        if nal_type == 32 && cra_count == 1 && cra_offset.is_none() {
            cra_offset = Some(offset);
        } else if nal_type == 21 {
            cra_count += 1;
            if cra_count == 2 && cra_offset.is_none() {
                cra_offset = Some(offset);
            }
        }
    }
    let cra_offset = cra_offset.expect("found second CRA access unit in stream");
    let cra_data = &data[cra_offset..];

    // Compute oracle hashes from FFmpeg starting from that CRA point.
    let mut oracle_cmd = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-f",
            "hevc",
            "-i",
            "-",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "framemd5",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("ffmpeg oracle process");
    {
        use std::io::Write;
        let mut stdin = oracle_cmd.stdin.take().unwrap();
        stdin.write_all(cra_data).unwrap();
    }
    let oracle_output = oracle_cmd.wait_with_output().unwrap();
    assert!(oracle_output.status.success(), "ffmpeg oracle decode");
    let oracle_text = String::from_utf8(oracle_output.stdout).unwrap();
    let expected: Vec<String> = oracle_text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.rsplit(',').next().unwrap().trim().to_owned())
        .collect();
    assert!(!expected.is_empty(), "FFmpeg produced output frames from CRA");

    // Test: decode stream up to the midstream CRA, call reset(), then decode from CRA.
    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");

    // Feed initial packets before the CRA.
    let pre_cra = &data[..cra_offset];
    let _ = decoder.send_packet(&Packet::new(0, TimeBase::new(1, 25), pre_cra.to_vec()));
    let mut pre_hashes = Vec::new();
    let _ = receive(decoder.as_mut(), &mut pre_hashes);

    // Seek / reset onto the CRA.
    decoder.reset().expect("decoder reset succeeds");

    // Feed the stream starting from the CRA.
    let mut cra_starts: Vec<usize> = Vec::new();
    let mut j = 0;
    while j < cra_data.len().saturating_sub(2) {
        if cra_data[j..].starts_with(&[0, 0, 1]) {
            cra_starts.push(j);
            j += 3;
        } else if cra_data[j..].starts_with(&[0, 0, 0, 1]) {
            cra_starts.push(j);
            j += 4;
        } else {
            j += 1;
        }
    }
    cra_starts.push(cra_data.len());

    let mut actual_hashes = Vec::new();
    for offsets in cra_starts.windows(2) {
        decoder
            .send_packet(&Packet::new(
                0,
                TimeBase::new(1, 25),
                cra_data[offsets[0]..offsets[1]].to_vec(),
            ))
            .expect("send_packet succeeds from CRA");
        receive(decoder.as_mut(), &mut actual_hashes).expect("receive frames");
    }
    decoder.flush().expect("decoder flush");
    receive(decoder.as_mut(), &mut actual_hashes).expect("receive flushed frames");

    assert_eq!(
        actual_hashes, expected,
        "decoded frames from mid-stream CRA after reset must match FFmpeg byte-exact"
    );
}
