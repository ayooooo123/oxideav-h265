//! Original FATE conformance streams, never renamed, capped, or rewritten.
//! FATE_SUITE defaults to ~/projects/fate-suite. FFmpeg/FFprobe are required.

mod fixture_bytes;

use oxideav_core::{CodecParameters, Decoder, Error, ExecutionContext, Frame, Packet, TimeBase};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

fn receive_frames(
    decoder: &mut dyn Decoder,
    out: &mut Vec<(Option<i64>, String)>,
) -> Result<(), String> {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let mut bytes = Vec::new();
                for plane in frame.planes {
                    bytes.extend(plane.data);
                }
                out.push((frame.pts, fixture_bytes::md5::hex(&bytes)));
            }
            Ok(_) => return Err("non-video frame".into()),
            Err(Error::NeedMore | Error::Eof) => return Ok(()),
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn receive(decoder: &mut dyn Decoder, hashes: &mut Vec<String>) -> Result<(), String> {
    let mut frames = Vec::new();
    receive_frames(decoder, &mut frames)?;
    hashes.extend(frames.into_iter().map(|(_, h)| h));
    Ok(())
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

static OPEN_GOP_STREAM: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();

fn ensure_open_gop_stream(dir: &Path) -> (PathBuf, PathBuf) {
    OPEN_GOP_STREAM
        .get_or_init(|| {
            fs::create_dir_all(dir).expect("create dir");
            let mp4_path = dir.join("open_gop_repro.mp4");
            let hevc_path = dir.join("open_gop_repro.hevc");
            let pid = std::process::id();
            let mp4_tmp = dir.join(format!("open_gop_repro.{pid}.tmp.mp4"));
            let hevc_tmp = dir.join(format!("open_gop_repro.{pid}.tmp.hevc"));

            if !mp4_path.exists() || !hevc_path.exists() {
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
                        "-tag:v",
                        "hvc1",
                    ])
                    .arg(&mp4_tmp)
                    .status()
                    .expect("ffmpeg with libx265 required");
                assert!(status.success(), "ffmpeg generated open-GOP stream");

                let status_hevc = Command::new("ffmpeg")
                    .args(["-nostdin", "-v", "error", "-y", "-i"])
                    .arg(&mp4_tmp)
                    .args(["-c:v", "copy", "-f", "hevc"])
                    .arg(&hevc_tmp)
                    .status()
                    .expect("ffmpeg extract hevc");
                assert!(status_hevc.success());

                fs::rename(&mp4_tmp, &mp4_path).expect("rename mp4");
                fs::rename(&hevc_tmp, &hevc_path).expect("rename hevc");
            }
            (mp4_path, hevc_path)
        })
        .clone()
}

fn probe_packet_info(mp4_path: &Path) -> Vec<(i64, bool)> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "packet=pts,flags",
            "-of",
            "csv=p=0",
        ])
        .arg(mp4_path)
        .output()
        .expect("ffprobe packet probe");
    assert!(probe.status.success());
    let probe_text = String::from_utf8(probe.stdout).unwrap();
    probe_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut parts = l.split(',');
            let raw_pts: i64 = parts.next().unwrap().trim().parse().unwrap();
            let flags = parts.next().unwrap_or("").trim();
            (raw_pts / 512, flags.contains('K'))
        })
        .collect()
}

fn oracle_framemd5(input_path: &Path) -> Vec<(i64, String)> {
    let oracle = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-copyts", "-i"])
        .arg(input_path)
        .args(["-f", "framemd5", "-"])
        .output()
        .expect("ffmpeg oracle");
    assert!(oracle.status.success());
    let oracle_text = String::from_utf8(oracle.stdout).unwrap();
    oracle_text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            let pts: i64 = parts[2].trim().parse().unwrap();
            let hash = parts[5].trim().to_owned();
            (pts, hash)
        })
        .collect()
}

fn split_access_units(data: &[u8]) -> Vec<usize> {
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

    let mut au_starts: Vec<usize> = Vec::new();
    let mut in_ps = false;
    for &off in &starts[..starts.len() - 1] {
        let sc = if data[off..].starts_with(&[0, 0, 0, 1]) { 4 } else { 3 };
        let nal_type = (data[off + sc] >> 1) & 0x3f;
        let rbsp = &data[off + sc + 2..];
        let first = if !rbsp.is_empty() { (rbsp[0] >> 7) & 1 } else { 0 };
        if nal_type == 32 || nal_type == 35 {
            au_starts.push(off);
            in_ps = true;
        } else if nal_type < 32 && first == 1 {
            if in_ps {
                in_ps = false;
            } else {
                au_starts.push(off);
            }
        }
    }
    au_starts.push(data.len());
    au_starts
}

#[test]
fn decode_from_midstream_cra_after_reset_matches_ffmpeg() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-cra-repro");
    fs::create_dir_all(&dir).unwrap();
    let (mp4_path, hevc_path) = ensure_open_gop_stream(&dir);

    let packet_info = probe_packet_info(&mp4_path);
    let all_expected = oracle_framemd5(&mp4_path);

    // Locate the second keyframe (CRA) in packet_info.
    let cra_idx = packet_info
        .iter()
        .enumerate()
        .filter(|(idx, (_, is_key))| *idx > 0 && *is_key)
        .map(|(idx, _)| idx)
        .next()
        .expect("second keyframe / CRA");
    let cra_target_pts = packet_info[cra_idx].0;
    let expected: Vec<(Option<i64>, String)> = all_expected
        .into_iter()
        .filter(|(pts, _)| *pts >= cra_target_pts)
        .map(|(pts, hash)| (Some(pts), hash))
        .collect();
    assert!(!expected.is_empty(), "FFmpeg produced output frames from CRA");

    let data = fs::read(&hevc_path).unwrap();
    let au_starts = split_access_units(&data);
    assert_eq!(au_starts.len() - 1, packet_info.len(), "AU count matches packet count");

    // Create decoder, decode initial GOP before CRA, then reset().
    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");

    for idx in 0..cra_idx {
        let au = &data[au_starts[idx]..au_starts[idx + 1]];
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au.to_vec());
        pkt.pts = Some(packet_info[idx].0);
        decoder.send_packet(&pkt).expect("pre-reset send_packet");
        let mut pre_frames = Vec::new();
        receive_frames(decoder.as_mut(), &mut pre_frames).expect("pre-reset receive_frames");
    }

    // Reset at seek point onto the CRA.
    decoder.reset().expect("decoder reset succeeds");

    // Feed from the CRA onwards, each packet stamped with its container PTS.
    let mut actual = Vec::new();
    for idx in cra_idx..packet_info.len() {
        let au = &data[au_starts[idx]..au_starts[idx + 1]];
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au.to_vec());
        pkt.pts = Some(packet_info[idx].0);
        decoder
            .send_packet(&pkt)
            .expect("send_packet succeeds from CRA");
        receive_frames(decoder.as_mut(), &mut actual).expect("receive frames");
    }
    decoder.flush().expect("decoder flush");
    receive_frames(decoder.as_mut(), &mut actual).expect("receive flushed frames");

    assert_eq!(
        actual, expected,
        "decoded frames from mid-stream CRA after reset must match FFmpeg pixels and PTS byte-exact"
    );
}

#[test]
fn untimed_b_frame_pair_packets_produce_none_pts_frames() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-cra-repro");
    fs::create_dir_all(&dir).unwrap();
    let (mp4_path, hevc_path) = ensure_open_gop_stream(&dir);

    let packet_info = probe_packet_info(&mp4_path);
    let all_expected = oracle_framemd5(&mp4_path);

    let cras: Vec<usize> = packet_info
        .iter()
        .enumerate()
        .filter(|(idx, (_, is_key))| *idx > 0 && *is_key)
        .map(|(idx, _)| idx)
        .collect();
    let second_cra = cras[1]; // 0 is first CRA, 1 is second CRA

    // Find two consecutive packets after the second CRA whose pts run backwards (a B-frame pair).
    let (p1, p2) = (second_cra + 1..packet_info.len() - 1)
        .find(|&i| packet_info[i].0 > packet_info[i + 1].0)
        .map(|i| (i, i + 1))
        .expect("consecutive packets whose pts run backwards");

    let untimed_pts = [packet_info[p1].0, packet_info[p2].0];
    let expected: Vec<(Option<i64>, String)> = all_expected
        .into_iter()
        .map(|(pts, hash)| {
            if untimed_pts.contains(&pts) {
                (None, hash)
            } else {
                (Some(pts), hash)
            }
        })
        .collect();

    let data = fs::read(&hevc_path).unwrap();
    let au_starts = split_access_units(&data);
    assert_eq!(au_starts.len() - 1, packet_info.len());

    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");
    let mut actual = Vec::new();

    for idx in 0..packet_info.len() {
        let au = &data[au_starts[idx]..au_starts[idx + 1]];
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au.to_vec());
        if idx == p1 || idx == p2 {
            pkt.pts = None;
        } else {
            pkt.pts = Some(packet_info[idx].0);
        }
        decoder.send_packet(&pkt).expect("send_packet");
        receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames");
    }
    decoder.flush().expect("flush");
    receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames flush");

    assert_eq!(actual, expected);
}

#[test]
fn two_access_units_in_one_packet_attaches_pts_only_to_first_au() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-cra-repro");
    fs::create_dir_all(&dir).unwrap();
    let (mp4_path, hevc_path) = ensure_open_gop_stream(&dir);

    let packet_info = probe_packet_info(&mp4_path);
    let all_expected = oracle_framemd5(&mp4_path);

    let data = fs::read(&hevc_path).unwrap();
    let au_starts = split_access_units(&data);
    assert_eq!(au_starts.len() - 1, packet_info.len());

    // Merge AU j and j+1 into one packet carrying AU j's pts.
    let j = 30;
    let untimed_pts = packet_info[j + 1].0;
    let expected: Vec<(Option<i64>, String)> = all_expected
        .into_iter()
        .map(|(pts, hash)| {
            if pts == untimed_pts {
                (None, hash)
            } else {
                (Some(pts), hash)
            }
        })
        .collect();

    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");
    let mut actual = Vec::new();

    let mut idx = 0;
    while idx < packet_info.len() {
        if idx == j {
            let mut merged = data[au_starts[j]..au_starts[j + 1]].to_vec();
            merged.extend_from_slice(&data[au_starts[j + 1]..au_starts[j + 2]]);
            let mut pkt = Packet::new(0, TimeBase::new(1, 25), merged);
            pkt.pts = Some(packet_info[j].0);
            decoder.send_packet(&pkt).expect("send merged packet");
            receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames");
            idx += 2;
        } else {
            let au = &data[au_starts[idx]..au_starts[idx + 1]];
            let mut pkt = Packet::new(0, TimeBase::new(1, 25), au.to_vec());
            pkt.pts = Some(packet_info[idx].0);
            decoder.send_packet(&pkt).expect("send packet");
            receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames");
            idx += 1;
        }
    }
    decoder.flush().expect("flush");
    receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames flush");

    assert_eq!(actual, expected);
}

#[test]
fn eos_nal_before_third_cra_drops_prior_pictures_and_preserves_pts() {
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let dir = root.join("hevc-cra-repro");
    fs::create_dir_all(&dir).unwrap();
    let (mp4_path, hevc_path) = ensure_open_gop_stream(&dir);

    let packet_info = probe_packet_info(&mp4_path);
    let all_expected = oracle_framemd5(&mp4_path);

    let cras: Vec<usize> = packet_info
        .iter()
        .enumerate()
        .filter(|(idx, (_, is_key))| *idx > 0 && *is_key)
        .map(|(idx, _)| idx)
        .collect();
    let third_cra = cras[2]; // 0 is first CRA, 1 is second CRA, 2 is third CRA

    let data = fs::read(&hevc_path).unwrap();
    let au_starts = split_access_units(&data);
    assert_eq!(au_starts.len() - 1, packet_info.len());

    // Append EOS NAL (00 00 01 48 01) to the AU before the third CRA.
    let mut modified_hevc = Vec::new();
    let mut packets = Vec::new();
    for idx in 0..packet_info.len() {
        let mut au = data[au_starts[idx]..au_starts[idx + 1]].to_vec();
        if idx == third_cra - 1 {
            au.extend_from_slice(&[0, 0, 1, 0x48, 0x01]);
        }
        modified_hevc.extend_from_slice(&au);
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), au);
        pkt.pts = Some(packet_info[idx].0);
        packets.push(pkt);
    }

    let modified_path = dir.join("open_gop_repro_eos.hevc");
    fs::write(&modified_path, &modified_hevc).expect("write modified hevc");

    let ffmpeg_modified = oracle_framemd5(&modified_path);
    let modified_hashes: std::collections::HashSet<String> =
        ffmpeg_modified.into_iter().map(|(_, h)| h).collect();

    // Assert the oracle md5s are unique.
    let mut unique_hashes = std::collections::HashSet::new();
    for (_, h) in &all_expected {
        assert!(unique_hashes.insert(h.clone()), "oracle hashes must be unique");
    }

    let expected: Vec<(Option<i64>, String)> = all_expected
        .into_iter()
        .filter(|(_, h)| modified_hashes.contains(h))
        .map(|(pts, h)| (Some(pts), h))
        .collect();

    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");
    let mut actual = Vec::new();
    for pkt in &packets {
        decoder.send_packet(pkt).expect("send packet");
        receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames");
    }
    decoder.flush().expect("flush");
    receive_frames(decoder.as_mut(), &mut actual).expect("receive_frames flush");

    assert_eq!(actual, expected);
}

#[test]
fn mvhevc_both_views_of_access_unit_carry_that_access_unit_pts() {
    const STREAM: &[u8] = include_bytes!("fixture_bytes/r462-mvhevc-3au.hevc");
    let data = STREAM;

    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i..].starts_with(&[0, 0, 1]) {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts.push(data.len());

    let mut au = Vec::new();
    let mut pending_non_vcl: Option<usize> = None;
    for &off in &starts[..starts.len() - 1] {
        let b0 = data[off + 3];
        let b1 = data[off + 4];
        let nal_type = (b0 >> 1) & 0x3f;
        let layer = ((b0 & 1) << 5) | (b1 >> 3);
        let first = (data[off + 5] >> 7) & 1;
        if nal_type >= 32 {
            if nal_type != 36 && nal_type != 37 && nal_type != 40 {
                pending_non_vcl.get_or_insert(off);
            }
        } else {
            if first == 1 && layer == 0 {
                au.push(pending_non_vcl.unwrap_or(off));
            }
            pending_non_vcl = None;
        }
    }
    au.push(data.len());
    assert_eq!(au.len() - 1, 3, "three access units in MV-HEVC stream");

    let mut decoder = oxideav_h265::make_decoder(&CodecParameters::video("h265".into()))
        .expect("make_decoder");
    let mut actual_pts = Vec::new();
    for (n, w) in au.windows(2).enumerate() {
        let mut pkt = Packet::new(0, TimeBase::new(1, 25), data[w[0]..w[1]].to_vec());
        pkt.pts = Some(n as i64 * 10);
        decoder.send_packet(&pkt).expect("send packet");
        let mut frames = Vec::new();
        receive_frames(decoder.as_mut(), &mut frames).expect("receive frames");
        actual_pts.extend(frames.into_iter().map(|(pts, _)| pts));
    }
    decoder.flush().expect("flush");
    let mut frames = Vec::new();
    receive_frames(decoder.as_mut(), &mut frames).expect("receive frames flush");
    actual_pts.extend(frames.into_iter().map(|(pts, _)| pts));

    // Output order is POC order within CVS: AU 0 (POC 0), AU 2 (POC 1), AU 1 (POC 2).
    // Both views (layer 0 and layer 1) of each AU must carry that AU's packet PTS.
    let expected_pts = vec![
        Some(0),
        Some(0),
        Some(20),
        Some(20),
        Some(10),
        Some(10),
    ];
    assert_eq!(actual_pts, expected_pts, "both views of each AU must carry that AU's PTS");
    for pair in actual_pts.chunks(2) {
        assert_eq!(pair[0], pair[1], "both views in one AU must share the same PTS");
    }
}
