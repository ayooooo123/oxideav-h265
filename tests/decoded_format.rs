//! The registry decoder reports the visible size and pixel layout of the
//! frame it last returned (oxideav-core `Decoder::output_video_dimensions`
//! / `output_pixel_format`): the conformance-cropped size, odd sizes
//! included, at 8, 10 and 12 bits. A new size or layout is reported with
//! the first frame that has it, not when its parameter sets arrive.
//!
//! `decoded_format/*.hevc` are three-frame libx265 streams (FFmpeg
//! `testsrc`, `bframes=2`) coded at 40×24 / 48×32 and cropped by their
//! conformance windows.

mod fixture_bytes;

use oxideav_core::{
    CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};

/// A stream and the size and layout of each of its frames.
struct Segment {
    bytes: &'static [u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    frames: usize,
}

fn segments() -> [Segment; 5] {
    use PixelFormat::*;
    [
        Segment {
            bytes: include_bytes!("decoded_format/a_444_33x17.hevc"),
            width: 33,
            height: 17,
            format: Yuv444P,
            frames: 3,
        },
        Segment {
            bytes: include_bytes!("decoded_format/b_gray12_47x31.hevc"),
            width: 47,
            height: 31,
            format: Gray12Le,
            frames: 3,
        },
        Segment {
            bytes: include_bytes!("decoded_format/c_422p10_34x17.hevc"),
            width: 34,
            height: 17,
            format: Yuv422P10Le,
            frames: 3,
        },
        Segment {
            bytes: fixture_bytes::I_THEN_P_HEVC,
            width: 64,
            height: 64,
            format: Yuv420P,
            frames: 2,
        },
        Segment {
            bytes: fixture_bytes::MAIN10_HEVC,
            width: 32,
            height: 32,
            format: Yuv420P10Le,
            frames: 2,
        },
    ]
}

/// The layout each frame of the concatenated segments must report.
fn expected() -> Vec<(u32, u32, PixelFormat)> {
    segments()
        .iter()
        .flat_map(|s| std::iter::repeat((s.width, s.height, s.format)).take(s.frames))
        .collect()
}

/// Asserts that `frame` holds a `w`×`h` picture of `format`: the plane
/// count, and each plane's visible row bytes and rows (this decoder packs
/// the cropped window without padding).
fn assert_geometry(frame: &VideoFrame, (w, h, format): (u32, u32, PixelFormat), at: usize) {
    use PixelFormat::*;
    let (bytes, sub_x, sub_y, count) = match format {
        Yuv420P => (1, 2, 2, 3),
        Yuv420P10Le => (2, 2, 2, 3),
        Yuv422P10Le => (2, 2, 1, 3),
        Yuv444P => (1, 1, 1, 3),
        Gray12Le => (2, 1, 1, 1),
        other => panic!("frame {at}: no plane layout for {other:?} in this test"),
    };
    let planes = frame.image_planes();
    assert_eq!(planes.len(), count, "frame {at}: plane count");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = if i == 0 {
            (w, h)
        } else {
            (w.div_ceil(sub_x), h.div_ceil(sub_y))
        };
        assert_eq!(
            plane.stride,
            pw as usize * bytes,
            "frame {at}: plane {i} stride"
        );
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "frame {at}: plane {i} rows"
        );
    }
}

fn decoder() -> Box<dyn Decoder> {
    oxideav_h265::make_decoder(&CodecParameters::video("h265".into())).expect("decoder")
}

fn report(dec: &dyn Decoder) -> Option<(u32, u32, PixelFormat)> {
    let (w, h) = dec.output_video_dimensions()?;
    Some((w, h, dec.output_pixel_format()?))
}

/// Receives every available frame, checking each against the next
/// expected layout right after it is returned.
fn receive_all(dec: &mut dyn Decoder, expected: &[(u32, u32, PixelFormat)], seen: &mut usize) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let want = expected[*seen];
                assert_eq!(report(dec), Some(want), "report after frame {seen}");
                assert_geometry(&frame, want, *seen);
                *seen += 1;
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// All five streams in one packet: every parameter set is parsed before
/// the first frame comes out, so the reports must follow the frames.
#[test]
fn each_frame_reports_its_own_size_and_layout() {
    let expected = expected();
    let stream: Vec<u8> = segments()
        .iter()
        .flat_map(|s| s.bytes.iter().copied())
        .collect();
    let mut dec = decoder();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream))
        .expect("send");
    // Before any frame is returned, the next frame in output order (the
    // first stream's), not the newest parameter sets.
    assert_eq!(
        report(&*dec),
        Some(expected[0]),
        "report before the first frame"
    );
    let mut seen = 0;
    receive_all(&mut *dec, &expected, &mut seen);
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
    // After the end, the last frame returned.
    assert_eq!(report(&*dec), expected.last().copied());
}

/// One NAL unit per packet, receiving after each, as a demuxer feeds it.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let expected = expected();
    let stream: Vec<u8> = segments()
        .iter()
        .flat_map(|s| s.bytes.iter().copied())
        .collect();
    let starts: Vec<usize> = (0..stream.len().saturating_sub(3))
        .filter(|&i| stream[i..i + 3] == [0, 0, 1])
        .collect();
    let mut dec = decoder();
    let mut seen = 0;
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = [&[0u8][..], &stream[start..end]].concat();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), nal))
            .expect("send");
        receive_all(&mut *dec, &expected, &mut seen);
    }
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
}
