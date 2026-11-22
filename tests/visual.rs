//! Headless visual showcase tests for `waterui-particle`.
//!
//! Renders deterministic particle scenes offscreen via [`ParticleSystem::render_offscreen`],
//! composites them onto dark opaque backdrops for inspection, verifies that particles
//! actually drew non-background pixels, and writes canonical
//! `<suite>/<case>/<stage>.png` artifacts under `WATERUI_TEST_ARTIFACTS_DIR`
//! (or under `CARGO_TARGET_TMPDIR/visual-gallery` when unset).

use core::{f32::consts::PI, num::NonZeroU32, time::Duration};
use std::{env, fs, path::PathBuf};

use waterui_core::Environment;
use waterui_graphics::{
    color::Color,
    gpu::GpuRuntime,
    offscreen::{OffscreenImage, OffscreenSize},
};
use waterui_particle::{ParticleShape, ParticleSystem};

const FRAME_STEP: Duration = Duration::from_nanos(16_666_667);

struct ShowcaseSpec {
    case: &'static str,
    stage: &'static str,
    width: u32,
    height: u32,
    frames: NonZeroU32,
    background: [u8; 3],
    system: fn() -> ParticleSystem,
}

fn rain_scene() -> ParticleSystem {
    ParticleSystem::new(8_000)
        .emit_from_rect(1.4, 0.08)
        .at(0.5, -0.04)
        .rate(480_000.0)
        .life(0.6, 0.8)
        .speed(2.4, 4.2)
        .angle(PI * 0.49, PI * 0.51)
        .size(0.0012, 0.0022)
        .color(
            Color::srgb_hex("#D5E8FF").with_opacity(0.65),
            Color::srgb_hex("#E8F5FF").with_opacity(0.0),
        )
        .gravity(0.0, 5.0)
        .wind(0.05, 0.0)
        .stretch_with_velocity()
        .softness(0.35)
}

fn flame_scene() -> ParticleSystem {
    ParticleSystem::new(3_000)
        .emit_from_rect(0.05, 0.0)
        .at(0.5, 0.82)
        .rate(180_000.0)
        .life(0.4, 0.8)
        .speed(0.5, 1.2)
        .angle(PI * 1.4, PI * 1.6)
        .size(0.03, 0.06)
        .color(
            Color::srgb_hex("#FFB433").with_opacity(0.6),
            Color::srgb_hex("#FF2A0D").with_opacity(0.0),
        )
        .gravity(0.0, -1.0)
        .additive()
        .softness(0.6)
}

fn fog_scene() -> ParticleSystem {
    ParticleSystem::new(2_000)
        .emit_from_rect(1.5, 0.2)
        .at(0.5, 1.1)
        .rate(40_000.0)
        .life(8.0, 12.0)
        .speed(0.02, 0.08)
        .angle(PI * 1.4, PI * 1.6)
        .size(0.1, 0.25)
        .color(
            Color::srgb_hex("#C8D7CC").with_opacity(0.18),
            Color::srgb_hex("#C8D7CC").with_opacity(0.0),
        )
        .gravity(0.0, -0.01)
        .wind(0.02, 0.0)
        .softness(1.0)
}

fn explosion_scene() -> ParticleSystem {
    ParticleSystem::new(20_000)
        .emit_from_circle(0.05)
        .at(0.5, 0.5)
        .rate(1_200_000.0)
        .life(0.8, 1.5)
        .speed(0.5, 3.0)
        .angle(0.0, PI * 2.0)
        .size(0.004, 0.01)
        .color(
            Color::srgb_hex("#FF7A00").with_opacity(1.0),
            Color::srgb_hex("#333333").with_opacity(1.0),
        )
        .gravity(0.0, 3.0)
        .shape(ParticleShape::Rect)
        .softness(0.0)
}

fn bounce_box_scene() -> ParticleSystem {
    ParticleSystem::new(6_000)
        .emit_from_circle(0.02)
        .at(0.5, 0.18)
        .rate(90_000.0)
        .life(4.0, 6.0)
        .speed(0.5, 1.4)
        .angle(0.0, PI * 2.0)
        .size(0.006, 0.014)
        .color(
            Color::srgb_hex("#85DBFF").with_opacity(0.95),
            Color::srgb_hex("#2EA4FF").with_opacity(0.2),
        )
        .gravity(0.0, 1.4)
        .turbulence(0.2)
        .collide_with_particles(0.01, 16.0)
        .collide_with_rect(0.08, 0.08, 0.84, 0.84)
        .collide_with_circle_obstacle(0.38, 0.34, 0.06)
        .collide_with_circle_obstacle(0.5, 0.36, 0.08)
        .collide_with_circle_obstacle(0.62, 0.34, 0.06)
        .bounce(0.82)
        .surface_friction(0.9)
        .softness(0.25)
}

fn artifacts_root() -> PathBuf {
    env::var_os("WATERUI_TEST_ARTIFACTS_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("visual-gallery"),
        PathBuf::from,
    )
}

fn composite_over_background(output: &OffscreenImage, background: [u8; 3]) -> Vec<u8> {
    output
        .premultiplied_rgba8()
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| {
            let alpha = u16::from(pixel[3]);
            let inv_alpha = 255_u16 - alpha;
            let red = u16::from(pixel[0]) + (u16::from(background[0]) * inv_alpha + 127) / 255;
            let green = u16::from(pixel[1]) + (u16::from(background[1]) * inv_alpha + 127) / 255;
            let blue = u16::from(pixel[2]) + (u16::from(background[2]) * inv_alpha + 127) / 255;
            [
                u8::try_from(red.min(255)).expect("clamped channel fits in u8"),
                u8::try_from(green.min(255)).expect("clamped channel fits in u8"),
                u8::try_from(blue.min(255)).expect("clamped channel fits in u8"),
                255,
            ]
        })
        .collect()
}

fn count_visible_particle_pixels(rgba8: &[u8], background: [u8; 3]) -> usize {
    rgba8
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| {
            pixel[0].abs_diff(background[0]) > 2
                || pixel[1].abs_diff(background[1]) > 2
                || pixel[2].abs_diff(background[2]) > 2
        })
        .count()
}

const fn showcase_specs() -> [ShowcaseSpec; 8] {
    [
        ShowcaseSpec {
            case: "rain",
            stage: "settled",
            width: 320,
            height: 480,
            frames: NonZeroU32::new(8).expect("non-zero frame count"),
            background: [0x0F, 0x17, 0x2A],
            system: rain_scene,
        },
        ShowcaseSpec {
            case: "flame",
            stage: "frame_02",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(2).expect("non-zero frame count"),
            background: [0x00, 0x00, 0x00],
            system: flame_scene,
        },
        ShowcaseSpec {
            case: "flame",
            stage: "frame_08",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(8).expect("non-zero frame count"),
            background: [0x00, 0x00, 0x00],
            system: flame_scene,
        },
        ShowcaseSpec {
            case: "flame",
            stage: "frame_24",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(24).expect("non-zero frame count"),
            background: [0x00, 0x00, 0x00],
            system: flame_scene,
        },
        ShowcaseSpec {
            case: "fog",
            stage: "frame_08",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(8).expect("non-zero frame count"),
            background: [0x08, 0x10, 0x12],
            system: fog_scene,
        },
        ShowcaseSpec {
            case: "fog",
            stage: "frame_24",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(24).expect("non-zero frame count"),
            background: [0x08, 0x10, 0x12],
            system: fog_scene,
        },
        ShowcaseSpec {
            case: "explosion",
            stage: "burst",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(8).expect("non-zero frame count"),
            background: [0x00, 0x00, 0x00],
            system: explosion_scene,
        },
        ShowcaseSpec {
            case: "bounce_box",
            stage: "collision",
            width: 360,
            height: 360,
            frames: NonZeroU32::new(12).expect("non-zero frame count"),
            background: [0x06, 0x14, 0x1F],
            system: bounce_box_scene,
        },
    ]
}

#[test]
fn particle_visual_gallery_renders_non_blank_frames() {
    let root = artifacts_root();
    let runtime = pollster::block_on(GpuRuntime::new())
        .expect("particle visual gallery requires a working GPU runtime");
    let env = Environment::new();

    for spec in showcase_specs() {
        let size = OffscreenSize::try_from_pixels(spec.width, spec.height)
            .expect("showcase frame size must be valid");
        let readback = (spec.system)()
            .render_offscreen(&runtime, size, &env, spec.frames, FRAME_STEP)
            .unwrap_or_else(|error| {
                panic!(
                    "particle scene {}/{} render failed: {error}",
                    spec.case, spec.stage
                )
            });
        let image = OffscreenImage::from_readback(&readback);
        let rgba8 = composite_over_background(&image, spec.background);
        let visible = count_visible_particle_pixels(&rgba8, spec.background);
        assert!(
            visible > 0,
            "particle scene {}/{} produced no visible particles over background",
            spec.case,
            spec.stage
        );

        let output_dir = root.join("particle").join(spec.case);
        fs::create_dir_all(&output_dir).expect("visual gallery output directory must be creatable");
        let output_path = output_dir.join(format!("{}.png", spec.stage));
        OffscreenImage::new(image.width(), image.height(), rgba8)
            .save_png(&output_path)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", output_path.display()));
    }
}
