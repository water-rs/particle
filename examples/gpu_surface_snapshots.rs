//! Offscreen engine snapshot example for the particle effect.

use core::{f32::consts::PI, num::NonZeroU32};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use waterui_core::Environment;
use waterui_graphics::{
    cherenkov::{ColorSpace, LinearDisplayP3, Readback, Srgb},
    color::Color,
    offscreen::{OffscreenImage, OffscreenSize},
};
use waterui_particle::{ParticleShape, ParticleSystem};

struct SnapshotSpec {
    name: &'static str,
    width: u32,
    height: u32,
    frames: NonZeroU32,
    background: [u8; 3],
    system: ParticleSystem,
}

fn rain_scene() -> ParticleSystem {
    ParticleSystem::new(8_000)
        .emit_from_rect(1.4, 0.08)
        .at(0.5, -0.04)
        .rate(480_000.0)
        .life(0.6, 0.8)
        .speed(2.4, 4.2)
        .angle(PI * 0.49, PI * 0.51)
        .size(0.0008, 0.0015)
        .color(
            Color::srgb_hex("#D5E8FF").with_opacity(0.45),
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
            Color::srgb_hex("#C8D7CC").with_opacity(0.12),
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
        .size(0.003, 0.008)
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

/// Composites the premultiplied readback like the engine's presentation pass:
/// source-over in linear Display P3, so additive blends saturate the same way
/// they do on screen.
fn composite_over_opaque_background(readback: &Readback, background: [u8; 3]) -> OffscreenImage {
    let background = LinearDisplayP3::from_linear_srgb(Srgb::to_linear_srgb([
        f32::from(background[0]) / 255.0,
        f32::from(background[1]) / 255.0,
        f32::from(background[2]) / 255.0,
    ]));
    let rgba8 = readback
        .pixels
        .iter()
        .flat_map(|[red, green, blue, alpha]| {
            let linear = LinearDisplayP3::to_linear_srgb([
                red + (1.0 - alpha) * background[0],
                green + (1.0 - alpha) * background[1],
                blue + (1.0 - alpha) * background[2],
            ]);
            [encode(linear[0]), encode(linear[1]), encode(linear[2]), 255]
        })
        .collect();

    OffscreenImage {
        width: readback.width,
        height: readback.height,
        rgba8,
    }
}

/// sRGB-encodes one linear channel.
fn encode(linear: f32) -> u8 {
    let clipped = linear.clamp(0.0, 1.0);
    let encoded = if clipped <= 0.003_130_8 {
        clipped * 12.92
    } else {
        1.055_f32.mul_add(clipped.powf(1.0 / 2.4), -0.055)
    };
    encode_channel(encoded)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to [0, 255] before the cast"
)]
fn encode_channel(unit: f32) -> u8 {
    (unit * 255.0).round() as u8
}

fn write_snapshot(output_dir: &Path, spec: SnapshotSpec) {
    let size = OffscreenSize::try_from_pixels(spec.width, spec.height)
        .expect("snapshot frame size must be valid");
    let env = Environment::new();
    let output = spec
        .system
        .render_offscreen(size, &env, spec.frames, Duration::from_secs_f64(1.0 / 60.0))
        .expect("particle snapshot render should succeed");

    OffscreenImage::from_readback(&output)
        .save_png(output_dir.join(format!("{}.raw.png", spec.name)))
        .expect("raw particle png write should succeed");

    composite_over_opaque_background(&output, spec.background)
        .save_png(output_dir.join(format!("{}.png", spec.name)))
        .expect("composited particle png write should succeed");
}

fn run() {
    let output_dir = env::args_os().nth(1).map(PathBuf::from).expect(
        "usage: cargo run -p waterui-particle --example gpu_surface_snapshots -- <output-dir>",
    );
    fs::create_dir_all(&output_dir).expect("snapshot output directory must be creatable");

    let snapshots = [
        SnapshotSpec {
            name: "rain",
            width: 540,
            height: 960,
            frames: NonZeroU32::new(54).expect("non-zero literal"),
            background: [0x0F, 0x17, 0x2A],
            system: rain_scene(),
        },
        SnapshotSpec {
            name: "flame",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(60).expect("non-zero literal"),
            background: [0x00, 0x00, 0x00],
            system: flame_scene(),
        },
        SnapshotSpec {
            name: "fog",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(480).expect("non-zero literal"),
            background: [0x08, 0x10, 0x12],
            system: fog_scene(),
        },
        SnapshotSpec {
            name: "explosion",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(36).expect("non-zero literal"),
            background: [0x00, 0x00, 0x00],
            system: explosion_scene(),
        },
        SnapshotSpec {
            name: "bounce_box",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(180).expect("non-zero literal"),
            background: [0x06, 0x14, 0x1F],
            system: bounce_box_scene(),
        },
    ];

    for spec in snapshots {
        write_snapshot(&output_dir, spec);
    }
}

fn main() {
    run();
}
