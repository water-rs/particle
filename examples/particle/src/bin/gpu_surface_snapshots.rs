use std::env;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;

use particle_example::{explosion_system, flame_system, rain_system};
use waterui::prelude::Environment;
use waterui_graphics::{
    cherenkov::{ColorSpace, LinearDisplayP3, Readback, Srgb},
    offscreen::{OffscreenImage, OffscreenSize},
};
use waterui_particle::ParticleSystem;

struct SnapshotSpec {
    name: &'static str,
    width: u32,
    height: u32,
    frames: NonZeroU32,
    background: [u8; 3],
    system: fn() -> ParticleSystem,
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
    let output = (spec.system)()
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
    let output_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: cargo run -p particle-example --bin gpu_surface_snapshots -- <output-dir>");
    fs::create_dir_all(&output_dir).expect("snapshot output directory must be creatable");

    let snapshots = [
        SnapshotSpec {
            name: "rain",
            width: 540,
            height: 960,
            frames: NonZeroU32::new(72).expect("non-zero literal"),
            background: [0x0F, 0x17, 0x2A],
            system: rain_system,
        },
        SnapshotSpec {
            name: "flame",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(60).expect("non-zero literal"),
            background: [0x00, 0x00, 0x00],
            system: flame_system,
        },
        SnapshotSpec {
            name: "explosion",
            width: 600,
            height: 600,
            frames: NonZeroU32::new(36).expect("non-zero literal"),
            background: [0x00, 0x00, 0x00],
            system: explosion_system,
        },
    ];

    for spec in snapshots {
        write_snapshot(&output_dir, spec);
    }
}

fn main() {
    run();
}
