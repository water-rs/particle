use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use particle_example::{explosion_system, flame_system, rain_system};
use waterui::prelude::Environment;
use waterui_graphics::{
    gpu::GpuRuntime,
    offscreen::{OffscreenImage, OffscreenSize},
};
use waterui_particle::ParticleSystem;

struct SnapshotSpec {
    name: &'static str,
    width: u32,
    height: u32,
    background: [u8; 3],
    system: fn() -> ParticleSystem,
}

fn composite_over_opaque_background(pixels: &[u8], background: [u8; 3]) -> Vec<u8> {
    pixels
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
                red.min(255) as u8,
                green.min(255) as u8,
                blue.min(255) as u8,
                255,
            ]
        })
        .collect()
}

fn write_snapshot(runtime: &GpuRuntime, output_dir: &Path, spec: SnapshotSpec) {
    let size = OffscreenSize::try_from_pixels(spec.width, spec.height)
        .expect("snapshot frame size must be valid");
    let env = Environment::new();
    let output = (spec.system)()
        .render_offscreen(
            runtime,
            size,
            &env,
            core::num::NonZeroU32::MIN,
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        )
        .expect("particle snapshot render should succeed");
    let output = OffscreenImage::from_readback(&output);

    output
        .save_png(output_dir.join(format!("{}.raw.png", spec.name)))
        .expect("raw particle png write should succeed");

    let composited = OffscreenImage {
        width: output.width,
        height: output.height,
        rgba8: composite_over_opaque_background(&output.premultiplied_rgba8(), spec.background),
    };
    composited
        .save_png(output_dir.join(format!("{}.png", spec.name)))
        .expect("composited particle png write should succeed");
}

async fn run() {
    let output_dir = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: cargo run -p particle-example --bin gpu_surface_snapshots -- <output-dir>");
    fs::create_dir_all(&output_dir).expect("snapshot output directory must be creatable");
    let runtime = GpuRuntime::new()
        .await
        .expect("particle snapshot export requires a working GPU runtime");

    let snapshots = [
        SnapshotSpec {
            name: "rain",
            width: 540,
            height: 960,
            background: [0x0F, 0x17, 0x2A],
            system: rain_system,
        },
        SnapshotSpec {
            name: "flame",
            width: 600,
            height: 600,
            background: [0x00, 0x00, 0x00],
            system: flame_system,
        },
        SnapshotSpec {
            name: "explosion",
            width: 600,
            height: 600,
            background: [0x00, 0x00, 0x00],
            system: explosion_system,
        },
    ];

    for spec in snapshots {
        write_snapshot(&runtime, &output_dir, spec);
    }
}

fn main() {
    pollster::block_on(run());
}
