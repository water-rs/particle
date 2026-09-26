//! Build-time compiled shaders for the particle system.

use shaderloom::CompiledShader;

pub const COMPUTE_SHADER: CompiledShader =
    include!(concat!(env!("OUT_DIR"), "/particle_compute.rs"));
pub const RENDER_SHADER: CompiledShader = include!(concat!(env!("OUT_DIR"), "/particle_render.rs"));

/// The particle shaders each declare exactly one bind group.
pub(crate) fn bind_group_layout(
    shader: &CompiledShader,
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    let [layout]: [_; 1] = shader
        .create_bind_group_layouts(device)
        .try_into()
        .expect("particle shader must declare one bind group");
    layout
}
