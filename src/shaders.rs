//! WGSL shader sources and their wgpu declarations.

use waterui_graphics::wgpu;

/// Compiled particle simulation compute module.
pub fn compute_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particle Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("particle_compute.wgsl").into()),
    })
}

/// Compiled particle render module.
pub fn render_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Particle Render Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("particle_render.wgsl").into()),
    })
}

/// Bind group layout shared by the three simulation compute pipelines.
pub fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    const STORAGE: wgpu::BindingType = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    const READ_ONLY: wgpu::BindingType = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Particle Compute Bind Group Layout"),
        entries: &[
            entry(0, READ_ONLY), // uniforms
            entry(1, READ_ONLY), // particle_source
            entry(2, STORAGE),   // particle_target
            entry(3, READ_ONLY), // circle_obstacles
            entry(4, STORAGE),   // cell_heads
            entry(5, STORAGE),   // particle_links
        ],
    })
}

/// Bind group layout of the render pipeline's uniforms buffer.
pub fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Particle Render Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}
