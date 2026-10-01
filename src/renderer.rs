//! GPU renderer for particle simulation and visualization.

use crate::{
    EmitterShape,
    config::{BlendMode, ParticleConfig, ParticleShape},
    gpu::{
        CollisionUniforms, GpuCircleObstacle, GpuParticle, InteractionUniforms, ShaderVec2,
        ShaderVec4, Uniforms,
    },
    shaders::{COMPUTE_SHADER, RENDER_SHADER},
};
use encase::{ShaderSize, StorageBuffer};
use num_traits::ToPrimitive;
use shaderloom::{CompiledShader, CompiledShaderModule, ShaderStage};
use std::mem::offset_of;
use std::sync::mpsc::{self, Receiver, Sender};
use waterui_core::{Computed, Environment, Signal, SignalExt, flatten_signal};
use waterui_graphics::{
    color::WorkingColor,
    gpu::{Context, Frame, GpuContent},
    wgpu,
};

/// Resolved particle configuration ready for GPU.
#[derive(Clone, Debug)]
pub struct ResolvedParticleConfig {
    pub max_particles: u32,
    pub emitter_pos: [f32; 2],
    pub emitter_shape: EmitterShape,
    pub emit_rate: f32,
    pub gravity: [f32; 2],
    pub wind: [f32; 2],
    pub turbulence: f32,
    pub drag: f32,
    pub interaction_enabled: bool,
    pub interaction_radius: f32,
    pub interaction_strength: f32,
    pub collision_enabled: bool,
    pub collision_bounds: [f32; 4],
    pub collision_restitution: f32,
    pub collision_surface_friction: f32,
    pub collision_circle_obstacles: Vec<[f32; 3]>,
    pub life_range: [f32; 2],
    pub speed_range: [f32; 2],
    pub angle_range: [f32; 2],
    pub size_range: [f32; 2],
    pub spin_range: [f32; 2],
    pub color_start: WorkingColor,
    pub color_end: WorkingColor,
    pub stretch_with_velocity: bool,
    pub blend_mode: BlendMode,
    pub softness: f32,
    pub shape: ParticleShape,
}

pub struct ParticleFeed {
    max_particles: u32,
    config: ParticleConfig,
    color_start: Computed<WorkingColor>,
    color_end: Computed<WorkingColor>,
    updates: Sender<ResolvedParticleConfig>,
}

impl ParticleFeed {
    fn snapshot(&self) -> ResolvedParticleConfig {
        ResolvedParticleConfig {
            max_particles: self.max_particles,
            collision_circle_obstacles: self
                .config
                .collision
                .circle_obstacles
                .iter()
                .map(|obstacle| obstacle.value.snapshot())
                .collect(),
            emitter_pos: self.config.emitter.position.snapshot(),
            emitter_shape: self.config.emitter.shape.snapshot(),
            emit_rate: self.config.emitter.rate.snapshot(),
            gravity: self.config.environment.gravity.snapshot(),
            wind: self.config.environment.wind.snapshot(),
            turbulence: self.config.environment.turbulence.snapshot(),
            drag: self.config.environment.drag.snapshot(),
            collision_enabled: self.config.collision.enabled,
            collision_bounds: self.config.collision.bounds.snapshot(),
            collision_restitution: self.config.collision.restitution.snapshot(),
            collision_surface_friction: self.config.collision.surface_friction.snapshot(),
            interaction_enabled: self.config.interaction.enabled,
            interaction_radius: self.config.interaction.radius.snapshot(),
            interaction_strength: self.config.interaction.strength.snapshot(),
            life_range: self.config.particle.life.snapshot(),
            speed_range: self.config.particle.speed.snapshot(),
            angle_range: self.config.particle.angle.snapshot(),
            size_range: self.config.particle.size.snapshot(),
            spin_range: self.config.particle.spin.snapshot(),
            color_start: self.color_start.snapshot(),
            color_end: self.color_end.snapshot(),
            stretch_with_velocity: self.config.particle.stretch_with_velocity,
            blend_mode: self.config.blend_mode,
            softness: self.config.particle.softness.snapshot(),
            shape: self.config.particle.shape,
        }
    }

    pub(crate) fn pump(&self) {
        // The UI hook and renderer share the view's lifetime.
        self.updates
            .send(self.snapshot())
            .expect("particle renderer disconnected");
    }
}

#[derive(Clone, Copy, Debug)]
struct InteractionGrid {
    width: u32,
    height: u32,
    cell_count: u32,
}

const PARTICLE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 8] = [
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, pos) as u64,
        shader_location: 0,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, vel) as u64,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x2,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, life) as u64,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, max_life) as u64,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, size) as u64,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, rotation) as u64,
        shader_location: 5,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, rot_speed) as u64,
        shader_location: 6,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        offset: offset_of!(GpuParticle, color) as u64,
        shader_location: 7,
        format: wgpu::VertexFormat::Float32x4,
    },
];

#[expect(
    clippy::cast_possible_truncation,
    reason = "the statically defined particle uniform is far smaller than usize on supported targets"
)]
const PARTICLE_UNIFORM_SIZE: usize = <Uniforms as ShaderSize>::SHADER_SIZE.get() as usize;

const fn encode_emitter_size(shape: EmitterShape) -> ShaderVec2 {
    match shape {
        EmitterShape::Point => ShaderVec2::ZERO,
        EmitterShape::Rect { width, height } => ShaderVec2::new(width, height),
        EmitterShape::Circle { radius } => ShaderVec2::new(radius, -1.0),
    }
}

const fn resolved_linear_color(color: WorkingColor) -> ShaderVec4 {
    ShaderVec4::from_array(color.components)
}

fn write_obstacles(queue: &wgpu::Queue, buffer: &wgpu::Buffer, obstacles: &[[f32; 3]]) {
    let obstacle_data: Vec<_> = if obstacles.is_empty() {
        vec![GpuCircleObstacle::default()]
    } else {
        obstacles
            .iter()
            .map(|obstacle| {
                GpuCircleObstacle::new(ShaderVec2::new(obstacle[0], obstacle[1]), obstacle[2])
            })
            .collect()
    };
    let mut collision_data = StorageBuffer::new(Vec::new());
    collision_data
        .write(&obstacle_data)
        .expect("failed to encode particle collision obstacle buffer");
    queue.write_buffer(buffer, 0, collision_data.as_ref());
}

fn interaction_grid(config: &ResolvedParticleConfig) -> InteractionGrid {
    if !config.interaction_enabled || config.interaction_radius <= 0.0 {
        return InteractionGrid {
            width: 1,
            height: 1,
            cell_count: 1,
        };
    }

    let ideal_dim = f32_to_u32_ceil(1.0 / config.interaction_radius);
    let max_dim = f32_to_u32_ceil(u32_to_f32(config.max_particles).sqrt().max(1.0));
    let dim = ideal_dim.clamp(1, max_dim);

    InteractionGrid {
        width: dim,
        height: dim,
        cell_count: dim * dim,
    }
}

fn max_interaction_grid_cells(max_particles: u32) -> u32 {
    let dimension = f32_to_u32_ceil(u32_to_f32(max_particles).sqrt().max(1.0));
    dimension * dimension
}

const fn particle_shape_code(shape: ParticleShape) -> u32 {
    match shape {
        ParticleShape::Circle => 0,
        ParticleShape::Rect => 1,
    }
}

fn particle_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    let particle_size = u64::try_from(core::mem::size_of::<GpuParticle>())
        .expect("GpuParticle size must fit into wgpu's u64 buffer addressing");
    assert_eq!(
        particle_size,
        <GpuParticle as ShaderSize>::SHADER_SIZE.get(),
        "GpuParticle Rust layout must match shader storage stride"
    );

    wgpu::VertexBufferLayout {
        array_stride: particle_size,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &PARTICLE_VERTEX_ATTRIBUTES,
    }
}

const fn blend_state(blend_mode: BlendMode) -> wgpu::BlendState {
    match blend_mode {
        BlendMode::Alpha => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        BlendMode::Additive => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        },
    }
}

/// GPU renderer for particle systems.
pub struct ParticleRenderer {
    updates: Receiver<ResolvedParticleConfig>,
    resolved_config: ResolvedParticleConfig,
    clear_grid_pipeline: Option<wgpu::ComputePipeline>,
    count_grid_pipeline: Option<wgpu::ComputePipeline>,
    prefix_grid_pipeline: Option<wgpu::ComputePipeline>,
    scatter_grid_pipeline: Option<wgpu::ComputePipeline>,
    sort_grid_pipeline: Option<wgpu::ComputePipeline>,
    simulate_pipeline: Option<wgpu::ComputePipeline>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    particle_buffers: [Option<wgpu::Buffer>; 2],
    collision_buffer: Option<wgpu::Buffer>,
    cell_counts_buffer: Option<wgpu::Buffer>,
    cell_offsets_buffer: Option<wgpu::Buffer>,
    cell_cursors_buffer: Option<wgpu::Buffer>,
    particle_links_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    compute_bind_groups: [Option<wgpu::BindGroup>; 2],
    render_bind_group: Option<wgpu::BindGroup>,
    current_particle_buffer_index: usize,
    seed_stream: Option<core::cell::RefCell<fastrand::Rng>>,
}

impl ParticleRenderer {
    /// Creates a renderer from a resolved test fixture.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub fn new(config: ResolvedParticleConfig) -> Self {
        Self::with_config(config, mpsc::channel().1)
    }

    pub(crate) fn reactive(
        max_particles: u32,
        config: ParticleConfig,
        env: &Environment,
    ) -> (ParticleFeed, Self) {
        let (updates, receiver) = mpsc::channel();
        let start_env = env.clone();
        let end_env = env.clone();
        let feed = ParticleFeed {
            max_particles,
            color_start: flatten_signal(
                config
                    .particle
                    .color_start
                    .clone()
                    .map(move |color| color.resolve(&start_env)),
            ),
            color_end: flatten_signal(
                config
                    .particle
                    .color_end
                    .clone()
                    .map(move |color| color.resolve(&end_env)),
            ),
            config,
            updates,
        };
        let renderer = Self::with_config(feed.snapshot(), receiver);
        (feed, renderer)
    }

    fn with_config(
        resolved_config: ResolvedParticleConfig,
        updates: Receiver<ResolvedParticleConfig>,
    ) -> Self {
        Self {
            updates,
            resolved_config,
            clear_grid_pipeline: None,
            count_grid_pipeline: None,
            prefix_grid_pipeline: None,
            scatter_grid_pipeline: None,
            sort_grid_pipeline: None,
            simulate_pipeline: None,
            render_pipeline: None,
            particle_buffers: std::array::from_fn(|_| None),
            collision_buffer: None,
            cell_counts_buffer: None,
            cell_offsets_buffer: None,
            cell_cursors_buffer: None,
            particle_links_buffer: None,
            uniform_buffer: None,
            compute_bind_groups: std::array::from_fn(|_| None),
            render_bind_group: None,
            current_particle_buffer_index: 0,
            seed_stream: None,
        }
    }

    /// Pins the emitter's per-frame seed to a private generator, so renders
    /// are reproducible instead of sharing the global RNG stream.
    pub(crate) const fn pin_seed_stream(&mut self, seed: u64) {
        self.seed_stream = Some(core::cell::RefCell::new(fastrand::Rng::with_seed(seed)));
    }

    const fn particle_buffer(&self, index: usize) -> &wgpu::Buffer {
        self.particle_buffers[index]
            .as_ref()
            .expect("particle buffer must exist before render")
    }

    const fn compute_bind_group(&self, index: usize) -> &wgpu::BindGroup {
        self.compute_bind_groups[index]
            .as_ref()
            .expect("compute bind group must exist before render")
    }

    fn update_uniforms(
        &self,
        config: &ResolvedParticleConfig,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
        delta: std::time::Duration,
    ) {
        let buffer = self
            .uniform_buffer
            .as_ref()
            .expect("uniform buffer must exist before render");
        let time = elapsed.as_secs_f32();
        let dt = delta.as_secs_f32().min(0.1);
        let grid = interaction_grid(config);

        let uniforms = Uniforms {
            time,
            dt,
            seed: self
                .seed_stream
                .as_ref()
                .map_or_else(|| fastrand::u32(..), |rng| rng.borrow_mut().u32(..)),
            max_particles: config.max_particles,
            gravity: config.gravity.into(),
            wind: config.wind.into(),
            emitter_pos: config.emitter_pos.into(),
            emitter_size: encode_emitter_size(config.emitter_shape),
            emit_rate: config.emit_rate,
            turbulence: config.turbulence,
            drag: config.drag,
            stretch_factor: if config.stretch_with_velocity {
                1.0
            } else {
                0.0
            },
            softness: config.softness,
            interaction: InteractionUniforms::new(
                config.interaction_enabled,
                grid.width,
                grid.height,
                config.interaction_radius,
                config.interaction_strength,
            ),
            collision: CollisionUniforms::new(
                config.collision_enabled,
                config.collision_restitution,
                config.collision_surface_friction,
                config.collision_bounds.into(),
                u32::try_from(config.collision_circle_obstacles.len())
                    .expect("particle collision obstacle count must fit into u32"),
            ),
            life_range: config.life_range.into(),
            speed_range: config.speed_range.into(),
            angle_range: config.angle_range.into(),
            size_range: config.size_range.into(),
            spin_range: config.spin_range.into(),
            color_start: resolved_linear_color(config.color_start),
            color_end: resolved_linear_color(config.color_end),
            shape: particle_shape_code(config.shape),
            viewport_width: width,
            viewport_height: height,
        };

        let mut uniform_data = StorageBuffer::new([0; PARTICLE_UNIFORM_SIZE]);
        uniform_data
            .write(&uniforms)
            .expect("failed to write particle uniform buffer");
        queue.write_buffer(buffer, 0, uniform_data.as_ref());
    }

    fn encode_simulation_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_index: usize,
        config: &ResolvedParticleConfig,
    ) -> usize {
        let target_index = 1 - source_index;
        let grid = interaction_grid(config);
        let bind_group = self.compute_bind_group(source_index);

        for (pipeline, label, workgroups) in [
            (
                self.clear_grid_pipeline
                    .as_ref()
                    .expect("clear-grid pipeline must exist before render"),
                "Particle Clear Grid Pass",
                grid.cell_count.div_ceil(64),
            ),
            (
                self.count_grid_pipeline
                    .as_ref()
                    .expect("count-grid pipeline must exist before render"),
                "Particle Count Grid Pass",
                config.max_particles.div_ceil(64),
            ),
            (
                self.prefix_grid_pipeline
                    .as_ref()
                    .expect("prefix-grid pipeline must exist before render"),
                "Particle Prefix Grid Pass",
                1,
            ),
            (
                self.scatter_grid_pipeline
                    .as_ref()
                    .expect("scatter-grid pipeline must exist before render"),
                "Particle Scatter Grid Pass",
                config.max_particles.div_ceil(64),
            ),
            (
                self.sort_grid_pipeline
                    .as_ref()
                    .expect("sort-grid pipeline must exist before render"),
                "Particle Sort Grid Pass",
                grid.cell_count.div_ceil(64),
            ),
        ] {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let pipeline = self
                .simulate_pipeline
                .as_ref()
                .expect("simulation pipeline must exist before render");
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Simulate Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(config.max_particles.div_ceil(64), 1, 1);
        }

        target_index
    }
}

fn fill_mapped_buffer(buffer: &wgpu::Buffer, value: u8) {
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .expect("freshly mapped buffer must expose its range")
        .slice(..)
        .fill(value);
    buffer.unmap();
}

/// Creates the shader's bind group layouts and asserts there is exactly one.
fn single_bind_group_layout(
    shader: &CompiledShader,
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    let mut layouts = shader.create_bind_group_layouts(device);
    assert_eq!(
        layouts.len(),
        1,
        "{label} must declare exactly one bind group"
    );
    layouts
        .pop()
        .expect("a shader with one bind group layout must yield one layout")
}

/// Creates the shader's render-stage modules and its single bind group layout.
fn single_bind_group_render_stages(
    shader: &CompiledShader,
    device: &wgpu::Device,
    label: &str,
    vertex_entry_point: &str,
    fragment_entry_point: &str,
) -> (
    CompiledShaderModule,
    CompiledShaderModule,
    wgpu::BindGroupLayout,
) {
    let (vertex_shader, fragment_shader) =
        shader.create_render_stages(device, vertex_entry_point, fragment_entry_point);
    (
        vertex_shader,
        fragment_shader,
        single_bind_group_layout(shader, device, label),
    )
}

impl GpuContent for ParticleRenderer {
    #[expect(
        clippy::too_many_lines,
        reason = "GPU setup is one render-thread resource graph whose local handles encode dependency order"
    )]
    fn setup(&mut self, ctx: &Context<'_>) {
        let config = &self.resolved_config;
        let device = ctx.device;
        let particle_size = <GpuParticle as ShaderSize>::SHADER_SIZE.get();
        let buffer_size = particle_size * u64::from(config.max_particles);
        let collision_stride = <GpuCircleObstacle as ShaderSize>::SHADER_SIZE.get();
        let collision_count = config.collision_circle_obstacles.len().max(1);
        let particle_buffers = std::array::from_fn(|index| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match index {
                    0 => "Particle Buffer A",
                    _ => "Particle Buffer B",
                }),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            fill_mapped_buffer(&buffer, 0);
            Some(buffer)
        });

        let uniform_size = <Uniforms as ShaderSize>::SHADER_SIZE.get();
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Uniforms"),
            size: uniform_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let collision_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Collision Obstacles"),
            size: collision_stride
                * u64::try_from(collision_count)
                    .expect("particle obstacle count must fit into wgpu buffer addressing"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        write_obstacles(
            ctx.queue,
            &collision_buffer,
            &config.collision_circle_obstacles,
        );

        let u32_size = u64::try_from(core::mem::size_of::<u32>())
            .expect("u32 size must fit into wgpu's u64 buffer addressing");
        let grid_cells_size =
            u64::from(max_interaction_grid_cells(config.max_particles)) * u32_size;
        let cell_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Interaction Cell Counts"),
            size: grid_cells_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        fill_mapped_buffer(&cell_counts_buffer, 0);
        let cell_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Interaction Cell Offsets"),
            size: grid_cells_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        fill_mapped_buffer(&cell_offsets_buffer, 0);
        let cell_cursors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Interaction Cell Cursors"),
            size: grid_cells_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        fill_mapped_buffer(&cell_cursors_buffer, 0);
        let particle_links_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Interaction Links"),
            size: u64::from(config.max_particles) * u32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        fill_mapped_buffer(&particle_links_buffer, 0);

        let clear_grid_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "clear_grid");
        let count_grid_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "count_grid");
        let prefix_grid_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "prefix_grid");
        let scatter_grid_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "scatter_grid");
        let sort_grid_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "sort_grid");
        let simulate_shader =
            COMPUTE_SHADER.create_entry_point(device, ShaderStage::Compute, "simulate_particles");
        let compute_bind_group_layout =
            single_bind_group_layout(&COMPUTE_SHADER, device, "the particle compute shader");
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Compute PL"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });
        let compute_pipeline_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let make_grid_pipeline = |shader: &CompiledShaderModule, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&compute_pipeline_layout),
                module: shader.module(),
                entry_point: Some(shader.entry_point()),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let clear_grid_pipeline =
            make_grid_pipeline(&clear_grid_shader, "Particle Clear Grid Pipeline");
        let count_grid_pipeline =
            make_grid_pipeline(&count_grid_shader, "Particle Count Grid Pipeline");
        let prefix_grid_pipeline =
            make_grid_pipeline(&prefix_grid_shader, "Particle Prefix Grid Pipeline");
        let scatter_grid_pipeline =
            make_grid_pipeline(&scatter_grid_shader, "Particle Scatter Grid Pipeline");
        let sort_grid_pipeline =
            make_grid_pipeline(&sort_grid_shader, "Particle Sort Grid Pipeline");
        let simulate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Simulate Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: simulate_shader.module(),
            entry_point: Some(simulate_shader.entry_point()),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let compute_pipeline_error = pollster::block_on(compute_pipeline_scope.pop());
        assert!(
            compute_pipeline_error.is_none(),
            "particle compute pipeline creation failed: {compute_pipeline_error:?}"
        );
        let compute_bind_groups = std::array::from_fn(|source_index| {
            let target_index = 1 - source_index;
            Some(
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(match source_index {
                        0 => "Particle Compute BG A->B",
                        _ => "Particle Compute BG B->A",
                    }),
                    layout: &compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: particle_buffers[source_index]
                                .as_ref()
                                .expect("source particle buffer must exist")
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: particle_buffers[target_index]
                                .as_ref()
                                .expect("target particle buffer must exist")
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: collision_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: cell_counts_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: cell_offsets_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: cell_cursors_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: particle_links_buffer.as_entire_binding(),
                        },
                    ],
                }),
            )
        });

        let (vertex_shader, fragment_shader, render_bind_group_layout) =
            single_bind_group_render_stages(
                &RENDER_SHADER,
                device,
                "the particle render shader",
                "vs_main",
                "fs_main",
            );
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Render PL"),
                bind_group_layouts: &[Some(&render_bind_group_layout)],
                immediate_size: 0,
            });
        let render_pipeline_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: vertex_shader.module(),
                entry_point: Some(vertex_shader.entry_point()),
                buffers: &[Some(particle_vertex_layout())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_shader.module(),
                entry_point: Some(fragment_shader.entry_point()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.format,
                    blend: Some(blend_state(config.blend_mode)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let render_pipeline_error = pollster::block_on(render_pipeline_scope.pop());
        assert!(
            render_pipeline_error.is_none(),
            "particle render pipeline creation failed: {render_pipeline_error:?}"
        );
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Render BG"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.particle_buffers = particle_buffers;
        self.collision_buffer = Some(collision_buffer);
        self.cell_counts_buffer = Some(cell_counts_buffer);
        self.cell_offsets_buffer = Some(cell_offsets_buffer);
        self.cell_cursors_buffer = Some(cell_cursors_buffer);
        self.particle_links_buffer = Some(particle_links_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.clear_grid_pipeline = Some(clear_grid_pipeline);
        self.count_grid_pipeline = Some(count_grid_pipeline);
        self.prefix_grid_pipeline = Some(prefix_grid_pipeline);
        self.scatter_grid_pipeline = Some(scatter_grid_pipeline);
        self.sort_grid_pipeline = Some(sort_grid_pipeline);
        self.simulate_pipeline = Some(simulate_pipeline);
        self.render_pipeline = Some(render_pipeline);
        self.compute_bind_groups = compute_bind_groups;
        self.render_bind_group = Some(render_bind_group);
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let mut obstacles_changed = false;
        for config in self.updates.try_iter() {
            obstacles_changed |= config.collision_circle_obstacles
                != self.resolved_config.collision_circle_obstacles;
            self.resolved_config = config;
        }
        let config = &self.resolved_config;
        if obstacles_changed {
            write_obstacles(
                frame.queue,
                self.collision_buffer
                    .as_ref()
                    .expect("collision buffer must exist before render"),
                &config.collision_circle_obstacles,
            );
        }
        self.update_uniforms(
            config,
            frame.queue,
            frame.width,
            frame.height,
            frame.elapsed,
            frame.delta,
        );

        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Particle Encoder"),
            });
        let render_buffer_index =
            self.encode_simulation_passes(&mut encoder, self.current_particle_buffer_index, config);

        {
            let pipeline = self
                .render_pipeline
                .as_ref()
                .expect("render pipeline must exist before render");
            let bind_group = self
                .render_bind_group
                .as_ref()
                .expect("render bind group must exist before render");
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.set_vertex_buffer(0, self.particle_buffer(render_buffer_index).slice(..));
            rpass.draw(0..6, 0..config.max_particles);
        }

        frame.queue.submit(std::iter::once(encoder.finish()));
        self.current_particle_buffer_index = render_buffer_index;
        frame.request_redraw();
    }
}

fn f32_to_u32_ceil(value: f32) -> u32 {
    value
        .ceil()
        .to_u32()
        .expect("particle dimension must be representable as u32")
}

fn u32_to_f32(value: u32) -> f32 {
    value
        .to_f32()
        .expect("particle count must be representable as f32")
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    use super::ResolvedParticleConfig;
    use super::{ParticleRenderer, blend_state, encode_emitter_size, resolved_linear_color};
    use crate::{
        EmitterShape,
        config::{BlendMode, ParticleConfig},
        gpu::{ShaderVec2, ShaderVec4},
    };
    #[cfg(not(target_arch = "wasm32"))]
    use crate::{
        ParticleShape,
        gpu::{CollisionUniforms, GpuParticle, InteractionUniforms, Uniforms},
    };
    #[cfg(not(target_arch = "wasm32"))]
    use encase::{ShaderSize, StorageBuffer};
    use waterui_core::{Binding, Environment, SignalExt};
    #[cfg(not(target_arch = "wasm32"))]
    use waterui_graphics::{
        cherenkov::{FrameTime, Offscreen, OffscreenFormat},
        gpu::{
            Context, Frame, GpuContent, GpuContentView, GpuRuntime, RedrawHandle, SharedGpuContext,
        },
        offscreen::OffscreenImage,
    };
    use waterui_graphics::{color::WorkingColor, wgpu};

    /// The shared runtime's context: the shared device requests the features
    /// the shaderloom passthrough modules and cherenkov both need.
    #[cfg(not(target_arch = "wasm32"))]
    fn test_gpu() -> std::sync::Arc<SharedGpuContext> {
        pollster::block_on(GpuRuntime::new())
            .expect("particle GPU tests require a GPU runtime")
            .context()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn test_gpu_context(gpu: &SharedGpuContext) -> Context<'_> {
        Context {
            adapter: gpu.adapter(),
            device: gpu.device(),
            queue: gpu.queue(),
            format: wgpu::TextureFormat::Rgba16Float,
            redraw: RedrawHandle::new(|| {}),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn opaque_white() -> WorkingColor {
        WorkingColor {
            components: [1.0; 4],
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn particle_test_config(max_particles: u32) -> ResolvedParticleConfig {
        ResolvedParticleConfig {
            max_particles,
            emitter_pos: [0.5, 0.5],
            emitter_shape: EmitterShape::Point,
            emit_rate: 0.0,
            gravity: [0.0, 0.0],
            wind: [0.0, 0.0],
            turbulence: 0.0,
            drag: 1.0,
            interaction_enabled: false,
            interaction_radius: 0.0,
            interaction_strength: 0.0,
            collision_enabled: false,
            collision_bounds: [0.0, 0.0, 1.0, 1.0],
            collision_restitution: 1.0,
            collision_surface_friction: 1.0,
            collision_circle_obstacles: Vec::new(),
            life_range: [1.0, 1.0],
            speed_range: [0.0, 0.0],
            angle_range: [0.0, 0.0],
            size_range: [0.1, 0.1],
            spin_range: [0.0, 0.0],
            color_start: opaque_white(),
            color_end: opaque_white(),
            stretch_with_velocity: false,
            blend_mode: BlendMode::Alpha,
            softness: 0.0,
            shape: ParticleShape::Circle,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn simulate_and_read_particles(
        renderer: &ParticleRenderer,
        ctx: &Context<'_>,
        particle_count: u32,
        label: &str,
    ) -> Vec<u8> {
        use std::sync::mpsc;

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let config = renderer.resolved_config.clone();
        let target_index = renderer.encode_simulation_passes(
            &mut encoder,
            renderer.current_particle_buffer_index,
            &config,
        );

        let buffer_size =
            <GpuParticle as ShaderSize>::SHADER_SIZE.get() * u64::from(particle_count);
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            renderer.particle_buffer(target_index),
            0,
            &readback,
            0,
            buffer_size,
        );
        ctx.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("particle compute readback callback dropped")
            .expect("particle compute readback mapping failed");

        let mapped = slice
            .get_mapped_range()
            .expect("particle compute readback range is mapped");
        let bytes = mapped.to_vec();
        drop(mapped);
        readback.unmap();
        bytes
    }

    #[test]
    fn alpha_mode_uses_premultiplied_blending() {
        assert_eq!(
            blend_state(BlendMode::Alpha),
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
        );
    }

    #[test]
    fn circle_emitters_use_disk_encoding() {
        assert_eq!(
            encode_emitter_size(EmitterShape::Circle { radius: 0.25 }),
            ShaderVec2::new(0.25, -1.0)
        );
        assert_eq!(
            encode_emitter_size(EmitterShape::Rect {
                width: 0.4,
                height: 0.2,
            }),
            ShaderVec2::new(0.4, 0.2)
        );
    }

    #[test]
    fn resolved_colors_keep_hdr_headroom() {
        let color = WorkingColor {
            components: [0.5, 1.0, 1.5, 0.4],
        };

        assert_eq!(
            resolved_linear_color(color),
            ShaderVec4::new(0.5, 1.0, 1.5, 0.4)
        );
    }

    #[test]
    fn reactive_config_crosses_render_thread_channel() {
        let rate = Binding::f32(100.0);
        let mut config = ParticleConfig::default();
        config.emitter.rate = rate.computed();
        let (feed, renderer) = ParticleRenderer::reactive(128, config, &Environment::new());
        rate.set(240.0);
        feed.pump();
        let updated = renderer.updates.try_recv().expect("updated configuration");
        assert!((updated.emit_rate - 240.0).abs() < f32::EPSILON);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn render_pass_draws_when_particle_buffer_is_prefilled() {
        use waterui_graphics::offscreen::OffscreenSize;

        struct PrefilledParticleRenderer {
            inner: ParticleRenderer,
        }

        impl GpuContent for PrefilledParticleRenderer {
            fn setup(&mut self, ctx: &Context<'_>) {
                self.inner.setup(ctx);

                let mut particle = GpuParticle::default();
                particle.pos = ShaderVec2::new(0.5, 0.5);
                particle.vel = ShaderVec2::ZERO;
                particle.life = 1.0;
                particle.max_life = 1.0;
                particle.size = 0.25;
                particle.rotation = 0.0;
                particle.rot_speed = 0.0;
                particle.color = ShaderVec4::ONE;

                let mut particle_data = StorageBuffer::new(Vec::new());
                particle_data
                    .write(&vec![particle])
                    .expect("prefilled particle buffer encoding must succeed");
                ctx.queue
                    .write_buffer(self.inner.particle_buffer(0), 0, particle_data.as_ref());
            }

            fn render(&mut self, frame: &mut Frame<'_>) {
                self.inner.render(frame);
            }
        }

        let renderer = PrefilledParticleRenderer {
            inner: ParticleRenderer::new(ResolvedParticleConfig {
                size_range: [0.25, 0.25],
                color_start: WorkingColor {
                    components: [1.0, 0.5, 0.0, 1.0],
                },
                color_end: WorkingColor {
                    components: [1.0, 0.5, 0.0, 1.0],
                },
                ..particle_test_config(1)
            }),
        };

        let size = OffscreenSize::try_from_pixels(256, 256).expect("test size must be valid");
        let runtime =
            pollster::block_on(GpuRuntime::new()).expect("GPU test requires a GPU runtime");
        let engine = runtime.engine().expect("GPU test requires an engine");
        let pixels = (size.width(), size.height());
        let surface = engine
            .surface(Offscreen::new(pixels, OffscreenFormat::LinearF16))
            .expect("offscreen surface must be created");
        let content = GpuContentView::new(renderer).take_engine_content(|| {});
        surface.update(|tx| {
            tx[surface.root()].content(engine.gpu_content(pixels, content));
        });
        engine
            .render(FrameTime::now())
            .expect("engine frame must render");
        let output = OffscreenImage::from_readback(
            &surface.readback().expect("offscreen readback must succeed"),
        );

        assert_eq!(output.width, 256);
        assert_eq!(output.height, 256);
        assert_eq!(output.rgba8.len(), 256 * 256 * 4);
        let center = (128 * 256 + 128) * 4;
        let [red, green, blue, alpha] = output.rgba8[center..center + 4]
            .try_into()
            .expect("RGBA pixel");
        assert_eq!(alpha, 255);
        assert!(
            red > green && green > blue,
            "the engine must composite the orange particle"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compute_pass_writes_live_particles_to_storage_buffer() {
        let mut renderer = ParticleRenderer::new(ResolvedParticleConfig {
            emit_rate: 1_000_000.0,
            color_end: WorkingColor {
                components: [1.0, 1.0, 1.0, 0.0],
            },
            ..particle_test_config(256)
        });

        let gpu = test_gpu();
        let ctx = test_gpu_context(&gpu);
        renderer.setup(&ctx);
        let config = renderer.resolved_config.clone();
        renderer.update_uniforms(
            &config,
            ctx.queue,
            256,
            256,
            std::time::Duration::from_secs_f32(1.0 / 60.0),
            std::time::Duration::from_secs_f32(1.0 / 60.0),
        );

        let mapped = simulate_and_read_particles(
            &renderer,
            &ctx,
            config.max_particles,
            "particle_compute_buffer_test_readback",
        );
        assert!(
            mapped.iter().any(|byte| *byte != 0),
            "compute pass should write non-zero particle data"
        );

        let stride = usize::try_from(<GpuParticle as ShaderSize>::SHADER_SIZE.get())
            .expect("GpuParticle shader size must fit usize");
        let mut live_particles = 0usize;
        for chunk in mapped.chunks_exact(stride) {
            let life = f32::from_ne_bytes(chunk[16..20].try_into().expect("life bytes must exist"));
            let max_life =
                f32::from_ne_bytes(chunk[20..24].try_into().expect("max life bytes must exist"));
            let size = f32::from_ne_bytes(chunk[24..28].try_into().expect("size bytes must exist"));
            let pos_x = f32::from_ne_bytes(chunk[0..4].try_into().expect("pos x bytes must exist"));
            let pos_y = f32::from_ne_bytes(chunk[4..8].try_into().expect("pos y bytes must exist"));
            if life.is_finite() && max_life.is_finite() && size.is_finite() && life > 0.0 {
                assert!(max_life > 0.0, "live particle must have positive max_life");
                assert!(size > 0.0, "live particle must have positive size");
                assert!(
                    (0.0..=1.0).contains(&pos_x) && (0.0..=1.0).contains(&pos_y),
                    "live particle position should stay normalized, got ({pos_x}, {pos_y})"
                );
                live_particles += 1;
            }
        }
        assert!(
            live_particles > 0,
            "compute pass should spawn at least one live particle"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compute_pass_applies_bounds_collision_on_gpu() {
        let mut renderer = ParticleRenderer::new(ResolvedParticleConfig {
            collision_enabled: true,
            collision_bounds: [0.0, 0.0, 1.0, 1.0],
            collision_restitution: 0.5,
            collision_surface_friction: 0.25,
            ..particle_test_config(1)
        });

        let gpu = test_gpu();
        let ctx = test_gpu_context(&gpu);
        renderer.setup(&ctx);

        let mut particle = GpuParticle::default();
        particle.pos = ShaderVec2::new(0.95, 0.5);
        particle.vel = ShaderVec2::new(2.0, 1.0);
        particle.life = 1.0;
        particle.max_life = 1.0;
        particle.size = 0.1;
        particle.rotation = 0.0;
        particle.rot_speed = 0.0;
        particle.color = ShaderVec4::ONE;

        let mut particle_data = StorageBuffer::new(Vec::new());
        particle_data
            .write(&vec![particle])
            .expect("prefilled collision particle encoding must succeed");
        ctx.queue
            .write_buffer(renderer.particle_buffer(0), 0, particle_data.as_ref());

        let uniforms = Uniforms {
            dt: 0.1,
            max_particles: 1,
            collision: CollisionUniforms::new(
                true,
                0.5,
                0.25,
                ShaderVec4::new(0.0, 0.0, 1.0, 1.0),
                0,
            ),
            size_range: ShaderVec2::new(0.1, 0.1),
            color_start: ShaderVec4::ONE,
            color_end: ShaderVec4::ONE,
            ..Uniforms::default()
        };
        let mut uniform_data = StorageBuffer::new(Vec::new());
        uniform_data
            .write(&uniforms)
            .expect("collision test uniform encoding must succeed");
        ctx.queue.write_buffer(
            renderer
                .uniform_buffer
                .as_ref()
                .expect("uniform buffer must exist after setup"),
            0,
            uniform_data.as_ref(),
        );

        let buffer_size = <GpuParticle as ShaderSize>::SHADER_SIZE.get();
        let mapped = simulate_and_read_particles(
            &renderer,
            &ctx,
            1,
            "particle_collision_buffer_test_readback",
        );
        let updated = bytemuck::pod_read_unaligned::<GpuParticle>(
            &mapped[..usize::try_from(buffer_size).expect("particle buffer size must fit usize")],
        );
        assert!(
            (updated.pos.x() - 0.9).abs() < 0.0001,
            "particle should clamp against the right wall, got {}",
            updated.pos.x()
        );
        assert!(
            (updated.vel.x() + 1.0).abs() < 0.0001,
            "particle should bounce on the x axis, got {}",
            updated.vel.x()
        );
        assert!(
            (updated.vel.y() - 0.25).abs() < 0.0001,
            "particle tangential velocity should be damped on collision, got {}",
            updated.vel.y()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compute_pass_applies_circle_obstacle_collision_on_gpu() {
        let mut renderer = ParticleRenderer::new(ResolvedParticleConfig {
            collision_circle_obstacles: vec![[0.3, 0.4, 0.08], [0.8, 0.5, 0.05]],
            collision_restitution: 0.5,
            collision_surface_friction: 0.25,
            ..particle_test_config(1)
        });

        let gpu = test_gpu();
        let ctx = test_gpu_context(&gpu);
        renderer.setup(&ctx);

        let mut particle = GpuParticle::default();
        particle.pos = ShaderVec2::new(0.86, 0.5);
        particle.vel = ShaderVec2::new(-1.0, 0.4);
        particle.life = 1.0;
        particle.max_life = 1.0;
        particle.size = 0.05;
        particle.color = ShaderVec4::ONE;

        let mut particle_data = StorageBuffer::new(Vec::new());
        particle_data
            .write(&vec![particle])
            .expect("prefilled obstacle collision particle encoding must succeed");
        ctx.queue
            .write_buffer(renderer.particle_buffer(0), 0, particle_data.as_ref());

        let uniforms = Uniforms {
            dt: 0.0,
            max_particles: 1,
            collision: CollisionUniforms::new(
                false,
                0.5,
                0.25,
                ShaderVec4::new(0.0, 0.0, 1.0, 1.0),
                2,
            ),
            size_range: ShaderVec2::new(0.05, 0.05),
            color_start: ShaderVec4::ONE,
            color_end: ShaderVec4::ONE,
            ..Uniforms::default()
        };
        let mut uniform_data = StorageBuffer::new(Vec::new());
        uniform_data
            .write(&uniforms)
            .expect("obstacle collision test uniform encoding must succeed");
        ctx.queue.write_buffer(
            renderer
                .uniform_buffer
                .as_ref()
                .expect("uniform buffer must exist after setup"),
            0,
            uniform_data.as_ref(),
        );

        let buffer_size = <GpuParticle as ShaderSize>::SHADER_SIZE.get();
        let mapped = simulate_and_read_particles(
            &renderer,
            &ctx,
            1,
            "particle_circle_collision_buffer_test_readback",
        );
        let updated = bytemuck::pod_read_unaligned::<GpuParticle>(
            &mapped[..usize::try_from(buffer_size).expect("particle buffer size must fit usize")],
        );
        assert!(
            (updated.pos.x() - 0.9).abs() < 0.0001,
            "particle should clamp against the second obstacle surface, got {}",
            updated.pos.x()
        );
        assert!(
            (updated.vel.x() - 0.5).abs() < 0.0001,
            "particle should bounce away from the obstacle, got {}",
            updated.vel.x()
        );
        assert!(
            (updated.vel.y() - 0.1).abs() < 0.0001,
            "particle tangential velocity should be damped against the obstacle, got {}",
            updated.vel.y()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn compute_pass_applies_particle_neighbor_interaction_on_gpu() {
        let mut renderer = ParticleRenderer::new(ResolvedParticleConfig {
            interaction_enabled: true,
            interaction_radius: 0.02,
            interaction_strength: 20.0,
            ..particle_test_config(2)
        });

        let gpu = test_gpu();
        let ctx = test_gpu_context(&gpu);
        renderer.setup(&ctx);

        let mut first = GpuParticle::default();
        first.pos = ShaderVec2::new(0.5, 0.5);
        first.life = 1.0;
        first.max_life = 1.0;
        first.size = 0.02;
        first.color = ShaderVec4::ONE;

        let mut second = GpuParticle::default();
        second.pos = ShaderVec2::new(0.53, 0.5);
        second.life = 1.0;
        second.max_life = 1.0;
        second.size = 0.02;
        second.color = ShaderVec4::ONE;

        let mut particle_data = StorageBuffer::new(Vec::new());
        particle_data
            .write(&vec![first, second])
            .expect("neighbor interaction particle encoding must succeed");
        ctx.queue
            .write_buffer(renderer.particle_buffer(0), 0, particle_data.as_ref());

        let uniforms = Uniforms {
            dt: 0.1,
            max_particles: 2,
            interaction: InteractionUniforms::new(true, 2, 2, 0.02, 20.0),
            size_range: ShaderVec2::new(0.02, 0.02),
            color_start: ShaderVec4::ONE,
            color_end: ShaderVec4::ONE,
            ..Uniforms::default()
        };
        let mut uniform_data = StorageBuffer::new(Vec::new());
        uniform_data
            .write(&uniforms)
            .expect("neighbor interaction uniform encoding must succeed");
        ctx.queue.write_buffer(
            renderer
                .uniform_buffer
                .as_ref()
                .expect("uniform buffer must exist after setup"),
            0,
            uniform_data.as_ref(),
        );

        let mapped = simulate_and_read_particles(
            &renderer,
            &ctx,
            2,
            "particle_neighbor_interaction_test_readback",
        );
        let stride = usize::try_from(<GpuParticle as ShaderSize>::SHADER_SIZE.get())
            .expect("GpuParticle shader size must fit usize");
        let first_updated = bytemuck::pod_read_unaligned::<GpuParticle>(&mapped[..stride]);
        let second_updated =
            bytemuck::pod_read_unaligned::<GpuParticle>(&mapped[stride..(2 * stride)]);
        assert!(
            first_updated.vel.x() < 0.0,
            "first particle should be pushed left, got {}",
            first_updated.vel.x()
        );
        assert!(
            second_updated.vel.x() > 0.0,
            "second particle should be pushed right, got {}",
            second_updated.vel.x()
        );
    }

    /// A standalone device for the benchmark; Metal in this VM exposes no
    /// timestamp counters, so submission-to-completion wall time is measured
    /// instead of query-set deltas.
    #[cfg(not(target_arch = "wasm32"))]
    fn benchmark_gpu() -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .expect("interaction benchmark requires a GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: shaderloom::required_features(adapter.features()),
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("interaction benchmark requires a device");
        (adapter, device, queue)
    }

    /// Median GPU time of the simulation passes over eight frames, in
    /// nanoseconds, with every particle alive and spread over the viewport.
    #[cfg(not(target_arch = "wasm32"))]
    fn simulation_pass_gpu_ns(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
    ) -> f64 {
        use std::sync::mpsc;
        use std::time::Duration;

        let (_sender, receiver) = mpsc::channel();
        let config = ResolvedParticleConfig {
            interaction_enabled: true,
            interaction_radius: 0.01,
            interaction_strength: 16.0,
            life_range: [1.0e9, 1.0e9],
            ..particle_test_config(particle_count)
        };
        let mut renderer = ParticleRenderer::with_config(config.clone(), receiver);
        let ctx = Context {
            adapter,
            device,
            queue,
            format: wgpu::TextureFormat::Rgba16Float,
            redraw: RedrawHandle::new(|| {}),
        };
        renderer.setup(&ctx);

        let row_count = particle_count / 1024;
        let particles: Vec<GpuParticle> = (0..particle_count)
            .map(|index| {
                let mut particle = GpuParticle::default();
                let x = super::u32_to_f32(index % 1024) / 1023.0;
                let y = super::u32_to_f32(index / 1024) / super::u32_to_f32(row_count);
                particle.pos = ShaderVec2::new(0.96f32.mul_add(x, 0.02), 0.96f32.mul_add(y, 0.02));
                particle.life = 1.0e9;
                particle.max_life = 1.0e9;
                particle.size = 0.006;
                particle.color = ShaderVec4::ONE;
                particle
            })
            .collect();
        let mut particle_data = StorageBuffer::new(Vec::new());
        particle_data
            .write(&particles)
            .expect("benchmark particle encoding must succeed");
        queue.write_buffer(renderer.particle_buffer(0), 0, particle_data.as_ref());

        let mut source_index = 0;
        let mut samples = Vec::new();
        for frame in 0..8u32 {
            renderer.update_uniforms(
                &config,
                queue,
                600,
                600,
                Duration::from_secs_f64(f64::from(frame) / 60.0),
                Duration::from_secs_f64(1.0 / 60.0),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interaction benchmark encoder"),
            });
            let target_index =
                renderer.encode_simulation_passes(&mut encoder, source_index, &config);
            let submitted = std::time::Instant::now();
            queue.submit([encoder.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            source_index = target_index;
            samples.push(submitted.elapsed());
        }
        samples.sort();
        samples[samples.len() / 2].as_secs_f64() * 1.0e9
    }

    /// Manual benchmark: `cargo test interaction_gpu_time -- --ignored --nocapture`.
    /// Reports the GPU time of the simulation passes (interaction grid build,
    // neighbor impulses and integration) per frame at two particle counts.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "manual GPU benchmark"]
    fn interaction_gpu_time_scales_with_neighbors() {
        let (adapter, device, queue) = benchmark_gpu();
        for &particle_count in &[6_000_u32, 24_000] {
            let median_ns = simulation_pass_gpu_ns(&adapter, &device, &queue, particle_count);
            println!(
                "simulation passes at {particle_count} particles: {:.3} ms GPU",
                median_ns / 1.0e6
            );
        }
    }
}
