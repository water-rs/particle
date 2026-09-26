//! `ParticleSystem` View with ergonomic API.

use crate::{
    EmitterShape,
    config::{BlendMode, CircleObstacleConfig, ParticleConfig},
    renderer::{ParticleFeed, ParticleRenderer},
};
use core::num::NonZeroU32;
use std::time::{Duration, Instant};
use waterui_core::{Computed, Environment, IntoSignal, IntoSignalF32, SignalExt, View};
use waterui_graphics::{
    cherenkov::{FrameTime, Offscreen, OffscreenFormat, Readback},
    color::Color,
    gpu::{GpuContentView, GpuRuntime},
    offscreen::{OffscreenError, OffscreenSize},
};

/// High-performance GPU particle system.
///
/// Use the flat modifier-chain API to configure the particle system,
/// then use it as a View in your UI hierarchy.
#[derive(Clone, Debug)]
pub struct ParticleSystem {
    max_particles: u32,
    pub(crate) config: ParticleConfig,
}

fn computed_f32(signal: impl IntoSignalF32 + 'static) -> Computed<f32> {
    signal.into_signal_f32().computed()
}

fn computed_color(signal: impl IntoSignal<Color> + 'static) -> Computed<Color> {
    signal.into_signal().computed()
}

fn computed_pair(
    start: impl IntoSignalF32 + 'static,
    end: impl IntoSignalF32 + 'static,
) -> Computed<[f32; 2]> {
    start
        .into_signal_f32()
        .zip(&end.into_signal_f32())
        .map(<[f32; 2]>::from)
        .computed()
}

impl ParticleSystem {
    /// Create a new particle system with a maximum particle count.
    #[must_use]
    pub fn new(max_particles: u32) -> Self {
        Self {
            max_particles,
            config: ParticleConfig::default(),
        }
    }

    /// Set emission rate (particles per second).
    #[must_use]
    pub fn rate(mut self, rate: impl IntoSignalF32 + 'static) -> Self {
        self.config.emitter.rate = computed_f32(rate);
        self
    }

    /// Set emitter as a single point.
    #[must_use]
    pub fn emit_from_point(mut self) -> Self {
        self.config.emitter.shape = Computed::constant(EmitterShape::Point);
        self
    }

    /// Set emitter as a rectangle.
    #[must_use]
    pub fn emit_from_rect(
        mut self,
        width: impl IntoSignalF32 + 'static,
        height: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.emitter.shape = width
            .into_signal_f32()
            .zip(&height.into_signal_f32())
            .map(|(width, height)| EmitterShape::Rect { width, height })
            .computed();
        self
    }

    /// Set emitter position (normalized coordinates 0.0-1.0).
    #[must_use]
    pub fn at(mut self, x: impl IntoSignalF32 + 'static, y: impl IntoSignalF32 + 'static) -> Self {
        self.config.emitter.position = computed_pair(x, y);
        self
    }

    /// Set particle lifespan range.
    #[must_use]
    pub fn life(
        mut self,
        start: impl IntoSignalF32 + 'static,
        end: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.particle.life = computed_pair(start, end);
        self
    }

    /// Set particle speed range.
    #[must_use]
    pub fn speed(
        mut self,
        start: impl IntoSignalF32 + 'static,
        end: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.particle.speed = computed_pair(start, end);
        self
    }

    /// Set particle emission angle range (in radians).
    #[must_use]
    pub fn angle(
        mut self,
        start: impl IntoSignalF32 + 'static,
        end: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.particle.angle = computed_pair(start, end);
        self
    }

    /// Set particle size range.
    #[must_use]
    pub fn size(
        mut self,
        start: impl IntoSignalF32 + 'static,
        end: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.particle.size = computed_pair(start, end);
        self
    }

    /// Set particle start and end colors.
    #[must_use]
    pub fn color(
        mut self,
        start: impl IntoSignal<Color> + 'static,
        end: impl IntoSignal<Color> + 'static,
    ) -> Self {
        self.config.particle.color_start = computed_color(start);
        self.config.particle.color_end = computed_color(end);
        self
    }

    /// Enable motion blur (stretch particles based on velocity).
    #[must_use]
    pub const fn stretch_with_velocity(mut self) -> Self {
        self.config.particle.stretch_with_velocity = true;
        self
    }

    /// Set gravity vector.
    #[must_use]
    pub fn gravity(
        mut self,
        x: impl IntoSignalF32 + 'static,
        y: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.environment.gravity = computed_pair(x, y);
        self
    }

    /// Set wind vector.
    #[must_use]
    pub fn wind(
        mut self,
        x: impl IntoSignalF32 + 'static,
        y: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.environment.wind = computed_pair(x, y);
        self
    }

    /// Keep particles inside the normalized viewport `[0, 0]..[1, 1]`.
    #[must_use]
    pub fn collide_with_viewport(mut self) -> Self {
        self.config.collision.enabled = true;
        self.config.collision.bounds = Computed::constant([0.0, 0.0, 1.0, 1.0]);
        self
    }

    /// Keep particles inside a normalized rectangle.
    #[must_use]
    pub fn collide_with_rect(
        mut self,
        x: impl IntoSignalF32 + 'static,
        y: impl IntoSignalF32 + 'static,
        width: impl IntoSignalF32 + 'static,
        height: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.collision.enabled = true;
        self.config.collision.bounds = x
            .into_signal_f32()
            .zip(&y.into_signal_f32())
            .zip(&width.into_signal_f32())
            .zip(&height.into_signal_f32())
            .map(|(((x, y), width), height)| [x, y, x + width, y + height])
            .computed();
        self
    }

    /// Add a circular obstacle collider in normalized coordinates.
    #[must_use]
    pub fn collide_with_circle_obstacle(
        mut self,
        x: impl IntoSignalF32 + 'static,
        y: impl IntoSignalF32 + 'static,
        radius: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config
            .collision
            .circle_obstacles
            .push(CircleObstacleConfig {
                value: x
                    .into_signal_f32()
                    .zip(&y.into_signal_f32())
                    .zip(&radius.into_signal_f32())
                    .map(|((x, y), radius)| [x, y, radius])
                    .computed(),
            });
        self
    }

    /// Set the fraction of normal velocity preserved after a collision.
    #[must_use]
    pub fn bounce(mut self, restitution: impl IntoSignalF32 + 'static) -> Self {
        self.config.collision.restitution = computed_f32(restitution);
        self
    }

    /// Set the fraction of tangential velocity preserved after a collision.
    #[must_use]
    pub fn surface_friction(mut self, value: impl IntoSignalF32 + 'static) -> Self {
        self.config.collision.surface_friction = computed_f32(value);
        self
    }

    /// Enable pure-GPU particle-particle interaction using a neighbor grid.
    #[must_use]
    pub fn collide_with_particles(
        mut self,
        radius: impl IntoSignalF32 + 'static,
        strength: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.interaction.enabled = true;
        self.config.interaction.radius = computed_f32(radius);
        self.config.interaction.strength = computed_f32(strength);
        self
    }

    /// Set additive blending mode.
    #[must_use]
    pub const fn additive(mut self) -> Self {
        self.config.blend_mode = BlendMode::Additive;
        self
    }

    /// Set edge softness (0.0=hard, 1.0=soft).
    #[must_use]
    pub fn softness(mut self, value: impl IntoSignalF32 + 'static) -> Self {
        self.config.particle.softness = computed_f32(value);
        self
    }

    /// Set turbulence strength.
    #[must_use]
    pub fn turbulence(mut self, value: impl IntoSignalF32 + 'static) -> Self {
        self.config.environment.turbulence = computed_f32(value);
        self
    }

    /// Set velocity damping factor normalized to a 60 FPS baseline.
    ///
    /// `1.0` keeps velocity unchanged, lower values damp motion over time
    /// without changing behavior across frame rates.
    #[must_use]
    pub fn drag(mut self, value: impl IntoSignalF32 + 'static) -> Self {
        self.config.environment.drag = computed_f32(value);
        self
    }

    /// Set emitter as a disk with the given radius.
    #[must_use]
    pub fn emit_from_circle(mut self, radius: impl IntoSignalF32 + 'static) -> Self {
        self.config.emitter.shape = radius
            .into_signal_f32()
            .map(|radius| EmitterShape::Circle { radius })
            .computed();
        self
    }

    /// Set particle shape.
    #[must_use]
    pub const fn shape(mut self, shape: crate::config::ParticleShape) -> Self {
        self.config.particle.shape = shape;
        self
    }

    /// Set initial particle spin speed range (radians/sec).
    #[must_use]
    pub fn spin(
        mut self,
        start: impl IntoSignalF32 + 'static,
        end: impl IntoSignalF32 + 'static,
    ) -> Self {
        self.config.particle.spin = computed_pair(start, end);
        self
    }

    fn renderer(self, env: &Environment) -> (ParticleFeed, ParticleRenderer) {
        ParticleRenderer::reactive(self.max_particles, self.config, env)
    }

    /// Simulates frames on the engine and reads premultiplied linear Display P3 pixels.
    ///
    /// An initial frame sets up the simulation, then `frame_count` steps advance it.
    /// The caller supplies the simulation interval; no wall-clock waiting is required.
    /// Convert the returned HDR pixels with `OffscreenImage::from_readback` for an SDR export.
    ///
    /// # Errors
    /// Returns engine initialization, surface creation, rendering, or readback errors.
    pub fn render_offscreen(
        self,
        runtime: &GpuRuntime,
        size: OffscreenSize,
        env: &Environment,
        frame_count: NonZeroU32,
        frame_interval: Duration,
    ) -> Result<Readback, OffscreenError> {
        let (feed, renderer) = self.renderer(env);
        let engine = runtime.engine()?;
        let pixels = (size.width(), size.height());
        let surface = engine.surface(Offscreen::new(pixels, OffscreenFormat::LinearF16))?;
        let content = GpuContentView::new(renderer).take_engine_content(|| {});
        surface.update(|tx| {
            tx[surface.root()].content(engine.gpu_content(pixels, content));
        });
        let start = Instant::now();
        engine.render(FrameTime::at(start))?;
        for frame in 1..=frame_count.get() {
            feed.pump();
            engine.render(FrameTime::at(start + frame_interval * frame))?;
        }
        Ok(surface.readback()?)
    }
}

impl View for ParticleSystem {
    fn body(self, env: &Environment) -> impl View {
        let (feed, renderer) = self.renderer(env);
        GpuContentView::new(renderer).on_frame(move || feed.pump())
    }
}

/// Convenience constructor for [`ParticleSystem`] with `max_particles` capacity.
#[must_use]
pub fn particles(max_particles: u32) -> ParticleSystem {
    ParticleSystem::new(max_particles)
}
