//! Semantics checks for the `particle` component.
//!
//! Two layers live here:
//!
//! - The crate-level assertions below, which run at the pinned
//!   engine/API revisions — that the component is a `View` and its body
//!   builds a GPU content view.
//! - The mounted-runtime checks in [`mounted`], behind the `host-e2e`
//!   feature. They mount the view through `waterui-testing` against a
//!   Hydrolysis backend — the image role and label the component exposes —
//!   and they cannot build yet: `waterui-testing` pulls `hydrolysis`, and
//!   no hydrolysis or hydrolysis-m3 revision compiles against the pinned
//!   waterui revision (`163fbb3`), since both still implement the retired
//!   `Scene2D` contract. Enabling the feature today fails this target with
//!   unresolved imports, by design: the dev-deps named in `Cargo.toml` go
//!   in once a hydrolysis revision compiles against the new API.

use waterui_core::{Environment, View};
use waterui_particle::ParticleSystem;

#[test]
fn particle_system_is_a_view_whose_body_builds() {
    fn assert_view(view: impl View) -> impl View {
        view
    }
    let view = assert_view(ParticleSystem::new(128).rate(48.0));
    // `body` creates the feed/renderer pair without touching a GPU.
    let _ = view.body(&Environment::new());
}

/// The mounted-runtime half of the semantics contract. Enabled by the
/// `host-e2e` feature once the test host can build at the pins.
#[cfg(feature = "host-e2e")]
mod mounted {
    use waterui::ViewExt as _;
    use waterui::accessibility::AccessibilityRole;
    use waterui::graphics::color::Srgb;
    use waterui_particle::ParticleSystem;
    use waterui_testing::{Role, SemanticApp};

    fn particle_system_view() -> impl waterui::View {
        ParticleSystem::new(128)
            .emit_from_rect(0.4, 0.2)
            .at(0.5, 0.2)
            .rate(48.0)
            .life(0.6, 1.0)
            .speed(0.15, 0.25)
            .angle(1.3, 1.8)
            .size(0.015, 0.03)
            .color(Srgb::WHITE, Srgb::new(1.0, 0.6, 0.1))
            .width(180.0)
            .height(120.0)
            .a11y_role(AccessibilityRole::Image)
            .a11y_label("Particle system")
    }

    #[waterui::test(particle_system_view)]
    fn particle_system_exposes_accessibility_image(app: &mut SemanticApp) {
        app.query()
            .role(Role::IMAGE)
            .label("Particle system")
            .assert_exists();
    }
}
