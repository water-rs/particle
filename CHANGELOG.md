# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/water-rs/particle/compare/v0.2.0...v0.3.0) - 2026-11-14

### Added

- port to the render-target-neutral recording contract

### Fixed

- *(deps)* resolve gpu-allocator against windows 0.62
- depend on shaderloom's runtime API only ([#37](https://github.com/water-rs/particle/pull/37))
- *(deps)* repin waterui at the MSRV-compatible shaderloom declaration
- *(deps)* repin waterui at the passthrough-shader device fix
- *(style)* satisfy rustfmt 1.99
- repin hydrolysis to 14d5e231, m3 to fcb8b9e8; patch all registry waterkit crates to 123b0581
- *(ci)* pin the toolchain without disarming the MSRV gate ([#28](https://github.com/water-rs/particle/pull/28))

### Other

- update release-plz rev for Rust 1.100 compatibility
- release from main and retire dev ([#51](https://github.com/water-rs/particle/pull/51))
- *(deps)* point migrated Batch 1 git dependencies to main ([#49](https://github.com/water-rs/particle/pull/49)) ([#50](https://github.com/water-rs/particle/pull/50))
- take offscreen readback types from waterui_graphics::gpu
- follow waterui dev again
- resolve gpu-allocator against the windows version wgpu-hal uses
- lock the wgpu fork at the revision the framework builds DX12 with
- render offscreen frames at an unpresented frame time
- read waterui structs through their accessors
- move to shaderloom 0.2.1 for the explicit Metal target
- pass the release pull request's gate without CI
- *(deps)* pin water-rs/waterui to dev at e76ad1e
- Merge branch 'dev' into chore/1924-surface-wake
- follow cherenkov surface-wake signature (water-rs/waterui#1924)
- drop unused patch entries and route wgpu through waterui-graphics
- repin framework graph for scene recovery ([#35](https://github.com/water-rs/particle/pull/35))
- *(deps)* pin hydrolysis, hydrolysis-m3 and video-gpu at the repinned heads
- pin cherenkov to 52236561
- install SPIRV-Tools through the shared Linux deps action
- install SPIRV-Tools for cherenkov-gpu's shader build on non-Apple legs
- repin to the Cherenkov cutover and adapt to wgpu 30
- [**breaking**] render particles through engine GPU content
- repin the vello fork to the glifo canonical-location fix
- repin water-rs dependencies to merged dev heads
- repin water-rs dependencies to unified migration heads
- repin hydrolysis to accessibility import fix
- refresh lockfile for final branch revs
- repin migrated dependency branch heads
- pin migrated dependency branch revs
- [**breaking**] migrate to nami snapshot API
- consume the 0.5.0 framework wave from crates.io ([#27](https://github.com/water-rs/particle/pull/27))

## [0.1.1](https://github.com/water-rs/particle/compare/v0.1.0...v0.1.1) - 2026-09-11

### Fixed

- *(ci)* install the same Linux packages for the release preflight

### Other

- link the test graph to the waterui 0.4 release commit
- *(deps)* waterui-graphics 0.4 (and waterui-text/-testing 0.4 where used)
- resolve gpu-allocator against windows 0.62, as wgpu-hal does
- depend on the released waterui crates instead of the monorepo dev branch
