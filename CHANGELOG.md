# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/water-rs/particle/compare/v0.1.0...v0.1.1) - 2026-09-11

### Fixed

- *(ci)* install the same Linux packages for the release preflight

### Other

- link the test graph to the waterui 0.4 release commit
- *(deps)* waterui-graphics 0.4 (and waterui-text/-testing 0.4 where used)
- resolve gpu-allocator against windows 0.62, as wgpu-hal does
- depend on the released waterui crates instead of the monorepo dev branch
