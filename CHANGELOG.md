# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Repository scaffold: packaging, tooling, CI, data contracts, fake sources, typed adapter stubs,
  spec tests, design records and backlog.
- USGS source adapter using `dataretrieval.waterdata`, including discharge and stage retrieval,
  SI unit conversion, station discovery, metadata, and cassette-backed contract tests.

### Changed
- Minimum supported Python is 3.12 (CI tests 3.12–3.14): the latest `numpy`, `zarr` and
  `icechunk` releases require it, and `icechunk` has no Python 3.10 wheels.
