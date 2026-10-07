# Changelog

## Unreleased

- `quack` now runs the `.gitignore` phase once per reindex, off the write path, and skips it entirely when `gitignore: false` (MAR-144).
