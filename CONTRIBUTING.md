# Contributing

1. Fork the repo
2. Create a feature branch
3. Run `cargo test` and `cargo clippy --all-targets`
4. Submit a PR

## Code Style

- `cargo fmt` for formatting
- `cargo clippy` for linting
- All public APIs need doc comments

## Testing

- Unit tests alongside source files
- Integration tests in `tests/`
- Benchmarks in `benches/` (Criterion)
