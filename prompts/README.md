# Prompt inventory

These Markdown files are the prompt source of truth. Rust includes them at compile time, so
editing a file changes the next build without introducing runtime file lookup.

- `presets/` defines what one review lane investigates. The universal defaults are
  `correctness`, `security`, `performance`, and `simplicity`; the other files are opt-in.
- `protocol/` defines reviewer, debate, synthesis, delegation, and output contracts. Tokens such
  as `{{RUBRIC}}` are filled by the small renderer in `src/prompts.rs`.

Generic agent-loop contracts live with the code that interprets them under
[`crates/nitpicker-agent/prompts/`](../crates/nitpicker-agent/prompts/). Keeping those files inside
the library crate also makes them part of its published package.

Run `cargo test --workspace` after changing prompts. Tests verify that all protocol template
tokens resolve and that preset selection invariants still hold.
