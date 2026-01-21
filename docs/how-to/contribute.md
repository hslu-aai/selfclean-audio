# Contribute to the Project

We welcome issues, discussions, and pull requests. This guide covers the
recommended workflow for changes that touch code, experiments, or docs.

## Before You Start

- Check existing issues and discussions to avoid duplicates.
- If you are proposing a larger change, open a discussion with context and scope.

## Development Setup (Docker)

The repo is Docker-first to avoid dependency drift.

1. Start the stack:
   - `make up`
2. Open a shell inside the container:
   - `make bash`
3. Run the full check suite before opening a PR:
   - `make check`

## Documentation Updates

- Edit Markdown files under `docs/`.
- Build docs locally to verify:
  - `make docs`

## Submitting Changes

- Keep PRs focused and explain the motivation.
- Include outputs or plots when changing experimental behavior.
- Link to the issue or discussion that motivated the change.
