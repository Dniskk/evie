# evie

A framework for training custom LLMs.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run basedpyright
```

## Development

The project follows these standards:
- **Formatter**: ruff
- **Linter**: ruff with comprehensive rule set
- **Type checker**: basedpyright
- **Docstring style**: Google style

## Project Structure

```
evie/
├── src/
│   └── evie/          # Main package
├── tests/             # Test suite
└── pyproject.toml     # Project configuration
```