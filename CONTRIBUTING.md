# Contributing to evie

Thank you for your interest in contributing to evie! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

### Getting Started

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/evie.git
cd evie
```

2. **Install dependencies**

```bash
uv sync
```

3. **Verify your setup**

```bash
# Run tests
uv run pytest

# Check code formatting
uv run ruff format --check .

# Check linting
uv run ruff check .

# Run type checker
uv run basedpyright
```

## Code Standards

### Code Style

We use the following tools to maintain code quality:

- **Formatter**: [ruff](https://docs.astral.sh/ruff/) for code formatting
- **Linter**: ruff with comprehensive rule set (see `pyproject.toml`)
- **Type Checker**: [basedpyright](https://docs.basedpyright.com/) for static type checking
- **Docstrings**: Google style

Before submitting a PR, ensure your code passes all checks:

```bash
# Format code
uv run ruff format .

# Fix auto-fixable linting issues
uv run ruff check --fix .

# Type check
uv run basedpyright
```

### Type Hints

All functions and methods should have complete type hints:

```python
def process_tokens(tokens: list[str], max_length: int) -> list[str]:
    """Process tokens to fit within max length.

    Args:
        tokens: List of token strings to process.
        max_length: Maximum number of tokens to return.

    Returns:
        Processed list of tokens.
    """
    return tokens[:max_length]
```

### Docstrings

Use Google-style docstrings when helpful for understanding complex functionality:

```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 10,
) -> dict[str, float]:
    """Train the model on the provided data.

    Args:
        model: The neural network model to train.
        data_loader: DataLoader providing training batches.
        epochs: Number of training epochs. Defaults to 10.

    Returns:
        Dictionary containing training metrics:
        - "loss": Final training loss
        - "accuracy": Final training accuracy

    Raises:
        ValueError: If epochs is less than 1.

    Example:
        >>> model = TransformerModel()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> metrics = train_model(model, loader, epochs=5)
        >>> print(f"Loss: {metrics['loss']:.4f}")
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    # Training implementation...
```

## Testing

Write tests for all new functionality:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_transformer.py

# Run with coverage
uv run pytest --cov=evie
```

### Test Structure

Tests should mirror the `src/` directory structure. For example:

```
src/
  evie/
    models/
      transformer.py

tests/
  evie/
    models/
      test_transformer.py
```

This makes it easy to find tests for any given module.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) with issue number references:

### Format

```
<type>(<issue-number>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, whitespace, etc.)
- `refactor`: Code refactoring without changing functionality
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates, etc.

### Examples

```bash
# Feature with issue number
feat(5): implement multi-head attention mechanism

# Bug fix
fix(12): correct gradient accumulation in training loop

# Documentation
docs(6): add dataset preprocessing guide

# Multiple paragraphs
feat(8): add learning rate scheduler

Implement cosine annealing with warmup for better training
stability. The scheduler gradually increases LR during warmup
then decreases following a cosine curve.

Closes #8
```

### Guidelines

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- Keep the first line under 72 characters
- Reference the issue number in parentheses after the type

## Pull Request Process

### Branch Naming

Create a new branch for each issue following this pattern:

```
<your-name>/<issue-number>-<short-description>
```

Examples:
- `alice/5-implement-transformer`
- `bob/12-fix-attention-mask`
- `charlie/6-add-dataset-docs`

### Creating a PR

Create a pull request using GitHub's interface or your preferred method. Ensure your PR:
- Has a clear title following conventional commits format
- Describes what changes were made and why
- References the issue number it addresses
- Assigns `Dniskk` as the reviewer

You can use `gh pr create` or the GitHub web interface to create PRs.

### PR Requirements

Before your PR can be merged:

- ✅ All tests pass
- ✅ Code is formatted (`ruff format`)
- ✅ No linting errors (`ruff check`)
- ✅ Type checking passes (`basedpyright`)
- ✅ New functionality includes tests
- ✅ Documentation is updated
- ✅ Commit messages follow conventional commits format
- ✅ PR has been reviewed and approved

### PR Description Template

```markdown
## Summary

Brief description of what this PR does.

## Changes

- Bullet point list of changes
- Another change

## Test Plan

- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing performed

## Documentation

- [ ] Updated relevant documentation
- [ ] Added docstrings to new code

Closes #<issue-number>
```

## Documentation

When you make changes, update the relevant documentation:

- **Code docs**: Add/update docstrings
- **Architecture**: Update `docs/architecture/` if design changes
- **Decisions**: Create an ADR in `docs/decisions/` for significant choices
- **Learnings**: Document insights in `docs/learnings/`
- **Training**: Update `docs/training/` for training-related changes

See [docs/README.md](docs/README.md) for the documentation structure.

## Claude Code Skills

The `.claude/skills/` directory contains helpful reference guides for common tasks:

- **commit/** - Conventional commits format and examples
- **pr/** - Pull request creation process and requirements

These skills provide quick reference for project conventions when working with Claude Code.

## Getting Help

- Check existing [issues](https://github.com/Dniskk/evie/issues)
- Read the [documentation](docs/README.md)
- Review `.claude/skills/` for common workflows
- Ask questions in issue comments

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
