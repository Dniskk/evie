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

3. **Install pre-commit hooks**

Pre-commit hooks automatically run quality checks before each commit:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the git hooks
pre-commit install
```

What the pre-commit hooks do:
- **ruff format**: Automatically formats your Python code
- **ruff lint**: Checks for code quality issues and fixes auto-fixable ones
- **basedpyright**: Runs type checking on your code
- **pytest**: Runs the test suite to ensure nothing breaks
- **File checks**: Detects secrets, removes trailing whitespace, fixes file endings

When you commit, these checks run automatically. If any check fails:
- Fix the reported issues
- Stage the changes with `git add`
- Try committing again

To run the hooks manually on all files:

```bash
pre-commit run --all-files
```

To skip hooks (not recommended, only for emergencies):

```bash
git commit --no-verify -m "your message"
```

4. **Verify your setup**

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
- `style`: Code style changes (formatting, missing semicolons, etc.)
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

## Continuous Integration

Every push to `main` and every pull request triggers our CI pipeline on GitHub Actions. The pipeline runs four parallel jobs:

1. **Test Suite**: Runs pytest with coverage reporting
2. **Lint Check**: Runs ruff lint to catch code quality issues
3. **Format Check**: Verifies code is properly formatted with ruff
4. **Type Check**: Runs basedpyright for static type checking

All four jobs must pass before a PR can be merged. You can see the status of these checks on your PR page.

### Running CI Checks Locally

To ensure your PR will pass CI, run these commands before pushing:

```bash
# Run all checks at once with pre-commit
pre-commit run --all-files

# Or run individually:
uv run pytest tests/ -v --cov=evie
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

If any check fails in CI:
1. Review the error message in the GitHub Actions log
2. Fix the issue locally
3. Run the failing check locally to verify
4. Push the fix

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

## Getting Help

- Check existing [issues](https://github.com/Dniskk/evie/issues)
- Read the [documentation](docs/README.md)
- Ask questions in issue comments

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the project
- Show empathy towards other contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
