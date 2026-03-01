---
name: commit
description: Create commits following evie's conventional commits format
disable-model-invocation: true
---

# Conventional Commits

evie uses conventional commits to maintain a clear and organized commit history.

## Format

```
<type>(<issue-number>): <description>
```

## Commit Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring without changing functionality
- `test` - Adding or updating tests
- `chore` - Maintenance tasks, dependency updates, etc.

## Guidelines

- Use present tense: "add feature" not "added feature"
- Use imperative mood: "move cursor to..." not "moves cursor to..."
- Keep the first line under 72 characters
- Always reference the issue number in parentheses after the type
- For multi-line commits, include a blank line between the header and body

## Examples

Single line:
```
feat(5): implement multi-head attention mechanism
```

Multi-line:
```
feat(8): add learning rate scheduler

Implement cosine annealing with warmup for better training stability.
The scheduler gradually increases LR during warmup then decreases
following a cosine curve.

Closes #8
```

Bug fix:
```
fix(12): correct gradient accumulation in training loop
```

Documentation:
```
docs(6): add dataset preprocessing guide
```

## Closing Issues

If your commit closes an issue, add to the commit body or title:

```
Closes #<issue-number>
```

This automatically closes the issue when the commit is merged.
