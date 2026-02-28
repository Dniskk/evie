---
name: commit
description: Create a conventional commit following evie's format
argument-hint: [type] [issue-number] [message]
disable-model-invocation: true
---

# Conventional Commit

Create a commit following evie's conventional commits format.

## Usage

```bash
/commit <type> <issue-number> <message>
```

## Types

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

## Example

```bash
/commit feat 5 implement multi-head attention mechanism
```

Creates commit: `feat(5): implement multi-head attention mechanism`

## Format

All commits follow the pattern: `<type>(<issue-number>): <message>`

- Use present tense ("add" not "added")
- Use imperative mood ("move" not "moves")
- Keep first line under 72 characters
- Reference the issue number
