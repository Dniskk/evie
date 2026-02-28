---
name: pr
description: Create a pull request for evie following project conventions
argument-hint: [issue-number] [title]
disable-model-invocation: true
---

# Create Pull Request

Create a pull request following evie's conventions.

## Usage

```bash
/pr <issue-number> <title>
```

## Example

```bash
/pr 5 implement transformer architecture
```

This creates a PR titled `feat(5): implement transformer architecture` that closes issue #5 and assigns Dniskk as reviewer.

## Steps

1. Verify you're on a feature branch (not main)
2. Push branch to remote if not already pushed
3. Create PR with:
   - Title: `feat($ARGUMENTS[0]): $ARGUMENTS[1+]`
   - Body: `Closes #$ARGUMENTS[0]`
   - Reviewer: Dniskk
