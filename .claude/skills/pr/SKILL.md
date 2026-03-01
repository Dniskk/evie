---
name: pr
description: Create a pull request following evie's conventions
disable-model-invocation: true
---

# Creating a Pull Request

When creating a pull request for evie, follow these steps:

## Branch Requirements

- Branch should follow the naming convention: `<your-name>/<issue-number>-<short-description>`
- Examples: `alice/5-implement-transformer`, `bob/12-fix-attention-mask`

## PR Title

Use conventional commits format for the title:

```
feat(<issue-number>): <description>
```

Examples:
- `feat(5): implement multi-head attention`
- `fix(12): correct gradient accumulation`
- `docs(6): add dataset preprocessing guide`

## PR Body

Include:
- What changes were made and why
- Reference the issue: `Closes #<issue-number>`

## Requirements Before Merge

Your PR must satisfy:
- ✅ All tests pass
- ✅ Code is formatted (`ruff format`)
- ✅ No linting errors (`ruff check`)
- ✅ Type checking passes (`basedpyright`)
- ✅ New functionality includes tests
- ✅ Documentation is updated
- ✅ Branch name follows convention
- ✅ PR has been reviewed and approved

## Assigning Reviewers

Assign `Dniskk` as the reviewer.
