#!/bin/bash
# Helper script for creating PRs
set -euo pipefail

ISSUE_NUM="${1:-}"
shift || true
TITLE="$*"

if [ -z "$ISSUE_NUM" ] || [ -z "$TITLE" ]; then
    echo "Usage: pr.sh <issue-number> <title>"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# Ensure we're on a feature branch
if [ "$CURRENT_BRANCH" = "main" ]; then
    echo "Error: Cannot create PR from main branch"
    exit 1
fi

# Push branch if not already pushed
if ! git rev-parse --verify "origin/$CURRENT_BRANCH" &>/dev/null; then
    echo "📤 Pushing branch to remote..."
    git push -u origin "$CURRENT_BRANCH"
fi

# Create PR
echo "🚀 Creating pull request..."
gh pr create \
    --title "feat($ISSUE_NUM): $TITLE" \
    --body "Closes #$ISSUE_NUM" \
    --reviewer Dniskk

echo "✅ Pull request created!"
