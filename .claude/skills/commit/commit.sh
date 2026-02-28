#!/bin/bash
# Helper script for conventional commits
set -euo pipefail

TYPE="${1:-}"
ISSUE_NUM="${2:-}"
shift 2 || true
MESSAGE="$*"

if [ -z "$TYPE" ] || [ -z "$ISSUE_NUM" ] || [ -z "$MESSAGE" ]; then
    echo "Usage: commit.sh <type> <issue-number> <message>"
    echo ""
    echo "Types: feat, fix, docs, style, refactor, test, chore"
    exit 1
fi

# Validate type
case "$TYPE" in
    feat|fix|docs|style|refactor|test|chore)
        ;;
    *)
        echo "Error: Invalid type '$TYPE'"
        echo "Valid types: feat, fix, docs, style, refactor, test, chore"
        exit 1
        ;;
esac

# Create commit
COMMIT_MSG="$TYPE($ISSUE_NUM): $MESSAGE"
git commit -m "$COMMIT_MSG"

echo "✅ Committed: $COMMIT_MSG"
