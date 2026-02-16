#!/bin/bash
# AURALIS Quality Check — adapted from MeetMind pipeline
# Run all quality gates before commit

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log_pass() { echo -e "${GREEN}✅ PASS${NC} — $1"; }
log_fail() { echo -e "${RED}❌ FAIL${NC} — $1"; FAILED=1; }
log_info() { echo -e "${BLUE}🔍${NC} $1"; }
log_section() { echo -e "\n${YELLOW}━━━ $1 ━━━${NC}"; }

cd "$PROJECT_DIR"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════╗"
echo "║   🎛️  AURALIS Quality Check          ║"
echo "╚══════════════════════════════════════╝"
echo -e "${NC}"

# ─── SEC-001: Secret scanning ───
log_section "SEC-001: Secret Scanning"
if command -v gitleaks &>/dev/null; then
    if gitleaks detect --source . --config .gitleaks.toml --no-banner 2>/dev/null; then
        log_pass "No secrets found"
    else
        log_fail "Secrets detected! Fix before committing."
    fi
else
    log_info "gitleaks not installed — skipping (install: brew install gitleaks)"
fi

# ─── LINT: Ruff check ───
log_section "LINT: Ruff Check"
if uv run ruff check auralis/ tests/ 2>/dev/null; then
    log_pass "No lint errors"
else
    log_fail "Lint errors found. Run: uv run ruff check --fix"
fi

# ─── FORMAT: Ruff format ───
log_section "FORMAT: Code Formatting"
if uv run ruff format --check auralis/ tests/ 2>/dev/null; then
    log_pass "Code properly formatted"
else
    log_fail "Formatting issues. Run: uv run ruff format"
fi

# ─── TYPES: MyPy strict ───
log_section "TYPES: MyPy Strict"
if uv run mypy auralis/ 2>/dev/null; then
    log_pass "No type errors"
else
    log_fail "Type errors found. Fix type annotations."
fi

# ─── TESTS: Pytest with coverage ───
log_section "TESTS: Pytest + Coverage"
if uv run pytest --cov=auralis --cov-report=term-missing --tb=short -q 2>/dev/null; then
    log_pass "All tests passed"
else
    log_fail "Tests failed"
fi

# ─── CODE-001: File size check ───
log_section "CODE-001: File Size Check"
OVERSIZED=0
while IFS= read -r -d '' file; do
    lines=$(wc -l < "$file" | tr -d ' ')
    if [ "$lines" -gt 800 ]; then
        echo -e "  ${RED}⚠️  $file: $lines lines (max 800)${NC}"
        OVERSIZED=1
    elif [ "$lines" -gt 500 ]; then
        echo -e "  ${YELLOW}⚠️  $file: $lines lines (soft limit 500)${NC}"
    fi
done < <(find auralis/ -name "*.py" -print0)

if [ "$OVERSIZED" -eq 0 ]; then
    log_pass "All files within size limits"
else
    log_fail "Files exceed 800-line hard limit"
fi

# ─── Summary ───
echo ""
echo -e "${YELLOW}━━━ Summary ━━━${NC}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}🎉 All quality gates passed!${NC}"
    exit 0
else
    echo -e "${RED}💥 Some quality gates failed. Fix before committing.${NC}"
    exit 1
fi
