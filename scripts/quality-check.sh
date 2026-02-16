#!/usr/bin/env bash
# ============================================================================
# AURALIS Quality Gate — Development Standards
# ============================================================================
# Usage: ./scripts/quality-check.sh [--fix]
#
# Verifies ALL development standards:
#   SEC-001: No hardcoded secrets (gitleaks)
#   SEC-002: No sensitive defaults in code
#   CODE-001: File size limits (≤500 soft, ≤800 hard)
#   TEST-001: Coverage ≥70%
#   DOC-001: Type hints & linting (mypy --strict)
#   LINT: 0 errors (ruff)
#   FORMAT: 100% formatted (ruff format)
# ============================================================================

set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

PASSED=0
FAILED=0
WARNINGS=0

pass() { ((PASSED++)); echo -e "  ${GREEN}✅ PASS${NC} — $1"; }
fail() { ((FAILED++)); echo -e "  ${RED}❌ FAIL${NC} — $1"; }
warn() { ((WARNINGS++)); echo -e "  ${YELLOW}⚠️  WARN${NC} — $1"; }
info() { echo -e "  ${BLUE}ℹ️  INFO${NC} — $1"; }

# ============================================================================
# SECTION 1: SECURITY CHECKS
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🔒 SECURITY CHECKS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# SEC-001: Secrets scan
if command -v gitleaks &> /dev/null; then
    GITLEAKS_CONFIG=""
    if [[ -f "$ROOT_DIR/.gitleaks.toml" ]]; then
        GITLEAKS_CONFIG="--config=$ROOT_DIR/.gitleaks.toml"
    fi
    if gitleaks detect --source="$ROOT_DIR" $GITLEAKS_CONFIG --no-banner --no-color 2>/dev/null; then
        pass "SEC-001: No secrets detected (gitleaks)"
    else
        fail "SEC-001: Secrets found in code! Run: gitleaks detect --verbose"
    fi
else
    warn "SEC-001: gitleaks not installed — skipping secrets scan"
    info "Install with: brew install gitleaks"
fi

# SEC-002: No hardcoded AWS credentials
echo ""
HARDCODED=$(grep -rn "AKIA\|aws_secret_access_key\|sk-[a-zA-Z0-9]\{20,\}" \
    "$ROOT_DIR/auralis/" 2>/dev/null || true)
if [[ -z "$HARDCODED" ]]; then
    pass "SEC-002: No hardcoded AWS/API keys in source"
else
    fail "SEC-002: Hardcoded credentials found:"
    echo "$HARDCODED"
fi

# SEC-003: .env not committed
if git -C "$ROOT_DIR" ls-files --cached | grep -q "\.env$"; then
    fail "SEC-003: .env file is tracked by git!"
else
    pass "SEC-003: .env not tracked by git"
fi

# ============================================================================
# SECTION 2: PYTHON QUALITY
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🐍 PYTHON QUALITY${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$ROOT_DIR"

# LINT: ruff check
echo ""
if $FIX_MODE; then
    info "Running ruff with --fix..."
    uv run ruff check auralis/ tests/ --fix 2>/dev/null || true
fi
RUFF_OUTPUT=$(uv run ruff check auralis/ tests/ 2>&1 || true)
if echo "$RUFF_OUTPUT" | grep -q "All checks passed\|^$"; then
    pass "LINT: ruff — 0 errors"
else
    RUFF_COUNT=$(echo "$RUFF_OUTPUT" | grep -oE 'Found [0-9]+ error' | head -1)
    fail "LINT: ruff — ${RUFF_COUNT:-errors found}"
    echo "$RUFF_OUTPUT" | head -20
fi

# FORMAT: ruff format
echo ""
if $FIX_MODE; then
    info "Running ruff format..."
    uv run ruff format auralis/ tests/ 2>/dev/null || true
fi
FORMAT_OUTPUT=$(uv run ruff format --check auralis/ tests/ 2>&1)
FORMAT_EXIT=$?
if [[ $FORMAT_EXIT -eq 0 ]]; then
    pass "FORMAT: ruff format — 100% formatted"
else
    FORMAT_COUNT=$(echo "$FORMAT_OUTPUT" | grep -c "would be reformatted" 2>/dev/null || echo "?")
    fail "FORMAT: $FORMAT_COUNT file(s) need formatting. Run: uv run ruff format auralis/ tests/"
fi

# TYPE SAFETY: mypy --strict
echo ""
MYPY_OUTPUT=$(uv run mypy auralis/ 2>&1 || true)
MYPY_ERRORS=$(echo "$MYPY_OUTPUT" | grep ": error:" | wc -l | tr -d ' ')
if [[ "$MYPY_ERRORS" -eq 0 ]] || echo "$MYPY_OUTPUT" | grep -q "Success"; then
    pass "TYPES: mypy --strict — 0 errors"
else
    fail "TYPES: mypy --strict — $MYPY_ERRORS error(s)"
    echo "$MYPY_OUTPUT" | grep ": error:" | head -10
fi

# TESTS: pytest + coverage
echo ""
PYTEST_OUTPUT=$(uv run pytest tests/ --cov=auralis --cov-report=term-missing --cov-report=html -q 2>&1 || true)
if echo "$PYTEST_OUTPUT" | grep -q "passed"; then
    PASSED_TESTS=$(echo "$PYTEST_OUTPUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "?")
    FAILED_TESTS=$(echo "$PYTEST_OUTPUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")

    if [[ "$FAILED_TESTS" -gt 0 ]]; then
        fail "TESTS: $FAILED_TESTS tests failed"
    else
        pass "TESTS: $PASSED_TESTS tests passed"
    fi

    COVERAGE=$(echo "$PYTEST_OUTPUT" | grep "TOTAL" | awk '{print $NF}' | tr -d '%')
    if [[ -n "$COVERAGE" ]]; then
        if [[ "$COVERAGE" -ge 70 ]]; then
            pass "COVERAGE: ${COVERAGE}% (≥70% required)"
        else
            fail "COVERAGE: ${COVERAGE}% — below 70% minimum!"
        fi
    else
        warn "COVERAGE: Could not determine coverage percentage"
    fi
else
    fail "TESTS: pytest failed to run"
    echo "$PYTEST_OUTPUT" | tail -10
fi

# ============================================================================
# SECTION 3: CODE QUALITY
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  📏 CODE QUALITY${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$ROOT_DIR"

# CODE-001: File size
echo ""
BIG_FILES_HARD=0
BIG_FILES_SOFT=0
while IFS= read -r file; do
    lines=$(wc -l < "$file" | tr -d ' ')
    if [[ "$lines" -gt 800 ]]; then
        fail "CODE-001: $file — $lines lines (HARD limit: 800)"
        ((BIG_FILES_HARD++))
    elif [[ "$lines" -gt 500 ]]; then
        warn "CODE-001: $file — $lines lines (SOFT limit: 500)"
        ((BIG_FILES_SOFT++))
    fi
done < <(find "$ROOT_DIR/auralis" -name "*.py" -type f 2>/dev/null)

if [[ "$BIG_FILES_HARD" -eq 0 && "$BIG_FILES_SOFT" -eq 0 ]]; then
    pass "CODE-001: All Python files ≤500 lines"
elif [[ "$BIG_FILES_HARD" -eq 0 ]]; then
    info "CODE-001: $BIG_FILES_SOFT files above soft limit (500) but within hard limit (800)"
fi

# ANTI-PATTERN: No bare except
echo ""
BARE_EXCEPT=$(grep -rn "except:" "$ROOT_DIR/auralis/" 2>/dev/null | grep -v "except:  #" || true)
if [[ -z "$BARE_EXCEPT" ]]; then
    pass "ANTI-PATTERN: No bare 'except:' clauses"
else
    fail "ANTI-PATTERN: Bare 'except:' found — use specific exceptions"
    echo "$BARE_EXCEPT"
fi

# ANTI-PATTERN: No print()
PRINTS=$(grep -rn "^[[:space:]]*print(" "$ROOT_DIR/auralis/" 2>/dev/null || true)
if [[ -z "$PRINTS" ]]; then
    pass "ANTI-PATTERN: No print() in source — use structlog"
else
    fail "ANTI-PATTERN: print() found — use structlog instead"
    echo "$PRINTS"
fi

# ============================================================================
# SECTION 4: PROJECT STRUCTURE
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  📂 PROJECT STRUCTURE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

REQUIRED_DIRS=(
    "auralis/api"
    "auralis/api/routes"
    "auralis/ear"
    "auralis/console"
    "tests"
    "web"
    "scripts"
)

ALL_DIRS_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        fail "STRUCTURE: Missing directory — $dir"
        ALL_DIRS_OK=false
    fi
done
if $ALL_DIRS_OK; then
    pass "STRUCTURE: All required directories present"
fi

REQUIRED_FILES=(
    ".gitignore"
    "README.md"
    "pyproject.toml"
    ".env.example"
)

ALL_FILES_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        fail "STRUCTURE: Missing file — $f"
        ALL_FILES_OK=false
    fi
done
if $ALL_FILES_OK; then
    pass "STRUCTURE: All required project files present"
fi

# ============================================================================
# RESULTS
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  📊 QUALITY GATE RESULTS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}✅ Passed:${NC}  $PASSED"
echo -e "  ${RED}❌ Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}⚠️  Warns:${NC}   $WARNINGS"
echo ""

if [[ "$FAILED" -eq 0 ]]; then
    echo -e "  ${GREEN}🏆 ALL QUALITY GATES PASSED — Ready to commit!${NC}"
    echo ""
    exit 0
else
    echo -e "  ${RED}🚫 $FAILED GATE(S) FAILED — Fix before committing!${NC}"
    echo -e "  ${YELLOW}💡 Run with --fix to auto-fix formatting: ./scripts/quality-check.sh --fix${NC}"
    echo ""
    exit 1
fi
