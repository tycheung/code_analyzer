# Code Analysis Metrics Documentation

This document provides a comprehensive breakdown of all metrics analyzed by the system.

## Table of Contents
- [Complexity Metrics](#complexity-metrics)
- [Code Quality Metrics](#code-quality-metrics)
- [Architectural Metrics](#architectural-metrics)
- [Security Metrics](#security-metrics)
- [Change Probability Analysis](#change-probability-analysis)

## Complexity Metrics

### Cyclomatic Complexity (McCabe)
Measures code complexity by counting independent paths through code.

**Elements counted**:
- if/else statements
- elif statements
- for/while loops
- try/catch/except blocks
- and/or operators
- function definitions
- class definitions
- eval/exec usage
- conditional returns
- list comprehensions and generator expressions

**Base calculation**: 
Starts at 1 and adds 1 for each control flow element

### Cognitive Complexity
Measures how difficult code is to understand.

**Factors Analyzed**:
- Nesting depth (measured in indentation levels of 4 spaces)
- Recursion (method calling itself)
- Boolean operators (and/or/not)
- Exception handling (catch/except)
- Control flow breaks (break/continue/return/raise)

**Scoring**:
- Base points for each pattern match
- Additional points based on nesting level
- Higher weights for complex patterns like recursion

### Halstead Metrics
**Components**:
- Operators: Mathematical, logical, and control flow operators
- Operands: Variables, constants, and string literals

**Calculated Metrics**:
- Program Length (N) = Total operators + Total operands
- Vocabulary Size (n) = Unique operators + Unique operands
- Volume (V) = N × log2(n)
- Difficulty (D) = (Unique operators/2) × (Total operands/Unique operands)
- Effort (E) = D × V

### Maintainability Index
**Formula**: 171 - 5.2ln(Halstead Volume) - 0.23(Cyclomatic Complexity) - 16.2ln(LOC) + 50sin(√(2.4 × Comment Ratio))

**Range**: 0-100 (higher is better)

## Code Quality Metrics

### Code Patterns
**Detected Issues**:
- Magic numbers: Non-zero numeric literals not followed by .0 (`\b\d+\b(?!\.0*\b)`)
- Large try blocks: Try blocks exceeding 30 lines (`try:(?:[^}]*){30,}catch`)
- Boolean traps: Boolean parameters in function signatures (`\w+\s*\([^)]*bool\s+\w+[^)]*\)`)
- Long parameter lists: Parameter lists exceeding 120 characters (`\([^)]{120,}\)`)
- TODO counts: Explicit TODO markers in comments

**Implementation Details**:
Each pattern is implemented using regular expressions for consistent detection across the codebase. The patterns are designed to catch common code quality issues while minimizing false positives.

### Line Metrics
- Lines of code
- Comment lines
- Blank lines
- File size

### Code Duplication
**Analysis**:
- Minimum lines threshold for duplication
- Block-based comparison
- Cross-file detection

## Architectural Metrics

### Interface Analysis
**Patterns Detected**:
- Interface definitions (interface/protocol keywords)
- Abstract class declarations
- Abstract method decorators

### Component Coupling
**Measurements**:
- Import analysis
- External vs internal dependency ratio
- Module-level coupling

### Layering
**Violations Detected**:
- Deep import patterns (../../)
- Skip-level access
- Layer boundary violations

### Circular Dependencies
**Analysis**:
- Module import cycles
- Package-level circular dependencies
- Import chain analysis

## Security Metrics

### Vulnerability Detection
**Patterns**:
- SQL Injection:
  - Execute calls with string concatenation
  - Raw input usage in queries
  - F-string SELECT statements with WHERE clauses
  - Pattern: `execute\s*\(.*?\+.*?\)|raw_input\s*\(.*?\)|f["\']SELECT.*?WHERE.*?\{.*?\}["\']`

- Command Injection:
  - os.system() calls
  - subprocess.call/Popen usage
  - Pattern: `os\.system\(|subprocess\.call\(`

- Unsafe Evaluation:
  - eval() usage
  - exec() usage
  - Pattern: `eval\(.*\)|exec\(.*\)`

- Unsafe Deserialization:
  - pickle.loads/load calls
  - yaml.load usage without safe loader
  - Pattern: `pickle\.loads?\(|yaml\.load\(`

### Secret Detection
**Patterns**:
- Hardcoded passwords: Direct password assignments (`password\s*=\s*[\'"][^\'"]+[\'""]`)
- API keys: Hardcoded API key assignments (`api_key\s*=\s*[\'"][^\'"]+[\'""]`)
- Direct credential assignments in code
- Pattern matches both quoted strings containing sensitive data

### Unsafe Patterns
**Detection**:
- Vulnerable imports (telnetlib, pickle, marshal, subprocess)
- Temporary file usage
- Unsafe regex patterns
- Shell command execution

## Change Probability Analysis

### Git History Analysis
**Metrics**:
- Change frequency
- Last modified timestamp
- Contributor count
- Historical changes

### Churn Calculation
**Factors**:
- Lines added/removed over time
- Change patterns
- Modification frequency

### Risk Assessment
**Formula Components**:
- 30% Change frequency (normalized to 0-1)
- 30% Recency factor (exponential decay over 1 year)
- 20% Contributor count (normalized by 10)
- 20% Churn rate (normalized by 1000)

**Output**: Risk score from 0-100

---

The system combines these metrics to provide a comprehensive analysis of code quality, security, and maintainability. Each metric is calculated based on static analysis of the codebase and its Git history, providing actionable insights for improvement.