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

**Calculation**: Number of decision points + 1

**Elements counted**:
- if/else statements
- for/while loops
- case statements
- catch blocks
- boolean operators (and/or)

**Thresholds**:
- 1-5: Low complexity
- 6-10: Moderate complexity
- 11-20: High complexity
- >20: Very high complexity (should be refactored)

### Cognitive Complexity
Measures how difficult code is to understand.

**Factors**:
- Nesting depth (weighted more heavily)
- Control flow breaks
- Recursion
- Multiple conditions

**Calculation**: Base points + nesting penalties

**Scoring**:
- +1 for each control flow statement
- +1 per level of nesting
- +2 for interrupting control flow (break/continue)
- +2 for recursion

### Halstead Metrics
**Components**:
- n1 = unique operators
- n2 = unique operands
- N1 = total operators
- N2 = total operands

**Calculated Metrics**:
- Program Length (N) = N1 + N2
- Vocabulary Size (n) = n1 + n2
- Volume (V) = N × log2(n)
- Difficulty (D) = (n1/2) × (N2/n2)
- Effort (E) = D × V

## Code Quality Metrics

### Lines of Code Metrics
**Raw Counts**:
- Physical lines
- Logical lines
- Comment lines
- Blank lines

**Derived Metrics**:
- Comment density (comments/code ratio)
- Average line length
- Maximum line length

### Maintainability Index
**Formula**: 171 - 5.2ln(HV) - 0.23CC - 16.2ln(LOC) + 50sin(√(2.4CR))
- HV = Halstead Volume
- CC = Cyclomatic Complexity
- LOC = Lines of Code
- CR = Comment Ratio

**Scale**: 0-100

**Interpretation**:
- >85: Highly maintainable
- 65-85: Moderately maintainable
- <65: Difficult to maintain

### Code Duplication
**Detection Methods**:
- Token-based comparison
- Line-by-line matching
- AST comparison

**Metrics**:
- Duplicate blocks count
- Lines of duplicated code
- Duplication percentage
- Cross-file duplication

## Architectural Metrics

### Component Coupling
**Types Measured**:
- Afferent coupling (incoming dependencies)
- Efferent coupling (outgoing dependencies)
- Instability (ratio of efferent coupling)

**Scope**:
- Class-level coupling
- Package-level coupling
- Module-level coupling

### Abstraction Metrics
**Interface Analysis**:
- Interface count
- Abstract class count
- Implementation ratio

**Abstraction Level**:
- Abstract components ratio
- Interface adherence
- Dependency inversion compliance

### Layering Violations
**Detection**:
- Skip-level access
- Circular dependencies
- Layer isolation breaches

**Metrics**:
- Violation count
- Violation severity
- Affected components

## Security Metrics

### Vulnerability Detection
**Types**:
- SQL Injection patterns
- Command injection risks
- Path traversal vulnerabilities
- Unsafe deserialization

**Metrics**:
- Vulnerability count by type
- Severity ratings
- Location tracking

### Secret Detection
**Patterns**:
- API keys
- Passwords
- Authentication tokens
- Private keys

**Analysis**:
- Pattern matching
- Entropy analysis
- Known format detection

### Code Pattern Analysis
**Unsafe Patterns**:
- eval() usage
- shell execution
- unsafe regex
- temporary file usage

**Security Features**:
- Input validation presence
- Output encoding usage
- Security header implementation

## Change Probability Analysis

### Git History Metrics
**Temporal Analysis**:
- Change frequency
- Last modified date
- Change patterns

**Contributor Analysis**:
- Number of contributors
- Contribution distribution
- Author expertise level

### Churn Metrics
**Code Churn**:
- Lines added/removed
- Churn rate over time
- Churn by component

**Stability Metrics**:
- Change coupling
- Change impact
- Refactoring frequency

### Risk Assessment
**Factors**:
- Complexity trends
- Bug frequency
- Test coverage
- Documentation completeness

**Composite Metrics**:
- Risk score calculation
- Stability prediction
- Maintenance forecast

---

Each metric contributes to a comprehensive understanding of code quality and helps identify specific areas for improvement. The system combines these metrics to provide actionable insights and recommendations for codebase improvement.