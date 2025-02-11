# Comparative Analysis Guide

The Code Quality Analyzer provides comparative analysis features through its FastAPI interface, enabling you to track code quality changes over time and compare different components of your codebase.

## Overview

### Analysis Types
1. Directory Comparison - Compare metrics between different directories
2. Snapshot Comparison - Compare changes between Git references
3. Historical Trend Analysis - Track metrics over time

### Available Metrics
1. Code Complexity
   - Cyclomatic complexity
   - Cognitive complexity
   - Halstead metrics (volume, difficulty, effort, vocabulary, length)
   - Maintainability index
   - Change risk score
   - Maximum nesting depth
2. Security
   - SQL injection vulnerabilities
   - Hardcoded secrets
   - Unsafe regex patterns
   - Vulnerable imports
   - Shell injection detection
   - Temporary file usage
   - Unsafe deserialization patterns
3. Code Quality
   - Magic numbers
   - Large try blocks
   - Boolean traps
   - Long parameter lists
   - TODO counts
   - Duplication metrics
4. Architecture
   - Interface and abstract class count
   - Component coupling
   - Layering violations
   - Circular dependencies

## Using the API

### Directory Comparison

Compare metrics between different directories in your codebase:

```python
import aiohttp
import asyncio

async def compare_directories(repo_url: str, dir1: str, dir2: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': repo_url,
                'options': {
                    'comparison': {
                        'enabled': True,
                        'type': 'directory',
                        'options': {
                            'dir1': dir1,
                            'dir2': dir2
                        },
                        'metrics': ['complexity', 'security', 'architecture']
                    }
                }
            }
        ) as response:
            return await response.json()
```

### Snapshot Comparison

Compare metrics between different Git references:

```python
async def compare_snapshots(repo_url: str, baseline: str, target: str, component_path: str = None):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': repo_url,
                'options': {
                    'comparison': {
                        'enabled': True,
                        'type': 'snapshot',
                        'options': {
                            'baseline': baseline,
                            'target': target,
                            'component': component_path
                        }
                    }
                }
            }
        ) as response:
            return await response.json()
```

### Historical Trend Analysis

Track metrics over time:

```python
async def analyze_trends(repo_url: str, start_date: str, end_date: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': repo_url,
                'options': {
                    'comparison': {
                        'enabled': True,
                        'type': 'historical',
                        'options': {
                            'startDate': start_date,
                            'endDate': end_date,
                            'interval': 'weekly'
                        }
                    }
                }
            }
        ) as response:
            return await response.json()
```

## ML-Enhanced Analysis

The analyzer supports machine learning enhanced analysis for deployment risk assessment:

```python
async def analyze_with_ml(repo_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': repo_url,
                'options': {
                    'mlAnalysis': {
                        'enabled': True,
                        'teamAvailability': {
                            'team1': [['09:00', '17:00']],
                            'team2': [['12:00', '20:00']]
                        },
                        'analyzeDeploymentWindows': True,
                        'analyzeResourceRequirements': True,
                        'analyzeRollbackRisks': True,
                        'analyzeIncidentPrediction': True
                    }
                }
            }
        ) as response:
            return await response.json()
```

## Additional Features

### PDF Report Generation

Enable PDF report generation in your analysis request:

```python
'options': {
    'generatePdf': True,
    'comparison': {
        'enabled': True,
        # ... other comparison options
    }
}
```

### JSON Export

Export detailed metrics in JSON format:

```python
'options': {
    'exportJson': True,
    'comparison': {
        'enabled': True,
        # ... other comparison options
    }
}
```

## Analysis Report Structure

The analysis produces a comprehensive report including:

### 1. Basic Metrics
```json
{
    "files": {
        "file_path": {
            "lines_code": 100,
            "lines_comment": 20,
            "lines_blank": 10,
            "complexity": {
                "cyclomatic_complexity": 15,
                "cognitive_complexity": 12,
                "maintainability_index": 75,
                "change_risk": 45
            },
            "security": {
                "potential_sql_injections": 0,
                "hardcoded_secrets": 1,
                "unsafe_regex": 0,
                "vulnerable_imports": []
            },
            "architecture": {
                "interface_count": 2,
                "abstract_class_count": 1,
                "layering_violations": 0,
                "component_coupling": 0.4
            }
        }
    }
}
```

### 2. Comparison Results
```json
{
    "comparison": {
        "differences": {
            "complexity": {
                "cyclomatic": -7,
                "cognitive": -5,
                "maintainability": 3
            },
            "security": {
                "vulnerabilities": -2,
                "severity_score": -0.5
            }
        }
    }
}
```

## Error Handling

The API returns detailed error information in case of failures:

```json
{
    "detail": "Error message describing what went wrong"
}
```

Common error scenarios:
1. Invalid repository URL
2. Invalid Git references
3. Analysis timeout
4. Permission issues
5. Invalid comparison options

## Best Practices

1. Analysis Scope
   - Focus on specific components or directories for faster results
   - Use appropriate comparison types based on your needs
   - Consider using ML analysis for critical deployments

2. Performance Optimization
   - Set appropriate minimum line thresholds for duplication detection
   - Use component-specific analysis when possible
   - Enable verbose logging only when needed
   - Configure ignored directories (default: node_modules, venv, .git, __pycache__, build, dist, target, bin, obj)

3. File Analysis
   - Configure supported code extensions (default: .py, .js, .jsx, .ts, .tsx, .java, .cpp, .c, .h, .hpp, .cs, .rb, .php, .go, .rs, .swift)
   - Use binary file detection to skip non-text files
   - Handle file encoding issues with UTF-8 and error ignore options

4. Change Analysis
   - Utilize Git history for change probability analysis
   - Track contributor metrics
   - Monitor churn rates
   - Consider recency of changes in risk assessment

5. Report Generation
   - Enable PDF generation for comprehensive reports
   - Use JSON export for programmatic analysis
   - Save reports for historical tracking
   - Leverage logging for detailed analysis tracking