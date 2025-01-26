# Comparative Analysis Guide

The Code Quality Analyzer provides robust comparative analysis features through its FastAPI interface and CLI, enabling you to track code quality changes over time and compare different components of your codebase.

## Overview

### Analysis Types
1. Directory Comparison - Compare different parts of your codebase
2. Before/After Analysis - Compare changes between Git references
3. Historical Trend Analysis - Track metrics over time

### Available Metrics
1. Code Complexity
   - Cyclomatic complexity
   - Cognitive complexity
   - Halstead metrics
   - Maintainability index
2. Security
   - Vulnerability count
   - Severity scores
   - Security debt
3. Code Quality
   - Duplication metrics
   - Test coverage
   - Documentation coverage
4. Architecture
   - Component coupling
   - Dependency metrics
   - Layer violations

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
                        'metrics': ['complexity', 'security', 'maintainability']
                    }
                }
            }
        ) as response:
            return await response.json()

# Usage example
result = await compare_directories(
    'https://github.com/org/repo',
    'src/old-module',
    'src/new-module'
)
```

### Before/After Analysis

Compare metrics between different Git references:

```python
async def compare_versions(repo_url: str, baseline: str, target: str):
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
                            'component': 'src/core'  # Optional
                        }
                    }
                }
            }
        ) as response:
            return await response.json()

# Compare main branch with previous version
result = await compare_versions(
    'https://github.com/org/repo',
    'main~1',
    'main'
)
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

# Analyze trends over the last quarter
result = await analyze_trends(
    'https://github.com/org/repo',
    '2024-01-01',
    '2024-03-31'
)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Code Quality Check

on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install aiohttp asyncio
          
      - name: Run analysis
        run: |
          python -c "
          import asyncio
          import aiohttp
          import os
          
          async def analyze():
              async with aiohttp.ClientSession() as session:
                  response = await session.post(
                      'http://localhost:8000/analyze',
                      json={
                          'repoUrl': os.environ['GITHUB_REPOSITORY'],
                          'options': {
                              'comparison': {
                                  'enabled': True,
                                  'type': 'snapshot',
                                  'options': {
                                      'baseline': os.environ['GITHUB_BASE_REF'],
                                      'target': os.environ['GITHUB_HEAD_REF']
                                  }
                              }
                          }
                      }
                  )
                  result = await response.json()
                  # Add quality gates
                  if result['comparison']['differences']['complexity']['cyclomatic'] > 20:
                      exit(1)
          
          asyncio.run(analyze())
          "
```

### Quality Gates

Implement quality gates to enforce standards:

```python
async def check_quality_gates(comparison_result: dict) -> bool:
    thresholds = {
        'complexity': {
            'cyclomatic': 20,
            'cognitive': 15
        },
        'security': {
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 2
        },
        'maintainability': {
            'index_decrease': -5
        }
    }
    
    differences = comparison_result['differences']
    
    # Check complexity thresholds
    if differences['complexity']['cyclomatic'] > thresholds['complexity']['cyclomatic']:
        raise ValueError(f"Cyclomatic complexity increase exceeds threshold")
    
    if differences['complexity']['cognitive'] > thresholds['complexity']['cognitive']:
        raise ValueError(f"Cognitive complexity increase exceeds threshold")
    
    # Check security thresholds
    security_diff = differences['security']
    if security_diff['critical_vulnerabilities'] > thresholds['security']['critical_vulnerabilities']:
        raise ValueError(f"Critical security vulnerabilities introduced")
    
    # Check maintainability
    if differences['maintainability']['index'] < thresholds['maintainability']['index_decrease']:
        raise ValueError(f"Maintainability decreased significantly")
    
    return True
```

## Analysis Report Structure

The analysis produces a comprehensive report including:

### 1. Summary
```json
{
    "summary": {
        "status": "passed",
        "total_changes": 42,
        "critical_changes": 0,
        "improvement_suggestions": 3
    }
}
```

### 2. Detailed Metrics
```json
{
    "metrics": {
        "complexity": {
            "before": {
                "cyclomatic": 245,
                "cognitive": 180
            },
            "after": {
                "cyclomatic": 238,
                "cognitive": 175
            },
            "difference": {
                "cyclomatic": -7,
                "cognitive": -5
            }
        }
    }
}
```

### 3. Recommendations
```json
{
    "recommendations": [
        {
            "type": "refactoring",
            "priority": "high",
            "description": "Consider splitting complex method in UserService",
            "location": "src/services/UserService.ts:126"
        }
    ]
}
```

## Best Practices

### Comparison Strategy
1. Regular Baseline Updates
   - Update baseline references regularly
   - Document baseline selection criteria
   - Keep historical baseline data

2. Granular Analysis
   - Compare specific components
   - Focus on critical paths
   - Track related metrics together

3. Context Awareness
   - Consider development phase
   - Account for team size
   - Factor in technical debt goals

### Performance Optimization
1. Use Async Operations
   - Leverage FastAPI's async capabilities
   - Implement concurrent analysis where possible
   - Handle timeouts appropriately

2. Caching Strategy
   - Cache analysis results
   - Implement incremental analysis
   - Store historical data efficiently

3. Resource Management
   - Clean up temporary files
   - Monitor memory usage
   - Implement rate limiting

## Troubleshooting

### Common Issues

1. Analysis Timeout
   - Increase timeout settings
   - Reduce analysis scope
   - Check repository size
   - Verify network connectivity

2. Memory Issues
   - Implement batch processing
   - Optimize cache usage
   - Monitor resource utilization
   - Use streaming responses

3. Inconsistent Results
   - Verify Git references
   - Check file encodings
   - Validate input parameters
   - Review analysis configuration

### Error Handling

```python
async def handle_analysis_error(error: Exception):
    error_mapping = {
        'GitCommandError': 'Repository access failed',
        'AnalysisTimeout': 'Analysis took too long',
        'InvalidReference': 'Git reference not found'
    }
    
    error_type = error.__class__.__name__
    message = error_mapping.get(error_type, 'Unknown error occurred')
    
    logging.error(f"{message}: {str(error)}")
    return {
        'status': 'error',
        'error_type': error_type,
        'message': message,
        'details': str(error)
    }
```
