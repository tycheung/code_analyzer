# CI/CD Integration Guide

This guide explains how to integrate the Code Quality Analyzer into your CI/CD pipelines for automated code quality checks.

## Overview

The Code Quality Analyzer can be integrated into your CI/CD workflow to:
- Analyze code quality on pull requests
- Generate reports for each build
- Track metrics over time
- Enforce quality standards
- Automate deployment analysis

## GitHub Actions

### Basic Integration

```yaml
name: Code Quality Check

on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      
      - name: Install dependencies
        run: poetry install --extras "all"
      
      - name: Run analysis
        run: |
          poetry run code_analyzer ${{ github.repository }} \
            --generate-pdf \
            --export-json \
            --include-test-analysis \
            --include-trends
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: analysis-reports
          path: |
            ./reports/*.pdf
            ./metrics.json
```

### Advanced Configuration

```yaml
name: Advanced Code Analysis

on:
  pull_request:
    types: [opened, synchronize]
  push:
    branches: [main]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for trends
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pypoetry
          key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
      
      - name: Install dependencies
        run: poetry install --extras "all"
      
      - name: Run comparative analysis
        if: github.event_name == 'pull_request'
        run: |
          poetry run code_analyzer ${{ github.repository }} \
            --generate-pdf \
            --export-json \
            --include-test-analysis \
            --include-trends \
            --trend-window 3months \
            --trend-granularity weekly
      
      - name: Check quality gates
        run: poetry run python .github/scripts/check_quality.py
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: analysis-reports
          path: |
            ./reports/*.pdf
            ./metrics.json
```

### Quality Gate Script Example

```python
# .github/scripts/check_quality.py
import json
import sys

def check_quality_gates():
    with open('metrics.json') as f:
        metrics = json.load(f)
    
    # Define quality gates
    gates = {
        'max_complexity': 25,
        'min_test_coverage': 0.7,
        'max_security_issues': 0,
        'max_architecture_issues': 5
    }
    
    # Check metrics
    failures = []
    summary = metrics['summary']
    
    if summary['avg_complexity'] > gates['max_complexity']:
        failures.append(f"Average complexity {summary['avg_complexity']} exceeds maximum {gates['max_complexity']}")
    
    if summary['test_coverage'] < gates['min_test_coverage']:
        failures.append(f"Test coverage {summary['test_coverage']} below minimum {gates['min_test_coverage']}")
    
    if summary['security_issues'] > gates['max_security_issues']:
        failures.append(f"Found {summary['security_issues']} security issues")
    
    if summary['architecture_issues'] > gates['max_architecture_issues']:
        failures.append(f"Found {summary['architecture_issues']} architecture issues")
    
    # Report results
    if failures:
        print("Quality gate failures:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    
    print("All quality gates passed!")
    sys.exit(0)

if __name__ == '__main__':
    check_quality_gates()
```

## GitLab CI

### Basic Configuration

```yaml
image: python:3.9

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache/
    - .venv/

before_script:
  - python -V
  - pip install poetry
  - poetry config virtualenvs.in-project true
  - poetry install --extras "all"

analyze:
  script:
    - poetry run code_analyzer $CI_PROJECT_URL \
        --generate-pdf \
        --export-json \
        --include-test-analysis
  artifacts:
    paths:
      - reports/
      - metrics.json
    reports:
      junit: reports/junit.xml
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

## Azure DevOps

### Pipeline Configuration

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'
    addToPath: true

- script: |
    curl -sSL https://install.python-poetry.org | python3 -
    poetry install --extras "all"
  displayName: 'Install dependencies'

- script: |
    poetry run code_analyzer $(Build.Repository.Uri) \
      --generate-pdf \
      --export-json \
      --include-test-analysis
  displayName: 'Run analysis'

- task: PublishBuildArtifacts@1
  inputs:
    pathToPublish: 'reports'
    artifactName: 'analysis-reports'
```

## Jenkins Pipeline

### Jenkinsfile Example

```groovy
pipeline {
    agent any
    
    environment {
        PATH = "$PATH:/usr/local/bin"
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m pip install --user poetry
                    poetry install --extras "all"
                '''
            }
        }
        
        stage('Analyze') {
            steps {
                sh '''
                    poetry run code_analyzer ${GIT_URL} \
                        --generate-pdf \
                        --export-json \
                        --include-test-analysis
                '''
            }
        }
        
        stage('Process Results') {
            steps {
                archiveArtifacts artifacts: 'reports/**, metrics.json'
            }
        }
    }
}
```

## Best Practices

1. Pipeline Configuration
   - Cache dependencies
   - Use specific Python version
   - Archive reports as artifacts
   - Set appropriate triggers

2. Quality Gates
   - Define clear thresholds
   - Include critical metrics
   - Document exceptions
   - Update regularly

3. Resource Usage
   - Configure appropriate timeouts
   - Manage artifact retention
   - Clean up temporary files
   - Monitor pipeline duration

4. Reporting
   - Archive PDF reports
   - Export metrics as JSON
   - Generate trend analysis
   - Track historical data

## Troubleshooting

### Common Issues

1. Pipeline Timeouts
   - Increase timeout limits
   - Optimize analysis scope
   - Cache dependencies
   - Use incremental analysis

2. Resource Constraints
   - Adjust memory limits
   - Optimize artifact storage
   - Clean workspace regularly
   - Use efficient triggers

3. Report Generation
   - Verify file permissions
   - Check output paths
   - Ensure sufficient space
   - Handle path separators

### Solutions

1. For Timeouts
   - Configure step timeouts
   - Split analysis steps
   - Optimize configuration
   - Use selective analysis

2. For Resources
   - Set resource limits
   - Clean old artifacts
   - Use efficient caching
   - Optimize triggers

3. For Reports
   - Create output directories
   - Set correct permissions
   - Handle path formats
   - Manage file cleanup