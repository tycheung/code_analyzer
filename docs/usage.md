# Usage Guide

This guide covers how to use the Code Quality Analyzer effectively.

## Command Line Interface

### Basic Usage
```bash
code_analyzer <repository_url> [options]
```

### Web Server Usage
```bash
# Start the FastAPI server
code_analyzer-web

# Custom port
code_analyzer-web --port 8080

# Start without opening browser
code_analyzer-web --no-browser
```

### Options
- `--output-dir`: Directory to clone the repository into (default: ./repo_clone)
- `--min-duplicate-lines`: Minimum number of lines to consider as duplicate code (default: 6)
- `--export-json`: Export metrics to JSON file
- `--generate-pdf`: Generate PDF report
- `--pdf-output`: Directory for PDF report output (default: ./report)
- `--verbose`: Enable verbose output

#### New Analysis Options
- `--include-test-analysis`: Include test coverage analysis (default: true)
- `--include-trends`: Include historical trend analysis (default: true)
- `--trend-window`: Time window for trend analysis (choices: 1month, 3months, 6months, 1year)
- `--trend-granularity`: Data granularity for trends (choices: daily, weekly, monthly)

### Web API Endpoints

#### Analysis Endpoint
```bash
POST /analyze
```

Request body:
```json
{
  "repoUrl": "https://github.com/username/repo",
  "options": {
    "verbose": false,
    "generatePdf": true,
    "exportJson": true,
    "minLines": 6,
    "comparison": {
      "enabled": false,
      "type": null,
      "options": {},
      "metrics": []
    },
    "includeTestAnalysis": true,
    "includeTrends": true,
    "mlAnalysis": {
      "enabled": true,
      "teamAvailability": {
        "developer1": [["09:00", "17:00"]],
        "developer2": [["10:00", "18:00"]]
      },
      "analyzeDeploymentWindows": true,
      "analyzeResourceRequirements": true,
      "analyzeRollbackRisks": true,
      "analyzeIncidentPrediction": true,
      "generateHistoricalTrends": true
    }
  }
}
```

#### Download Endpoint
```bash
GET /download/{file_path}
```

### Interactive Documentation
When running the web server, you can access:
- Swagger UI documentation at `/docs`
- ReDoc documentation at `/redoc`

### Examples

#### Using the CLI
```bash
# Basic analysis with test coverage and trends
code_analyzer https://github.com/username/repo

# Generate comprehensive PDF report
code_analyzer https://github.com/username/repo \
    --generate-pdf \
    --include-test-analysis \
    --include-trends \
    --trend-window 6months \
    --trend-granularity weekly

# Custom output directory and export JSON
code_analyzer https://github.com/username/repo \
    --output-dir ./analysis \
    --export-json \
    --generate-pdf \
    --pdf-output ./reports

# Adjust duplicate detection and analysis scope
code_analyzer https://github.com/username/repo \
    --min-duplicate-lines 10 \
    --trend-window 3months \
    --trend-granularity daily
```

#### Using the API
```python
import aiohttp
import asyncio
import json

async def analyze_repo():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': 'https://github.com/username/repo',
                'options': {
                    'generatePdf': True,
                    'exportJson': True,
                    'includeTestAnalysis': True,
                    'includeTrends': True,
                    'mlAnalysis': {
                        'enabled': True,
                        'teamAvailability': {
                            'developer1': [['09:00', '17:00']]
                        },
                        'generateHistoricalTrends': True
                    }
                }
            }
        ) as response:
            result = await response.json()
            print(result)

# Run the analysis
asyncio.run(analyze_repo())
```

#### Using Python API Directly
```python
from code_analyzer.analyzers import CodebaseAnalyzer
from code_analyzer.reporters import CodeMetricsPDFGenerator

async def analyze_repo():
    # Initialize analyzer
    analyzer = CodebaseAnalyzer()
    
    # Analyze repository
    repo_dir = await analyzer.clone_repo("https://github.com/username/repo", "./repo_clone")
    await analyzer.analyze_git_history(repo_dir)
    await analyzer.scan_directory(repo_dir)
    
    # Collect metrics data
    metrics_data = {
        'code_metrics': await analyzer.get_metrics(),
        'test_coverage': await analyzer.get_test_coverage(),
        'trends': await analyzer.get_historical_trends()
    }
    
    # Generate enhanced report
    pdf_generator = CodeMetricsPDFGenerator("./report/analysis.pdf")
    pdf_generator.generate_pdf(metrics_data)

# Run the analysis
asyncio.run(analyze_repo())
```

### Working with Metrics
```python
async def work_with_metrics(analyzer):
    # Access metrics for specific files
    for file_path, metrics in analyzer.stats.items():
        print(f"\nFile: {file_path}")
        print(f"Lines of code: {metrics.lines_code}")
        print(f"Complexity: {metrics.complexity.cyclomatic_complexity}")
        print(f"Cognitive complexity: {metrics.complexity.cognitive_complexity}")
        print(f"Security issues: {metrics.security.potential_sql_injections}")
        print(f"Has tests: {bool(metrics.test_coverage_files)}")
        
        # Access new architecture metrics
        print(f"Component coupling: {metrics.architecture.component_coupling}")
        print(f"Abstraction level: {metrics.architecture.abstraction_level}")
        print(f"Circular dependencies: {len(metrics.architecture.circular_dependencies)}")
    
    # Get overall statistics
    overall_metrics = await analyzer.get_metrics()
    print(f"Total files: {overall_metrics['summary']['total_files']}")
    print(f"Total lines: {overall_metrics['summary']['total_lines']}")
    print(f"Test coverage: {overall_metrics['summary']['test_coverage']*100:.1f}%")
```

## Generated Reports

### PDF Report Structure
1. Executive Summary
   - Key metrics and trends
   - Important findings
   - Risk assessment
   - Test coverage overview
2. Detailed Analysis
   - Code complexity and cognitive complexity
   - Security vulnerabilities
   - Architecture issues
   - Code duplication
   - Change probability
   - Historical trends
3. Test Coverage Analysis
   - Coverage metrics
   - Untested components
   - Test quality assessment
4. Trend Analysis
   - Complexity evolution
   - Code growth patterns
   - Issue trends
   - Quality patterns
5. Recommendations
   - Security improvements
   - Code quality suggestions
   - Architecture recommendations
   - Testing improvements

### JSON Export Format
```json
{
    "files": {
        "file_path": {
            "lines": {
                "code": 100,
                "comment": 20,
                "blank": 5
            },
            "complexity": {
                "cyclomatic": 5,
                "cognitive": 8,
                "maintainability": 75,
                "halstead_metrics": {
                    "volume": 250,
                    "difficulty": 12,
                    "effort": 3000
                }
            },
            "security": {
                "sql_injections": 0,
                "hardcoded_secrets": 1,
                "unsafe_regex": 0,
                "vulnerable_imports": []
            },
            "architecture": {
                "interfaces": 2,
                "abstract_classes": 1,
                "component_coupling": 0.4,
                "abstraction_level": 0.6,
                "circular_dependencies": []
            },
            "test_coverage": {
                "has_tests": true,
                "coverage_files": ["test_file.py"]
            },
            "code_patterns": {
                "magic_numbers": 3,
                "large_methods": 1
            }
        }
    },
    "summary": {
        "total_files": 10,
        "total_lines": 1000,
        "avg_complexity": 4.5,
        "security_issues": 2,
        "architecture_issues": 1,
        "test_coverage": 0.8
    }
}
```

## Best Practices

### 1. Regular Analysis
- Run analysis before major releases
- Monitor trends over time
- Set quality thresholds
- Integrate with CI/CD pipelines
- Track test coverage trends

### 2. API Usage
- Use async/await for better performance
- Handle errors appropriately
- Implement proper request validation
- Monitor server resources
- Cache frequently accessed metrics

### 3. Interpreting Results
- Focus on high-risk areas
- Prioritize security issues
- Consider change probability
- Track metrics over time
- Monitor test coverage trends
- Analyze complexity evolution

### 4. ML-Based Analysis
- Provide accurate team availability data
- Review deployment windows regularly
- Monitor prediction confidence scores
- Validate resource predictions
- Update historical data regularly

## Troubleshooting

### Common Issues

1. Analysis takes too long
   - Use smaller output directories
   - Increase min-duplicate-lines
   - Skip certain directories
   - Consider using async operations
   - Optimize trend analysis window

2. PDF generation fails
   - Verify all required metrics are available
   - Check write permissions
   - Ensure sufficient memory for large reports
   - Validate historical data completeness
   - Check template customization

3. Memory issues
   - Analyze smaller portions
   - Increase available memory
   - Use cleanup options
   - Monitor server resources
   - Optimize trend data storage

4. API-specific issues
   - Check endpoint availability
   - Verify request format
   - Handle connection timeouts
   - Implement proper error handling
   - Validate ML analysis options

5. Test Analysis issues
   - Verify test file detection
   - Check test naming patterns
   - Validate coverage calculation
   - Ensure test file access
   - Review coverage thresholds

6. Trend Analysis issues
   - Verify git history access
   - Check date range validity
   - Validate granularity settings
   - Monitor data point density
   - Handle sparse history gracefully