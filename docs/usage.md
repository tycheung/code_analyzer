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
# Basic analysis
code_analyzer https://github.com/username/repo

# Generate PDF report
code_analyzer https://github.com/username/repo --generate-pdf

# Custom output directory and export JSON
code_analyzer https://github.com/username/repo \
    --output-dir ./analysis \
    --export-json \
    --generate-pdf \
    --pdf-output ./reports

# Adjust duplicate detection
code_analyzer https://github.com/username/repo \
    --min-duplicate-lines 10
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
                    'exportJson': True
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
from code_analyzer.reporters import PDFReportGenerator

async def analyze_repo():
    # Initialize analyzer
    analyzer = CodebaseAnalyzer()
    
    # Analyze repository
    repo_dir = await analyzer.clone_repo("https://github.com/username/repo", "./repo_clone")
    await analyzer.analyze_git_history(repo_dir)
    await analyzer.scan_directory(repo_dir)
    
    # Generate report
    report_generator = PDFReportGenerator()
    await report_generator.generate_pdf(
        analyzer.stats,
        "https://github.com/username/repo",
        "./report"
    )

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
        print(f"Security issues: {metrics.security.potential_sql_injections}")
    
    # Get overall statistics
    overall_metrics = await analyzer.get_metrics()
    print(f"Total files: {overall_metrics['summary']['total_files']}")
    print(f"Total lines: {overall_metrics['summary']['total_lines']}")
```

## Generated Reports

### PDF Report Structure
1. Executive Summary
   - Key metrics
   - Important findings
   - Risk assessment
2. Detailed Analysis
   - Code complexity
   - Security vulnerabilities
   - Architecture issues
   - Code duplication
   - Change probability
3. Recommendations
   - Security improvements
   - Code quality suggestions
   - Architecture recommendations

### JSON Export Format
```json
{
    "files": {
        "file_path": {
            "lines_code": 100,
            "lines_comment": 20,
            "complexity": {
                "cyclomatic_complexity": 5,
                "maintainability_index": 75
            },
            "security": {
                "potential_sql_injections": 0,
                "hardcoded_secrets": 1
            }
        }
    },
    "summary": {
        "total_files": 10,
        "total_lines": 1000,
        "security_issues": 2
    }
}
```

## Best Practices

### 1. Regular Analysis
- Run analysis before major releases
- Monitor trends over time
- Set quality thresholds
- Integrate with CI/CD pipelines

### 2. API Usage
- Use async/await for better performance
- Handle errors appropriately
- Implement proper request validation
- Monitor server resources

### 3. Interpreting Results
- Focus on high-risk areas
- Prioritize security issues
- Consider change probability
- Track metrics over time

## Troubleshooting

### Common Issues

1. Analysis takes too long
   - Use smaller output directories
   - Increase min-duplicate-lines
   - Skip certain directories
   - Consider using async operations

2. PDF generation fails
   - Verify LaTeX installation
   - Check template customization
   - Ensure write permissions
   - Check async operation completion

3. Memory issues
   - Analyze smaller portions
   - Increase available memory
   - Use cleanup options
   - Monitor server resources

4. API-specific issues
   - Check endpoint availability
   - Verify request format
   - Handle connection timeouts
   - Implement proper error handling