# Code Quality Analyzer

A comprehensive static code analysis tool that helps development teams maintain high code quality and optimize deployment strategies. It analyzes GitHub repositories to provide insights about code complexity, security vulnerabilities, architectural issues, and potential deployment risks. This project is for fun/personal use and in alpha stage, there may be incompatibilities.

## What it Does

The Code Quality Analyzer helps you:
- Identify code quality issues before they become problems
- Make data-driven deployment decisions
- Track code health over time
- Optimize team resources and deployment timing
- Predict and prevent potential deployment issues
- Monitor test coverage and trends

Perfect for:
- Development teams wanting to maintain code quality
- DevOps engineers planning deployments
- Team leads making resource allocation decisions
- Organizations tracking code health metrics
- CI/CD pipeline quality checks

## Key Benefits

- **Comprehensive Analysis**: Get deep insights into code quality, complexity, and potential issues
- **Deployment Optimization**: Use ML-based analysis to choose the best deployment times and resource allocation
- **Historical Tracking**: Monitor how your codebase evolves over time
- **Team Efficiency**: Make informed decisions about team resources and deployment timing
- **Easy Integration**: Works with major CI/CD platforms and development workflows

## Goals and Differentiation

While tools like SonarQube, CodeClimate, and Codacy excel at traditional code quality analysis, this Code Quality Analyzer takes a different approach by focusing on the human aspects of software development. The primary goals are:

### Team-Centric Analysis
- Understand how different team combinations affect deployment success
- Model individual developer expertise with specific components
- Track team performance patterns across different time windows
- Optimize code review assignments based on component familiarity

### Deployment Intelligence
- Predict optimal deployment windows based on team availability and historical success
- Estimate resource requirements for specific changes
- Assess rollback risks based on component changes and team coverage
- Provide confidence levels for deployment recommendations

### Historical Learning
- Build knowledge base of successful deployment patterns
- Track team evolution and component ownership
- Monitor how code changes impact deployment success
- Learn from past incidents and recoveries

### Traditional Quality Metrics
While focusing on team and deployment aspects, this repo still provide essential code quality metrics:
- Complexity analysis
- Security scanning
- Architecture evaluation
- Test coverage tracking

The key difference is that these metrics are analyzed in the context of team dynamics and deployment success, not just as standalone indicators. This tool is designed to assist in decision-making and foster collaboration. It serves as a guiding framework rather than a rigid, step-by-step directive. Users should interpret its output with discernment and avoid over-reliance. Always consider additional context and perspectives when making decisions.

This tool should not be used as a metric for hiring, firing, or making other employment-related decisions.

## Features

- Deep code analysis including:
  - Complexity metrics (cyclomatic, cognitive, Halstead)
  - Security vulnerability detection
  - Architecture analysis
  - Code duplication detection
  - Change probability analysis
  - Test coverage analysis
  - Historical trend analysis
- Git history analysis
- Enhanced PDF report generation
- Machine learning-based deployment analysis
- Support for multiple programming languages
- Detailed metrics and recommendations
- Modern FastAPI-based web interface
- Async/await support throughout
- Comprehensive comparative analysis
- CI/CD integration

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Poetry (dependency management)
- Git
- 4GB RAM minimum (8GB recommended)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install
git clone https://github.com/yourusername/code-quality-analyzer.git
cd code-quality-analyzer
poetry install --extras "all"

# Start the FastAPI server
poetry run code_analyzer-web
```

Visit `http://localhost:8000/docs` for interactive API documentation.

For detailed installation instructions, see the [Installation Guide](docs/installation.md).

## Usage

### Command Line

```bash
# Basic analysis
poetry run code_analyzer https://github.com/username/repo

# Generate PDF report with all features
poetry run code_analyzer https://github.com/username/repo \
    --generate-pdf \
    --include-test-analysis \
    --include-trends

# Full analysis with all options
poetry run code_analyzer https://github.com/username/repo \
    --output-dir ./analysis \
    --generate-pdf \
    --pdf-output ./reports \
    --export-json \
    --min-duplicate-lines 10 \
    --trend-window 6months \
    --trend-granularity weekly \
    --verbose
```

### FastAPI Web Interface

```bash
# Start the server
poetry run code_analyzer-web

# Custom port
poetry run code_analyzer-web --port 8080

# Start without browser
poetry run code_analyzer-web --no-browser
```

### API Usage

```python
import aiohttp
import asyncio

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

asyncio.run(analyze_repo())
```

For complete usage instructions, see the [Usage Guide](docs/usage.md).

## Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Usage Guide](docs/usage.md) - Complete usage documentation
- [API Reference](docs/api.md) - FastAPI endpoints and models
- [Comparative Analysis Guide](docs/comparative_analysis.md) - Advanced comparison features
- [Metrics Documentation](docs/metrics_explanation.md) - Available metrics and their meaning
- [ML Analysis Guide](docs/ml_analysis.md) - Machine learning features

## Architecture

The analyzer is built with modern Python practices and tools:

- **FastAPI** - Modern, fast web framework for building APIs
- **Poetry** - Dependency management and packaging
- **async/await** - Asynchronous operations throughout
- **Pydantic** - Data validation using Python type annotations
- **GitPython** - Git repository interaction
- **ReportLab** - PDF report generation
- **recharts** - Interactive data visualization
- **Machine Learning** - Statistical analysis for deployment optimization

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
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      
      - name: Install dependencies
        run: poetry install --extras "all"
      
      - name: Run analysis
        run: poetry run code_analyzer-web &
             sleep 5 &&
             poetry run python ci/analyze.py
```

For more integration examples, see the [CI/CD Guide](docs/ci_cd.md).

## Development

### Setting Up Development Environment

```bash
# Install with development dependencies
poetry install --with dev,docs,ml

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Build documentation
poetry run sphinx-build -b html docs/source docs/build
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=code_analyzer

# Run specific test category
poetry run pytest -m "not integration"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Package management by [Poetry](https://python-poetry.org/)
- Documentation built with [Sphinx](https://www.sphinx-doc.org/)
- Visualizations powered by [recharts](https://recharts.org/)

## Project Status

Current version: 0.1.0

- [x] FastAPI Integration
- [x] Async Support
- [x] Basic Analysis
- [x] PDF Reports
- [x] Comparative Analysis
- [x] Machine Learning Integration
- [x] Test Coverage Analysis
- [x] Historical Trends
- [ ] Real-time Analysis
- [ ] Plugin System

## Roadmap and Future Features

### Enhanced ML Analysis with Bayesian Methods  

The next major upgrade to the ML analysis system will incorporate Bayesian statistical methods to provide more accurate and personalized deployment predictions. This new approach will:  

- **Learn from Experience**: Build personalized models for each team member based on their deployment history and expertise.  
- **Handle Uncertainty Better**: Provide confidence ranges for predictions instead of single-point estimates.  
- **Adapt to Team Dynamics**: Automatically adjust to changing team patterns and relationships.  
- **Make Better Use of Limited Data**: Start making useful predictions even with limited historical data.  
- **Give More Nuanced Insights**: Account for complex relationships between team members, components, and deployment timing.  

The new system will allow teams to:  
- Get more accurate deployment window recommendations.  
- Better understand prediction confidence.  
- See how different team combinations affect success probability.  
- Track how deployment success patterns evolve over time.  

This update aims to make deployment recommendations even more useful for real-world development teams.

### Frontend Improvements  

Currently, the system uses a static HTML page as the frontend. While this provides a simple and lightweight interface, it lacks interactivity and flexibility. Moving to a framework like **React** or **Vue.js** would allow for:  

- A **more dynamic and responsive UI** with real-time updates.  
- Better **user experience** with interactive dashboards and visualizations.  
- **Scalability** for future feature expansions, including improved filtering and customization options.  

### Additional Integrations  

To enhance usability, integrations such as **calendar integrations** are being considered to better align deployment planning with team member availability. This would allow for:  

- **Automated scheduling** based on team member availability.  
- **Conflict detection** to avoid deployments during unavailable time slots.  
- **Seamless coordination** with tools like Google Calendar, Outlook, or Slack for real-time notifications.  

These enhancements aim to make the system more intuitive and practical for engineering teams managing complex deployments.