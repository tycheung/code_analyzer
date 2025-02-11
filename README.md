# Code Quality Analyzer

A tool for analyzing code quality, complexity, and maintainability of GitHub repositories, powered by FastAPI and modern Python async capabilities. This project is for fun/personal use and in alpha stage, there may be incompatibilities.

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

For detailed installation instructions, see our [Installation Guide](docs/installation.md).

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

For complete usage instructions, see our [Usage Guide](docs/usage.md).

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

For more integration examples, see our [CI/CD Guide](docs/ci_cd.md).

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