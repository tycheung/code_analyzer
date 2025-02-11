# Installation Guide

This guide provides detailed instructions for installing the Code Quality Analyzer and its dependencies.

## System Requirements

- Python 3.9 or higher
- Git
- 4GB free disk space (recommended)
- Internet connection for cloning repositories
- 4GB RAM minimum (8GB recommended for large repositories)
- Sufficient CPU cores for parallel analysis

## Step-by-Step Installation

### 1. Python Environment Setup

First, ensure you have Python 3.9 or higher installed:
```bash
python --version
```

### 2. Installing Poetry
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Unix, macOS, WSL
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Installing the Package

Clone the repository and install using Poetry:
```bash
# Clone the repository
git clone https://github.com/yourusername/code-quality-analyzer.git
cd code-quality-analyzer

# Install dependencies with Poetry
poetry install

# Install with web interface support
poetry install --extras "web"

# Install with ML analysis support
poetry install --extras "ml"

# Install with all optional features
poetry install --extras "all"
```

### 4. ML Dependencies (Optional)

If you plan to use ML-based analysis features:

```bash
# Install ML dependencies
poetry install --extras "ml"

# Download required data models
poetry run code_analyzer-download-models

# Verify ML installation
poetry run code_analyzer-verify-ml
```

### 5. Verifying Installation

Run the following commands to verify the installation:
```bash
# Activate poetry shell
poetry shell

# Verify code_analyzer is installed
code_analyzer --version

# Start the FastAPI server
code_analyzer-web

# Run tests
poetry run pytest

# Try analyzing a repository with all features
code_analyzer https://github.com/username/repository \
    --generate-pdf \
    --include-test-analysis \
    --include-trends
```

## Troubleshooting

### Common Issues

1. **FastAPI Server Issues**
   - Verify uvicorn is installed: `poetry show uvicorn`
   - Check port availability
   - Ensure proper permissions
   - Verify FastAPI dependencies
   - Check CORS configuration
   - Verify WebSocket support

2. **PDF Generation Fails**
   - Check write permissions for output directory
   - Verify sufficient memory
   - Ensure all required fonts are available
   - Check chart generation dependencies

3. **Git Clone Errors**
   - Verify Git is installed: `git --version`
   - Check repository URL
   - Ensure you have internet access
   - Check async operation handling
   - Verify sufficient disk space
   - Check Git LFS support if needed

4. **Python Import Errors**
   - Verify Poetry environment is activated
   - Ensure all dependencies are installed
   - Check FastAPI and its dependencies
   - Verify Python version compatibility
   - Check ML dependencies if using ML features

5. **ML Analysis Issues**
   - Verify ML dependencies are installed
   - Check model downloads
   - Ensure sufficient RAM
   - Check input data format
   - Verify statistical model compatibility

6. **Memory Issues**
   - Check available system RAM
   - Monitor memory usage during analysis
   - Adjust batch sizes for large repositories
   - Configure swap space if needed
   - Consider using memory-efficient options

### Getting Help
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the documentation
- Check FastAPI documentation for API-specific issues
- Review ML documentation for analysis issues

## Updating

To update to the latest version:
```bash
git pull origin main
poetry install
poetry run code_analyzer-update-models  # If using ML features
```

## Development Setup

For development, install with all dependencies:
```bash
# Install with development dependencies
poetry install --with dev,docs,ml

# Install pre-commit hooks
poetry run pre-commit install

# Verify installations
poetry run python -c "import fastapi; print(fastapi.__version__)"

# Setup development environment
poetry run code_analyzer-setup-dev

# Run development tests
poetry run pytest tests/
poetry run pytest tests/ml/  # ML-specific tests
```