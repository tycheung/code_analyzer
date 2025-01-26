# Installation Guide

This guide provides detailed instructions for installing the Code Quality Analyzer and its dependencies.

## System Requirements
- Python 3.9 or higher
- Git
- LaTeX distribution (for PDF report generation)
- 2GB free disk space (recommended)
- Internet connection for cloning repositories

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

# Install with all optional features
poetry install --extras "all"
```

### 4. LaTeX Installation

LaTeX is required for generating PDF reports.

#### Windows
1. Download MiKTeX from https://miktex.org/download
2. Run the installer
3. Select "Install MiKTeX for anyone who uses this computer"
4. In MiKTeX Console:
   - Open MiKTeX Console
   - Go to Updates
   - Check for updates
   - Install any available updates

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install texlive-full
sudo apt-get install texlive-latex-extra
```

#### macOS
```bash
# Using Homebrew
brew install basictex
# Refresh PATH
eval "$(/usr/libexec/path_helper)"
# Install additional packages
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
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

# Try analyzing a repository
code_analyzer https://github.com/username/repository --generate-pdf
```

## Troubleshooting

### Common Issues

1. **FastAPI Server Issues**
   - Verify uvicorn is installed: `poetry show uvicorn`
   - Check port availability
   - Ensure proper permissions
   - Verify FastAPI dependencies

2. **PDF Generation Fails**
   - Ensure LaTeX is properly installed
   - Check if required LaTeX packages are installed
   - Try running `pdflatex` from command line
   - Verify async operations complete properly

3. **Git Clone Errors**
   - Verify Git is installed: `git --version`
   - Check repository URL
   - Ensure you have internet access
   - Check async operation handling

4. **Python Import Errors**
   - Verify Poetry environment is activated
   - Ensure all dependencies are installed
   - Check FastAPI and its dependencies
   - Verify Python version compatibility

### Getting Help
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the documentation
- Check FastAPI documentation for API-specific issues

## Updating

To update to the latest version:
```bash
git pull origin main
poetry install
```

## Development Setup

For development, install with all dependencies:
```bash
# Install with development dependencies
poetry install --with dev,docs

# Install pre-commit hooks
poetry run pre-commit install

# Verify FastAPI installation
poetry run python -c "import fastapi; print(fastapi.__version__)"
```