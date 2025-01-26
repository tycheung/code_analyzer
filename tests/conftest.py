"""
Global test fixtures and configuration.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Generator, Dict

# Add src directory to Python path to allow imports
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from code_analyzer.models.metrics import (
    CodeMetrics, SecurityMetrics, ComplexityMetrics,
    ArchitectureMetrics, ChangeProbability
)

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    test_dir = tempfile.mkdtemp(prefix='code_analyzer_test_')
    yield test_dir
    # Cleanup after test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture
def sample_repo(temp_dir: str) -> str:
    """Create a sample repository structure with test files."""
    repo_dir = Path(temp_dir) / "sample_repo"
    repo_dir.mkdir(parents=True)
    
    # Create sample Python file
    python_file = repo_dir / "example.py"
    python_file.write_text("""
def risky_function(user_input):
    # TODO: Add input validation
    query = f"SELECT * FROM users WHERE id = {user_input}"
    password = "hardcoded_secret123"
    return eval(query)

class TestClass:
    def test_method(self):
        pass
""")
    
    # Create sample JavaScript file
    js_file = repo_dir / "script.js"
    js_file.write_text("""
function processData(data) {
    eval(data);  // Unsafe eval
    var password = 'another_secret';
    return data;
}
""")
    
    return str(repo_dir)

@pytest.fixture
def mock_git_history() -> Dict:
    """Create mock git history data."""
    return {
        'example.py': {
            'frequency': 5,
            'last_modified': datetime.now(),
            'contributors': {'user1', 'user2'},
            'churn_rate': 2.5
        },
        'script.js': {
            'frequency': 3,
            'last_modified': datetime.now(),
            'contributors': {'user1'},
            'churn_rate': 1.5
        }
    }

@pytest.fixture
def sample_metrics() -> Dict[str, CodeMetrics]:
    """Create sample metrics for testing."""
    metrics = {}
    
    # Create metrics for two files
    for i, filename in enumerate(['example.py', 'script.js']):
        metrics[filename] = CodeMetrics(
            lines_code=100 + i * 10,
            lines_comment=20 + i * 5,
            lines_blank=10 + i,
            security=SecurityMetrics(
                potential_sql_injections=i,
                hardcoded_secrets=1,
                unsafe_regex=1
            ),
            complexity=ComplexityMetrics(
                cyclomatic_complexity=5 + i,
                maintainability_index=80 - i * 10,
                change_risk=50 + i * 10
            ),
            architecture=ArchitectureMetrics(
                interface_count=i,
                abstract_class_count=i,
                layering_violations=i
            ),
            change_probability=ChangeProbability(
                file_path=filename,
                change_frequency=10 + i,
                last_modified=datetime.now(),
                contributors={'user1', 'user2'}[:i+1],
                churn_rate=100 + i * 50
            )
        )
    
    return metrics

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "latex: mark test as requiring LaTeX installation"
    )
    config.addinivalue_line(
        "markers",
        "git: mark test as requiring Git installation"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (takes more than 1 second)"
    )

@pytest.fixture(autouse=True)
def _check_latex(request):
    """Skip tests marked with 'latex' if LaTeX is not installed."""
    if request.node.get_closest_marker('latex'):
        latex_installed = shutil.which('pdflatex') is not None
        if not latex_installed:
            pytest.skip('LaTeX is not installed')

@pytest.fixture(autouse=True)
def _check_git(request):
    """Skip tests marked with 'git' if Git is not installed."""
    if request.node.get_closest_marker('git'):
        git_installed = shutil.which('git') is not None
        if not git_installed:
            pytest.skip('Git is not installed')

@pytest.fixture
def assert_no_warnings():
    """Fixture to ensure no warnings are emitted during test."""
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield
        assert not warning_list, f"Test produced warnings: {[str(w.message) for w in warning_list]}"