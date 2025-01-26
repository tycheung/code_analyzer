import os
import pytest
from pathlib import Path
from datetime import datetime
from code_analyzer.analyzers import CodebaseAnalyzer
from code_analyzer.models.metrics import CodeMetrics, SecurityMetrics, ArchitectureMetrics

# Create a fixture directory for test files
@pytest.fixture
def test_files_dir(tmp_path):
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def sample_python_file(test_files_dir):
    file_content = """
def risky_function(user_input):
    # TODO: Add input validation
    query = f"SELECT * FROM users WHERE id = {user_input}"
    password = "hardcoded_secret123"
    return eval(query)

class TestClass:
    def test_method(self):
        pass
"""
    file_path = test_files_dir / "test.py"
    file_path.write_text(file_content)
    return file_path

@pytest.fixture
def analyzer():
    return CodebaseAnalyzer()

def test_analyze_security(analyzer, sample_python_file):
    with open(sample_python_file, 'r') as f:
        content = f.read()
    
    security_metrics = analyzer.analyze_security(content)
    assert isinstance(security_metrics, SecurityMetrics)
    assert security_metrics.potential_sql_injections >= 1
    assert security_metrics.hardcoded_secrets >= 1
    assert security_metrics.unsafe_regex >= 1

def test_analyze_architecture(analyzer, sample_python_file):
    with open(sample_python_file, 'r') as f:
        content = f.read()
    
    arch_metrics = analyzer.analyze_architecture(content)
    assert isinstance(arch_metrics, ArchitectureMetrics)
    assert arch_metrics.interface_count == 0
    assert arch_metrics.abstract_class_count == 0

def test_analyze_file(analyzer, sample_python_file):
    metrics = analyzer.analyze_file(str(sample_python_file))
    assert isinstance(metrics, CodeMetrics)
    assert metrics.lines_code > 0
    assert metrics.lines_comment > 0
    assert metrics.todo_count == 1

def test_binary_file_detection(analyzer, test_files_dir):
    # Create a binary file
    binary_file = test_files_dir / "binary.dat"
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03')
    
    assert analyzer.is_binary_file(str(binary_file))

def test_complexity_calculation(analyzer, sample_python_file):
    with open(sample_python_file, 'r') as f:
        content = f.read()
    
    complexity = analyzer.analyze_complexity(content)
    assert complexity.cyclomatic_complexity > 1
    assert complexity.max_nesting_depth >= 0
    assert complexity.cognitive_complexity >= 0

def test_change_probability(analyzer, sample_python_file):
    change_prob = analyzer.analyze_change_probability(str(sample_python_file))
    assert change_prob.file_path == str(sample_python_file)
    assert isinstance(change_prob.last_modified, datetime)
    assert isinstance(change_prob.contributors, set)
    assert change_prob.churn_rate >= 0

def test_scan_directory(analyzer, test_files_dir):
    # Create multiple test files
    (test_files_dir / "test1.py").write_text("def test(): pass")
    (test_files_dir / "test2.py").write_text("class Test: pass")
    (test_files_dir / "ignore.txt").write_text("Not a code file")
    
    analyzer.scan_directory(str(test_files_dir))
    assert len(analyzer.stats) == 2  # Only .py files should be analyzed