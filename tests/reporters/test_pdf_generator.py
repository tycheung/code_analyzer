import pytest
from pathlib import Path
from datetime import datetime, time
from unittest.mock import Mock, patch
from pylatex import Document
from code_analyzer.reporters import PDFReportGenerator
from code_analyzer.models.metrics import (
    CodeMetrics, SecurityMetrics, ComplexityMetrics,
    ArchitectureMetrics, ChangeProbability, DeploymentWindow,
    ResourceAllocation, RollbackPrediction, IncidentPrediction
)

@pytest.fixture
def sample_stats():
    """Create sample code metrics for testing."""
    stats = {}
    for i in range(3):
        metrics = CodeMetrics(
            lines_code=100 + i * 10,
            lines_comment=20 + i * 5,
            lines_blank=10 + i,
            security=SecurityMetrics(
                potential_sql_injections=i,
                hardcoded_secrets=i,
                unsafe_regex=i,
                vulnerable_imports=[f"vulnerable_pkg_{j}" for j in range(i)]
            ),
            complexity=ComplexityMetrics(
                cyclomatic_complexity=5 + i,
                maintainability_index=80 - i * 10,
                change_risk=50 + i * 10
            ),
            architecture=ArchitectureMetrics(
                interface_count=i,
                abstract_class_count=i,
                layering_violations=i,
                component_coupling=0.5 + i * 0.1
            ),
            change_probability=ChangeProbability(
                file_path=f"file{i}.py",
                change_frequency=10 + i,
                last_modified=datetime.now(),
                contributors=set([f"user{j}" for j in range(i+1)]),
                churn_rate=100 + i * 50
            )
        )
        stats[f"file{i}.py"] = metrics
    return stats

@pytest.fixture
def mock_ml_system():
    """Create a mock ML system for testing."""
    mock = Mock()
    mock.analyze_deployment.return_value = {
        'overall_confidence': 0.85,
        'optimal_windows': [
            DeploymentWindow(
                start_time=time(9, 0),
                end_time=time(10, 0),
                risk_score=0.2,
                team_availability=0.9,
                historical_success_rate=0.95
            )
        ],
        'resource_prediction': ResourceAllocation(
            recommended_team_size=3,
            required_skills={'python', 'devops'},
            estimated_support_duration=4.5,
            confidence_score=0.8,
            explanation="Team size based on complexity",
            feature_importance={'complexity': 0.6},
            top_contributors=[{'feature': 'complexity', 'impact': 0.6}],
            feature_interactions=[]
        ),
        'rollback_prediction': RollbackPrediction(
            probability=0.15,
            risk_factors={'complexity': 0.3},
            severity_level='low',
            critical_files=['file1.py'],
            mitigation_suggestions=['Review test coverage'],
            recommended_reviewers=['user1', 'user2'],  # Added this line
            confidence_score=0.9,
            prediction_quality={'sufficient_data': True},
            data_completeness=0.95,
            explanation="Low rollback risk",
            feature_importance={'complexity': 0.4},
            top_contributors=[],
            feature_interactions=[],
            similar_cases=['prev_deploy_1'],
            historical_pattern={'success_rate': 0.9}
        ),
        'incident_prediction': IncidentPrediction(
            probability=0.1,
            potential_areas=['Test Coverage'],
            severity_level='low',
            estimated_resolution_time=2.0,
            confidence_score=0.85,
            explanation="Low incident risk",
            feature_importance={'complexity': 0.3},
            top_contributors=[],
            feature_interactions=[]
        ),
        'window_explanation': "Optimal window selected based on team availability",
        'rollback_explanation': "Low risk due to good test coverage",
        'resource_explanation': "Team size based on project complexity",
        'incident_explanation': "Low incident risk due to stable codebase",
        'feature_importances': {
            'cross_model': {'complexity': 0.5},
            'by_model': {}
        }
    }
    return mock

@pytest.fixture
def pdf_generator(mock_ml_system):
    """Create PDF generator with mock ML system."""
    return PDFReportGenerator(ml_system=mock_ml_system)

def test_document_setup(pdf_generator):
    """Test proper document setup with required packages."""
    doc = Document()
    pdf_generator._setup_document(doc)
    
    # Extract package names from PyLaTeX packages
    package_names = set()
    for pkg in doc.packages:
        pkg_str = str(pkg)
        if "Arguments('" in pkg_str:
            # Extract package name from Arguments('package_name', ...)
            start = pkg_str.find("Arguments('") + len("Arguments('")
            end = pkg_str.find("'", start)
            if end != -1:
                pkg_name = pkg_str[start:end]
                package_names.add(pkg_name)
        elif '{' in pkg_str:
            # Fallback for other formats
            try:
                pkg_name = pkg_str.split('{')[1].split('}')[0]
                package_names.add(pkg_name)
            except IndexError:
                pass
    
    # Check each required package is included
    required_packages = {'inputenc', 'geometry', 'graphicx', 'xcolor'}
    for required_pkg in required_packages:
        assert any(required_pkg in pkg_name for pkg_name in package_names), \
            f"Required package '{required_pkg}' not found in {package_names}"

def test_prepare_metrics_data(pdf_generator, sample_stats):
    """Test metrics data preparation."""
    data = pdf_generator.prepare_metrics_data(sample_stats)
    
    # Check required keys
    required_keys = {
        'TOTAL_LOC', 'TOTAL_FILES', 'PRIMARY_LANGUAGES',
        'MAINTAINABILITY_SCORE', 'KEY_FINDINGS', 'SECURITY_FINDINGS'
    }
    assert required_keys.issubset(data.keys())
    
    # Validate values
    assert data['TOTAL_LOC'].isdigit()
    assert data['TOTAL_FILES'].isdigit()
    assert len(data['PRIMARY_LANGUAGES']) > 0
    assert float(data['MAINTAINABILITY_SCORE'].rstrip('/100')) <= 100

def test_security_findings_generation(pdf_generator, sample_stats):
    """Test security findings section generation."""
    findings = pdf_generator._generate_security_findings(sample_stats)
    
    # Check LaTeX formatting
    assert findings.count('\\item') >= 1
    assert findings.count('{') == findings.count('}')
    assert '\\color{warning}' in findings or '\\color{critical}' in findings
    
    # Check content
    has_issues = any(
        m.security.potential_sql_injections > 0 or
        m.security.hardcoded_secrets > 0
        for m in sample_stats.values()
    )
    if has_issues:
        assert any(x in findings for x in ['SQL Injection', 'Hardcoded Secrets'])
    else:
        assert 'No significant security issues' in findings

@pytest.mark.integration
def test_chart_generation(pdf_generator, sample_stats, tmp_path):
    """Test chart generation functionality."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    charts = pdf_generator.generate_charts(sample_stats, str(tmp_path))
    
    # Check that all expected charts are generated
    assert set(charts.keys()) == {
        'COMPLEXITY_CHART',
        'MAINTAINABILITY_CHART',
        'DISTRIBUTION_CHART'
    }
    
    # Verify files exist and are not empty
    charts_dir = tmp_path / 'charts'
    for chart_path in charts.values():
        chart_file = charts_dir / Path(chart_path).name
        assert chart_file.exists()
        assert chart_file.stat().st_size > 0

@pytest.mark.integration
def test_full_pdf_generation(pdf_generator, sample_stats, tmp_path):
    """Test complete PDF generation process."""
    try:
        pdf_generator.generate_pdf(
            stats=sample_stats,
            repo_url="https://github.com/test/repo",
            output_dir=str(tmp_path)
        )
        
        # Check PDF generation
        pdf_file = tmp_path / "report.pdf"
        assert pdf_file.exists()
        assert pdf_file.stat().st_size > 0
        
        # Check cleanup
        assert not (tmp_path / "report.aux").exists()
        assert not (tmp_path / "report.log").exists()
        assert not (tmp_path / "report.toc").exists()
        
    except FileNotFoundError:
        pytest.skip("LaTeX not installed - skipping PDF generation test")

def test_ml_data_preparation(pdf_generator, sample_stats, mock_ml_system):
    """Test ML-related data preparation."""
    data = pdf_generator._prepare_ml_data(sample_stats)
    
    # Check required ML-related keys
    required_keys = {
        'DEPLOYMENT_CONFIDENCE',
        'DEPLOYMENT_WINDOWS_TABLE',
        'RESOURCE_REQUIREMENTS',
        'INCIDENT_PROBABILITY',
        'INCIDENT_SEVERITY'
    }
    assert required_keys.issubset(data.keys())
    
    # Check value formatting
    assert data['DEPLOYMENT_CONFIDENCE'].endswith('\\%')
    assert data['INCIDENT_SEVERITY'] in ['Low', 'Medium', 'High', 'Unknown']

def test_error_handling(pdf_generator, sample_stats):
    """Test error handling in PDF generation."""
    with patch.object(Document, 'generate_pdf', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            pdf_generator.generate_pdf(
                stats=sample_stats,
                repo_url="https://github.com/test/repo",
                output_dir="/nonexistent"
            )

def test_latex_escaping(pdf_generator):
    """Test LaTeX special character escaping."""
    test_cases = [
        ('simple', 'simple'),  # No special chars
        ('text&', 'text\\&'),  # Single special char
        ('text%', 'text\\%'),  # Single special char
        ('text$', 'text\\$'),  # Single special char
        ('text_', 'text\\_'),  # Single special char
        ('\\begin{item}', '\\begin{item}'),  # LaTeX command
        ('test%test', 'test\\%test'),  # Special char in middle
        ('test_test', 'test\\_test'),  # Special char in middle
        ('test&test', 'test\\&test'),  # Special char in middle
        ('test$test', 'test\\$test'),  # Special char in middle
        ('test#test', 'test\\#test'),  # Special char in middle
        ('a_{b}', 'a_{b}'),  # Math mode subscript - no escaping
        ('\\textbf{test}', '\\textbf{test}'),  # LaTeX command with braces
        ('test_{1}', 'test_{1}'),  # Math mode with number
        ('a_{b}_{c}', 'a_{b}_{c}'),  # Multiple subscripts
        ('test_1', 'test\\_1'),  # Regular underscore
        ('\\section{test_1}', '\\section{test_1}'),  # Command with underscore
    ]
    
    for input_text, expected in test_cases:
        result = pdf_generator._escape_latex(input_text)
        assert result == expected, f"Failed on input '{input_text}': expected '{expected}' but got '{result}'"