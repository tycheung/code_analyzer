import pytest
from pathlib import Path
from datetime import datetime
from code_analyzer.reporters import PDFReportGenerator
from code_analyzer.models.metrics import (
    CodeMetrics, SecurityMetrics, ComplexityMetrics,
    ArchitectureMetrics, ChangeProbability
)

@pytest.fixture
def sample_stats():
    stats = {}
    # Create sample metrics for a few files
    for i in range(3):
        metrics = CodeMetrics(
            lines_code=100 + i * 10,
            lines_comment=20 + i * 5,
            lines_blank=10 + i,
            security=SecurityMetrics(
                potential_sql_injections=i,
                hardcoded_secrets=i,
                unsafe_regex=i
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
def pdf_generator():
    return PDFReportGenerator()

def test_prepare_metrics_data(pdf_generator, sample_stats):
    """Test preparation of metrics data."""
    data = pdf_generator.prepare_metrics_data(sample_stats)
    
    # Check basic structure
    assert isinstance(data, dict)
    assert 'TOTAL_LOC' in data
    assert 'TOTAL_FILES' in data
    assert 'PRIMARY_LANGUAGES' in data
    assert 'MAINTAINABILITY_SCORE' in data
    assert 'SIZE_METRICS_TABLE' in data
    
    # Check data types
    assert isinstance(data['TOTAL_LOC'], str)
    assert isinstance(data['TOTAL_FILES'], str)
    assert isinstance(data['PRIMARY_LANGUAGES'], str)
    assert isinstance(data['MAINTAINABILITY_SCORE'], str)
    
    # Check value ranges
    assert float(data['MAINTAINABILITY_SCORE'].rstrip('/100')) <= 100
    assert int(data['TOTAL_FILES'].replace(',', '')) == len(sample_stats)
    assert isinstance(data['SIZE_METRICS_TABLE'], str)
    assert '\\\\' in data['SIZE_METRICS_TABLE']  # Check for LaTeX line endings

def test_generate_security_findings(pdf_generator, sample_stats):
    """Test generation of security findings section."""
    findings = pdf_generator._generate_security_findings(sample_stats)
    
    # Check basic structure
    assert isinstance(findings, str)
    assert '\\item' in findings
    
    # Check that LaTeX color commands are properly formatted
    assert '{\\color{warning}' in findings or '{\\color{critical}' in findings
    
    # Check content based on sample data
    found_issues = any(
        m.security.potential_sql_injections > 0 or
        m.security.hardcoded_secrets > 0 or
        len(m.security.vulnerable_imports) > 0 or
        m.security.unsafe_regex > 0
        for m in sample_stats.values()
    )
    
    if found_issues:
        assert 'SQL Injection' in findings or 'Hardcoded Secrets' in findings
    else:
        assert 'No significant security issues detected' in findings
        
    # Check LaTeX formatting
    assert findings.count('\\item') >= 1  # At least one finding
    assert findings.count('{') == findings.count('}')  # Balanced braces
    assert '\\color' not in findings or '{\\color' in findings  # Proper color formatting

@pytest.mark.integration
def test_generate_risk_table(pdf_generator, sample_stats):
    """Test generation of risk analysis table."""
    # Add a high-risk file to the sample data
    high_risk_metrics = CodeMetrics(
        lines_code=100,
        lines_comment=20,
        lines_blank=10,
        complexity=ComplexityMetrics(
            cyclomatic_complexity=10,
            maintainability_index=60,
            change_risk=75.0  # High risk
        ),
        change_probability=ChangeProbability(
            file_path="high_risk.py",
            change_frequency=20,
            last_modified=datetime.now(),
            contributors={'user1', 'user2', 'user3'},
            churn_rate=200.0
        )
    )
    sample_stats['high_risk.py'] = high_risk_metrics
    
    table = pdf_generator._generate_risk_table(sample_stats)
    
    # Check basic structure
    assert isinstance(table, str)
    assert table.count('&') > 0  # Should contain column separators
    
    # Check content
    assert 'high_risk.py' in table
    assert '75.0' in table  # Risk score should be present
    assert '20' in table    # Change frequency should be present
    
    # Check table structure
    rows = [row.strip() for row in table.split('\\\\') if row.strip()]
    
    # Each row should have exactly 2 '&' symbols (3 columns)
    for row in rows:
        assert row.count('&') == 2, f"Row '{row}' does not have exactly 3 columns"
        
        # Split into columns and validate
        columns = [col.strip() for col in row.split('&')]
        assert len(columns) == 3, f"Row '{row}' does not have 3 columns"
        
        # Validate number formats
        try:
            risk_score = float(columns[1])
            freq = float(columns[2])
            assert 0 <= risk_score <= 100, f"Invalid risk score: {risk_score}"
            assert freq >= 0, f"Invalid frequency: {freq}"
        except ValueError as e:
            assert False, f"Invalid number format in row: {row}"
    
    # Check we have 1-5 rows total
    assert 1 <= len(rows) <= 5, f"Found {len(rows)} rows, expected between 1 and 5"
    
    # Check proper LaTeX escaping
    assert '\\%' in table or '%' not in table  # Properly escaped percentages

def test_generate_detailed_metrics(pdf_generator, sample_stats):
    metrics = pdf_generator._generate_detailed_metrics(sample_stats)
    assert isinstance(metrics, str)
    assert '\\subsection' in metrics
    assert 'Complexity Metrics' in metrics
    assert 'Code Quality Metrics' in metrics
    assert 'Change Probability Analysis' in metrics

@pytest.mark.integration
def test_generate_pdf(pdf_generator, sample_stats, tmp_path):
    """Test full PDF generation process."""
    try:
        # Generate the PDF
        pdf_generator.generate_pdf(
            stats=sample_stats,
            repo_url="https://github.com/test/repo",
            output_dir=str(tmp_path)
        )
        
        # Check that PDF was generated
        assert (tmp_path / "report.pdf").exists()
        
        # Check that all charts were generated
        charts_dir = tmp_path / "charts"
        assert charts_dir.exists()
        assert (charts_dir / "complexity_dist.png").exists()
        assert (charts_dir / "maintainability_pie.png").exists()
        assert (charts_dir / "file_distribution.png").exists()
        
        # Check chart file sizes (ensure they're not empty)
        assert (charts_dir / "complexity_dist.png").stat().st_size > 0
        assert (charts_dir / "maintainability_pie.png").stat().st_size > 0
        assert (charts_dir / "file_distribution.png").stat().st_size > 0
        
        # Check temporary files were cleaned up
        assert not (tmp_path / "report.aux").exists()
        assert not (tmp_path / "report.log").exists()
        assert not (tmp_path / "report.toc").exists()
        assert not (tmp_path / "report.out").exists()
        
    except FileNotFoundError:
        pytest.skip("LaTeX not installed - skipping PDF generation test")
    except Exception as e:
        # If the test fails, print the LaTeX log if available
        log_file = tmp_path / "report.log"
        if log_file.exists():
            print("\nLaTeX log contents:")
            print(log_file.read_text(encoding='utf-8'))
        
        # Also print directory contents to help debug
        print("\nDirectory contents:")
        print("\nRoot directory:")
        print(list(tmp_path.glob('*')))
        print("\nCharts directory:")
        if (tmp_path / "charts").exists():
            print(list((tmp_path / "charts").glob('*')))
        
        raise  # Re-raise the exception