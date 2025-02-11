from datetime import datetime, time
from typing import Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import pytest
from unittest.mock import MagicMock, Mock, patch, ANY
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.graphics.shapes import Drawing
from copy import deepcopy

from code_analyzer.models.metrics import (
    CodeMetrics, SecurityMetrics, ArchitectureMetrics,
    ComplexityMetrics, ChangeProbability, DeploymentWindow,
    ResourceAllocation, RollbackPrediction, IncidentPrediction,
    DeploymentAnalysis
)

@pytest.fixture
def mock_security_metrics():
    """Create mock security metrics."""
    return SecurityMetrics(
        potential_sql_injections=2,
        hardcoded_secrets=1,
        unsafe_regex=3,
        vulnerable_imports=['telnetlib', 'pickle'],
        insecure_patterns={
            'shell_injection': 1,
            'temp_file': 2,
            'unsafe_deserialization': 1
        }
    )

@pytest.fixture
def mock_architecture_metrics():
    """Create mock architecture metrics."""
    return ArchitectureMetrics(
        circular_dependencies=[['module1', 'module2'], ['module3', 'module4']],
        abstraction_level=0.65,
        component_coupling=0.45,
        interface_count=5,
        abstract_class_count=3,
        layering_violations=2
    )

@pytest.fixture
def mock_complexity_metrics():
    """Create mock complexity metrics."""
    return ComplexityMetrics(
        cyclomatic_complexity=12,
        max_nesting_depth=4,
        cognitive_complexity=15,
        halstead_metrics={
            'volume': 1000.0,
            'difficulty': 25.0,
            'effort': 25000.0,
            'vocabulary': 50,
            'length': 200
        },
        maintainability_index=75.0,
        change_risk=65.0
    )

@pytest.fixture
def mock_change_probability():
    """Create mock change probability metrics."""
    return ChangeProbability(
        file_path="src/main.py",
        change_frequency=10,
        last_modified=datetime.now(),
        contributors={'dev1', 'dev2'},
        churn_rate=0.75
    )

@pytest.fixture
def mock_code_metrics(
    mock_security_metrics,
    mock_architecture_metrics,
    mock_complexity_metrics,
    mock_change_probability
):
    """Create mock code metrics."""
    metrics = CodeMetrics(
        lines_code=100,
        lines_comment=20,
        lines_blank=10,
        classes=['TestClass'],
        functions=['test_func'],
        routes=['/api/test'],
        imports=['os', 'sys'],
        dependencies={'dep1', 'dep2'},
        complexity=mock_complexity_metrics,
        architecture=mock_architecture_metrics,
        security=mock_security_metrics,
        file_size=5000,
        avg_function_length=15.5,
        max_function_length=30,
        todo_count=2,
        test_coverage_files={'test_main.py'},
        code_patterns={
            'magic_numbers': 5,
            'large_try_blocks': 1,
            'boolean_traps': 2
        },
        change_probability=mock_change_probability
    )
    
    # Add history data
    metrics.history = [
        {
            'date': '2024-01',
            'avg_complexity': 10.5,
            'avg_cognitive_complexity': 12.0,
            'total_lines': 1000,
            'code_lines': 800,
            'security_issues': 5,
            'architecture_issues': 3
        },
        {
            'date': '2024-02',
            'avg_complexity': 11.0,
            'avg_cognitive_complexity': 13.0,
            'total_lines': 1200,
            'code_lines': 950,
            'security_issues': 4,
            'architecture_issues': 2
        }
    ]
    
    return metrics

@pytest.fixture
def mock_deployment_window():
    """Create mock deployment window."""
    return DeploymentWindow(
        start_time=time(10, 0),
        end_time=time(12, 0),
        risk_score=0.25,
        team_availability=0.85,
        historical_success_rate=0.92,
        required_team_size=3,
        required_skills=['Python', 'DevOps'],
        estimated_duration=2.5,
        system_load={'cpu': 0.45, 'memory': 0.55},
        concurrent_deployments=['service1'],
        explanation="Optimal window based on team availability and system load",
        feature_importance={'team_size': 0.3, 'system_load': 0.4},
        top_contributors=[
            {'feature': 'team_size', 'impact': 0.3, 'deviation': 0.1}
        ],
        feature_interactions=[
            {'features': ['team_size', 'system_load'], 'importance': 0.25}
        ],
        prediction_quality={'sufficient_data': True},
        data_completeness=0.95
    )

@pytest.fixture
def mock_resource_allocation():
    """Create mock resource allocation."""
    return ResourceAllocation(
        recommended_team_size=3,
        required_skills={'Python', 'DevOps'},
        estimated_support_duration=4.5,
        confidence_score=0.85,
        explanation="Resource allocation based on complexity and scope",
        feature_importance={'complexity': 0.4, 'scope': 0.3},
        top_contributors=[
            {'feature': 'complexity', 'impact': 0.4, 'deviation': 0.1}
        ],
        feature_interactions=[
            {'features': ['complexity', 'scope'], 'importance': 0.2}
        ]
    )

@pytest.fixture
def mock_rollback_prediction():
    """Create mock rollback prediction."""
    return RollbackPrediction(
        probability=0.15,
        risk_factors={'complexity': 0.3, 'dependencies': 0.2},
        severity_level='low',
        critical_files=['src/core.py'],
        mitigation_suggestions=['Increase test coverage'],
        recommended_reviewers=['dev1'],
        confidence_score=0.88,
        prediction_quality={'sufficient_data': True},
        data_completeness=0.95,
        explanation="Low rollback risk based on analysis",
        feature_importance={'complexity': 0.3, 'test_coverage': 0.4},
        top_contributors=[
            {'feature': 'complexity', 'impact': 0.3, 'deviation': 0.1}
        ],
        feature_interactions=[
            {'features': ['complexity', 'test_coverage'], 'importance': 0.2}
        ],
        similar_cases=['deployment123'],
        historical_pattern={'success_rate': 0.9}
    )

@pytest.fixture
def mock_incident_prediction():
    """Create mock incident prediction."""
    return IncidentPrediction(
        probability=0.12,
        potential_areas=['API', 'Database'],
        severity_level='low',
        estimated_resolution_time=1.5,
        confidence_score=0.87,
        explanation="Low incident probability based on analysis",
        feature_importance={'complexity': 0.25, 'test_coverage': 0.35},
        top_contributors=[
            {'feature': 'complexity', 'impact': 0.25, 'deviation': 0.1}
        ],
        feature_interactions=[
            {'features': ['complexity', 'test_coverage'], 'importance': 0.15}
        ]
    )

@pytest.fixture
def mock_deployment_analysis(
    mock_deployment_window,
    mock_rollback_prediction,
    mock_resource_allocation,
    mock_incident_prediction
):
    """Create mock deployment analysis."""
    return DeploymentAnalysis(
        optimal_windows=[mock_deployment_window],
        rollback_prediction=mock_rollback_prediction,
        resource_prediction=mock_resource_allocation,
        incident_prediction=mock_incident_prediction,
        overall_confidence=0.85,
        confidence_breakdown={
            'window': 0.9,
            'rollback': 0.88,
            'resource': 0.85,
            'incident': 0.87
        },
        feature_importances={
            'cross_model': {
                'complexity': 0.35,
                'test_coverage': 0.3
            },
            'by_model': {
                'window': {'team_size': 0.3},
                'rollback': {'complexity': 0.3}
            }
        },
        system_explanation="Comprehensive deployment analysis results",
        key_insights=[
            {
                'aspect': 'deployment_window',
                'feature': 'team_size',
                'impact': 0.3,
                'deviation': 0.1
            }
        ],
        data_completeness=0.95,
        prediction_quality={'sufficient_data': True}
    )

@pytest.fixture
def pdf_generator(tmp_path):
    """Create PDF generator instance with temporary output path."""
    from code_analyzer.reporters.pdf_generator import CodeMetricsPDFGenerator
    output_path = tmp_path / "test_report.pdf"
    return CodeMetricsPDFGenerator(str(output_path))

class TestCodeMetricsPDFGenerator:
    """Test cases for CodeMetricsPDFGenerator class."""

    def test_initialization(self, pdf_generator):
        """Test proper initialization of PDF generator."""
        assert pdf_generator.pagesize == A4
        assert isinstance(pdf_generator.doc, SimpleDocTemplate)
        assert pdf_generator.story == []
        assert pdf_generator.styles is not None
        assert isinstance(pdf_generator.styles, getSampleStyleSheet().__class__)
        
    def test_custom_styles_setup(self, pdf_generator):
        """Test custom styles are properly initialized."""
        style_names = {
            'MainTitle', 'SectionTitle', 'SubsectionTitle',
            'CustomBody', 'MetricValue'
        }
        for style_name in style_names:
            assert style_name in pdf_generator.styles.byName
            
    def test_generate_pdf_creates_file(self, pdf_generator, mock_code_metrics):
        """Test PDF file is actually created."""
        pdf_generator.generate_pdf({"file1.py": mock_code_metrics})
        assert os.path.exists(pdf_generator.output_path)
        assert os.path.getsize(pdf_generator.output_path) > 0
        
    def test_generate_pdf_with_deployment(
        self, pdf_generator, mock_code_metrics, mock_deployment_analysis
    ):
        """Test PDF generation with deployment analysis."""
        pdf_generator.generate_pdf(
            {"file1.py": mock_code_metrics},
            mock_deployment_analysis
        )
        assert os.path.exists(pdf_generator.output_path)
        assert os.path.getsize(pdf_generator.output_path) > 0
        
    @pytest.mark.parametrize("section_method", [
        "_generate_summary_section",
        "_generate_security_section",
        "_generate_architecture_section",
        "_generate_complexity_section",
        "_generate_testing_section",
        "_generate_trends_section"
    ])
    def test_section_generation(
        self, pdf_generator, mock_code_metrics, section_method
    ):
        """Test each section generator method."""
        # Get the method
        method = getattr(pdf_generator, section_method)
        
        # Create metrics dict
        metrics = {"file1.py": mock_code_metrics}
        
        # Call the method
        method(metrics)
        
        # Check that content was added to story
        assert len(pdf_generator.story) > 0
        
    def test_deployment_section(
        self, pdf_generator, mock_deployment_analysis
    ):
        """Test deployment section generation."""
        pdf_generator._generate_deployment_section(mock_deployment_analysis)
        assert len(pdf_generator.story) > 0
        
    def test_add_title(self, pdf_generator):
        """Test title addition."""
        title = "Test Title"
        pdf_generator._add_title(title)
        assert len(pdf_generator.story) > 0
        assert isinstance(pdf_generator.story[0], Paragraph)
        
    def test_add_paragraph(self, pdf_generator):
        """Test paragraph addition."""
        text = "Test paragraph"
        pdf_generator._add_paragraph(text)
        assert len(pdf_generator.story) > 0
        assert isinstance(pdf_generator.story[0], Paragraph)
        
    def test_add_table(self, pdf_generator):
        """Test table addition."""
        data = [["Header"], ["Data"]]
        pdf_generator._add_table(data)
        assert len(pdf_generator.story) > 0
        
    def test_header_footer(self, pdf_generator):
        """Test header and footer generation."""
        canvas_mock = Mock()
        doc_mock = Mock()
        doc_mock.pagesize = A4
        
        pdf_generator._header_footer(canvas_mock, doc_mock)
        
        # Verify canvas methods were called
        canvas_mock.saveState.assert_called_once()
        canvas_mock.setFont.assert_called()
        canvas_mock.drawString.assert_called()
        canvas_mock.restoreState.assert_called_once()
        
    def test_generate_cover_page(self, pdf_generator, mock_code_metrics):
        """Test cover page generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_cover_page(metrics)
        assert len(pdf_generator.story) > 0
        
    def test_generate_appendix(self, pdf_generator, mock_code_metrics):
        """Test appendix generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_appendix(metrics)
        assert len(pdf_generator.story) > 0
        
    def test_error_handling(self, pdf_generator):
        """Test error handling in PDF generation."""
        with pytest.raises(TypeError):
            pdf_generator.generate_pdf(None)
            
    def test_large_dataset(self, pdf_generator, mock_code_metrics):
        """Test PDF generation with a large dataset."""
        # Create a large metrics dictionary
        large_metrics = {
            f"file{i}.py": mock_code_metrics
            for i in range(100)
        }
        pdf_generator.generate_pdf(large_metrics)
        assert os.path.exists(pdf_generator.output_path)
        assert os.path.getsize(pdf_generator.output_path) > 0

class TestSecuritySection:
    """Test cases for security section generation."""
    
    def test_security_overview(self, pdf_generator, mock_code_metrics):
        """Test security overview generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_security_section(metrics)
        
        # Check that overview paragraph was added
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("security issues" in str(p).lower() for p in paragraphs)
        
    def test_security_table_generation(self, pdf_generator, mock_code_metrics):
        """Test security issues table generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_security_section(metrics)
        
        # Find security table in story
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        assert len(tables) > 0
        
        # Verify table contains expected columns
        first_table = tables[0]
        table_data = first_table._cellvalues
        expected_headers = ['Vulnerability Type', 'Count', 'Severity']
        assert all(header in table_data[0] for header in expected_headers)
        
    def test_security_chart_generation(self, pdf_generator, mock_code_metrics):
        """Test security chart generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_security_section(metrics)
        
        # Check for chart presence
        drawings = [
            item for item in pdf_generator.story 
            if isinstance(item, Drawing)
        ]
        assert len(drawings) > 0
        
    def test_vulnerable_files_listing(self, pdf_generator, mock_code_metrics):
        """Test listing of files with security issues."""
        # Create metrics with known security issues
        metrics = {
            "file1.py": mock_code_metrics,
            "file2.py": deepcopy(mock_code_metrics)
        }
        # Modify second file to have more security issues
        metrics["file2.py"].security.potential_sql_injections = 3
        
        pdf_generator._generate_security_section(metrics)
        
        # Check for vulnerable files table
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        vulnerable_table = tables[1]  # Second table should be vulnerable files
        assert any("file2.py" in str(row) for row in vulnerable_table._cellvalues)
        
    def test_no_security_issues(self, pdf_generator, mock_code_metrics):
        """Test handling of case with no security issues."""
        # Create metrics with no security issues
        clean_metrics = deepcopy(mock_code_metrics)
        clean_metrics.security.potential_sql_injections = 0
        clean_metrics.security.hardcoded_secrets = 0
        clean_metrics.security.unsafe_regex = 0
        clean_metrics.security.vulnerable_imports = []
        
        metrics = {"clean_file.py": clean_metrics}
        
        pdf_generator._generate_security_section(metrics)
        
        # Check for appropriate "no issues" message
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("no" in str(p).lower() and "security issues" in str(p).lower() 
                  for p in paragraphs)
        
    def test_security_recommendations(self, pdf_generator, mock_code_metrics):
        """Test security recommendations generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_security_section(metrics)
        
        # Check for recommendations
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("recommend" in str(p).lower() for p in paragraphs)
        
    def test_high_risk_security_issues(self, pdf_generator, mock_code_metrics):
        """Test handling of high-risk security issues."""
        # Create metrics with high-risk issues
        high_risk_metrics = deepcopy(mock_code_metrics)
        high_risk_metrics.security.potential_sql_injections = 5
        high_risk_metrics.security.hardcoded_secrets = 4
        
        metrics = {"high_risk.py": high_risk_metrics}
        
        pdf_generator._generate_security_section(metrics)
        
        # Check for high-risk warnings
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        severity_table = tables[0]
        assert any("HIGH" in str(row) for row in severity_table._cellvalues)
        
    def test_insecure_patterns_analysis(self, pdf_generator, mock_code_metrics):
        """Test analysis of insecure patterns."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_security_section(metrics)
        
        # Check for patterns analysis
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("pattern" in str(p).lower() for p in paragraphs)
        
        # Check pattern details
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        patterns_table = tables[-1]  # Last table should be patterns
        assert any("shell injection" in str(row).lower() 
                  for row in patterns_table._cellvalues)
        
class TestArchitectureSection:
    """Test cases for architecture section generation."""
    
    def test_architecture_overview(self, pdf_generator, mock_code_metrics):
        """Test architecture overview generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("architecture" in str(p).lower() for p in paragraphs)
        
    def test_metrics_overview_table(self, pdf_generator, mock_code_metrics):
        """Test architecture metrics table generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        metrics_table = tables[0]
        
        # Verify table contains key metrics
        expected_metrics = ['Component Coupling', 'Abstraction Level', 'Layering Violations']
        table_data = metrics_table._cellvalues
        assert all(any(metric.lower() in str(row).lower() for row in table_data) 
                  for metric in expected_metrics)
        
    def test_coupling_chart_generation(self, pdf_generator, mock_code_metrics):
        """Test component coupling chart generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        drawings = [
            item for item in pdf_generator.story 
            if isinstance(item, Drawing)
        ]
        assert len(drawings) > 0
        
    def test_dependency_analysis(self, pdf_generator, mock_code_metrics):
        """Test dependency analysis generation."""
        # Create metrics with circular dependencies
        metrics = {
            "file1.py": mock_code_metrics,
            "file2.py": deepcopy(mock_code_metrics)
        }
        metrics["file2.py"].architecture.circular_dependencies.append(['module5', 'module6'])
        
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for dependency analysis content
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("dependency" in str(p).lower() for p in paragraphs)
        
        # Check for circular dependencies table
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        assert any(any("circular" in str(row).lower() 
                      for row in table._cellvalues) 
                  for table in tables)
        
    def test_no_architectural_issues(self, pdf_generator, mock_code_metrics):
        """Test handling of case with no architectural issues."""
        clean_metrics = deepcopy(mock_code_metrics)
        clean_metrics.architecture.circular_dependencies = []
        clean_metrics.architecture.layering_violations = 0
        
        metrics = {"clean_file.py": clean_metrics}
        
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for appropriate "no issues" message
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("no" in str(p).lower() and "dependencies" in str(p).lower() 
                  for p in paragraphs)
        
    def test_layering_analysis(self, pdf_generator, mock_code_metrics):
        """Test layering analysis generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for layering analysis content
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("layer" in str(p).lower() for p in paragraphs)
        
    def test_abstraction_analysis(self, pdf_generator, mock_code_metrics):
        """Test abstraction analysis generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for abstraction chart
        drawings = [
            item for item in pdf_generator.story 
            if isinstance(item, Drawing)
        ]
        assert len(drawings) >= 2  # Should have at least coupling and abstraction charts
        
    def test_high_coupling_warning(self, pdf_generator, mock_code_metrics):
        """Test warnings for high component coupling."""
        # Create metrics with high coupling
        high_coupling_metrics = deepcopy(mock_code_metrics)
        high_coupling_metrics.architecture.component_coupling = 0.85
        
        metrics = {"high_coupling.py": high_coupling_metrics}
        
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for high coupling warnings
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        coupling_table = tables[0]
        assert any("HIGH" in str(row) for row in coupling_table._cellvalues)
        
    def test_architecture_recommendations(self, pdf_generator, mock_code_metrics):
        """Test architecture recommendations generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for recommendations
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("recommend" in str(p).lower() for p in paragraphs)
        
    def test_multiple_files_analysis(self, pdf_generator, mock_code_metrics):
        """Test architecture analysis with multiple files."""
        metrics = {
            f"file{i}.py": deepcopy(mock_code_metrics)
            for i in range(5)
        }
        
        # Modify some metrics to create variety
        metrics["file1.py"].architecture.component_coupling = 0.8
        metrics["file2.py"].architecture.abstraction_level = 0.3
        metrics["file3.py"].architecture.layering_violations = 5
        
        pdf_generator._generate_architecture_section(metrics)
        
        # Check for comprehensive analysis
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        assert len(tables) >= 3  # Should have multiple analysis tables

class TestComplexitySection:
    """Test cases for complexity section generation."""
    
    def test_complexity_overview(self, pdf_generator, mock_code_metrics):
        """Test complexity overview generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any("complexity" in str(p).lower() for p in paragraphs)
        
    def test_complexity_metrics_table(self, pdf_generator, mock_code_metrics):
        """Test complexity metrics table generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        metrics_table = tables[0]
        
        # Verify table contains key complexity metrics
        expected_metrics = [
            'Cyclomatic Complexity',
            'Cognitive Complexity',
            'Max Nesting Depth',
            'Maintainability Index'
        ]
        table_data = metrics_table._cellvalues
        assert all(any(metric.lower() in str(row).lower() for row in table_data) 
                  for metric in expected_metrics)
        
    def test_complexity_distribution_chart(self, pdf_generator, mock_code_metrics):
        """Test complexity distribution chart generation."""
        # Create metrics with varied complexity
        metrics = {
            f"file{i}.py": deepcopy(mock_code_metrics)
            for i in range(5)
        }
        
        # Modify complexities to create distribution
        metrics["file1.py"].complexity.cyclomatic_complexity = 5  # Low
        metrics["file2.py"].complexity.cyclomatic_complexity = 15  # Medium
        metrics["file3.py"].complexity.cyclomatic_complexity = 25  # High
        metrics["file4.py"].complexity.cyclomatic_complexity = 35  # Very High
        
        pdf_generator._generate_complexity_section(metrics)
        
        # Check for chart presence
        drawings = [
            item for item in pdf_generator.story 
            if isinstance(item, Drawing)
        ]
        assert len(drawings) > 0
        
    def test_high_complexity_files(self, pdf_generator, mock_code_metrics):
        """Test identification and reporting of high complexity files."""
        # Create metrics with some high complexity files
        metrics = {
            "simple.py": deepcopy(mock_code_metrics),
            "complex.py": deepcopy(mock_code_metrics),
            "very_complex.py": deepcopy(mock_code_metrics)
        }
        
        metrics["complex.py"].complexity.cyclomatic_complexity = 20
        metrics["very_complex.py"].complexity.cyclomatic_complexity = 30
        
        pdf_generator._generate_complexity_section(metrics)
        
        # Check for high complexity files table
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        high_complexity_table = [
            table for table in tables 
            if any("complex.py" in str(row) for row in table._cellvalues)
        ]
        assert len(high_complexity_table) > 0
        
    def test_halstead_metrics_analysis(self, pdf_generator, mock_code_metrics):
        """Test Halstead metrics analysis."""
        # Ensure mock metrics has Halstead data
        mock_code_metrics.complexity.halstead_metrics = {
            'volume': 1000.0,
            'difficulty': 25.0,
            'effort': 25000.0,
            'vocabulary': 50,
            'length': 200
        }
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        # Verify Halstead metrics table
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        halstead_table = [
            table for table in tables 
            if any('Halstead' in str(row) for row in table._cellvalues)
        ]
        assert len(halstead_table) > 0
        
        # Verify metrics are present
        table_content = str(halstead_table[0]._cellvalues)
        for metric in ['Volume', 'Difficulty', 'Effort', 'Vocabulary', 'Length']:
            assert metric in table_content
        
    def test_change_risk_analysis(self, pdf_generator, mock_code_metrics):
        """Test change risk analysis generation."""
        # Create metrics with varied change risks
        metrics = {
            "low_risk.py": deepcopy(mock_code_metrics),
            "high_risk.py": deepcopy(mock_code_metrics)
        }
        
        metrics["high_risk.py"].complexity.change_risk = 85.0
        
        pdf_generator._generate_complexity_section(metrics)
        
        # Check for risk analysis content
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        risk_table = [
            table for table in tables 
            if any("Risk" in str(row) for row in table._cellvalues)
        ]
        assert len(risk_table) > 0
        
    def test_maintainability_analysis(self, pdf_generator, mock_code_metrics):
        """Test maintainability index analysis."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        # Check for maintainability content
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        assert any(
            any("Maintainability" in str(row) for row in table._cellvalues)
            for table in tables
        )
        
    def test_cognitive_complexity_analysis(self, pdf_generator, mock_code_metrics):
        """Test cognitive complexity analysis."""
        # Create metrics with varied cognitive complexity
        metrics = {
            "simple.py": deepcopy(mock_code_metrics),
            "complex.py": deepcopy(mock_code_metrics)
        }
        
        metrics["complex.py"].complexity.cognitive_complexity = 25
        
        pdf_generator._generate_complexity_section(metrics)
        
        # Check for cognitive complexity content
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        assert any(
            any("Cognitive" in str(row) for row in table._cellvalues)
            for table in tables
        )
        
    def test_complexity_recommendations(self, pdf_generator, mock_code_metrics):
        """Test complexity recommendations generation."""
        # Configure metrics to trigger recommendations
        mock_code_metrics.complexity.cyclomatic_complexity = 25
        mock_code_metrics.complexity.cognitive_complexity = 20
        mock_code_metrics.complexity.max_nesting_depth = 6
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        # Verify recommendations
        paragraphs = [p for p in pdf_generator.story if isinstance(p, Paragraph)]
        recommendations = []
        for p in paragraphs:
            if hasattr(p, 'text'):  # Get the actual text content
                if isinstance(p.text, str):
                    recommendations.append(p.text)
                    
        recommendation_text = ' '.join(recommendations)
        assert len(recommendations) > 0
        assert 'complex' in recommendation_text.lower()
        assert 'nest' in recommendation_text.lower()
        
    def test_zero_complexity_handling(self, pdf_generator, mock_code_metrics):
        """Test handling of zero complexity metrics."""
        # Create metrics with zero complexity
        mock_code_metrics.complexity.cyclomatic_complexity = 0
        mock_code_metrics.complexity.cognitive_complexity = 0
        mock_code_metrics.complexity.max_nesting_depth = 0
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_complexity_section(metrics)
        
        # Verify appropriate handling
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        complexity_text = ' '.join(paragraphs)
        assert 'low' in complexity_text.lower()
        assert 'complexity' in complexity_text.lower()

class TestTestingSection:
    """Test cases for testing metrics section generation."""
    
    def test_testing_overview_generation(self, pdf_generator, mock_code_metrics):
        """Test testing overview generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_testing_section(metrics)
        
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        assert any('test' in text.lower() for text in paragraphs)
        
    def test_coverage_metrics_table_generation(self, pdf_generator, mock_code_metrics):
        """Test generation of coverage metrics table."""
        metrics = {
            "file1.py": mock_code_metrics,
            "file2.py": deepcopy(mock_code_metrics)
        }
        
        # Set different test coverage
        metrics["file1.py"].test_coverage_files = {"test_file1.py"}
        metrics["file2.py"].test_coverage_files = set()
        
        pdf_generator._generate_testing_section(metrics)
        
        # Find coverage table
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        coverage_table = tables[0]._cellvalues
        
        # Verify table headers and content
        assert any("Coverage" in row[0] for row in coverage_table)
        assert "1/2" in str(coverage_table)  # One out of two files has tests
        assert "50.0%" in str(coverage_table)  # 50% coverage
        
    def test_coverage_distribution_visualization(self, pdf_generator, mock_code_metrics):
        """Test coverage distribution chart generation."""
        # Create metrics with varied test coverage
        metrics = {
            f"file{i}.py": deepcopy(mock_code_metrics)
            for i in range(5)
        }
        
        # Set up different coverage scenarios
        for i, file_metrics in enumerate(metrics.values()):
            if i < 3:  # 3 files with tests
                file_metrics.test_coverage_files = {f"test_file{i}.py"}
            else:  # 2 files without tests
                file_metrics.test_coverage_files = set()
        
        pdf_generator._generate_testing_section(metrics)
        
        # Check for pie chart
        drawings = [item for item in pdf_generator.story if isinstance(item, Drawing)]
        assert len(drawings) > 0
        
    def test_critical_files_without_tests(self, pdf_generator, mock_code_metrics):
        """Test identification of critical files lacking tests."""
        # Create metrics with some high-complexity untested files
        metrics = {
            "tested.py": deepcopy(mock_code_metrics),
            "untested_complex.py": deepcopy(mock_code_metrics),
            "untested_critical.py": deepcopy(mock_code_metrics)
        }
        
        # Configure test coverage and complexity
        metrics["tested.py"].test_coverage_files = {"test_tested.py"}
        
        metrics["untested_complex.py"].test_coverage_files = set()
        metrics["untested_complex.py"].complexity.cyclomatic_complexity = 20
        
        metrics["untested_critical.py"].test_coverage_files = set()
        metrics["untested_critical.py"].complexity.cyclomatic_complexity = 25
        metrics["untested_critical.py"].security.potential_sql_injections = 2
        
        pdf_generator._generate_testing_section(metrics)
        
        # Check for critical files table
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        critical_files_table = next(
            (table for table in tables 
             if any('critical' in str(row).lower() for row in table._cellvalues)),
            None
        )
        
        assert critical_files_table is not None
        table_content = str(critical_files_table._cellvalues)
        assert 'untested_complex.py' in table_content
        assert 'untested_critical.py' in table_content
        
    def test_code_quality_patterns_analysis(self, pdf_generator, mock_code_metrics):
        """Test analysis of code quality patterns."""
        # Prepare metrics with quality patterns
        mock_code_metrics.code_patterns.update({
            'magic_numbers': 8,
            'large_try_blocks': 2,
            'boolean_traps': 4,
            'long_parameter_list': 3
        })
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_testing_section(metrics)
        
        # Verify patterns table
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        patterns_table = next(
            (table for table in tables 
             if any('Pattern' in str(row) for row in table._cellvalues)),
            None
        )
        
        assert patterns_table is not None
        table_content = str(patterns_table._cellvalues)
        assert 'magic numbers' in table_content.lower()
        assert 'boolean traps' in table_content.lower()
        
    def test_testing_recommendations_generation(self, pdf_generator, mock_code_metrics):
        """Test generation of testing recommendations."""
        metrics = {
            "file1.py": deepcopy(mock_code_metrics),
            "file2.py": deepcopy(mock_code_metrics)
        }
        
        # Set up scenarios that should trigger recommendations
        metrics["file2.py"].test_coverage_files = set()
        metrics["file2.py"].complexity.cyclomatic_complexity = 20
        metrics["file2.py"].change_probability.churn_rate = 0.8
        
        pdf_generator._generate_testing_section(metrics)
        
        # Verify recommendations content
        paragraphs = [item for item in pdf_generator.story if isinstance(item, Paragraph)]
        recommendations_text = ' '.join(str(p) for p in paragraphs)
        
        assert "coverage" in recommendations_text.lower()
        assert "complex" in recommendations_text.lower()
        assert "frequently" in recommendations_text.lower()
        
    def test_zero_test_coverage_handling(self, pdf_generator, mock_code_metrics):
        """Test handling of zero test coverage scenario."""
        # Create metrics with no test coverage
        metrics = {
            f"file{i}.py": deepcopy(mock_code_metrics)
            for i in range(3)
        }
        
        # Remove all test coverage
        for metrics_obj in metrics.values():
            metrics_obj.test_coverage_files = set()
        
        pdf_generator._generate_testing_section(metrics)
        
        # Verify appropriate warnings
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        warning_text = ' '.join(paragraphs)
        
        assert any(
            ('0%' in text or 'no test coverage' in text.lower())
            for text in paragraphs
        )
        
        # Verify recommendations for zero coverage
        assert any(
            ('coverage' in text.lower() and 'recommend' in text.lower())
            for text in paragraphs
        )
        
    def test_full_test_coverage_handling(self, pdf_generator, mock_code_metrics):
        """Test handling of full test coverage scenario."""
        # Create metrics with full test coverage
        metrics = {
            f"file{i}.py": deepcopy(mock_code_metrics)
            for i in range(3)
        }
        
        # Add test coverage for all files
        for metrics_obj in metrics.values():
            metrics_obj.test_coverage_files = {f"test_{metrics_obj.__class__.__name__}.py"}
        
        pdf_generator._generate_testing_section(metrics)
        
        # Find coverage table
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        coverage_table = next(
            table for table in tables 
            if any('Coverage' in str(row) for row in table._cellvalues)
        )
        table_content = str(coverage_table._cellvalues)
        
        # Verify 100% coverage is shown
        assert '100' in table_content
        
        # Verify positive recognition of full coverage
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        content_text = ' '.join(paragraphs)
        assert any(
            text.lower().find('100%') >= 0 or 
            text.lower().find('full coverage') >= 0 or
            text.lower().find('complete coverage') >= 0
            for text in paragraphs
        )
        
    def test_test_quality_assessment(self, pdf_generator, mock_code_metrics):
        """Test quality assessment of existing tests."""
        # Add test quality metrics
        mock_code_metrics.code_patterns.update({
            'large_test_methods': 3,
            'missing_assertions': 2,
            'test_data_duplication': 4
        })
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_testing_section(metrics)
        
        # Verify quality assessment content
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        quality_content = ' '.join(str(table._cellvalues) for table in tables)
        
        assert 'test' in quality_content.lower()
        assert any(
            pattern in quality_content.lower() 
            for pattern in ['large', 'missing', 'duplication']
        )

class TestTrendsSection:
    """Test cases for historical trends section generation."""
    
    def test_trends_overview(self, pdf_generator, mock_code_metrics):
        """Test trends overview generation."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        # Verify overview content
        story_content = [str(item) for item in pdf_generator.story]
        assert any("trend" in text.lower() for text in story_content)
        
    def test_complexity_evolution_chart(self, pdf_generator, mock_code_metrics):
        """Test complexity evolution chart generation."""
        # Update mock history data
        mock_code_metrics.history = [
            {
                'date': '2024-01',
                'avg_complexity': 10.5,
                'avg_cognitive_complexity': 12.0,
                'total_lines': 1000,
                'code_lines': 800,
                'security_issues': 5,
                'architecture_issues': 3
            },
            {
                'date': '2024-02',
                'avg_complexity': 11.0,
                'avg_cognitive_complexity': 13.0,
                'total_lines': 1200,
                'code_lines': 950,
                'security_issues': 4,
                'architecture_issues': 2
            }
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        # Check for chart presence
        drawings = [item for item in pdf_generator.story if isinstance(item, Drawing)]
        assert len(drawings) > 0
        
    def test_code_size_trends(self, pdf_generator, mock_code_metrics):
        """Test code size trends visualization."""
        mock_code_metrics.history = [
            {
                'date': f'2024-{i:02d}',
                'avg_complexity': 10.0 + i,
                'avg_cognitive_complexity': 12.0 + i,
                'total_lines': 1000 + i*100,
                'code_lines': 800 + i*80,
                'security_issues': max(0, 5 - i),
                'architecture_issues': max(0, 3 - i)
            }
            for i in range(4)
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        drawings = [item for item in pdf_generator.story if isinstance(item, Drawing)]
        assert len(drawings) >= 2
        
    def test_issue_trends(self, pdf_generator, mock_code_metrics):
        """Test issue trends analysis."""
        mock_code_metrics.history = [
            {
                'date': f'2024-{i:02d}',
                'avg_complexity': 10.0 + i,
                'avg_cognitive_complexity': 12.0 + i,
                'total_lines': 1000 + i*100,
                'code_lines': 800 + i*80,
                'security_issues': max(0, 5 - i),
                'architecture_issues': max(0, 3 - i)
            }
            for i in range(4)
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        drawings = [item for item in pdf_generator.story if isinstance(item, Drawing)]
        assert len(drawings) >= 3  # Should have complexity, size, and issues charts
        
    def test_trend_analysis_table(self, pdf_generator, mock_code_metrics):
        """Test trend analysis table generation."""
        mock_code_metrics.history = [
            {
                'date': '2024-01',
                'avg_complexity': 10.5,
                'avg_cognitive_complexity': 12.0,
                'total_lines': 1000,
                'code_lines': 800,
                'security_issues': 5,
                'architecture_issues': 3
            },
            {
                'date': '2024-02',
                'avg_complexity': 11.0,
                'avg_cognitive_complexity': 13.0,
                'total_lines': 1200,
                'code_lines': 950,
                'security_issues': 4,
                'architecture_issues': 2
            }
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        tables = [item for item in pdf_generator.story if isinstance(item, Table)]
        assert any('Change' in str(table._cellvalues) for table in tables)
        
    def test_trend_insights(self, pdf_generator, mock_code_metrics):
        """Test trend insights generation."""
        # Ensure history data shows significant changes
        mock_code_metrics.history = [
            {
                'date': '2024-01',
                'avg_complexity': 10.0,
                'avg_cognitive_complexity': 12.0,
                'total_lines': 1000,
                'code_lines': 800,
                'security_issues': 2,
                'architecture_issues': 1
            },
            {
                'date': '2024-02',
                'avg_complexity': 15.0,  # Significant increase
                'avg_cognitive_complexity': 18.0,
                'total_lines': 1500,
                'code_lines': 1200,
                'security_issues': 6,    # Significant increase
                'architecture_issues': 4
            }
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        # Verify insights about significant changes
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        insights_text = ' '.join(paragraphs)
        assert any(('complexity' in text.lower() and 'increase' in text.lower()) 
                  for text in paragraphs)
        
    def test_no_history_handling(self, pdf_generator, mock_code_metrics):
        """Test handling of missing historical data."""
        mock_code_metrics.history = []
        metrics = {"file1.py": mock_code_metrics}
        
        pdf_generator._generate_trends_section(metrics)
        
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        assert any('no historical data' in text.lower() for text in paragraphs)
        
    def test_trend_calculation(self, pdf_generator, mock_code_metrics):
        """Test trend calculation and categorization."""
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        tables = [
            item for item in pdf_generator.story 
            if isinstance(item, Table)
        ]
        trends_table = [
            table for table in tables 
            if any("Change" in str(row) for row in table._cellvalues)
        ]
        assert len(trends_table) > 0
        
        # Verify trend calculations
        table_data = trends_table[0]._cellvalues
        assert any(
            any(("+" in str(cell) or "-" in str(cell)) and "%" in str(cell) 
                for cell in row)
            for row in table_data
        )
        
    def test_rapid_growth_warning(self, pdf_generator, mock_code_metrics):
        """Test detection and warning of rapid codebase growth."""
        mock_code_metrics.history = [
            {
                'date': '2024-01',
                'total_lines': 1000,
                'code_lines': 800,
                'avg_complexity': 10.0,
                'security_issues': 2,
                'architecture_issues': 1
            },
            {
                'date': '2024-02',
                'total_lines': 2000,  # 100% growth
                'code_lines': 1600,
                'avg_complexity': 12.0,
                'security_issues': 3,
                'architecture_issues': 2
            }
        ]
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        paragraphs = [str(item) for item in pdf_generator.story if isinstance(item, Paragraph)]
        assert any(('rapid' in text.lower() and 'growth' in text.lower()) 
                  for text in paragraphs)
        
    def test_multiple_trend_periods(self, pdf_generator, mock_code_metrics):
        """Test analysis of multiple trend periods."""
        # Add more historical periods
        mock_code_metrics.history.extend([
            {
                'date': date,
                'avg_complexity': 12.0 + i,
                'avg_cognitive_complexity': 15.0 + i,
                'total_lines': 1600 + i*200,
                'code_lines': 1300 + i*150,
                'security_issues': max(0, 3 - i),
                'architecture_issues': max(0, 2 - i)
            }
            for i, date in enumerate(['2024-05', '2024-06', '2024-07'])
        ])
        
        metrics = {"file1.py": mock_code_metrics}
        pdf_generator._generate_trends_section(metrics)
        
        # Verify longer-term trend analysis
        drawings = [
            item for item in pdf_generator.story 
            if isinstance(item, Drawing)
        ]
        assert len(drawings) >= 3  # Should have multiple trend charts
        
        # Check for long-term trend insights
        paragraphs = [
            item for item in pdf_generator.story 
            if isinstance(item, Paragraph)
        ]
        assert any(
            "trend" in str(p).lower() and ("increase" in str(p).lower() or 
            "decrease" in str(p).lower())
            for p in paragraphs
        )