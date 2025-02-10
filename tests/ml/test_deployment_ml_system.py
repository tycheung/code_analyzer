import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from datetime import time
from pathlib import Path

from code_analyzer.ml.deployment_ml_system import DeploymentMLSystem
from code_analyzer.models.metrics import (
    DeploymentWindow, RollbackPrediction, ResourceAllocation,
    IncidentPrediction, DeploymentFeedback, DeploymentAnalysis
)

@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    return {
        'files': {
            'app.py': {
                'complexity': {
                    'cyclomatic_complexity': 15,
                    'cognitive_complexity': 12,
                    'change_risk': 0.7
                },
                'security': {
                    'potential_sql_injections': 1,
                    'hardcoded_secrets': 0,
                    'vulnerable_imports': ['unsafe_lib']
                },
                'architecture': {
                    'component_coupling': 0.6
                },
                'metrics': Mock(
                    lines_code=150,
                    lines_comment=30,
                    test_coverage_files={'test_app.py'}
                )
            },
            'utils.py': {
                'complexity': {
                    'cyclomatic_complexity': 8,
                    'cognitive_complexity': 6,
                    'change_risk': 0.3
                },
                'security': {
                    'potential_sql_injections': 0,
                    'hardcoded_secrets': 1,
                    'vulnerable_imports': []
                },
                'architecture': {
                    'component_coupling': 0.4
                },
                'metrics': Mock(
                    lines_code=80,
                    lines_comment=25,
                    test_coverage_files=set()
                )
            }
        },
        'summary': {
            'security_issues': 2,
            'duplication': {
                'total_duplicates': 15
            }
        }
    }

@pytest.fixture
def sample_team_availability() -> Dict[str, List[time]]:
    return {
        'dev1': [(time(9), time(17))],
        'dev2': [(time(10), time(18))],
        'dev3': [(time(8), time(16))]
    }

@pytest.fixture
def deployment_ml() -> DeploymentMLSystem:
    """Create a DeploymentMLSystem instance for testing."""
    return DeploymentMLSystem()

@pytest.fixture
def mock_window() -> Mock:
    window = Mock(spec=DeploymentWindow)
    window.start_time = time(10)
    window.end_time = time(12)
    window.risk_score = 0.3
    window.team_availability = 0.8
    window.historical_success_rate = 0.85
    window.feature_importance = {'hour_of_day': 0.5, 'team_size': 0.5}
    window.top_contributors = [
        {'feature': 'hour_of_day', 'impact': 0.5, 'deviation': 1.2}
    ]
    window.explanation = "Test window explanation"
    window.confidence_score = 0.8
    window.data_completeness = 0.9
    window.prediction_quality = {
        'sufficient_data': True,
        'balanced_features': True
    }
    return window

class TestDeploymentMLSystem:
    def test_init(self):
        """Test initialization of DeploymentMLSystem."""
        system = DeploymentMLSystem()
        
        assert system.window_predictor is not None
        assert system.rollback_predictor is not None
        assert system.resource_predictor is not None
        assert system.incident_predictor is not None
        assert system.version is not None
        assert system.min_confidence_threshold > 0
        assert isinstance(system.metrics_history, list)

    def test_record_deployment_feedback(
        self,
        deployment_ml: DeploymentMLSystem,
        sample_metrics: Dict[str, Any]
    ):
        """Test recording deployment feedback."""
        feedback = Mock(spec=DeploymentFeedback)
        feedback.metrics = sample_metrics
        feedback.success = True
        feedback.rollback_occurred = False
        feedback.issues_encountered = []
        
        deployment_ml.record_deployment_feedback(feedback)
        
        # Check if feedback was recorded in all predictors
        assert len(deployment_ml.window_predictor.deployment_history) == 1
        assert len(deployment_ml.rollback_predictor.deployment_history) == 1
        assert len(deployment_ml.resource_predictor.deployment_history) == 1
        assert len(deployment_ml.incident_predictor.deployment_history) == 1

    def test_analyze_deployment(
        self,
        deployment_ml: DeploymentMLSystem,
        sample_metrics: Dict[str, Any],
        sample_team_availability: Dict[str, List[time]],
        mock_window: Mock
    ):
        """Test full deployment analysis."""
        # Mock predictor outputs
        deployment_ml.window_predictor.predict_optimal_windows = Mock(
            return_value=[mock_window]
        )
        deployment_ml.rollback_predictor.predict_rollback = Mock(
            return_value=Mock(
                probability=0.3,
                confidence_score=0.8,
                feature_importance={'complexity': 0.5},
                explanation="Test rollback explanation",
                top_contributors=[{'feature': 'complexity', 'impact': 0.5, 'deviation': 1.0}]
            )
        )
        deployment_ml.resource_predictor.predict_resources = Mock(
            return_value=Mock(
                recommended_team_size=3,
                confidence_score=0.7,
                feature_importance={'team_size': 0.6},
                explanation="Test resource explanation",
                top_contributors=[{'feature': 'team_size', 'impact': 0.6, 'deviation': 0.8}]
            )
        )
        deployment_ml.incident_predictor.predict_incidents = Mock(
            return_value=Mock(
                probability=0.2,
                confidence_score=0.75,
                feature_importance={'security': 0.4},
                explanation="Test incident explanation",
                top_contributors=[{'feature': 'security', 'impact': 0.4, 'deviation': 0.5}]
            )
        )
        
        result = deployment_ml.analyze_deployment(
            sample_metrics,
            sample_team_availability
        )
        
        # Test result structure
        assert isinstance(result, DeploymentAnalysis)
        assert len(result.optimal_windows) > 0
        assert result.rollback_prediction is not None
        assert result.resource_prediction is not None
        assert result.incident_prediction is not None
        assert 0 <= result.overall_confidence <= 1
        assert isinstance(result.confidence_breakdown, dict)
        assert isinstance(result.feature_importances, dict)
        assert isinstance(result.system_explanation, str)
        assert isinstance(result.key_insights, list)

    def test_invalid_metrics_handling(
        self,
        deployment_ml: DeploymentMLSystem,
        sample_team_availability: Dict[str, List[time]]
    ):
        """Test handling of invalid metrics."""
        # Test with empty metrics
        empty_result = deployment_ml.analyze_deployment(
            {'files': {}},
            sample_team_availability
        )
        assert empty_result.error_message is not None
        assert empty_result.overall_confidence == 0.0
        
        # Test with None values
        none_metrics = {
            'files': {
                'test.py': {
                    'complexity': None,
                    'security': None,
                    'architecture': None
                }
            }
        }
        none_result = deployment_ml.analyze_deployment(
            none_metrics,
            sample_team_availability
        )
        assert none_result.error_message is not None
        assert none_result.data_completeness == 0.0

    def test_model_persistence(self, deployment_ml: DeploymentMLSystem, tmp_path: Path):
        """Test model saving and loading."""
        # Create mock models
        deployment_ml.window_predictor.model = Mock()
        deployment_ml.rollback_predictor.model = Mock()
        deployment_ml.resource_predictor.model = Mock()
        deployment_ml.incident_predictor.model = Mock()
        
        # Test save
        test_dir = tmp_path / "test_models"
        deployment_ml.save_models(str(test_dir))
        assert (test_dir / "window_model.joblib").exists()
        assert (test_dir / "rollback_model.joblib").exists()
        assert (test_dir / "resource_model.joblib").exists()
        assert (test_dir / "incident_model.joblib").exists()
        
        # Test load
        new_system = DeploymentMLSystem()
        new_system.load_models(str(test_dir))
        assert new_system.window_predictor.model is not None
        assert new_system.rollback_predictor.model is not None
        assert new_system.resource_predictor.model is not None
        assert new_system.incident_predictor.model is not None

    def test_system_metrics(
        self,
        deployment_ml: DeploymentMLSystem,
        sample_metrics: Dict[str, Any]
    ):
        """Test system metrics calculation."""
        # Add some deployment history
        feedback = Mock(spec=DeploymentFeedback)
        feedback.metrics = sample_metrics
        feedback.success = True
        feedback.rollback_occurred = False
        feedback.issues_encountered = []
        feedback.confidence_score = 0.8
        
        deployment_ml.record_deployment_feedback(feedback)
        
        # Test accuracy calculations
        accuracies = deployment_ml._calculate_predictor_accuracies()
        assert isinstance(accuracies, dict)
        assert all(0 <= v <= 1 for v in accuracies.values())
        
        # Test confidence calculations
        confidence = deployment_ml._calculate_confidence_scores()
        assert isinstance(confidence, dict)
        assert all(0 <= v <= 1 for v in confidence.values())
        
        # Test feature coverage
        coverage = deployment_ml._calculate_feature_coverage()
        assert isinstance(coverage, dict)
        assert all(0 <= v <= 1 for v in coverage.values())

    def test_error_handling(
        self,
        deployment_ml: DeploymentMLSystem,
        sample_team_availability: Dict[str, List[time]]
    ):
        """Test error handling in various scenarios."""
        # Test with invalid team availability
        invalid_avail = {'dev1': 'invalid'}
        result = deployment_ml.analyze_deployment(
            {'files': {}},
            invalid_avail
        )
        assert result.error_message is not None
        assert result.overall_confidence == 0.0
        
        # Test with missing required fields
        missing_fields = {
            'wrong_key': {}
        }
        result = deployment_ml.analyze_deployment(
            missing_fields,
            sample_team_availability
        )
        assert result.error_message is not None
        assert result.data_completeness == 0.0

    def test_feature_importance_aggregation(self, deployment_ml: DeploymentMLSystem):
        """Test feature importance aggregation."""
        importances = {
            'model1': {'feature1': 0.5, 'feature2': 0.3},
            'model2': {'feature1': 0.4, 'feature3': 0.6}
        }
        
        result = deployment_ml._aggregate_feature_importances(importances)
        
        assert 'cross_model' in result
        assert 'by_model' in result
        assert result['cross_model']['feature1'] == pytest.approx(0.45)
        assert len(result['cross_model']) == 3

if __name__ == '__main__':
    pytest.main([__file__])