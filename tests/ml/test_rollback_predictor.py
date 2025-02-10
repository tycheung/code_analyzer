import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

from code_analyzer.ml.rollback_predictor import RollbackPredictor
from code_analyzer.ml.base_explainer import MLExplainer
from code_analyzer.models.metrics import RollbackPrediction, DeploymentWindow

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
                    'component_coupling': 0.6,
                    'circular_dependencies': []
                },
                'metrics': Mock(
                    lines_code=150,
                    lines_comment=30,
                    test_coverage_files={'test_app.py'},
                    change_probability=Mock(
                        last_modified=datetime.now() - timedelta(days=5),
                        churn_rate=0.4,
                        contributors={'dev1', 'dev2'}
                    )
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
                    'component_coupling': 0.4,
                    'circular_dependencies': []
                },
                'metrics': Mock(
                    lines_code=80,
                    lines_comment=25,
                    test_coverage_files=set(),
                    change_probability=Mock(
                        last_modified=datetime.now() - timedelta(days=15),
                        churn_rate=0.2,
                        contributors={'dev2', 'dev3'}
                    )
                )
            }
        }
    }

@pytest.fixture
def sample_deployment_window() -> DeploymentWindow:
    return Mock(
        spec=DeploymentWindow,
        risk_score=0.5,
        team_availability=0.7,
        historical_success_rate=0.8
    )

@pytest.fixture
def predictor() -> RollbackPredictor:
    """Create a RollbackPredictor instance for testing."""
    return RollbackPredictor()

@pytest.fixture
def mock_ml_explainer():
    explainer = Mock(spec=MLExplainer)
    explainer.explain_prediction.return_value = {
        'feature_importance': {
            'cyclomatic_complexity_mean': 0.3,
            'security_vulnerabilities': 0.2,
            'test_coverage': 0.2,
            'deploy_window_risk': 0.15,
            'team_availability': 0.15
        },
        'top_contributors': [
            {'feature': 'cyclomatic_complexity_mean', 'impact': 0.3, 'deviation': 1.8},
            {'feature': 'security_vulnerabilities', 'impact': 0.2, 'deviation': 1.2},
            {'feature': 'test_coverage', 'impact': -0.2, 'deviation': -1.5}
        ],
        'explanation': 'Test rollback prediction explanation',
        'confidence_factors': {
            'extreme_values': 1,
            'importance_concentration': 0.3,
            'confidence_penalty': 0.1,
            'confidence_factors': {
                'balanced_feature_contribution': True,
                'extreme_values_present': False,
                'high_importance_concentration': False
            }
        }
    }
    explainer.get_feature_interactions.return_value = [
        {'features': ('cyclomatic_complexity_mean', 'test_coverage'), 'strength': 0.4},
        {'features': ('security_vulnerabilities', 'deploy_window_risk'), 'strength': 0.3}
    ]
    return explainer

class TestRollbackPredictor:
    def test_init(self):
        """Test initialization of RollbackPredictor."""
        predictor = RollbackPredictor()
        assert isinstance(predictor.model, RandomForestClassifier)
        assert isinstance(predictor.explainer, MLExplainer)
        assert predictor.explainer.model_type == "classifier"
        assert len(predictor.feature_names) > 0
        assert predictor.min_samples_for_training > 0
        assert predictor.training_frequency > 0

    def test_extract_features(self, sample_metrics, sample_deployment_window):
        """Test feature extraction from metrics."""
        predictor = RollbackPredictor()
        features = predictor._extract_features(sample_metrics, sample_deployment_window)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(predictor.feature_names),)
        assert np.all(features >= 0)
        
        # Test window-specific features
        assert features[6] == sample_deployment_window.risk_score
        assert features[7] == sample_deployment_window.team_availability
        
        # Test empty metrics
        empty_features = predictor._extract_features({'files': {}}, sample_deployment_window)
        assert np.all(empty_features == 0)

    def test_predict_rollback(
        self,
        predictor: RollbackPredictor,
        sample_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test full rollback prediction with explainability."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        result = predictor.predict_rollback(sample_metrics, sample_deployment_window)
        
        # Test core prediction attributes
        assert isinstance(result, RollbackPrediction)
        assert 0 <= result.probability <= 1
        assert isinstance(result.risk_factors, dict)
        assert isinstance(result.severity_level, str)
        assert isinstance(result.critical_files, list)
        
        # Test mitigation and review attributes
        assert isinstance(result.mitigation_suggestions, list)
        assert isinstance(result.recommended_reviewers, list)
        
        # Test confidence attributes
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.prediction_quality, dict)
        assert 0 <= result.data_completeness <= 1
        
        # Test explainability attributes
        assert isinstance(result.explanation, str)
        assert isinstance(result.feature_importance, dict)
        assert isinstance(result.top_contributors, list)
        assert isinstance(result.feature_interactions, list)
        
        # Test historical context
        assert isinstance(result.similar_cases, list)
        assert isinstance(result.historical_pattern, dict)

    def test_identify_critical_files(
        self,
        predictor: RollbackPredictor,
        sample_metrics: Dict[str, Any]
    ):
        """Test identification of critical files."""
        critical_files = predictor._identify_critical_files(
            sample_metrics,
            [{'feature': 'cyclomatic_complexity', 'deviation': 2.0}]
        )
        
        assert isinstance(critical_files, list)
        assert len(critical_files) <= 5  # Should limit to top 5
        assert 'app.py' in critical_files  # Should identify high-risk file

    def test_calculate_comprehensive_confidence(
        self,
        predictor: RollbackPredictor,
        sample_metrics: Dict[str, Any],
        mock_ml_explainer: MLExplainer
    ):
        """Test confidence score calculation."""
        confidence = predictor._calculate_comprehensive_confidence(
            sample_metrics,
            mock_ml_explainer.explain_prediction.return_value,
            5  # number of similar cases
        )
        
        assert 0 <= confidence <= 1
        
        # Test with extreme values
        mock_ml_explainer.explain_prediction.return_value['confidence_factors'] = {
            'extreme_values': 3,
            'importance_concentration': 0.6,
            'confidence_penalty': 0.2
        }
        reduced_confidence = predictor._calculate_comprehensive_confidence(
            sample_metrics,
            mock_ml_explainer.explain_prediction.return_value,
            5
        )
        assert reduced_confidence < confidence

    def test_analyze_historical_pattern(self, predictor: RollbackPredictor):
        """Test historical pattern analysis."""
        mock_deployments = [
            Mock(
                rollback_occurred=True,
                actual_deployment_time=2.5,
                actual_support_hours=4.0
            ),
            Mock(
                rollback_occurred=False,
                actual_deployment_time=1.8,
                actual_support_hours=2.5
            )
        ]
        
        pattern = predictor._analyze_historical_pattern(mock_deployments)
        
        assert isinstance(pattern, dict)
        assert 'historical_rollback_rate' in pattern
        assert 'avg_deployment_time' in pattern
        assert 'avg_support_hours' in pattern
        assert pattern['historical_rollback_rate'] == 0.5

    def test_assess_prediction_quality(
        self,
        predictor: RollbackPredictor,
        mock_ml_explainer: MLExplainer
    ):
        """Test prediction quality assessment."""
        quality = predictor._assess_prediction_quality(
            mock_ml_explainer.explain_prediction.return_value
        )
        
        assert isinstance(quality, dict)
        assert 'sufficient_data' in quality
        assert 'balanced_features' in quality
        assert 'normal_value_ranges' in quality
        assert 'stable_prediction' in quality
        assert all(isinstance(v, bool) for v in quality.values())

    def test_update_models_if_needed(
        self,
        predictor: RollbackPredictor,
        sample_metrics: Dict[str, Any]
    ):
        """Test model updating mechanism."""
        mock_feedback = Mock(
            metrics=sample_metrics,
            rollback_occurred=True
        )
        
        # Test with insufficient history
        predictor.deployment_history = [mock_feedback] * (predictor.min_samples_for_training - 1)
        predictor._update_models_if_needed()
        
        # Test with sufficient history
        predictor.deployment_history = [mock_feedback] * (predictor.min_samples_for_training + 1)
        with patch.object(RandomForestClassifier, 'fit') as mock_fit:
            predictor._update_models_if_needed()
            mock_fit.assert_called_once()
            X, y = mock_fit.call_args[0]
            assert X.shape[1] == len(predictor.feature_names)
            assert len(y) == len(predictor.deployment_history)

    def test_invalid_metrics_handling(
        self,
        predictor: RollbackPredictor,
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test handling of invalid metrics."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict_proba = Mock(return_value=np.array([[0.7, 0.3]]))
        
        # Test with empty metrics
        empty_result = predictor.predict_rollback(
            {'files': {}},
            sample_deployment_window
        )
        assert isinstance(empty_result, RollbackPrediction)
        assert empty_result.probability <= 0.5  # Should be conservative
        
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
        none_result = predictor.predict_rollback(
            none_metrics,
            sample_deployment_window
        )
        assert isinstance(none_result.risk_factors, dict)
        
        # Test with missing fields
        incomplete_metrics = {
            'files': {
                'test.py': {}
            }
        }
        incomplete_result = predictor.predict_rollback(
            incomplete_metrics,
            sample_deployment_window
        )
        assert isinstance(incomplete_result.mitigation_suggestions, list)

    def test_data_completeness(
        self,
        predictor: RollbackPredictor,
        sample_metrics: Dict[str, Any]
    ):
        """Test data completeness calculation."""
        completeness = predictor._calculate_data_completeness(sample_metrics)
        assert 0 <= completeness <= 1
        
        # Test with missing fields
        incomplete_metrics = {
            'files': {
                'test.py': {
                    'complexity': {'cyclomatic_complexity': 10}
                }
            }
        }
        incomplete_completeness = predictor._calculate_data_completeness(incomplete_metrics)
        assert incomplete_completeness < completeness

    def test_model_persistence(self, predictor: RollbackPredictor, tmp_path):
        """Test model saving and loading."""
        predictor.MODEL_PATH = tmp_path / "test_model.joblib"
        
        # Test save
        predictor._save_model()
        assert predictor.MODEL_PATH.exists()
        
        # Test load
        loaded_predictor = RollbackPredictor()
        loaded_predictor.MODEL_PATH = predictor.MODEL_PATH
        loaded_model = loaded_predictor._load_or_create_model()
        assert isinstance(loaded_model, RandomForestClassifier)

if __name__ == '__main__':
    pytest.main([__file__])