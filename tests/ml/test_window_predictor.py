import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from datetime import time
from sklearn.ensemble import RandomForestRegressor

from code_analyzer.ml.window_predictor import WindowPredictor
from code_analyzer.ml.base_explainer import MLExplainer
from code_analyzer.models.metrics import DeploymentWindow

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
def predictor() -> WindowPredictor:
    """Create a WindowPredictor instance for testing."""
    return WindowPredictor()

@pytest.fixture
def mock_ml_explainer():
    explainer = Mock(spec=MLExplainer)
    explainer.explain_prediction.return_value = {
        'feature_importance': {
            'cyclomatic_complexity_mean': 0.3,
            'security_issues': 0.2,
            'test_coverage': 0.2,
            'hour_of_day': 0.15,
            'team_size_ratio': 0.15
        },
        'top_contributors': [
            {'feature': 'cyclomatic_complexity_mean', 'impact': 0.3, 'deviation': 1.8},
            {'feature': 'security_issues', 'impact': 0.2, 'deviation': 1.2},
            {'feature': 'test_coverage', 'impact': -0.2, 'deviation': -1.5}
        ],
        'explanation': 'Test window prediction explanation',
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
        {'features': ('security_issues', 'hour_of_day'), 'strength': 0.3}
    ]
    return explainer

class TestWindowPredictor:
    def test_init(self):
        """Test initialization of WindowPredictor."""
        predictor = WindowPredictor()
        assert isinstance(predictor.model, RandomForestRegressor)
        assert isinstance(predictor.explainer, MLExplainer)
        assert predictor.explainer.model_type == "regressor"
        assert len(predictor.feature_names) > 0
        assert predictor.min_samples_for_training > 0
        assert predictor.training_frequency > 0

    def test_extract_features(self, sample_metrics):
        """Test feature extraction from metrics."""
        predictor = WindowPredictor()
        features = predictor._extract_features(sample_metrics, hour=10)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(predictor.feature_names),)
        assert np.all(features >= 0)
        assert features[-1] == 10  # hour feature
        
        # Test empty metrics
        empty_features = predictor._extract_features({'files': {}}, hour=12)
        assert np.all(empty_features == 0)

    def test_calculate_window_risk(self, predictor, sample_metrics):
        """Test window risk calculation."""
        # Test business hours
        business_risk = predictor._calculate_window_risk(10, sample_metrics)
        assert 0 <= business_risk <= 1
        assert business_risk > 0.5  # Higher risk during business hours
        
        # Test off hours
        off_hours_risk = predictor._calculate_window_risk(3, sample_metrics)
        assert 0 <= off_hours_risk <= 1
        assert off_hours_risk < business_risk  # Lower risk during off hours

    def test_calculate_team_availability(self, predictor, sample_team_availability):
        """Test team availability calculation."""
        # Test during working hours
        avail = predictor._calculate_team_availability(
            sample_team_availability,
            time(10),
            time(12)
        )
        assert 0 <= avail <= 1
        assert avail > 0  # Some team members should be available
        
        # Test outside working hours
        off_hours_avail = predictor._calculate_team_availability(
            sample_team_availability,
            time(20),
            time(22)
        )
        assert off_hours_avail == 0  # No team members available

    def test_predict_optimal_windows(
        self,
        predictor: WindowPredictor,
        sample_metrics: Dict[str, Any],
        sample_team_availability: Dict[str, List[time]],
        mock_ml_explainer: MLExplainer
    ):
        """Test full window prediction with explainability."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([0.8]))
        
        windows = predictor.predict_optimal_windows(
            sample_metrics,
            sample_team_availability
        )
        
        assert isinstance(windows, list)
        assert len(windows) > 0
        assert all(isinstance(w, DeploymentWindow) for w in windows)
        
        # Test window attributes
        window = windows[0]
        assert isinstance(window.start_time, time)
        assert isinstance(window.end_time, time)
        assert 0 <= window.risk_score <= 1
        assert 0 <= window.team_availability <= 1
        assert 0 <= window.historical_success_rate <= 1
        
        # Test required resources
        assert window.required_team_size >= 2
        assert isinstance(window.required_skills, list)
        assert window.estimated_duration >= 1.0
        
        # Test environment metrics
        assert isinstance(window.system_load, dict)
        assert isinstance(window.concurrent_deployments, list)
        
        # Test explainability
        assert isinstance(window.explanation, str)
        assert isinstance(window.feature_importance, dict)
        assert isinstance(window.top_contributors, list)
        assert isinstance(window.feature_interactions, list)
        
        # Test confidence metrics
        assert 0 <= window.confidence_score <= 1
        assert 0 <= window.data_completeness <= 1
        assert isinstance(window.prediction_quality, dict)

    def test_invalid_metrics_handling(
        self,
        predictor: WindowPredictor,
        sample_team_availability: Dict[str, List[time]],
        mock_ml_explainer: MLExplainer
    ):
        """Test handling of invalid metrics."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([0.7]))
        
        # Test with empty metrics
        empty_windows = predictor.predict_optimal_windows(
            {'files': {}},
            sample_team_availability
        )
        assert isinstance(empty_windows, list)
        if empty_windows:
            assert empty_windows[0].confidence_score < 0.5
        
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
        none_windows = predictor.predict_optimal_windows(
            none_metrics,
            sample_team_availability
        )
        assert isinstance(none_windows, list)
        if none_windows:
            assert none_windows[0].data_completeness == 0

    def test_update_models_if_needed(self, predictor: WindowPredictor):
        """Test model updating mechanism."""
        # Create mock deployment feedback
        mock_feedback = Mock(
            metrics={'files': {}},
            success=True
        )
        
        # Test with insufficient history
        predictor.deployment_history = [mock_feedback] * (predictor.min_samples_for_training - 1)
        predictor._update_models_if_needed()
        
        # Test with sufficient history
        predictor.deployment_history = [mock_feedback] * (predictor.min_samples_for_training + 1)
        with patch.object(RandomForestRegressor, 'fit') as mock_fit:
            predictor._update_models_if_needed()
            # Should be called because we have enough samples and it's a multiple of training_frequency
            if len(predictor.deployment_history) % predictor.training_frequency == 0:
                mock_fit.assert_called_once()

    def test_model_persistence(self, predictor: WindowPredictor, tmp_path):
        """Test model saving and loading."""
        predictor.MODEL_PATH = tmp_path / "test_model.joblib"
        
        # Test save
        predictor._save_model()
        assert predictor.MODEL_PATH.exists()
        
        # Test load
        loaded_predictor = WindowPredictor()
        loaded_predictor.MODEL_PATH = predictor.MODEL_PATH
        loaded_model = loaded_predictor._load_or_create_model()
        assert isinstance(loaded_model, RandomForestRegressor)

    def test_helper_methods(self, predictor: WindowPredictor, sample_metrics: Dict[str, Any]):
        """Test various helper calculation methods."""
        # Test test coverage calculation
        coverage = predictor._calculate_test_coverage(sample_metrics)
        assert 0 <= coverage <= 1
        
        # Test duplication ratio calculation
        dup_ratio = predictor._calculate_duplication_ratio(sample_metrics)
        assert dup_ratio > 0
        
        # Test team size estimation
        team_size = predictor._estimate_team_size(sample_metrics)
        assert team_size >= 2
        
        # Test skills identification
        skills = predictor._identify_required_skills(sample_metrics)
        assert isinstance(skills, list)
        assert len(skills) > 0
        
        # Test duration estimation
        duration = predictor._estimate_duration(sample_metrics)
        assert duration >= 2.0

if __name__ == '__main__':
    pytest.main([__file__])