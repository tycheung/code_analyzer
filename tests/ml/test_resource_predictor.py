import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor

from code_analyzer.ml.resource_predictor import ResourcePredictor
from code_analyzer.ml.base_explainer import MLExplainer
from code_analyzer.models.metrics import ResourceAllocation, DeploymentWindow

@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    return {
        'files': {
            'file1.py': {
                'complexity': {
                    'cyclomatic_complexity': 12,
                    'cognitive_complexity': 8,
                    'change_risk': 0.6
                },
                'security': {
                    'vulnerable_imports': ['insecure_lib'],
                    'potential_sql_injections': 1
                },
                'architecture': {
                    'component_coupling': 0.5,
                    'layering_violations': 1
                },
                'metrics': Mock(
                    lines_code=100,
                    lines_comment=20,
                    test_coverage_files={'test_file1.py'},
                    dependencies={'dep1', 'dep2'}
                )
            },
            'file2.py': {
                'complexity': {
                    'cyclomatic_complexity': 18,
                    'cognitive_complexity': 15,
                    'change_risk': 0.9
                },
                'security': {
                    'vulnerable_imports': [],
                    'potential_sql_injections': 0
                },
                'architecture': {
                    'component_coupling': 0.8,
                    'layering_violations': 2
                },
                'metrics': Mock(
                    lines_code=200,
                    lines_comment=30,
                    test_coverage_files=set(),
                    dependencies={'dep2', 'dep3'}
                )
            }
        },
        'summary': {
            'security_issues': 2
        }
    }

@pytest.fixture
def sample_deployment_window() -> DeploymentWindow:
    window = Mock(spec=DeploymentWindow)
    window.risk_score = 0.6
    window.team_availability = 0.8
    return window

@pytest.fixture
def sample_similar_deployments() -> List[Mock]:
    deployments = []
    for hours in [10, 15, 12]:
        deployment = Mock()
        deployment.actual_support_hours = hours
        deployments.append(deployment)
    return deployments

@pytest.fixture
def predictor() -> ResourcePredictor:
    """Create a ResourcePredictor instance for testing."""
    return ResourcePredictor()

@pytest.fixture
def mock_ml_explainer():
    explainer = Mock(spec=MLExplainer)
    explainer.explain_prediction.return_value = {
        'feature_importance': {
            'cyclomatic_complexity': 0.5,
            'security_issues': 0.3,
            'component_coupling': 0.2
        },
        'top_contributors': [
            {'feature': 'cyclomatic_complexity', 'impact': 0.5, 'deviation': 1.0},
            {'feature': 'security_issues', 'impact': 0.3, 'deviation': 0.8},
            {'feature': 'component_coupling', 'impact': 0.2, 'deviation': 1.6}
        ],
        'explanation': 'Test explanation',
        'confidence_factors': {
            'extreme_values': 0,
            'importance_concentration': 0.4,
            'confidence_penalty': 0.1
        }
    }
    explainer.get_feature_interactions.return_value = [
        {'features': ('cyclomatic_complexity', 'security_issues'), 'strength': 0.5}
    ]
    return explainer

class TestResourcePredictor:
    def test_init(self):
        """Test initialization of ResourcePredictor with explainer."""
        predictor = ResourcePredictor()
        assert isinstance(predictor.model, RandomForestRegressor)
        assert isinstance(predictor.explainer, MLExplainer)
        assert predictor.explainer.model_type == "regressor"
        assert hasattr(predictor, 'feature_names')
        assert len(predictor.feature_names) > 0

    def test_extract_features(self, sample_metrics, sample_deployment_window):
        """Test feature extraction from metrics."""
        predictor = ResourcePredictor()
        features = predictor._extract_features(sample_metrics, sample_deployment_window)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(predictor.feature_names),)
        assert np.all(features >= 0)
        
        # Test with window values
        assert features[6] == sample_deployment_window.risk_score  # deployment_window_risk
        assert features[7] == sample_deployment_window.team_availability  # team_availability
        
        # Test empty metrics
        empty_features = predictor._extract_features({'files': {}}, sample_deployment_window)
        assert np.all(empty_features == 0)

    def test_identify_required_skills(self, sample_metrics):
        """Test identification of required skills based on feature importance."""
        predictor = ResourcePredictor()
        
        skills = predictor._identify_required_skills(sample_metrics)
        
        assert isinstance(skills, set)
        assert 'senior_developer' in skills  # Due to high cyclomatic complexity
        assert 'security_expert' in skills   # Due to security issues
        assert 'architecture_specialist' in skills  # Due to high coupling
        assert 'domain_expert' in skills     # Due to high change risk
        assert 'test_engineer' in skills     # Due to test coverage

    def test_estimate_support_duration(self, sample_metrics, sample_similar_deployments):
        """Test support duration estimation with and without similar deployments."""
        predictor = ResourcePredictor()
        
        # Test with similar deployments
        duration_with_similar = predictor._estimate_support_duration(
            sample_metrics,
            sample_similar_deployments
        )
        assert duration_with_similar == pytest.approx(12.33, rel=0.1)
        
        # Test without similar deployments
        duration_without_similar = predictor._estimate_support_duration(
            sample_metrics,
            []
        )
        assert duration_without_similar > 2.0  # Base hours
        assert isinstance(duration_without_similar, float)

        # Test with invalid similar deployments
        invalid_deployments = [Mock()]  # Missing actual_support_hours
        fallback_duration = predictor._estimate_support_duration(
            sample_metrics,
            invalid_deployments
        )
        assert fallback_duration > 0

    def test_predict_resources(
        self,
        predictor: ResourcePredictor,
        sample_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test full resource prediction with explainability."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([3.5]))
        
        result = predictor.predict_resources(sample_metrics, sample_deployment_window)
        
        # Test basic attributes
        assert isinstance(result, ResourceAllocation)
        assert result.recommended_team_size == 4  # Should round up from 3.5
        assert isinstance(result.required_skills, set)
        assert isinstance(result.estimated_support_duration, float)
        assert 0 <= result.confidence_score <= 1
        
        # Test explainability outputs
        assert isinstance(result.explanation, str)
        assert isinstance(result.feature_importance, dict)
        assert isinstance(result.top_contributors, list)
        assert isinstance(result.feature_interactions, list)
        
        # Test specific skills based on metrics
        assert 'architecture_specialist' in result.required_skills  # Due to high coupling
        assert 'security_expert' in result.required_skills  # Due to security issues
        
        # Test content details
        assert len(result.feature_importance) > 0
        assert all(isinstance(v, float) for v in result.feature_importance.values())
        assert all(isinstance(c, dict) for c in result.top_contributors)
        assert all(isinstance(i, dict) for i in result.feature_interactions)

    def test_invalid_metrics_handling(
        self,
        predictor: ResourcePredictor,
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test handling of invalid metrics."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([2.0]))
        
        # Test with empty metrics
        empty_result = predictor.predict_resources(
            {'files': {}}, 
            sample_deployment_window
        )
        assert empty_result.recommended_team_size >= 2
        assert isinstance(empty_result.required_skills, set)
        assert empty_result.required_skills == set()
        
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
        none_result = predictor.predict_resources(
            none_metrics,
            sample_deployment_window
        )
        assert none_result.recommended_team_size >= 2
        assert isinstance(none_result.feature_importance, dict)
        
        # Test with missing required fields
        incomplete_metrics = {
            'files': {
                'test.py': {}
            }
        }
        incomplete_result = predictor.predict_resources(
            incomplete_metrics,
            sample_deployment_window
        )
        assert incomplete_result.recommended_team_size >= 2
        assert isinstance(incomplete_result.top_contributors, list)

    def test_helper_calculations(
        self,
        predictor: ResourcePredictor,
        sample_metrics: Dict[str, Any]
    ):
        """Test various helper calculation methods."""
        # Test test coverage calculation
        coverage = predictor._calculate_test_coverage(sample_metrics)
        assert coverage == pytest.approx(0.5)  # 1 out of 2 files has tests
        
        # Test documentation ratio calculation
        doc_ratio = predictor._calculate_documentation_ratio(sample_metrics)
        assert doc_ratio == pytest.approx(0.167, rel=0.01)  # (20 + 30) / (100 + 200)
        
        # Test dependency count calculation
        dep_count = predictor._calculate_dependency_count(sample_metrics)
        assert dep_count == 3  # unique dependencies: dep1, dep2, dep3

    def test_update_models_if_needed(self, sample_metrics):
        """Test model updating mechanism."""
        predictor = ResourcePredictor()
        
        # Create mock deployment with proper structure
        mock_deployment = Mock()
        mock_deployment.metrics = sample_metrics
        mock_deployment.team_size = 3
        
        # Test when not enough history
        predictor.deployment_history = [mock_deployment] * 9
        predictor._update_models_if_needed()
        
        # Test when enough history
        predictor.deployment_history = [mock_deployment] * 10
        with patch.object(RandomForestRegressor, 'fit') as mock_fit:
            predictor._update_models_if_needed()
            mock_fit.assert_called_once()
            # Verify feature extraction
            X, y = mock_fit.call_args[0]
            assert X.shape[1] == len(predictor.feature_names)
            assert len(y) == 10

    def test_confidence_calculation(
        self,
        predictor: ResourcePredictor,
        sample_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test confidence score calculation."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([3.0]))
        
        # Test normal case
        result = predictor.predict_resources(
            sample_metrics,
            sample_deployment_window
        )
        assert 0 <= result.confidence_score <= 1
        
        # Test with extreme values
        mock_ml_explainer.explain_prediction.return_value['confidence_factors'] = {
            'extreme_values': 3,
            'importance_concentration': 0.6,
            'confidence_penalty': 0.2
        }
        extreme_result = predictor.predict_resources(
            sample_metrics,
            sample_deployment_window
        )
        assert extreme_result.confidence_score < result.confidence_score

    def test_feature_interactions(
        self,
        predictor: ResourcePredictor,
        sample_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer
    ):
        """Test feature interaction analysis."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict = Mock(return_value=np.array([3.0]))
        
        result = predictor.predict_resources(
            sample_metrics,
            sample_deployment_window
        )
        
        assert isinstance(result.feature_interactions, list)
        assert len(result.feature_interactions) > 0
        for interaction in result.feature_interactions:
            assert 'features' in interaction
            assert 'strength' in interaction
            assert isinstance(interaction['features'], tuple)
            assert isinstance(interaction['strength'], float)

if __name__ == '__main__':
    pytest.main([__file__])