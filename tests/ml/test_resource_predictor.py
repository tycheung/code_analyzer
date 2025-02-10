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
                    'component_coupling': 0.5
                }
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
                    'component_coupling': 0.8
                }
            }
        },
        'summary': {
            'security_issues': 2
        }
    }

@pytest.fixture
def sample_deployment_window() -> DeploymentWindow:
    return DeploymentWindow(
        start_date="2025-01-01",
        end_date="2025-01-07",
        environment="production"
    )

@pytest.fixture
def sample_similar_deployments() -> List[Mock]:
    deployments = []
    for hours in [10, 15, 12]:
        deployment = Mock()
        deployment.actual_support_hours = hours
        deployments.append(deployment)
    return deployments

@pytest.fixture
def mock_ml_explainer():
    explainer = Mock(spec=MLExplainer)
    explainer.explain_prediction.return_value = {
        'feature_importance': {
            'mean_cyclomatic_complexity': 0.5,
            'security_issues': 0.3,
            'mean_component_coupling': 0.2
        },
        'top_contributors': [
            {'feature': 'mean_cyclomatic_complexity', 'impact': 0.5, 'deviation': 1.0},
            {'feature': 'security_issues', 'impact': 0.3, 'deviation': 0.8}
        ],
        'explanation': 'Test explanation',
        'confidence_factors': {
            'extreme_values': 0,
            'importance_concentration': 0.4,
            'confidence_penalty': 0.1
        }
    }
    explainer.get_feature_interactions.return_value = [
        {'features': ('mean_cyclomatic_complexity', 'security_issues'), 'strength': 0.5}
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

    def test_extract_features(self, sample_metrics):
        """Test feature extraction from metrics."""
        predictor = ResourcePredictor()
        features = predictor._extract_features(sample_metrics)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(predictor.feature_names),)
        assert np.all(features >= 0)
        
        # Test empty metrics
        empty_features = predictor._extract_features({'files': {}})
        assert np.all(empty_features == 0)

    def test_identify_required_skills(self, sample_metrics):
        """Test identification of required skills based on feature importance."""
        predictor = ResourcePredictor()
        predictor.explainer = mock_ml_explainer()
        
        skills = predictor._identify_required_skills(sample_metrics)
        
        assert isinstance(skills, set)
        assert 'senior_developer' in skills  # Due to high cyclomatic complexity
        assert 'security_expert' in skills   # Due to security issues
        assert 'architecture_specialist' in skills  # Due to high coupling
        assert 'domain_expert' in skills     # Due to high change risk

    def test_estimate_support_duration(self, sample_metrics, sample_similar_deployments, mock_ml_explainer):
        """Test support duration estimation with feature importance."""
        predictor = ResourcePredictor()
        predictor.explainer = mock_ml_explainer
        
        # Test with similar deployments
        duration_with_similar = predictor._estimate_support_duration(
            sample_metrics,
            sample_similar_deployments
        )
        assert duration_with_similar == pytest.approx(12.33, rel=0.1)
        
        # Test without similar deployments (using feature importance)
        duration_without_similar = predictor._estimate_support_duration(
            sample_metrics,
            []
        )
        assert duration_without_similar > 2.0  # Base hours
        assert isinstance(duration_without_similar, float)

    @patch.object(RandomForestRegressor, 'predict')
    def test_predict_resources(
        self,
        mock_predict,
        sample_metrics,
        sample_deployment_window,
        sample_similar_deployments,
        mock_ml_explainer
    ):
        """Test resource prediction with explainability."""
        predictor = ResourcePredictor()
        predictor.explainer = mock_ml_explainer
        mock_predict.return_value = np.array([3.5])
        
        # Mock internal methods
        predictor._find_similar_deployments = Mock(return_value=sample_similar_deployments)
        predictor._calculate_confidence_score = Mock(return_value=0.8)
        
        result = predictor.predict_resources(
            sample_metrics,
            sample_deployment_window
        )
        
        assert isinstance(result, ResourceAllocation)
        assert result.recommended_team_size == 4  # Should round up from 3.5
        assert isinstance(result.required_skills, set)
        assert isinstance(result.estimated_support_duration, float)
        assert 0 <= result.confidence_score <= 1
        
        # Test explainability components
        assert hasattr(result, 'explanation')
        assert isinstance(result.explanation, dict)
        assert 'feature_importance' in result.explanation
        assert 'top_contributors' in result.explanation
        assert 'confidence_factors' in result.explanation
        assert isinstance(result.explanation['feature_importance'], dict)
        assert len(result.explanation['top_contributors']) > 0
        assert all(k in result.explanation['confidence_factors'] 
                  for k in ['extreme_values', 'importance_concentration', 'confidence_penalty'])

    def test_update_models_if_needed(self):
        """Test model updating mechanism with feature extraction."""
        predictor = ResourcePredictor()
        
        # Create mock deployment with proper structure
        mock_deployment = Mock()
        mock_deployment.metrics = {
            'files': {
                'test.py': {
                    'complexity': {
                        'cyclomatic_complexity': 10,
                        'cognitive_complexity': 5,
                        'change_risk': 0.5
                    },
                    'security': {
                        'vulnerable_imports': [],
                        'potential_sql_injections': 0
                    },
                    'architecture': {
                        'component_coupling': 0.3
                    }
                }
            },
            'summary': {
                'security_issues': 0
            }
        }
        mock_deployment.team_size = 3
        
        # Test when not enough history
        predictor.deployment_history = [mock_deployment for _ in range(9)]
        predictor._update_models_if_needed()
        
        # Test when enough history
        predictor.deployment_history = [mock_deployment for _ in range(10)]
        with patch.object(RandomForestRegressor, 'fit') as mock_fit:
            predictor._update_models_if_needed()
            mock_fit.assert_called_once()
            # Verify feature extraction
            X, y = mock_fit.call_args[0]
            assert X.shape[1] == len(predictor.feature_names)
            assert len(y) == 10

    def test_confidence_score_calculation(
        self,
        predictor: ResourcePredictor,
        sample_metrics: Dict[str, Any],
        mock_ml_explainer: MLExplainer
    ):
        """Test confidence score calculation with explainability factors."""
        predictor.explainer = mock_ml_explainer
        
        # Get a prediction with explanation
        predictor.model.predict = Mock(return_value=np.array([3.5]))
        prediction = predictor.predict_resources(sample_metrics, Mock(spec=DeploymentWindow))
        
        # Verify confidence factors
        assert 'confidence_factors' in prediction.explanation
        confidence_factors = prediction.explanation['confidence_factors']
        assert isinstance(confidence_factors, dict)
        assert 'confidence_penalty' in confidence_factors
        
        # Verify the confidence score is affected by the explanation
        assert prediction.confidence_score < 1.0

    def test_edge_cases(self, sample_deployment_window, mock_ml_explainer):
        """Test edge cases with explainability."""
        predictor = ResourcePredictor()
        predictor.explainer = mock_ml_explainer
        
        # Test with empty metrics
        empty_metrics = {'files': {}}
        result = predictor.predict_resources(
            empty_metrics,
            sample_deployment_window
        )
        assert result.recommended_team_size >= 2  # Minimum team size
        assert 'feature_importance' in result.explanation
        
        # Test with extreme values
        extreme_metrics = {
            'files': {
                'test.py': {
                    'complexity': {
                        'cyclomatic_complexity': 1000,
                        'cognitive_complexity': 500,
                        'change_risk': 1.0
                    },
                    'security': {
                        'vulnerable_imports': ['a'] * 100,
                        'potential_sql_injections': 50
                    },
                    'architecture': {
                        'component_coupling': 1.0
                    }
                }
            },
            'summary': {
                'security_issues': 100
            }
        }
        result = predictor.predict_resources(
            extreme_metrics,
            sample_deployment_window
        )
        assert isinstance(result, ResourceAllocation)
        assert all(skill in result.required_skills for skill in [
            'senior_developer',
            'security_expert',
            'architecture_specialist',
            'domain_expert'
        ])
        # Verify extreme values are reflected in explanation
        assert any(contrib['deviation'] > 2.0 for contrib in result.explanation['top_contributors'])

if __name__ == '__main__':
    pytest.main([__file__])