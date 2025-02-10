import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

from code_analyzer.ml.incident_predictor import IncidentPredictor
from code_analyzer.ml.base_explainer import MLExplainer
from code_analyzer.models.metrics import (
    IncidentPrediction, DeploymentWindow, CodeMetrics,
    SecurityMetrics, ArchitectureMetrics, ComplexityMetrics,
    ChangeProbability, DeploymentFeedback
)

@pytest.fixture
def mock_metrics() -> Dict[str, Any]:
    """Create a mock metrics object that matches the implementation's expectations."""
    main_code_metrics = CodeMetrics(
        lines_code=500,
        lines_comment=100,
        test_coverage_files={'test_main.py'},
        change_probability=ChangeProbability(
            file_path='src/main.py',
            change_frequency=10,
            last_modified=datetime.now(),
            contributors={'dev1', 'dev2'},
            churn_rate=0.8
        ),
        complexity=ComplexityMetrics(
            cyclomatic_complexity=20,
            cognitive_complexity=15,
            change_risk=0.7
        ),
        security=SecurityMetrics(
            potential_sql_injections=1,
            hardcoded_secrets=2,
            vulnerable_imports=['unsafe_lib']
        ),
        architecture=ArchitectureMetrics(
            component_coupling=0.9,
            circular_dependencies=[['module_a', 'module_b', 'module_a']],
            layering_violations=2
        )
    )

    utils_code_metrics = CodeMetrics(
        lines_code=300,
        lines_comment=90,
        test_coverage_files=set(),
        change_probability=ChangeProbability(
            file_path='src/utils.py',
            change_frequency=5,
            last_modified=datetime.now(),
            contributors={'dev1'},
            churn_rate=0.2
        ),
        complexity=ComplexityMetrics(
            cyclomatic_complexity=10,
            cognitive_complexity=8,
            change_risk=0.3
        ),
        security=SecurityMetrics(
            potential_sql_injections=0,
            hardcoded_secrets=0,
            vulnerable_imports=[]
        ),
        architecture=ArchitectureMetrics(
            component_coupling=0.4,
            circular_dependencies=[],
            layering_violations=0
        )
    )

    return {
        'files': {
            'src/main.py': {
                'complexity': {
                    'cyclomatic_complexity': 20,
                    'cognitive_complexity': 15,
                    'change_risk': 0.7
                },
                'security': {
                    'vulnerable_imports': ['unsafe_lib'],
                    'potential_sql_injections': 1,
                    'hardcoded_secrets': 2
                },
                'architecture': {
                    'component_coupling': 0.9,
                    'circular_dependencies': ['module_a -> module_b -> module_a'],
                    'layering_violations': 2
                },
                'metrics': main_code_metrics
            },
            'src/utils.py': {
                'complexity': {
                    'cyclomatic_complexity': 10,
                    'cognitive_complexity': 8,
                    'change_risk': 0.3
                },
                'security': {
                    'vulnerable_imports': [],
                    'potential_sql_injections': 0,
                    'hardcoded_secrets': 0
                },
                'architecture': {
                    'component_coupling': 0.4,
                    'circular_dependencies': [],
                    'layering_violations': 0
                },
                'metrics': utils_code_metrics
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
    window.historical_success_rate = 0.9
    return window

@pytest.fixture
def predictor() -> IncidentPredictor:
    return IncidentPredictor()

@pytest.fixture
def mock_ml_explainer():
    explainer = Mock(spec=MLExplainer)
    explanation = {
        'feature_importance': {
            'cyclomatic_complexity_mean': 0.4,
            'security_vulnerabilities': 0.3,
            'component_coupling_mean': 0.2
        },
        'top_contributors': [
            {'feature': 'cyclomatic_complexity_mean', 'impact': 0.4, 'deviation': 2.0},
            {'feature': 'security_vulnerabilities', 'impact': 0.3, 'deviation': 1.5},
            {'feature': 'component_coupling_mean', 'impact': 0.2, 'deviation': 1.0}
        ],
        'explanation': 'Test explanation',
        'confidence_factors': {
            'extreme_values': 1,
            'importance_concentration': 0.4,
            'confidence_penalty': 0.1
        }
    }
    explainer.explain_prediction.return_value = explanation
    explainer.get_feature_interactions.return_value = [
        {'features': ('cyclomatic_complexity_mean', 'security_vulnerabilities'), 'strength': 0.5}
    ]
    return explainer

@pytest.fixture
def sample_similar_deployments():
    deployments = []
    for i in range(3):
        deployment = Mock()
        deployment.issues_encountered = True
        deployment.actual_support_hours = 4.0
        deployments.append(deployment)
    return deployments

class TestIncidentPredictor:
    def test_init(self, predictor: IncidentPredictor):
        """Test initialization of IncidentPredictor."""
        assert isinstance(predictor.model, RandomForestClassifier)
        assert isinstance(predictor.explainer, MLExplainer)
        assert predictor.explainer.model_type == "classifier"
        assert len(predictor.feature_names) == 13

    def test_extract_features(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow
    ):
        """Test feature extraction with all components."""
        features = predictor._extract_features(mock_metrics, sample_deployment_window)
        assert isinstance(features, np.ndarray)
        assert len(features) == len(predictor.feature_names)  # 13 features
        
        # Test feature calculation
        complexity_metrics = [m['complexity']['cyclomatic_complexity'] 
                            for m in mock_metrics['files'].values()]
        assert features[0] == pytest.approx(np.mean(complexity_metrics))
        assert features[1] == pytest.approx(max(complexity_metrics))

    def test_identify_potential_incident_areas(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any],
        mock_ml_explainer: MLExplainer
    ):
        """Test identification of potential risk areas."""
        top_contributors = mock_ml_explainer.explain_prediction.return_value['top_contributors']
        risk_areas = predictor._identify_potential_incident_areas(mock_metrics, top_contributors)
        
        assert isinstance(risk_areas, list)
        assert any("High complexity in src/main.py" in area for area in risk_areas)
        assert any("SQL injection risks in src/main.py" in area for area in risk_areas)
        assert any("High coupling in src/main.py" in area for area in risk_areas)
        assert any("Circular dependencies in src/main.py" in area for area in risk_areas)

    def test_predict_incident_severity(self, predictor: IncidentPredictor):
        """Test incident severity prediction."""
        explanation = {'top_contributors': [{'impact': 0.1}]}
        assert predictor._predict_incident_severity(0.2, explanation) == "low"
        
        explanation = {'top_contributors': [{'impact': 0.3}, {'impact': 0.3}]}
        assert predictor._predict_incident_severity(0.5, explanation) == "medium"
        
        explanation = {'top_contributors': [{'impact': 0.3}, {'impact': 0.3}, {'impact': 0.3}]}
        assert predictor._predict_incident_severity(0.8, explanation) == "high"

    def test_estimate_resolution_time(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any],
        sample_similar_deployments
    ):
        """Test resolution time estimation."""
        resolution_time = predictor._estimate_resolution_time(
            mock_metrics,
            sample_similar_deployments
        )
        assert isinstance(resolution_time, float)
        assert resolution_time == pytest.approx(4.0)

        # Test without similar deployments
        resolution_time_no_similar = predictor._estimate_resolution_time(
            mock_metrics,
            []
        )
        assert isinstance(resolution_time_no_similar, float)
        assert resolution_time_no_similar > 1.0

    @pytest.mark.integration
    def test_predict_incidents(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow,
        mock_ml_explainer: MLExplainer,
        sample_similar_deployments: List[Any]
    ):
        """Integration test for incident prediction."""
        predictor.model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        predictor._find_similar_deployments = Mock(return_value=sample_similar_deployments)
        predictor.explainer = mock_ml_explainer
        
        prediction = predictor.predict_incidents(mock_metrics, sample_deployment_window)
        
        assert isinstance(prediction, IncidentPrediction)
        assert prediction.probability == 0.7
        assert prediction.severity_level == "high"
        assert isinstance(prediction.potential_areas, list)
        assert isinstance(prediction.estimated_resolution_time, float)
        assert isinstance(prediction.confidence_score, float)
        assert 0 <= prediction.confidence_score <= 1
        assert isinstance(prediction.explanation, str)
        assert isinstance(prediction.feature_importance, dict)
        assert isinstance(prediction.top_contributors, list)
        assert isinstance(prediction.feature_interactions, list)

    def test_update_models_if_needed(self, predictor: IncidentPredictor, mock_metrics: Dict[str, Any]):
        """Test model updating mechanism."""
        # Create mock deployment with proper DeploymentFeedback structure
        deployment = DeploymentFeedback(
            deployment_id="test",
            actual_deployment_time=1.0,
            actual_support_hours=4.0,
            success=True,
            rollback_occurred=False,
            issues_encountered=["minor_issue"],
            support_tickets=[],
            team_size=3,
            start_time=datetime.now(),
            end_time=datetime.now(),
            affected_services=[]
        )
        deployment.metrics = mock_metrics
        
        predictor.deployment_history = [deployment] * 10
        
        original_fit = predictor.model.fit
        predictor.model.fit = Mock()
        
        try:
            predictor._update_models_if_needed()
            predictor.model.fit.assert_called_once()
            X, y = predictor.model.fit.call_args[0]
            assert X.shape[1] == len(predictor.feature_names)
            assert len(y) == 10
        finally:
            predictor.model.fit = original_fit

    def test_helper_calculations(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any]
    ):
        """Test various helper calculation methods."""
        # Test coupling calculation
        coupling = predictor._calculate_coupling(mock_metrics)
        assert coupling == pytest.approx(0.65)  # Average of 0.9 and 0.4

        # Test layering violations
        violations = predictor._calculate_layering_violations(mock_metrics)
        assert violations == 2

        # Test test coverage calculation
        coverage = predictor._calculate_test_coverage(mock_metrics)
        assert coverage == pytest.approx(0.5)  # 1 out of 2 files has tests

        # Test documentation ratio calculation
        doc_ratio = predictor._calculate_documentation_ratio(mock_metrics)
        assert doc_ratio == pytest.approx(0.2375)  # (100 + 90) / (500 + 300)

        # Test churn rate calculation
        churn = predictor._calculate_churn_rate(mock_metrics)
        assert churn == pytest.approx(0.5)  # Average of 0.8 and 0.2

    def test_feature_interactions(
        self,
        predictor: IncidentPredictor,
        mock_metrics: Dict[str, Any],
        mock_ml_explainer: MLExplainer,
        sample_deployment_window: DeploymentWindow
    ):
        """Test feature interaction analysis."""
        predictor.explainer = mock_ml_explainer
        predictor.model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        prediction = predictor.predict_incidents(mock_metrics, sample_deployment_window)
        
        # Test interactions structure
        assert hasattr(prediction, 'feature_interactions')
        assert len(prediction.feature_interactions) > 0
        for interaction in prediction.feature_interactions:
            assert 'features' in interaction
            assert 'strength' in interaction
            assert isinstance(interaction['features'], tuple)
            assert len(interaction['features']) == 2
            assert isinstance(interaction['strength'], float)
    def test_error_handling(
        self,
        predictor: IncidentPredictor,
        sample_deployment_window: DeploymentWindow
    ):
        """Test error handling for invalid inputs."""
        # Test with minimal valid metrics
        minimal_metrics = {
            'files': {
                'test.py': {
                    'complexity': {
                        'cyclomatic_complexity': 0,
                        'cognitive_complexity': 0,
                        'change_risk': 0
                    },
                    'security': {
                        'vulnerable_imports': [],
                        'potential_sql_injections': 0,
                        'hardcoded_secrets': 0
                    },
                    'architecture': {
                        'component_coupling': 0,
                        'circular_dependencies': [],
                        'layering_violations': 0
                    },
                    'metrics': CodeMetrics(
                        lines_code=0,
                        lines_comment=0,
                        test_coverage_files=set(),
                        change_probability=ChangeProbability(
                            file_path='test.py',
                            change_frequency=0,
                            last_modified=datetime.now(),
                            contributors=set(),
                            churn_rate=0
                        )
                    )
                }
            },
            'summary': {
                'security_issues': 0
            }
        }
        
        features = predictor._extract_features(minimal_metrics, sample_deployment_window)
        assert isinstance(features, np.ndarray)
        assert len(features) == len(predictor.feature_names)
        assert np.all(features[:10] == 0)  # All metrics should be zero
        assert features[10] == sample_deployment_window.risk_score  # Window risk
        assert features[11] == sample_deployment_window.team_availability  # Team availability

        # Test with completely empty metrics
        empty_features = predictor._extract_features({'files': {}}, sample_deployment_window)
        assert isinstance(empty_features, np.ndarray)
        assert len(empty_features) == len(predictor.feature_names)
        assert np.all(empty_features == 0)  # All features should be zero

        # Test with missing window
        no_window_features = predictor._extract_features(minimal_metrics, None)
        assert isinstance(no_window_features, np.ndarray)
        assert len(no_window_features) == len(predictor.feature_names)
        assert no_window_features[10] == 0.0  # Window risk should be zero
        assert no_window_features[11] == 0.0  # Team availability should be zero

    @pytest.mark.parametrize("invalid_metrics", [
        {},  # Empty dict
        {'files': None},  # None files
        {'files': {'test.py': None}},  # None file metrics
        {'files': {'test.py': {}}},  # Empty file metrics
    ])
    def test_invalid_metrics_handling(
        self,
        predictor: IncidentPredictor,
        invalid_metrics: Dict[str, Any],
        sample_deployment_window: DeploymentWindow
    ):
        """Test handling of various invalid metrics formats."""
        features = predictor._extract_features(invalid_metrics, sample_deployment_window)
        assert isinstance(features, np.ndarray)
        assert len(features) == len(predictor.feature_names)
        assert np.all(features == 0)  # Should return zeros for invalid inputs

if __name__ == '__main__':
    pytest.main([__file__])