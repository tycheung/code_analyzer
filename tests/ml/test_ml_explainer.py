import pytest
import numpy as np
from unittest.mock import Mock, patch
import shap
from typing import Dict, Any

from code_analyzer.ml.base_explainer import MLExplainer

@pytest.fixture
def sample_features():
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2]).reshape(1, -12)

@pytest.fixture
def feature_names():
    base_names = [
        'mean_cyclomatic_complexity',
        'mean_cognitive_complexity',
        'mean_change_risk',
        'mean_vulnerable_imports',
        'mean_component_coupling',
        'mean_circular_dependencies'
    ]
    max_names = [f'max_{name.split("_", 1)[1]}' for name in base_names]
    return base_names + max_names

@pytest.fixture
def mock_classifier():
    classifier = Mock()
    classifier.predict_proba.return_value = np.array([[0.3, 0.7]])
    return classifier

@pytest.fixture
def mock_explainer():
    return MLExplainer(model_type="classifier")

class TestMLExplainer:
    def test_init(self):
        """Test MLExplainer initialization."""
        explainer = MLExplainer(model_type="classifier")
        assert explainer.model_type == "classifier"
        
        explainer = MLExplainer(model_type="regressor")
        assert explainer.model_type == "regressor"

    @patch('shap.TreeExplainer')
    def test_explain_prediction_classifier(
        self,
        mock_shap_explainer,
        mock_explainer,
        sample_features,
        feature_names,
        mock_classifier
    ):
        """Test prediction explanation for classifier."""
        # Mock SHAP values
        mock_shap_values = [
            np.zeros_like(sample_features),
            np.random.random(sample_features.shape)
        ]
        mock_shap_explainer.return_value.shap_values.return_value = mock_shap_values

        explanation = mock_explainer.explain_prediction(
            mock_classifier,
            sample_features,
            feature_names
        )

        assert isinstance(explanation, dict)
        assert 'feature_importance' in explanation
        assert 'top_contributors' in explanation
        assert 'explanation' in explanation
        assert 'confidence_factors' in explanation
        
        # Verify feature importance structure
        assert len(explanation['feature_importance']) == len(feature_names)
        assert all(isinstance(v, float) for v in explanation['feature_importance'].values())
        
        # Verify top contributors
        assert len(explanation['top_contributors']) <= 5
        for contributor in explanation['top_contributors']:
            assert 'feature' in contributor
            assert 'impact' in contributor
            assert 'deviation' in contributor
            assert isinstance(contributor['impact'], float)
            assert isinstance(contributor['deviation'], float)

        # Verify natural language explanation
        assert isinstance(explanation['explanation'], str)
        assert "%" in explanation['explanation']  # Should contain probability
        
        # Verify confidence factors
        assert isinstance(explanation['confidence_factors'], dict)
        assert 'extreme_values' in explanation['confidence_factors']
        assert 'importance_concentration' in explanation['confidence_factors']
        assert 'confidence_penalty' in explanation['confidence_factors']

    @patch('shap.TreeExplainer')
    def test_explain_prediction_regressor(
        self,
        mock_shap_explainer,
        sample_features,
        feature_names
    ):
        """Test prediction explanation for regressor."""
        regressor = Mock()
        regressor.predict.return_value = np.array([3.5])
        explainer = MLExplainer(model_type="regressor")

        mock_shap_values = np.random.random(sample_features.shape)
        mock_shap_explainer.return_value.shap_values.return_value = mock_shap_values

        explanation = explainer.explain_prediction(
            regressor,
            sample_features,
            feature_names,
            prediction=3.5
        )

        assert isinstance(explanation, dict)
        assert "3.5" in explanation['explanation']

    @patch('shap.TreeExplainer')
    def test_get_feature_interactions(
        self,
        mock_shap_explainer,
        mock_explainer,
        sample_features,
        feature_names,
        mock_classifier
    ):
        """Test feature interaction analysis."""
        # Mock interaction values
        mock_interaction_values = [
            np.zeros((1, 12, 12)),
            np.random.random((1, 12, 12))
        ]
        mock_shap_explainer.return_value.shap_interaction_values.return_value = mock_interaction_values

        interactions = mock_explainer.get_feature_interactions(
            mock_classifier,
            sample_features,
            feature_names
        )

        assert isinstance(interactions, list)
        assert len(interactions) <= 5  # Should return top 5 interactions
        
        for interaction in interactions:
            assert 'features' in interaction
            assert 'strength' in interaction
            assert isinstance(interaction['features'], tuple)
            assert len(interaction['features']) == 2
            assert isinstance(interaction['strength'], float)

    def test_generate_natural_explanation(
        self,
        mock_explainer,
        feature_names
    ):
        """Test natural language explanation generation."""
        top_contributors = [
            (feature_names[0], 0.5, 2.5),
            (feature_names[1], -0.3, -1.5),
            (feature_names[2], 0.1, 0.5)
        ]

        # Test classifier explanation
        classifier_explanation = mock_explainer._generate_natural_explanation(
            0.7,
            top_contributors,
            "classifier"
        )
        assert isinstance(classifier_explanation, str)
        assert "70.0%" in classifier_explanation
        assert "standard deviations" in classifier_explanation

        # Test regressor explanation
        regressor_explanation = mock_explainer._generate_natural_explanation(
            3.5,
            top_contributors,
            "regressor"
        )
        assert isinstance(regressor_explanation, str)
        assert "3.50" in regressor_explanation

    def test_analyze_confidence_factors(self, mock_explainer):
        """Test confidence factor analysis."""
        contributors = [
            ("feature1", 0.5, 2.5),
            ("feature2", 0.3, 1.5),
            ("feature3", 0.2, 0.5)
        ]
        
        features = np.random.random((1, 12))
        
        confidence_analysis = mock_explainer._analyze_confidence_factors(
            features,
            contributors
        )
        
        assert isinstance(confidence_analysis, dict)
        assert 'extreme_values' in confidence_analysis
        assert 'importance_concentration' in confidence_analysis
        assert 'confidence_penalty' in confidence_analysis
        assert 'confidence_factors' in confidence_analysis
        
        # Verify confidence factors
        assert isinstance(confidence_analysis['extreme_values'], int)
        assert isinstance(confidence_analysis['importance_concentration'], float)
        assert isinstance(confidence_analysis['confidence_penalty'], float)
        assert isinstance(confidence_analysis['confidence_factors'], dict)

    def test_error_handling(
        self,
        mock_explainer,
        sample_features,
        feature_names,
        mock_classifier
    ):
        """Test error handling in feature interactions."""
        with patch('shap.TreeExplainer') as mock_shap:
            mock_shap.side_effect = Exception("SHAP error")
            
            # Should return empty list on error
            interactions = mock_explainer.get_feature_interactions(
                mock_classifier,
                sample_features,
                feature_names
            )
            assert interactions == []