from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap

class MLExplainer:
    """Centralized explainability for machine learning models."""

    def __init__(self, model_type: str = "classifier"):
        """
        Initialize explainer.
        Args:
            model_type: Either "classifier" or "regressor"
        """
        self.model_type = model_type

    def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation for a prediction."""
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        
        # Handle classifier vs regressor output
        if self.model_type == "classifier" and isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class

        # Compute feature importances
        importances = np.abs(shap_values).mean(0)
        feature_importance = dict(zip(feature_names, importances))

        # Find top contributing features
        contributions = []
        for i, fname in enumerate(feature_names):
            contributions.append((
                fname,
                shap_values[0][i],
                abs(features[0][i] - np.mean(features[:, i])) / (np.std(features[:, i]) or 1)
            ))

        top_contributors = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

        return {
            'feature_importance': feature_importance,
            'top_contributors': [
                {
                    'feature': feature,
                    'impact': float(impact),
                    'deviation': float(deviation)
                }
                for feature, impact, deviation in top_contributors[:5]
            ],
            'explanation': self._generate_natural_explanation(
                prediction if prediction is not None else (
                    model.predict_proba(features)[0][1] if self.model_type == "classifier"
                    else model.predict(features)[0]
                ),
                top_contributors[:3],
                self.model_type
            ),
            'confidence_factors': self._analyze_confidence_factors(features, top_contributors)
        }

    def get_feature_interactions(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze feature interactions."""
        try:
            explainer = shap.TreeExplainer(model)
            shap_interaction = explainer.shap_interaction_values(features)
            
            if isinstance(shap_interaction, list):
                shap_interaction = shap_interaction[1]
            
            interaction_matrix = np.abs(shap_interaction[0])
            np.fill_diagonal(interaction_matrix, 0)
            
            interactions = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if interaction_matrix[i, j] > 0:
                        interactions.append({
                            'features': (feature_names[i], feature_names[j]),
                            'strength': float(interaction_matrix[i, j])
                        })
            
            return sorted(interactions, key=lambda x: x['strength'], reverse=True)[:5]
            
        except Exception as e:
            return []

    def _generate_natural_explanation(
        self,
        prediction: float,
        top_contributors: List[Tuple[str, float, float]],
        model_type: str
    ) -> str:
        """Generate natural language explanation."""
        explanation = []
        
        # Overall prediction
        if model_type == "classifier":
            explanation.append(
                f"The model predicts a {prediction*100:.1f}% probability."
            )
        else:
            explanation.append(f"The model predicts a value of {prediction:.2f}.")
        
        # Explain top contributors
        explanation.append("\nThis prediction is primarily based on:")
        
        for feature, impact, deviation in top_contributors:
            feature_name = feature.replace('_', ' ').title()
            
            magnitude = (
                "significantly" if abs(deviation) > 2
                else "moderately" if abs(deviation) > 1
                else "slightly"
            )
            
            direction = "higher" if deviation > 0 else "lower"
            impact_direction = "increased" if impact > 0 else "decreased"
            
            explanation.append(
                f"- {feature_name} is {magnitude} {direction} than typical "
                f"({deviation:.1f} standard deviations), which {impact_direction} "
                f"the prediction by {abs(impact):.3f}"
            )

        return "\n".join(explanation)

    def _analyze_confidence_factors(
        self,
        features: np.ndarray,
        contributors: List[Tuple[str, float, float]]
    ) -> Dict[str, Any]:
        """Analyze factors affecting prediction confidence."""
        extreme_values = sum(1 for _, _, dev in contributors if abs(dev) > 2)
        impact_values = [abs(impact) for _, impact, _ in contributors]
        importance_concentration = max(impact_values) / sum(impact_values) if impact_values else 0
        
        confidence_penalty = 0.0
        if extreme_values > 2:
            confidence_penalty += 0.1 * (extreme_values - 2)
        if importance_concentration > 0.5:
            confidence_penalty += 0.1 * (importance_concentration - 0.5)
        
        return {
            'extreme_values': extreme_values,
            'importance_concentration': float(importance_concentration),
            'confidence_penalty': float(confidence_penalty),
            'confidence_factors': {
                'extreme_values_present': extreme_values > 0,
                'high_importance_concentration': importance_concentration > 0.5,
                'balanced_feature_contribution': importance_concentration < 0.3
            }
        }