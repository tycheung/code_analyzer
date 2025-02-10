from typing import Dict, List, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import RollbackPrediction, DeploymentWindow

class RollbackPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()
        self.explainer = MLExplainer(model_type="classifier")
        self.feature_names = [
            'cyclomatic_complexity',
            'cognitive_complexity',
            'change_risk',
            'security_issues',
            'component_coupling',
            'deploy_window_risk',
            'team_availability',
            'recent_changes',
            'test_coverage'
        ]

    def _extract_features(self, metrics: Dict[str, Any], window: DeploymentWindow = None) -> np.ndarray:
        """Extract features for rollback prediction."""
        features = []
        for file_metrics in metrics['files'].values():
            features.append([
                file_metrics['complexity']['cyclomatic_complexity'],
                file_metrics['complexity']['cognitive_complexity'],
                file_metrics['complexity']['change_risk'],
                file_metrics['security']['potential_sql_injections'] + 
                file_metrics['security']['hardcoded_secrets'] +
                len(file_metrics['security']['vulnerable_imports']),
                file_metrics['architecture']['component_coupling']
            ])
        
        # Calculate aggregate metrics
        avg_features = np.mean(features, axis=0) if features else np.zeros(5)
        
        # Add window-specific features if available
        if window:
            window_features = [
                window.risk_score,
                window.team_availability,
                self._calculate_recent_changes(metrics),
                self._calculate_test_coverage(metrics)
            ]
            return np.concatenate([avg_features, window_features])
        
        return avg_features

    def predict_rollback(
        self, 
        metrics: Dict[str, Any],
        deployment_window: DeploymentWindow
    ) -> RollbackPrediction:
        """Predict the probability of deployment rollback with explanations."""
        features = self._extract_features(metrics, deployment_window)
        features_reshaped = features.reshape(1, -1)
        
        # Get base prediction
        probability = self.model.predict_proba(features_reshaped)[0][1]
        
        # Get explanation using centralized explainer
        explanation = self.explainer.explain_prediction(
            self.model,
            features_reshaped,
            self.feature_names,
            probability
        )
        
        # Get feature interactions
        interactions = self.explainer.get_feature_interactions(
            self.model,
            features_reshaped,
            self.feature_names
        )
        
        # Calculate confidence score considering explainability factors
        confidence_score = self._calculate_confidence_score(
            explanation['confidence_factors']
        )

        return RollbackPrediction(
            probability=probability,
            risk_factors=self._analyze_risk_factors(explanation),
            mitigation_suggestions=self._generate_mitigation_suggestions(explanation),
            confidence_score=confidence_score,
            explanation=explanation['explanation'],
            feature_importance=explanation['feature_importance'],
            top_contributors=explanation['top_contributors'],
            feature_interactions=interactions
        )

    def _analyze_risk_factors(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk factors from explanation."""
        risk_factors = {
            'complexity': 0.0,
            'security': 0.0,
            'testing': 0.0,
            'dependencies': 0.0,
            'timing': 0.0
        }
        
        for contributor in explanation['top_contributors']:
            feature = contributor['feature']
            impact = abs(contributor['impact'])
            
            if 'complexity' in feature:
                risk_factors['complexity'] += impact
            elif 'security' in feature:
                risk_factors['security'] += impact
            elif 'test' in feature:
                risk_factors['testing'] += impact
            elif 'coupling' in feature or 'dependencies' in feature:
                risk_factors['dependencies'] += impact
            elif 'time' in feature or 'window' in feature:
                risk_factors['timing'] += impact

        total = sum(risk_factors.values())
        return {k: v/total for k, v in risk_factors.items()} if total > 0 else risk_factors

    def _generate_mitigation_suggestions(self, explanation: Dict[str, Any]) -> List[str]:
        """Generate mitigation suggestions based on explanation."""
        suggestions = []
        for contributor in explanation['top_contributors']:
            if abs(contributor['deviation']) > 1.5:
                feature = contributor['feature']
                if 'complexity' in feature:
                    suggestions.append(
                        f"Consider breaking down complex components - {feature} "
                        f"is {abs(contributor['deviation']):.1f} standard deviations above normal"
                    )
                elif 'security' in feature:
                    suggestions.append(
                        f"Address security concerns - {feature} shows elevated risk"
                    )
                elif 'test' in feature:
                    suggestions.append(
                        f"Improve test coverage - current coverage is "
                        f"{abs(contributor['deviation']):.1f} standard deviations below typical"
                    )
        return suggestions

    def _calculate_confidence_score(self, confidence_factors: Dict[str, Any]) -> float:
        """Calculate confidence score considering explainability factors."""
        base_confidence = super()._calculate_confidence_score(len(self.deployment_history))
        confidence_penalty = confidence_factors.get('confidence_penalty', 0.0)
        return max(0.0, min(1.0, base_confidence - confidence_penalty))