from typing import Dict, Any, List
import os
import joblib
from datetime import time
from .window_predictor import WindowPredictor
from .rollback_predictor import RollbackPredictor
from .resource_predictor import ResourcePredictor
from .incident_predictor import IncidentPredictor
from ..models.metrics import (
    DeploymentFeedback, DeploymentWindow, ResourceAllocation,
    RollbackPrediction, IncidentPrediction
)

class DeploymentMLSystem:
    """Orchestrates various ML predictors for deployment analysis."""
    
    def __init__(self):
        self.window_predictor = WindowPredictor()
        self.rollback_predictor = RollbackPredictor()
        self.resource_predictor = ResourcePredictor()
        self.incident_predictor = IncidentPredictor()

    def record_deployment_feedback(self, feedback: DeploymentFeedback) -> None:
        """Record deployment feedback across all predictors."""
        self.window_predictor.record_deployment_feedback(feedback)
        self.rollback_predictor.record_deployment_feedback(feedback)
        self.resource_predictor.record_deployment_feedback(feedback)
        self.incident_predictor.record_deployment_feedback(feedback)

    def analyze_deployment(
        self,
        metrics: Dict[str, Any],
        team_availability: Dict[str, List[time]]
    ) -> Dict[str, Any]:
        """Perform comprehensive deployment analysis with explainability."""
        # Get optimal deployment windows with explanations
        optimal_windows = self.window_predictor.predict_optimal_windows(
            metrics, team_availability
        )

        if not optimal_windows:
            return {
                'error': 'No suitable deployment windows found',
                'confidence': 0.0,
                'explanation': 'Could not identify suitable deployment windows based on available data.'
            }

        # Analyze best window
        best_window = optimal_windows[0]
        
        # Get predictions with explanations for each aspect
        rollback_prediction = self.rollback_predictor.predict_rollback(
            metrics, best_window
        )
        
        resource_prediction = self.resource_predictor.predict_resources(
            metrics, best_window
        )
        
        incident_prediction = self.incident_predictor.predict_incidents(
            metrics, best_window
        )

        # Calculate overall confidence with contributing factors
        confidence_scores = {
            'rollback': rollback_prediction.confidence_score,
            'resource': resource_prediction.confidence_score,
            'incident': incident_prediction.confidence_score,
            'window': best_window.historical_success_rate
        }
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)

        # Aggregate feature importances across all models
        feature_importances = self._aggregate_feature_importances({
            'window': best_window.feature_importance,
            'rollback': rollback_prediction.feature_importance,
            'resource': resource_prediction.feature_importance,
            'incident': incident_prediction.feature_importance
        })

        # Generate system-level explanation
        system_explanation = self._generate_system_explanation(
            optimal_windows[:3],
            rollback_prediction,
            resource_prediction,
            incident_prediction,
            feature_importances
        )

        return {
            'optimal_windows': optimal_windows[:3],  # Top 3 windows
            'rollback_prediction': rollback_prediction,
            'resource_prediction': resource_prediction,
            'incident_prediction': incident_prediction,
            'overall_confidence': overall_confidence,
            'confidence_breakdown': confidence_scores,
            'feature_importances': feature_importances,
            'system_explanation': system_explanation,
            'key_insights': self._extract_key_insights(
                optimal_windows[0],
                rollback_prediction,
                resource_prediction,
                incident_prediction
            )
        }

    def _aggregate_feature_importances(
        self,
        model_importances: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate feature importances across all models."""
        all_features = set()
        for importances in model_importances.values():
            all_features.update(importances.keys())

        aggregated = {
            'cross_model': {},
            'by_model': model_importances
        }

        # Calculate cross-model importance
        for feature in all_features:
            importance_sum = sum(
                imps.get(feature, 0)
                for imps in model_importances.values()
            )
            aggregated['cross_model'][feature] = importance_sum / len(model_importances)

        return aggregated

    def _generate_system_explanation(
        self,
        windows: List[DeploymentWindow],
        rollback: RollbackPrediction,
        resources: ResourceAllocation,
        incidents: IncidentPrediction,
        importances: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate comprehensive system-level explanation."""
        # Get top cross-model features
        top_features = sorted(
            importances['cross_model'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        explanation = [
            "# Deployment Analysis Summary",
            "\n## Overall Assessment",
            f"Optimal deployment window: {windows[0].start_time}-{windows[0].end_time}",
            f"Rollback risk: {rollback.probability*100:.1f}%",
            f"Required team size: {resources.recommended_team_size} developers",
            f"Incident probability: {incidents.probability*100:.1f}%",
            
            "\n## Key Contributing Factors",
            "The following factors most strongly influence these predictions:"
        ]

        for feature, importance in top_features:
            explanation.append(
                f"- {feature.replace('_', ' ').title()}: "
                f"Impact score of {importance:.3f}"
            )

        explanation.extend([
            "\n## Model-Specific Insights",
            "### Deployment Window Selection",
            windows[0].explanation,
            "\n### Rollback Risk Analysis",
            rollback.explanation,
            "\n### Resource Requirements",
            resources.explanation,
            "\n### Incident Prediction",
            incidents.explanation
        ])

        return "\n".join(explanation)

    def _extract_key_insights(
        self,
        window: DeploymentWindow,
        rollback: RollbackPrediction,
        resources: ResourceAllocation,
        incidents: IncidentPrediction
    ) -> List[Dict[str, Any]]:
        """Extract key insights from all predictions."""
        insights = []

        # Window insights
        if hasattr(window, 'top_contributors'):
            for contrib in window.top_contributors:
                if abs(contrib['impact']) > 0.1:
                    insights.append({
                        'aspect': 'deployment_window',
                        'feature': contrib['feature'],
                        'impact': contrib['impact'],
                        'deviation': contrib['deviation']
                    })

        # Rollback insights
        for contrib in rollback.top_contributors:
            if abs(contrib['impact']) > 0.1:
                insights.append({
                    'aspect': 'rollback_risk',
                    'feature': contrib['feature'],
                    'impact': contrib['impact'],
                    'deviation': contrib['deviation']
                })

        # Resource insights
        for contrib in resources.top_contributors:
            if abs(contrib['impact']) > 0.1:
                insights.append({
                    'aspect': 'resource_needs',
                    'feature': contrib['feature'],
                    'impact': contrib['impact'],
                    'deviation': contrib['deviation']
                })

        # Incident insights
        for contrib in incidents.top_contributors:
            if abs(contrib['impact']) > 0.1:
                insights.append({
                    'aspect': 'incident_risk',
                    'feature': contrib['feature'],
                    'impact': contrib['impact'],
                    'deviation': contrib['deviation']
                })

        return sorted(insights, key=lambda x: abs(x['impact']), reverse=True)

    def save_models(self, directory: str) -> None:
        """Save all ML models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        predictors = {
            'window': self.window_predictor,
            'rollback': self.rollback_predictor,
            'resource': self.resource_predictor,
            'incident': self.incident_predictor
        }
        
        for name, predictor in predictors.items():
            model_path = os.path.join(directory, f"{name}_model.joblib")
            joblib.dump(predictor.model, model_path)

    def load_models(self, directory: str) -> None:
        """Load all ML models from disk."""
        predictors = {
            'window': self.window_predictor,
            'rollback': self.rollback_predictor,
            'resource': self.resource_predictor,
            'incident': self.incident_predictor
        }
        
        for name, predictor in predictors.items():
            model_path = os.path.join(directory, f"{name}_model.joblib")
            if os.path.exists(model_path):
                predictor.model = joblib.load(model_path)