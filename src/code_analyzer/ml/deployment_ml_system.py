from typing import Dict, Any, List, Optional
import os
import time
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

from .window_predictor import WindowPredictor
from .rollback_predictor import RollbackPredictor
from .resource_predictor import ResourcePredictor
from .incident_predictor import IncidentPredictor
from ..models.metrics import (
    DeploymentFeedback, DeploymentWindow, ResourceAllocation,
    RollbackPrediction, IncidentPrediction, DeploymentAnalysis,
    MLSystemMetrics
)

class DeploymentMLSystem:
    """Orchestrates ML predictors for deployment analysis with comprehensive explainability."""
    
    MODELS_DIR = Path("models/deployment_ml")
    
    def __init__(self):
        """Initialize predictors and load models if available."""
        self.window_predictor = WindowPredictor()
        self.rollback_predictor = RollbackPredictor()
        self.resource_predictor = ResourcePredictor()
        self.incident_predictor = IncidentPredictor()
        
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.load_models()
        
        self.version = "1.0.0"
        self.min_confidence_threshold = 0.6
        self.metrics_history: List[MLSystemMetrics] = []

    def record_deployment_feedback(self, feedback: DeploymentFeedback) -> None:
        """Record deployment feedback with validation."""
        try:
            if not isinstance(feedback, DeploymentFeedback):
                raise ValueError("Invalid feedback object")
                
            # Record in each predictor
            predictors = [
                self.window_predictor,
                self.rollback_predictor,
                self.resource_predictor,
                self.incident_predictor
            ]
            
            for predictor in predictors:
                try:
                    predictor.record_deployment_feedback(feedback)
                except Exception as e:
                    print(f"Error recording feedback in {predictor.__class__.__name__}: {str(e)}")
            
            # Update system metrics
            self._update_system_metrics(feedback)
            
        except Exception as e:
            print(f"Error recording deployment feedback: {str(e)}")

    def analyze_deployment(
        self,
        metrics: Dict[str, Any],
        team_availability: Dict[str, List[time]]
    ) -> DeploymentAnalysis:
        """Perform comprehensive deployment analysis with error handling."""
        try:
            # Initial data validation
            if not self._validate_input_data(metrics, team_availability):
                return self._create_error_analysis("Invalid input data")

            # Get optimal deployment windows
            optimal_windows = self.window_predictor.predict_optimal_windows(
                metrics, team_availability
            )
            
            if not optimal_windows:
                return self._create_error_analysis(
                    "No suitable deployment windows found"
                )

            # Get best window and analyze all aspects
            best_window = optimal_windows[0]
            
            predictions = self._get_all_predictions(metrics, best_window)
            if predictions.get('error'):
                return self._create_error_analysis(predictions['error'])

            # Calculate confidence and aggregate features
            confidence_metrics = self._calculate_confidence_metrics(
                optimal_windows[0],
                predictions
            )
            
            feature_importances = self._aggregate_feature_importances(
                self._get_all_feature_importances(optimal_windows[0], predictions)
            )

            # Generate comprehensive explanation
            system_explanation = self._generate_system_explanation(
                optimal_windows[:3],
                predictions,
                feature_importances
            )

            # Extract key insights
            key_insights = self._extract_key_insights(
                optimal_windows[0],
                predictions
            )

            return DeploymentAnalysis(
                optimal_windows=optimal_windows[:3],
                rollback_prediction=predictions['rollback'],
                resource_prediction=predictions['resource'],
                incident_prediction=predictions['incident'],
                overall_confidence=confidence_metrics['overall'],
                confidence_breakdown=confidence_metrics['breakdown'],
                feature_importances=feature_importances,
                system_explanation=system_explanation,
                key_insights=key_insights,
                timestamp=time.time(),
                analysis_version=self.version,
                model_versions=self._get_model_versions(),
                data_completeness=self._calculate_data_completeness(metrics),
                prediction_quality=self._assess_prediction_quality(predictions)
            )

        except Exception as e:
            return self._create_error_analysis(f"Analysis error: {str(e)}")

    def _validate_input_data(
        self,
        metrics: Dict[str, Any],
        team_availability: Dict[str, List[time]]
    ) -> bool:
        """Validate input data completeness and format."""
        try:
            if not isinstance(metrics, dict) or not isinstance(team_availability, dict):
                return False
                
            # Check metrics structure
            if 'files' not in metrics or not isinstance(metrics['files'], dict):
                return False
                
            # Check team availability format
            for schedule in team_availability.values():
                if not isinstance(schedule, list):
                    return False
                    
                # Allow either tuple or list for time windows
                for time_window in schedule:
                    if not isinstance(time_window, (list, tuple)) or len(time_window) != 2:
                        return False
                    if not all(isinstance(t, (time, type(None))) for t in time_window):
                        return False
                        
            # Validate files have basic required structure but allow some to be invalid
            valid_files = 0
            for file_metrics in metrics['files'].values():
                if isinstance(file_metrics, dict) and all(
                    key in file_metrics for key in ['complexity', 'security', 'architecture']
                ):
                    valid_files += 1
                    
            return valid_files > 0  # Pass if we have at least one valid file
            
        except Exception:
            return False

    def _get_all_predictions(
        self,
        metrics: Dict[str, Any],
        window: DeploymentWindow
    ) -> Dict[str, Any]:
        """Get predictions from all models with error handling."""
        try:
            return {
                'rollback': self.rollback_predictor.predict_rollback(metrics, window),
                'resource': self.resource_predictor.predict_resources(metrics, window),
                'incident': self.incident_predictor.predict_incidents(metrics, window)
            }
        except Exception as e:
            return {'error': f"Prediction error: {str(e)}"}

    def _get_all_feature_importances(
        self,
        window: DeploymentWindow,
        predictions: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Get feature importances from all models."""
        try:
            return {
                'window': window.feature_importance,
                'rollback': predictions['rollback'].feature_importance,
                'resource': predictions['resource'].feature_importance,
                'incident': predictions['incident'].feature_importance
            }
        except Exception:
            return {}

    def _calculate_confidence_metrics(
        self,
        window: DeploymentWindow,
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate confidence metrics across all predictions."""
        try:
            confidence_scores = {
                'window': window.historical_success_rate,
                'rollback': predictions['rollback'].confidence_score,
                'resource': predictions['resource'].confidence_score,
                'incident': predictions['incident'].confidence_score
            }
            
            # Weight the scores based on their reliability
            weights = {
                'window': 0.3,
                'rollback': 0.3,
                'resource': 0.2,
                'incident': 0.2
            }
            
            overall = sum(
                score * weights[key]
                for key, score in confidence_scores.items()
            )
            
            return {
                'overall': overall,
                'breakdown': confidence_scores
            }
        except Exception:
            return {
                'overall': 0.0,
                'breakdown': {k: 0.0 for k in ['window', 'rollback', 'resource', 'incident']}
            }

    def _aggregate_feature_importances(
        self,
        model_importances: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate feature importances across models."""
        try:
            all_features = set()
            for importances in model_importances.values():
                if importances:
                    all_features.update(importances.keys())

            cross_model = {}
            for feature in all_features:
                importance_sum = sum(
                    imps.get(feature, 0)
                    for imps in model_importances.values()
                    if imps
                )
                cross_model[feature] = importance_sum / len(model_importances)

            return {
                'cross_model': cross_model,
                'by_model': model_importances
            }
        except Exception:
            return {'cross_model': {}, 'by_model': {}}

    def _generate_system_explanation(
        self,
        windows: List[DeploymentWindow],
        predictions: Dict[str, Any],
        importances: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate comprehensive system-level explanation."""
        try:
            # Get top cross-model features
            top_features = sorted(
                importances['cross_model'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            parts = [
                "# Deployment Analysis Summary",
                "\n## Overall Assessment",
                f"Optimal deployment window: {windows[0].start_time}-{windows[0].end_time}",
                f"Rollback risk: {predictions['rollback'].probability*100:.1f}%",
                f"Required team size: {predictions['resource'].recommended_team_size} developers",
                f"Incident probability: {predictions['incident'].probability*100:.1f}%",
                
                "\n## Key Contributing Factors",
                "The following factors most strongly influence these predictions:"
            ]

            for feature, importance in top_features:
                parts.append(
                    f"- {feature.replace('_', ' ').title()}: "
                    f"Impact score of {importance:.3f}"
                )

            parts.extend([
                "\n## Model-Specific Insights",
                "### Deployment Window Selection",
                windows[0].explanation,
                "\n### Rollback Risk Analysis",
                predictions['rollback'].explanation,
                "\n### Resource Requirements",
                predictions['resource'].explanation,
                "\n### Incident Prediction",
                predictions['incident'].explanation
            ])

            return "\n".join(parts)
        except Exception:
            return "Error generating system explanation"

    def _extract_key_insights(
        self,
        window: DeploymentWindow,
        predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from all predictions."""
        insights = []
        try:
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

            # Add insights from each prediction type
            prediction_types = {
                'rollback': predictions['rollback'],
                'resource': predictions['resource'],
                'incident': predictions['incident']
            }

            for aspect, prediction in prediction_types.items():
                for contrib in prediction.top_contributors:
                    if abs(contrib['impact']) > 0.1:
                        insights.append({
                            'aspect': f"{aspect}_analysis",
                            'feature': contrib['feature'],
                            'impact': contrib['impact'],
                            'deviation': contrib['deviation']
                        })

            return sorted(
                insights,
                key=lambda x: abs(x['impact']),
                reverse=True
            )
        except Exception:
            return []

    def _update_system_metrics(self, feedback: DeploymentFeedback) -> None:
        """Update system-wide ML performance metrics."""
        try:
            current_metrics = MLSystemMetrics(
                predictor_accuracies=self._calculate_predictor_accuracies(),
                confidence_scores=self._calculate_confidence_scores(),
                data_quality_scores=self._calculate_data_quality_scores(),
                historical_success_rate=self._calculate_historical_success_rate(),
                prediction_stability=self._calculate_prediction_stability(),
                calibration_score=self._calculate_calibration_score(),
                training_samples=self._get_training_sample_counts(),
                feature_coverage=self._calculate_feature_coverage(),
                recent_accuracy=self._calculate_recent_accuracy(),
                last_update_timestamp=time.time(),
                model_health_checks=self._perform_model_health_checks()
            )
            
            self.metrics_history.append(current_metrics)
            
            # Keep only recent history
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
                
        except Exception as e:
            print(f"Error updating system metrics: {str(e)}")

    def _calculate_predictor_accuracies(self) -> Dict[str, float]:
        """Calculate accuracy metrics for each predictor."""
        return {
            'window': self._calculate_window_accuracy(),
            'rollback': self._calculate_rollback_accuracy(),
            'resource': self._calculate_resource_accuracy(),
            'incident': self._calculate_incident_accuracy()
        }

    def _calculate_window_accuracy(self) -> float:
        """Calculate accuracy of window predictions."""
        try:
            if not self.window_predictor.deployment_history:
                return 0.0
                
            recent = self.window_predictor.deployment_history[-50:]
            return sum(1 for d in recent if d.success) / len(recent)
        except Exception:
            return 0.0

    def _calculate_rollback_accuracy(self) -> float:
        """Calculate accuracy of rollback predictions."""
        try:
            if not self.rollback_predictor.deployment_history:
                return 0.0
                
            recent = self.rollback_predictor.deployment_history[-50:]
            correct = sum(
                1 for d in recent
                if bool(d.rollback_occurred) == (d.predicted_rollback_probability > 0.5)
            )
            return correct / len(recent)
        except Exception:
            return 0.0

    def _calculate_resource_accuracy(self) -> float:
        """Calculate accuracy of resource predictions."""
        try:
            if not self.resource_predictor.deployment_history:
                return 0.0
                
            recent = self.resource_predictor.deployment_history[-50:]
            error_ratios = [
                abs(d.actual_support_hours - d.predicted_support_hours) / 
                max(1, d.actual_support_hours)
                for d in recent
            ]
            return 1 - min(1.0, np.mean(error_ratios))
        except Exception:
            return 0.0

    def _calculate_incident_accuracy(self) -> float:
        """Calculate accuracy of incident predictions."""
        try:
            if not self.incident_predictor.deployment_history:
                return 0.0
                
            recent = self.incident_predictor.deployment_history[-50:]
            correct = sum(
                1 for d in recent
                if bool(d.issues_encountered) == (d.predicted_incident_probability > 0.5)
            )
            return correct / len(recent)
        except Exception:
            return 0.0

    def _calculate_confidence_scores(self) -> Dict[str, float]:
        """Calculate current confidence scores for each predictor."""
        return {
            'window': self.window_predictor._calculate_data_completeness({}),
            'rollback': self.rollback_predictor._calculate_data_completeness({}),
            'resource': self.resource_predictor._calculate_data_completeness({}),
            'incident': self.incident_predictor._calculate_data_completeness({})
        }

    def _calculate_data_quality_scores(self) -> Dict[str, float]:
        """Calculate data quality metrics for each predictor."""
        return {
            'window': self._assess_data_quality(self.window_predictor),
            'rollback': self._assess_data_quality(self.rollback_predictor),
            'resource': self._assess_data_quality(self.resource_predictor),
            'incident': self._assess_data_quality(self.incident_predictor)
        }

    def _assess_data_quality(self, predictor: Any) -> float:
        """Assess data quality for a predictor."""
        try:
            if not predictor.deployment_history:
                return 0.0
                
            # Check for missing or invalid values
            valid_samples = sum(
                1 for d in predictor.deployment_history[-50:]
                if hasattr(d, 'metrics') and d.metrics
            )
            return valid_samples / 50
        except Exception:
            return 0.0

    def _calculate_historical_success_rate(self) -> float:
        """Calculate overall historical deployment success rate."""
        try:
            if not self.window_predictor.deployment_history:
                return 0.0
                
            recent = self.window_predictor.deployment_history[-100:]
            return sum(1 for d in recent if d.success) / len(recent)
        except Exception:
            return 0.0

    def _calculate_prediction_stability(self) -> float:
        """Calculate stability of predictions over time."""
        try:
            if len(self.metrics_history) < 2:
                return 1.0
                
            recent_accuracies = [
                m.predictor_accuracies
                for m in self.metrics_history[-10:]
            ]
            
            variations = []
            for predictor in ['window', 'rollback', 'resource', 'incident']:
                accuracies = [m[predictor] for m in recent_accuracies]
                variations.append(np.std(accuracies))
                
            return 1.0 - min(1.0, np.mean(variations))
        except Exception:
            return 0.0

    def _calculate_calibration_score(self) -> float:
        """Calculate calibration score across all predictors."""
        try:
            scores = []
            
            for predictor in [
                self.window_predictor,
                self.rollback_predictor,
                self.resource_predictor,
                self.incident_predictor
            ]:
                if predictor.deployment_history:
                    confidence_scores = [
                        d.confidence_score
                        for d in predictor.deployment_history[-50:]
                        if hasattr(d, 'confidence_score')
                    ]
                    if confidence_scores:
                        scores.append(np.mean(confidence_scores))
                        
            return np.mean(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _get_training_sample_counts(self) -> Dict[str, int]:
        """Get current training sample counts for each predictor."""
        return {
            'window': len(self.window_predictor.deployment_history),
            'rollback': len(self.rollback_predictor.deployment_history),
            'resource': len(self.resource_predictor.deployment_history),
            'incident': len(self.incident_predictor.deployment_history)
        }

    def _calculate_feature_coverage(self) -> Dict[str, float]:
        """Calculate feature coverage ratios for each predictor."""
        coverage = {}
        for name, predictor in [
            ('window', self.window_predictor),
            ('rollback', self.rollback_predictor),
            ('resource', self.resource_predictor),
            ('incident', self.incident_predictor)
        ]:
            try:
                if predictor.deployment_history:
                    features = len(predictor.feature_names)
                    available = sum(
                        1 for f in predictor.feature_names
                        if any(
                            hasattr(d.metrics, f)
                            for d in predictor.deployment_history[-10:]
                        )
                    )
                    coverage[name] = available / features
                else:
                    coverage[name] = 0.0
            except Exception:
                coverage[name] = 0.0
        return coverage

    def _calculate_recent_accuracy(self) -> Dict[str, List[float]]:
        """Calculate recent accuracy trends for each predictor."""
        return {
            'window': self._get_accuracy_trend(self.window_predictor),
            'rollback': self._get_accuracy_trend(self.rollback_predictor),
            'resource': self._get_accuracy_trend(self.resource_predictor),
            'incident': self._get_accuracy_trend(self.incident_predictor)
        }

    def _get_accuracy_trend(self, predictor: Any) -> List[float]:
        """Get accuracy trend for a predictor."""
        try:
            if not predictor.deployment_history:
                return [0.0]
                
            history = predictor.deployment_history[-50:]
            window_size = 10
            trends = []
            
            for i in range(0, len(history), window_size):
                window = history[i:i + window_size]
                accuracy = sum(1 for d in window if d.success) / len(window)
                trends.append(accuracy)
                
            return trends
        except Exception:
            return [0.0]

    def _perform_model_health_checks(self) -> Dict[str, bool]:
        """Perform health checks on all models."""
        return {
            'window': self._check_model_health(self.window_predictor),
            'rollback': self._check_model_health(self.rollback_predictor),
            'resource': self._check_model_health(self.resource_predictor),
            'incident': self._check_model_health(self.incident_predictor)
        }

    def _check_model_health(self, predictor: Any) -> bool:
        """Check health of a specific predictor."""
        try:
            # Check if model exists and has been trained
            if not predictor.model:
                return False
                
            # Check if we have recent feedback
            if not predictor.deployment_history:
                return False
                
            # Check if accuracy is above minimum threshold
            recent = predictor.deployment_history[-50:]
            if not recent:
                return False
                
            accuracy = sum(1 for d in recent if d.success) / len(recent)
            return accuracy >= 0.5
        except Exception:
            return False

    def _create_error_analysis(self, error_message: str) -> DeploymentAnalysis:
        """Create analysis object for error cases."""
        return DeploymentAnalysis(
            optimal_windows=[],
            rollback_prediction=None,
            resource_prediction=None,
            incident_prediction=None,
            overall_confidence=0.0,
            confidence_breakdown={},
            feature_importances={'cross_model': {}, 'by_model': {}},
            system_explanation="Analysis failed: " + error_message,
            key_insights=[],
            error_message=error_message,
            data_completeness=0.0,
            prediction_quality={'error': True}
        )

    def save_models(self, directory: str = None) -> None:
        """Save all ML models to disk."""
        try:
            save_dir = Path(directory) if directory else self.MODELS_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
            
            predictors = {
                'window': self.window_predictor,
                'rollback': self.rollback_predictor,
                'resource': self.resource_predictor,
                'incident': self.incident_predictor
            }
            
            for name, predictor in predictors.items():
                try:
                    model_path = save_dir / f"{name}_model.joblib"
                    joblib.dump(predictor.model, model_path)
                except Exception as e:
                    print(f"Error saving {name} model: {str(e)}")
                    
        except Exception as e:
            print(f"Error saving models: {str(e)}")

    def load_models(self, directory: str = None) -> None:
        """Load all ML models from disk."""
        try:
            load_dir = Path(directory) if directory else self.MODELS_DIR
            
            predictors = {
                'window': self.window_predictor,
                'rollback': self.rollback_predictor,
                'resource': self.resource_predictor,
                'incident': self.incident_predictor
            }
            
            for name, predictor in predictors.items():
                try:
                    model_path = load_dir / f"{name}_model.joblib"
                    if model_path.exists():
                        predictor.model = joblib.load(model_path)
                except Exception as e:
                    print(f"Error loading {name} model: {str(e)}")
                    
        except Exception as e:
            print(f"Error loading models: {str(e)}")

    def _get_model_versions(self) -> Dict[str, str]:
        """Get version information for all models."""
        return {
            'system': self.version,
            'window': getattr(self.window_predictor, 'version', 'unknown'),
            'rollback': getattr(self.rollback_predictor, 'version', 'unknown'),
            'resource': getattr(self.resource_predictor, 'version', 'unknown'),
            'incident': getattr(self.incident_predictor, 'version', 'unknown')
        }