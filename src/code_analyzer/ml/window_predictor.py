from datetime import time
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor

from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import DeploymentWindow

class WindowPredictor(BasePredictor):
    """Predicts optimal deployment windows with explainable ML."""
    
    MODEL_PATH = Path("models/window_predictor.joblib")
    
    def __init__(self):
        super().__init__()
        self.model = self._load_or_create_model()
        self.explainer = MLExplainer(model_type="regressor")
        self.feature_names = [
            'cyclomatic_complexity_mean',
            'cyclomatic_complexity_max',
            'cognitive_complexity_mean',
            'change_risk_mean',
            'security_issues',
            'component_coupling_mean',
            'duplication_ratio',
            'test_coverage',
            'team_size_ratio',
            'hour_of_day'
        ]
        self.min_samples_for_training = 50
        self.training_frequency = 10

    def _load_or_create_model(self) -> RandomForestRegressor:
        """Load existing model or create new one if not found."""
        try:
            return joblib.load(self.MODEL_PATH)
        except:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

    def _save_model(self) -> None:
        """Save trained model to disk."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)

    def _extract_features(self, metrics: Dict[str, Any], hour: Optional[int] = None) -> np.ndarray:
        """Extract features with robust error handling."""
        try:
            if not metrics or not isinstance(metrics.get('files'), dict):
                return np.zeros(len(self.feature_names))

            complexity_scores = []
            cognitive_scores = []
            risk_scores = []
            coupling_scores = []
            
            for file_metrics in metrics['files'].values():
                try:
                    complexity = file_metrics.get('complexity', {})
                    architecture = file_metrics.get('architecture', {})
                    
                    if isinstance(complexity, dict):
                        complexity_scores.append(
                            float(complexity.get('cyclomatic_complexity', 0))
                        )
                        cognitive_scores.append(
                            float(complexity.get('cognitive_complexity', 0))
                        )
                        risk_scores.append(
                            float(complexity.get('change_risk', 0))
                        )
                    
                    if isinstance(architecture, dict):
                        coupling_scores.append(
                            float(architecture.get('component_coupling', 0))
                        )
                except (TypeError, ValueError):
                    continue

            if not complexity_scores:
                return np.zeros(len(self.feature_names))

            summary = metrics.get('summary', {})
            features = [
                np.mean(complexity_scores),
                np.max(complexity_scores),
                np.mean(cognitive_scores),
                np.mean(risk_scores),
                float(summary.get('security_issues', 0)),
                np.mean(coupling_scores),
                self._calculate_duplication_ratio(metrics),
                self._calculate_test_coverage(metrics),
                len(metrics['files']),
                float(hour if hour is not None else 0)
            ]
            
            return np.array(features)
        except Exception:
            return np.zeros(len(self.feature_names))

    def _calculate_window_risk(self, hour: int, metrics: Dict[str, Any]) -> float:
        """Calculate risk score for a deployment window."""
        try:
            base_risk = 0.5
            
            # Time-based risk factors
            if 9 <= hour <= 17:  # Business hours
                base_risk += 0.2
            elif 0 <= hour <= 5:  # Early morning
                base_risk -= 0.2
            
            # Code complexity risk
            if metrics.get('files'):
                complexity_risk = sum(
                    1 for m in metrics['files'].values()
                    if isinstance(m.get('complexity'), dict) and 
                    m['complexity'].get('cyclomatic_complexity', 0) > 15
                ) / len(metrics['files'])
            else:
                complexity_risk = 0.0
            
            # Security risk
            summary = metrics.get('summary', {})
            total_files = len(metrics.get('files', {}))
            security_risk = (
                float(summary.get('security_issues', 0)) / 
                max(1, total_files)
            )
            
            return min(1.0, base_risk + complexity_risk * 0.3 + security_risk * 0.3)
        except Exception:
            return 0.5

    def _calculate_team_availability(
        self, 
        team_availability: Dict[str, List[time]],
        start: time,
        end: time
    ) -> float:
        """Calculate team availability score with validation."""
        try:
            if not team_availability:
                return 0.0
                
            available_count = 0
            total_count = len(team_availability)
            
            for member_schedule in team_availability.values():
                if not isinstance(member_schedule, list):
                    continue
                    
                for schedule in member_schedule:
                    if not isinstance(schedule, (list, tuple)) or len(schedule) != 2:
                        continue
                        
                    avail_start, avail_end = schedule
                    if (isinstance(avail_start, time) and 
                        isinstance(avail_end, time) and
                        avail_start <= start and avail_end >= end):
                        available_count += 1
                        break
            
            return available_count / total_count if total_count > 0 else 0.0
        except Exception:
            return 0.0

    def predict_optimal_windows(
        self, 
        metrics: Dict[str, Any],
        team_availability: Dict[str, List[time]]
    ) -> List[DeploymentWindow]:
        """Predict optimal deployment windows with comprehensive analysis."""
        windows = []
        
        for hour in range(24):
            try:
                start = time(hour=hour)
                end = time(hour=(hour + 2) % 24)
                
                # Extract and predict
                features = self._extract_features(metrics, hour)
                features_reshaped = features.reshape(1, -1)
                success_rate = float(self.model.predict(features_reshaped)[0])
                
                # Get explanation
                explanation = self.explainer.explain_prediction(
                    self.model,
                    features_reshaped,
                    self.feature_names,
                    success_rate
                )
                
                # Get feature interactions
                interactions = self.explainer.get_feature_interactions(
                    self.model,
                    features_reshaped,
                    self.feature_names
                )
                
                # Calculate metrics
                team_avail = self._calculate_team_availability(
                    team_availability,
                    start,
                    end
                )
                risk_score = self._calculate_window_risk(hour, metrics)
                
                # Calculate confidence
                confidence_score = self._calculate_comprehensive_confidence(
                    metrics,
                    explanation,
                    len(self.deployment_history)
                )
                
                windows.append(DeploymentWindow(
                    start_time=start,
                    end_time=end,
                    risk_score=risk_score,
                    team_availability=team_avail,
                    historical_success_rate=max(0.0, min(1.0, success_rate)),
                    required_team_size=self._estimate_team_size(metrics),
                    required_skills=self._identify_required_skills(metrics),
                    estimated_duration=self._estimate_duration(metrics),
                    system_load=self._get_system_load(hour),
                    concurrent_deployments=self._get_concurrent_deployments(start, end),
                    explanation=explanation['explanation'],
                    feature_importance=explanation['feature_importance'],
                    top_contributors=explanation['top_contributors'],
                    feature_interactions=interactions,
                    confidence_score=confidence_score,
                    data_completeness=self._calculate_data_completeness(metrics),
                    prediction_quality=self._assess_prediction_quality(explanation)
                ))
            except Exception as e:
                print(f"Error processing hour {hour}: {str(e)}")
                continue

        return sorted(
            windows,
            key=lambda w: (
                w.historical_success_rate * 0.4 +
                w.team_availability * 0.4 +
                (1 - w.risk_score) * 0.2
            ),
            reverse=True
        )

    def _calculate_test_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate test coverage with validation."""
        try:
            total_files = len(metrics.get('files', {}))
            if total_files == 0:
                return 0.0
                
            files_with_tests = sum(
                1 for m in metrics['files'].values()
                if hasattr(m.get('metrics'), 'test_coverage_files') and 
                m['metrics'].test_coverage_files
            )
            return files_with_tests / total_files
        except Exception:
            return 0.0

    def _calculate_duplication_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate code duplication ratio."""
        try:
            summary = metrics.get('summary', {})
            duplication = summary.get('duplication', {})
            total_files = len(metrics.get('files', {}))
            
            return (
                float(duplication.get('total_duplicates', 0)) / 
                max(1, total_files)
            )
        except Exception:
            return 0.0

    def _estimate_team_size(self, metrics: Dict[str, Any]) -> int:
        """Estimate required team size based on metrics."""
        try:
            complexity_score = np.mean([
                m.get('complexity', {}).get('cyclomatic_complexity', 0)
                for m in metrics.get('files', {}).values()
            ])
            return max(2, int(complexity_score / 10))
        except Exception:
            return 2

    def _identify_required_skills(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify required skills based on metrics."""
        skills = set()
        
        try:
            for file_metrics in metrics.get('files', {}).values():
                complexity = file_metrics.get('complexity', {})
                security = file_metrics.get('security', {})
                
                if complexity.get('cyclomatic_complexity', 0) > 15:
                    skills.add('senior_developer')
                if security.get('potential_sql_injections', 0) > 0:
                    skills.add('security_expert')
                if not file_metrics.get('test_coverage_files'):
                    skills.add('test_engineer')
        except Exception:
            skills.add('developer')  # Default skill
            
        return sorted(list(skills))

    def _estimate_duration(self, metrics: Dict[str, Any]) -> float:
        """Estimate deployment duration in hours."""
        try:
            base_duration = 2.0
            complexity_factor = sum(
                m.get('complexity', {}).get('cyclomatic_complexity', 0) > 15
                for m in metrics.get('files', {}).values()
            ) * 0.5
            
            return base_duration + complexity_factor
        except Exception:
            return 2.0

    def _get_system_load(self, hour: int) -> Dict[str, float]:
        """Get system load metrics for given hour."""
        try:
            # This would typically come from monitoring systems
            return {
                'cpu_utilization': 0.5,
                'memory_usage': 0.6,
                'network_load': 0.4
            }
        except Exception:
            return {}

    def _get_concurrent_deployments(self, start: time, end: time) -> List[str]:
        """Get list of concurrent deployments."""
        try:
            # This would typically come from deployment scheduling system
            return []
        except Exception:
            return []

    def _calculate_comprehensive_confidence(
        self,
        metrics: Dict[str, Any],
        explanation: Dict[str, Any],
        num_similar_cases: int
    ) -> float:
        """Calculate confidence score considering multiple factors."""
        try:
            # Base confidence from historical data
            base_confidence = super()._calculate_confidence_score(num_similar_cases)
            
            # Explainability confidence
            confidence_factors = explanation.get('confidence_factors', {})
            explainer_confidence = 1.0 - confidence_factors.get('confidence_penalty', 0.0)
            
            # Data completeness
            data_completeness = self._calculate_data_completeness(metrics)
            
            # Weighted combination
            return max(0.0, min(1.0, 
                base_confidence * 0.4 +
                explainer_confidence * 0.4 +
                data_completeness * 0.2
            ))
        except Exception:
            return 0.5

    def _assess_prediction_quality(self, explanation: Dict[str, Any]) -> Dict[str, bool]:
        """Assess various quality factors of the prediction."""
        try:
            confidence_factors = explanation.get('confidence_factors', {}).get('confidence_factors', {})
            return {
                'sufficient_data': len(self.deployment_history) >= self.min_samples_for_training,
                'balanced_features': confidence_factors.get('balanced_feature_contribution', False),
                'normal_value_ranges': not confidence_factors.get('extreme_values_present', True),
                'stable_prediction': confidence_factors.get('importance_concentration', 1.0) < 0.5
            }
        except Exception:
            return {
                'sufficient_data': False,
                'balanced_features': False,
                'normal_value_ranges': False,
                'stable_prediction': False
            }

    def _calculate_data_completeness(self, metrics: Dict[str, Any]) -> float:
        """Calculate completeness score for input data."""
        try:
            required_fields = {'complexity', 'security', 'architecture'}
            total_files = len(metrics.get('files', {}))
            if total_files == 0:
                return 0.0
                
            complete_files = sum(
                1 for file_metrics in metrics['files'].values()
                if all(
                    field in file_metrics and 
                    isinstance(file_metrics[field], dict) and 
                    file_metrics[field] is not None
                    for field in required_fields
                )
            )
            
            return complete_files / total_files
        except Exception:
            return 0.0

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        if len(self.deployment_history) < self.min_samples_for_training:
            return
            
        if len(self.deployment_history) % self.training_frequency != 0:
            return
            
        try:
            X = []
            y = []
            
            for feedback in self.deployment_history:
                try:
                    features = self._extract_features(feedback.metrics)
                    if features is not None and not np.isnan(features).any():
                        X.append(features)
                        y.append(float(feedback.success))
                except Exception:
                    continue
                    
            if len(X) >= self.min_samples_for_training:
                X = np.array(X)
                y = np.array(y)
                self.model.fit(X, y)
                self._save_model()
                
        except Exception as e:
            print(f"Error updating models: {str(e)}")
    
    def _calculate_data_completeness(self, metrics: Dict[str, Any]) -> float:
        """Calculate completeness score for input data."""
        try:
            required_fields = {'complexity', 'security', 'architecture'}
            total_files = len(metrics.get('files', {}))
            if total_files == 0:
                return 0.0
                
            complete_files = sum(
                1 for file_metrics in metrics['files'].values()
                if isinstance(file_metrics, dict) and all(
                    field in file_metrics and 
                    isinstance(file_metrics[field], dict) and 
                    file_metrics[field] is not None
                    for field in required_fields
                )
            )
            
            return complete_files / total_files
        except Exception:
            return 0.0

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        if len(self.deployment_history) < self.min_samples_for_training:
            return
            
        if len(self.deployment_history) % self.training_frequency != 0:
            return
            
        try:
            X = []
            y = []
            
            for feedback in self.deployment_history:
                try:
                    features = self._extract_features(feedback.metrics)
                    if features is not None and not np.isnan(features).any():
                        X.append(features)
                        y.append(float(feedback.success))
                except Exception:
                    continue
                    
            if len(X) >= self.min_samples_for_training:
                X = np.array(X)
                y = np.array(y)
                self.model.fit(X, y)
                self._save_model()
                
        except Exception as e:
            print(f"Error updating models: {str(e)}")