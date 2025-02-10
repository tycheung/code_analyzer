from datetime import time
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import DeploymentWindow

class WindowPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor()
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

    def _extract_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Extract features for deployment window prediction."""
        complexity_scores = []
        cognitive_scores = []
        risk_scores = []
        coupling_scores = []
        
        for file_metrics in metrics['files'].values():
            complexity_scores.append(
                file_metrics['complexity']['cyclomatic_complexity']
            )
            cognitive_scores.append(
                file_metrics['complexity']['cognitive_complexity']
            )
            risk_scores.append(
                file_metrics['complexity']['change_risk']
            )
            coupling_scores.append(
                file_metrics['architecture']['component_coupling']
            )
        
        if not complexity_scores:  # Handle empty metrics
            return np.zeros(len(self.feature_names))

        return np.array([
            np.mean(complexity_scores),
            np.max(complexity_scores),
            np.mean(cognitive_scores),
            np.mean(risk_scores),
            metrics['summary']['security_issues'],
            np.mean(coupling_scores),
            metrics['summary']['duplication']['total_duplicates'] / len(metrics['files']),
            self._calculate_test_coverage(metrics),
            len(metrics['files']),
            0  # Hour placeholder, filled during window analysis
        ])

    def _calculate_window_risk(self, hour: int, metrics: Dict[str, Any]) -> float:
        """Calculate risk score for a deployment window."""
        base_risk = 0.5
        
        # Higher risk during peak hours
        if 9 <= hour <= 17:
            base_risk += 0.2
        elif 0 <= hour <= 5:
            base_risk -= 0.2
        
        complexity_risk = sum(
            m['complexity']['cyclomatic_complexity'] > 15
            for m in metrics['files'].values()
        ) / len(metrics['files'])
        
        security_risk = metrics['summary']['security_issues'] / len(metrics['files'])
        
        return min(1.0, base_risk + complexity_risk * 0.3 + security_risk * 0.3)

    def _calculate_team_availability(
        self, 
        team_availability: Dict[str, List[time]],
        start: time,
        end: time
    ) -> float:
        """Calculate team availability score for a given time window."""
        available_count = 0
        total_count = len(team_availability)
        
        for member_schedule in team_availability.values():
            for avail_start, avail_end in member_schedule:
                if avail_start <= start and avail_end >= end:
                    available_count += 1
                    break
        
        return available_count / total_count if total_count > 0 else 0.0

    def predict_optimal_windows(
        self, 
        metrics: Dict[str, Any],
        team_availability: Dict[str, List[time]]
    ) -> List[DeploymentWindow]:
        """Predict optimal deployment windows with explanations."""
        base_features = self._extract_features(metrics)
        windows = []

        # Analyze each hour
        for hour in range(24):
            start = time(hour=hour)
            end = time(hour=(hour + 2) % 24)
            
            # Update hour-specific feature
            features = base_features.copy()
            features[-1] = hour
            features_reshaped = features.reshape(1, -1)
            
            # Get prediction and explanation
            success_rate = self.model.predict(features_reshaped)[0]
            explanation = self.explainer.explain_prediction(
                self.model,
                features_reshaped,
                self.feature_names,
                success_rate
            )
            
            # Calculate other metrics
            team_avail = self._calculate_team_availability(team_availability, start, end)
            risk_score = self._calculate_window_risk(hour, metrics)
            
            windows.append(DeploymentWindow(
                start_time=start,
                end_time=end,
                risk_score=risk_score,
                team_availability=team_avail,
                historical_success_rate=max(0.0, min(1.0, success_rate)),
                explanation=explanation['explanation'],
                feature_importance=explanation['feature_importance'],
                top_contributors=explanation['top_contributors']
            ))

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
        """Calculate test coverage metric."""
        total_files = len(metrics['files'])
        files_with_tests = sum(
            1 for m in metrics['files'].values()
            if hasattr(m, 'test_coverage_files') and m.test_coverage_files
        )
        return files_with_tests / total_files if total_files > 0 else 0.0

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        if len(self.deployment_history) % 10 == 0:  # Update every 10 new deployments
            X = np.array([self._extract_features(d) for d in self.deployment_history])
            y = np.array([d.success for d in self.deployment_history])
            self.model.fit(X, y)