from typing import Dict, List, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import IncidentPrediction, DeploymentWindow

class IncidentPredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()
        self.explainer = MLExplainer(model_type="classifier")
        self.feature_names = [
            'cyclomatic_complexity_mean',
            'cyclomatic_complexity_max',
            'cognitive_complexity_mean',
            'change_risk_mean',
            'security_vulnerabilities',
            'component_coupling_mean',
            'circular_dependencies',
            'layering_violations',
            'test_coverage',
            'documentation_ratio',
            'deployment_window_risk',
            'team_availability',
            'churn_rate'
        ]

    def _extract_features(self, metrics: Dict[str, Any], window: DeploymentWindow = None) -> np.ndarray:
        """Extract features for incident prediction."""
        if not metrics or 'files' not in metrics or not metrics['files']:
            return np.zeros(len(self.feature_names))

        complexity_features = []
        security_count = 0
        dependencies_count = 0
        
        for file_path, file_metrics in metrics['files'].items():
            try:
                # Complexity metrics
                complexity = file_metrics.get('complexity', {})
                complexity_features.append([
                    complexity.get('cyclomatic_complexity', 0),
                    complexity.get('cognitive_complexity', 0),
                    complexity.get('change_risk', 0)
                ])
                
                # Security issues
                security = file_metrics.get('security', {})
                security_count += (
                    security.get('potential_sql_injections', 0) +
                    security.get('hardcoded_secrets', 0) +
                    len(security.get('vulnerable_imports', []))
                )
                
                # Architecture issues
                architecture = file_metrics.get('architecture', {})
                dependencies_count += len(architecture.get('circular_dependencies', []))
            except (TypeError, AttributeError):
                continue

        if not complexity_features:
            return np.zeros(len(self.feature_names))

        complexity_features = np.array(complexity_features)
        
        try:
            features = [
                np.mean(complexity_features[:, 0]),  # cyclomatic mean
                np.max(complexity_features[:, 0]),   # cyclomatic max
                np.mean(complexity_features[:, 1]),  # cognitive mean
                np.mean(complexity_features[:, 2]),  # change risk mean
                security_count,
                self._calculate_coupling(metrics),
                dependencies_count,
                self._calculate_layering_violations(metrics),
                self._calculate_test_coverage(metrics),
                self._calculate_documentation_ratio(metrics),
                window.risk_score if window else 0.0,
                window.team_availability if window else 0.0,
                self._calculate_churn_rate(metrics)
            ]
            return np.array(features)
        except (TypeError, AttributeError, IndexError):
            return np.zeros(len(self.feature_names))
        
    def _identify_potential_incident_areas(
        self, 
        metrics: Dict[str, Any],
        top_contributors: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify areas likely to cause incidents."""
        risk_areas = []
        
        # Check top contributing factors
        for contributor in top_contributors:
            feature = contributor['feature']
            deviation = contributor['deviation']
            
            if abs(deviation) > 1.5:  # Significant deviation
                if 'complexity' in feature:
                    risk_areas.append("High code complexity")
                elif 'security' in feature:
                    risk_areas.append("Security vulnerabilities")
                elif 'coupling' in feature:
                    risk_areas.append("Tight component coupling")
                elif 'dependencies' in feature:
                    risk_areas.append("Complex dependencies")
                elif 'test' in feature:
                    risk_areas.append("Insufficient testing")
        
        # Check specific metrics
        for file_path, metrics in metrics['files'].items():
            if metrics['complexity']['cyclomatic_complexity'] > 15:
                risk_areas.append(f"High complexity in {file_path}")
            if metrics['security']['potential_sql_injections'] > 0:
                risk_areas.append(f"SQL injection risks in {file_path}")
            if metrics['architecture']['component_coupling'] > 0.8:
                risk_areas.append(f"High coupling in {file_path}")
            if len(metrics['architecture']['circular_dependencies']) > 0:
                risk_areas.append(f"Circular dependencies in {file_path}")
        
        return list(set(risk_areas))  # Remove duplicates

    def _predict_incident_severity(
        self, 
        probability: float,
        explanation: Dict[str, Any]
    ) -> str:
        """Predict the severity level of potential incidents."""
        if probability < 0.3:
            return "low"
        
        # Consider feature importance for severity
        high_risk_features = sum(
            1 for contrib in explanation['top_contributors']
            if abs(contrib['impact']) > 0.2
        )
        
        if probability > 0.7 or high_risk_features >= 3:
            return "high"
        return "medium"

    def _estimate_resolution_time(
        self, 
        metrics: Dict[str, Any],
        similar_deployments: List[Any]
    ) -> float:
        """Estimate time needed to resolve potential incidents."""
        if similar_deployments:
            resolution_times = []
            for deployment in similar_deployments:
                if deployment.issues_encountered:
                    resolution_times.append(deployment.actual_support_hours)
            if resolution_times:
                return np.mean(resolution_times)
        
        # Fallback estimation based on metrics
        base_time = 1.0
        complexity_factor = sum(
            m['complexity']['cyclomatic_complexity'] / 15
            for m in metrics['files'].values()
        )
        security_factor = metrics['summary']['security_issues'] * 0.5
        
        return base_time + complexity_factor + security_factor

    def predict_incidents(
        self, 
        metrics: Dict[str, Any],
        deployment_window: DeploymentWindow
    ) -> IncidentPrediction:
        """Predict potential post-deployment incidents with explanations."""
        features = self._extract_features(metrics, deployment_window)
        features_reshaped = features.reshape(1, -1)
        
        # Get base prediction
        probability = self.model.predict_proba(features_reshaped)[0][1]
        
        # Get explanation
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
        
        # Find similar deployments
        similar_deployments = self._find_similar_deployments(metrics)
        
        # Identify risk areas using explanation
        potential_areas = self._identify_potential_incident_areas(
            metrics,
            explanation['top_contributors']
        )
        
        # Predict severity and resolution time
        severity = self._predict_incident_severity(probability, explanation)
        resolution_time = self._estimate_resolution_time(metrics, similar_deployments)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            explanation['confidence_factors']
        )

        # Calculate confidence
        confidence_factors = explanation['confidence_factors']
        base_confidence = 0.8  # Start with reasonable base confidence
        if confidence_factors['extreme_values'] > 2:
            base_confidence *= 0.9
        if confidence_factors['importance_concentration'] > 0.5:
            base_confidence *= 0.9
        confidence_score = max(0.0, min(1.0, base_confidence - confidence_factors['confidence_penalty']))

        return IncidentPrediction(
            probability=probability,
            potential_areas=potential_areas,
            severity_level=severity,
            estimated_resolution_time=resolution_time,
            confidence_score=confidence_score,
            explanation=explanation['explanation'],
            feature_importance=explanation['feature_importance'],
            top_contributors=explanation['top_contributors'],
            feature_interactions=interactions
        )

    def _calculate_coupling(self, metrics: Dict[str, Any]) -> float:
        """Calculate average component coupling."""
        couplings = [
            m['architecture']['component_coupling']
            for m in metrics['files'].values()
        ]
        return np.mean(couplings) if couplings else 0.0

    def _calculate_layering_violations(self, metrics: Dict[str, Any]) -> int:
        """Calculate total layering violations."""
        return sum(
            m['architecture']['layering_violations']
            for m in metrics['files'].values()
        )

    def _calculate_test_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate test coverage metric."""
        total_files = len(metrics['files'])
        files_with_tests = sum(
            1 for m in metrics['files'].values()
            if m['metrics'].test_coverage_files
        )
        return files_with_tests / total_files if total_files > 0 else 0.0

    def _calculate_documentation_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate documentation ratio."""
        total_code = sum(m['metrics'].lines_code for m in metrics['files'].values())
        total_comments = sum(m['metrics'].lines_comment for m in metrics['files'].values())
        return total_comments / total_code if total_code > 0 else 0.0

    def _calculate_churn_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate code churn rate."""
        churn_rates = []
        for file_metrics in metrics['files'].values():
            if file_metrics['metrics'].change_probability:
                churn_rates.append(file_metrics['metrics'].change_probability.churn_rate)
        return np.mean(churn_rates) if churn_rates else 0.0

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        if len(self.deployment_history) % 10 == 0:
            X = np.array([self._extract_features(d.metrics) for d in self.deployment_history])
            y = np.array([bool(len(d.issues_encountered)) for d in self.deployment_history])
            self.model.fit(X, y)