from typing import Dict, Set, List, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import ResourceAllocation, DeploymentWindow

class ResourcePredictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor()
        self.explainer = MLExplainer(model_type="regressor")
        self.feature_names = [
            'cyclomatic_complexity',
            'cognitive_complexity',
            'change_risk',
            'security_issues',
            'component_coupling',
            'architecture_violations',
            'deployment_window_risk',
            'team_availability',
            'test_coverage',
            'documentation_ratio',
            'codebase_size',
            'dependency_count'
        ]

    def _extract_features(self, metrics: Dict[str, Any], window: DeploymentWindow = None) -> np.ndarray:
        """Extract features with robust error handling."""
        if not metrics or not isinstance(metrics.get('files'), dict):
            return np.zeros(len(self.feature_names))

        base_features = []
        for file_metrics in metrics['files'].values():
            try:
                complexity = file_metrics.get('complexity', {})
                security = file_metrics.get('security', {})
                architecture = file_metrics.get('architecture', {})
                
                base_features.append([
                    float(complexity.get('cyclomatic_complexity', 0)),
                    float(complexity.get('cognitive_complexity', 0)),
                    float(complexity.get('change_risk', 0)),
                    len(security.get('vulnerable_imports', [])),
                    float(architecture.get('component_coupling', 0)),
                    int(architecture.get('layering_violations', 0))
                ])
            except (TypeError, ValueError, AttributeError):
                continue

        if not base_features:
            return np.zeros(len(self.feature_names))

        # Calculate aggregate metrics
        avg_features = np.mean(base_features, axis=0)
        
        # Add window-specific and global metrics
        additional_features = [
            window.risk_score if window else 0.0,
            window.team_availability if window else 0.0,
            self._calculate_test_coverage(metrics),
            self._calculate_documentation_ratio(metrics),
            len(metrics.get('files', {})),
            self._calculate_dependency_count(metrics)
        ]
        
        return np.concatenate([avg_features, additional_features])

    def _identify_required_skills(self, metrics: Dict[str, Any]) -> Set[str]:
        """Identify required skills based on code analysis."""
        skills = set()
        
        for file_metrics in metrics.get('files', {}).values():
            try:
                complexity = file_metrics.get('complexity', {})
                security = file_metrics.get('security', {})
                architecture = file_metrics.get('architecture', {})
                
                if complexity.get('cyclomatic_complexity', 0) > 15:
                    skills.add('senior_developer')
                if security.get('potential_sql_injections', 0) > 0:
                    skills.add('security_expert')
                if architecture.get('component_coupling', 0) > 0.7:
                    skills.add('architecture_specialist')
                if complexity.get('change_risk', 0) > 0.8:
                    skills.add('domain_expert')
                if hasattr(file_metrics.get('metrics'), 'test_coverage_files'):
                    skills.add('test_engineer')
            except (TypeError, AttributeError):
                continue
        
        return skills

    def _estimate_support_duration(
        self, 
        metrics: Dict[str, Any],
        similar_deployments: List[Any]
    ) -> float:
        """Estimate required support duration."""
        if similar_deployments:
            try:
                return np.mean([d.actual_support_hours for d in similar_deployments])
            except (AttributeError, TypeError):
                pass
        
        # Fallback estimation based on metrics
        try:
            base_hours = 2.0
            complexity_factor = sum(
                m.get('complexity', {}).get('cyclomatic_complexity', 0) / 15
                for m in metrics.get('files', {}).values()
            )
            security_factor = metrics.get('summary', {}).get('security_issues', 0) * 0.5
            
            return base_hours + complexity_factor + security_factor
        except (AttributeError, TypeError):
            return base_hours

    def predict_resources(
        self, 
        metrics: Dict[str, Any],
        deployment_window: DeploymentWindow
    ) -> ResourceAllocation:
        """Predict required resources with explanations."""
        features = self._extract_features(metrics, deployment_window)
        features_reshaped = features.reshape(1, -1)
        
        # Get base prediction for team size
        predicted_size = self.model.predict(features_reshaped)[0]
        team_size = max(2, int(np.ceil(predicted_size)))  # Ensure minimum team size and round up
        
        # Get explanation
        explanation = self.explainer.explain_prediction(
            self.model,
            features_reshaped,
            self.feature_names,
            predicted_size
        )
        
        # Get feature interactions
        interactions = self.explainer.get_feature_interactions(
            self.model,
            features_reshaped,
            self.feature_names
        )
        
        # Get similar deployments and calculate support duration
        similar_deployments = self._find_similar_deployments(metrics)
        support_duration = self._estimate_support_duration(metrics, similar_deployments)
        
        # Identify required skills based on metrics and explanations
        required_skills = self._identify_required_skills(metrics)
        
        # Adjust skills based on top contributing factors
        for contributor in explanation.get('top_contributors', []):
            if contributor.get('deviation', 0) > 1.5:  # Significant deviation
                if 'security' in contributor.get('feature', ''):
                    required_skills.add('security_expert')
                elif 'architecture' in contributor.get('feature', ''):
                    required_skills.add('architecture_specialist')
                elif 'test' in contributor.get('feature', ''):
                    required_skills.add('test_engineer')

        # Calculate base confidence score
        num_relevant_samples = len(similar_deployments)
        confidence_score = self._calculate_confidence_score(num_relevant_samples)
        
        # Apply confidence penalties from explanation
        confidence_factors = explanation.get('confidence_factors', {})
        if confidence_factors:
            base_confidence = confidence_score or 0.8  # Default to 0.8 if no historical data
            if confidence_factors.get('extreme_values', 0) > 2:
                base_confidence *= 0.9
            if confidence_factors.get('importance_concentration', 0) > 0.5:
                base_confidence *= 0.9
            confidence_score = max(0.1, min(1.0, 
                base_confidence - confidence_factors.get('confidence_penalty', 0)))

        return ResourceAllocation(
            recommended_team_size=team_size,
            required_skills=required_skills,
            estimated_support_duration=support_duration,
            confidence_score=confidence_score,
            explanation=explanation['explanation'],
            feature_importance=explanation['feature_importance'],
            top_contributors=explanation['top_contributors'],
            feature_interactions=interactions
        )

    def _calculate_test_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate test coverage metric."""
        try:
            total_files = len(metrics.get('files', {}))
            if total_files == 0:
                return 0.0
            
            files_with_tests = sum(
                1 for file_metrics in metrics['files'].values()
                if file_metrics.get('metrics') and file_metrics['metrics'].test_coverage_files
            )
            return files_with_tests / total_files
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_documentation_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate documentation ratio."""
        try:
            total_code = 0
            total_comments = 0
            
            for file_metrics in metrics.get('files', {}).values():
                metrics_obj = file_metrics.get('metrics')
                if metrics_obj:
                    total_code += getattr(metrics_obj, 'lines_code', 0)
                    total_comments += getattr(metrics_obj, 'lines_comment', 0)
            
            return total_comments / total_code if total_code > 0 else 0.0
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_dependency_count(self, metrics: Dict[str, Any]) -> int:
        """Calculate total number of dependencies."""
        try:
            dependencies = set()
            for file_metrics in metrics.get('files', {}).values():
                metrics_obj = file_metrics.get('metrics')
                if metrics_obj and hasattr(metrics_obj, 'dependencies'):
                    dependencies.update(metrics_obj.dependencies)
            return len(dependencies)
        except (AttributeError, TypeError):
            return 0

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        try:
            if len(self.deployment_history) % 10 == 0:
                X = np.array([
                    self._extract_features(d.metrics) 
                    for d in self.deployment_history
                    if hasattr(d, 'metrics')
                ])
                y = np.array([
                    d.team_size 
                    for d in self.deployment_history
                    if hasattr(d, 'team_size')
                ])
                if len(X) > 0 and len(y) > 0:
                    self.model.fit(X, y)
        except (AttributeError, TypeError, ValueError) as e:
            print(f"Error updating models: {str(e)}")