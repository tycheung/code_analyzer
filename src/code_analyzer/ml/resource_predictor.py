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
        """Extract features for resource prediction."""
        base_features = []
        for file_metrics in metrics['files'].values():
            base_features.append([
                file_metrics['complexity']['cyclomatic_complexity'],
                file_metrics['complexity']['cognitive_complexity'],
                file_metrics['complexity']['change_risk'],
                len(file_metrics['security']['vulnerable_imports']),
                file_metrics['architecture']['component_coupling'],
                file_metrics['architecture']['layering_violations']
            ])

        if not base_features:  # Handle empty metrics
            return np.zeros(len(self.feature_names))

        # Calculate aggregate metrics
        avg_features = np.mean(base_features, axis=0)
        
        # Add window-specific and global metrics
        additional_features = [
            window.risk_score if window else 0.0,
            window.team_availability if window else 0.0,
            self._calculate_test_coverage(metrics),
            self._calculate_documentation_ratio(metrics),
            len(metrics['files']),
            self._calculate_dependency_count(metrics)
        ]
        
        return np.concatenate([avg_features, additional_features])

    def _identify_required_skills(self, metrics: Dict[str, Any]) -> Set[str]:
        """Identify required skills based on code analysis."""
        skills = set()
        
        for file_metrics in metrics['files'].values():
            if file_metrics['complexity']['cyclomatic_complexity'] > 15:
                skills.add('senior_developer')
            if file_metrics['security']['potential_sql_injections'] > 0:
                skills.add('security_expert')
            if file_metrics['architecture']['component_coupling'] > 0.7:
                skills.add('architecture_specialist')
            if file_metrics['complexity']['change_risk'] > 0.8:
                skills.add('domain_expert')
            if hasattr(file_metrics, 'test_coverage_files'):
                skills.add('test_engineer')
        
        return skills

    def _estimate_support_duration(
        self, 
        metrics: Dict[str, Any],
        similar_deployments: List[Any]
    ) -> float:
        """Estimate required support duration."""
        if similar_deployments:
            return np.mean([d.actual_support_hours for d in similar_deployments])
        
        # Fallback estimation based on metrics
        base_hours = 2.0
        complexity_factor = sum(
            m['complexity']['cyclomatic_complexity'] / 15
            for m in metrics['files'].values()
        )
        security_factor = metrics['summary']['security_issues'] * 0.5
        
        return base_hours + complexity_factor + security_factor

    def predict_resources(
        self, 
        metrics: Dict[str, Any],
        deployment_window: DeploymentWindow
    ) -> ResourceAllocation:
        """Predict required team size and skills with explanations."""
        features = self._extract_features(metrics, deployment_window)
        features_reshaped = features.reshape(1, -1)
        
        # Get base prediction for team size
        team_size = max(2, int(self.model.predict(features_reshaped)[0]))
        
        # Get explanation
        explanation = self.explainer.explain_prediction(
            self.model,
            features_reshaped,
            self.feature_names,
            team_size
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
        for contributor in explanation['top_contributors']:
            if contributor['deviation'] > 1.5:  # Significant deviation
                if 'security' in contributor['feature']:
                    required_skills.add('security_expert')
                elif 'architecture' in contributor['feature']:
                    required_skills.add('architecture_specialist')
                elif 'test' in contributor['feature']:
                    required_skills.add('test_engineer')

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            explanation['confidence_factors']
        )

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
        total_files = len(metrics['files'])
        files_with_tests = sum(
            1 for m in metrics['files'].values()
            if hasattr(m, 'test_coverage_files') and m.test_coverage_files
        )
        return files_with_tests / total_files if total_files > 0 else 0.0

    def _calculate_documentation_ratio(self, metrics: Dict[str, Any]) -> float:
        """Calculate documentation ratio."""
        total_code = sum(m.lines_code for m in metrics['files'].values())
        total_comments = sum(m.lines_comment for m in metrics['files'].values())
        return total_comments / total_code if total_code > 0 else 0.0

    def _calculate_dependency_count(self, metrics: Dict[str, Any]) -> int:
        """Calculate total number of dependencies."""
        dependencies = set()
        for file_metrics in metrics['files'].values():
            if hasattr(file_metrics, 'dependencies'):
                dependencies.update(file_metrics.dependencies)
        return len(dependencies)

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new data."""
        if len(self.deployment_history) % 10 == 0:
            X = np.array([self._extract_features(d) for d in self.deployment_history])
            y = np.array([d.team_size for d in self.deployment_history])
            self.model.fit(X, y)