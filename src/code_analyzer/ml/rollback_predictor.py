from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from .base_predictor import BasePredictor
from .base_explainer import MLExplainer
from ..models.metrics import RollbackPrediction, DeploymentWindow, DeploymentFeedback

class RollbackPredictor(BasePredictor):
    """Predicts deployment rollback likelihood with comprehensive explainability."""
    
    MODEL_PATH = Path("models/rollback_predictor.joblib")
    
    def __init__(self):
        super().__init__()
        self.model = self._load_or_create_model()
        self.explainer = MLExplainer(model_type="classifier")
        self.feature_names = [
            'cyclomatic_complexity_mean',
            'cyclomatic_complexity_max',
            'cognitive_complexity_mean',
            'change_risk_mean',
            'security_vulnerabilities',
            'component_coupling_mean',
            'deploy_window_risk',
            'team_availability',
            'recent_changes',
            'test_coverage',
            'documentation_coverage',
            'churn_rate'
        ]
        self.min_samples_for_training = 50
        self.training_frequency = 20

    def _load_or_create_model(self) -> RandomForestClassifier:
        """Load existing model or create new one if not found."""
        try:
            return joblib.load(self.MODEL_PATH)
        except:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

    def _extract_features(self, metrics: Dict[str, Any], window: Optional[DeploymentWindow] = None) -> np.ndarray:
        """Extract features with robust error handling."""
        if not metrics or not isinstance(metrics.get('files'), dict):
            return np.zeros(len(self.feature_names))

        try:
            complexity_features = []
            security_count = 0
            
            for file_metrics in metrics['files'].values():
                complexity = file_metrics.get('complexity', {})
                if isinstance(complexity, dict):
                    complexity_features.append([
                        float(complexity.get('cyclomatic_complexity', 0)),
                        float(complexity.get('cognitive_complexity', 0)),
                        float(complexity.get('change_risk', 0))
                    ])
                
                security = file_metrics.get('security', {})
                if isinstance(security, dict):
                    security_count += (
                        security.get('potential_sql_injections', 0) +
                        security.get('hardcoded_secrets', 0) +
                        len(security.get('vulnerable_imports', []))
                    )

            if not complexity_features:
                return np.zeros(len(self.feature_names))

            complexity_features = np.array(complexity_features)
            features = [
                np.mean(complexity_features[:, 0]),  # cyclomatic mean
                np.max(complexity_features[:, 0]),   # cyclomatic max
                np.mean(complexity_features[:, 1]),  # cognitive mean
                np.mean(complexity_features[:, 2]),  # change risk mean
                security_count,
                self._calculate_coupling(metrics),
                window.risk_score if window else 0.0,
                window.team_availability if window else 0.0,
                self._calculate_recent_changes(metrics),
                self._calculate_test_coverage(metrics),
                self._calculate_documentation_coverage(metrics),
                self._calculate_churn_rate(metrics)
            ]
            return np.array(features)
        except (TypeError, AttributeError, IndexError):
            return np.zeros(len(self.feature_names))

    def predict_rollback(
        self, 
        metrics: Dict[str, Any],
        deployment_window: DeploymentWindow
    ) -> RollbackPrediction:
        """Predict deployment rollback probability with comprehensive analysis."""
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
        
        # Identify critical files
        critical_files = self._identify_critical_files(metrics, explanation['top_contributors'])
        
        # Calculate confidence and severity
        confidence_score = self._calculate_comprehensive_confidence(
            metrics,
            explanation,
            len(similar_deployments)
        )
        severity_level = self._determine_severity_level(probability, explanation)
        
        # Generate historical patterns
        historical_pattern = self._analyze_historical_pattern(similar_deployments)
        
        return RollbackPrediction(
            probability=probability,
            risk_factors=self._analyze_risk_factors(explanation),
            severity_level=severity_level,
            critical_files=critical_files,
            mitigation_suggestions=self._generate_mitigation_suggestions(explanation),
            recommended_reviewers=self._identify_reviewers(metrics, critical_files),
            confidence_score=confidence_score,
            prediction_quality=self._assess_prediction_quality(explanation),
            data_completeness=self._calculate_data_completeness(metrics),
            explanation=explanation['explanation'],
            feature_importance=explanation['feature_importance'],
            top_contributors=explanation['top_contributors'],
            feature_interactions=interactions,
            similar_cases=[d.deployment_id for d in similar_deployments],
            historical_pattern=historical_pattern
        )

    def _identify_critical_files(
        self,
        metrics: Dict[str, Any],
        top_contributors: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify files most likely to cause rollback."""
        critical_files = []
        
        for file_path, file_metrics in metrics.get('files', {}).items():
            if not isinstance(file_metrics, dict):
                continue
                
            risk_score = 0
            
            # Check complexity
            if isinstance(file_metrics.get('complexity'), dict):
                complexity = file_metrics['complexity']
                if complexity.get('cyclomatic_complexity', 0) > 15:
                    risk_score += 1
                if complexity.get('cognitive_complexity', 0) > 20:
                    risk_score += 1
                    
            # Check security
            if isinstance(file_metrics.get('security'), dict):
                security = file_metrics['security']
                if security.get('potential_sql_injections', 0) > 0:
                    risk_score += 2
                if security.get('hardcoded_secrets', 0) > 0:
                    risk_score += 2
                    
            # Check architecture
            if isinstance(file_metrics.get('architecture'), dict):
                architecture = file_metrics['architecture']
                if architecture.get('component_coupling', 0) > 0.7:
                    risk_score += 1
                    
            if risk_score >= 2:
                critical_files.append(file_path)
                
        return sorted(critical_files, key=lambda x: len(x))[:5]  # Top 5 critical files

    def _calculate_comprehensive_confidence(
        self,
        metrics: Dict[str, Any],
        explanation: Dict[str, Any],
        num_similar_cases: int
    ) -> float:
        """Calculate confidence score considering multiple factors."""
        # Start with base confidence
        base_confidence = super()._calculate_confidence_score(num_similar_cases)
        
        # Consider explanation confidence factors
        confidence_factors = explanation.get('confidence_factors', {})
        explainer_confidence = 1.0 - confidence_factors.get('confidence_penalty', 0.0)
        
        # Consider data completeness
        data_completeness = self._calculate_data_completeness(metrics)
        
        # Weighted combination
        confidence_score = (
            0.4 * base_confidence +
            0.4 * explainer_confidence +
            0.2 * data_completeness
        )
        
        return max(0.0, min(1.0, confidence_score))

    def _determine_severity_level(
        self,
        probability: float,
        explanation: Dict[str, Any]
    ) -> str:
        """Determine severity level based on probability and impact."""
        if probability < 0.3:
            return "low"
            
        high_impact_features = sum(
            1 for contrib in explanation.get('top_contributors', [])
            if abs(contrib.get('impact', 0)) > 0.2
        )
        
        if probability >= 0.7 or high_impact_features >= 3:
            return "high"
        return "medium"

    def _analyze_historical_pattern(
        self,
        similar_deployments: List[DeploymentFeedback]
    ) -> Dict[str, float]:
        """Analyze patterns in similar historical deployments."""
        if not similar_deployments:
            return {}
            
        total = len(similar_deployments)
        rollback_count = sum(1 for d in similar_deployments if d.rollback_occurred)
        
        return {
            'historical_rollback_rate': rollback_count / total,
            'avg_deployment_time': np.mean([d.actual_deployment_time for d in similar_deployments]),
            'avg_support_hours': np.mean([d.actual_support_hours for d in similar_deployments])
        }

    def _assess_prediction_quality(self, explanation: Dict[str, Any]) -> Dict[str, bool]:
        """Assess various quality factors of the prediction."""
        confidence_factors = explanation.get('confidence_factors', {}).get('confidence_factors', {})
        return {
            'sufficient_data': len(self.deployment_history) >= self.min_samples_for_training,
            'balanced_features': confidence_factors.get('balanced_feature_contribution', False),
            'normal_value_ranges': not confidence_factors.get('extreme_values_present', True),
            'stable_prediction': confidence_factors.get('importance_concentration', 1.0) < 0.5
        }

    def _calculate_data_completeness(self, metrics: Dict[str, Any]) -> float:
        """Calculate completeness score for input data."""
        required_fields = {'complexity', 'security', 'architecture'}
        total_files = len(metrics.get('files', {}))
        if total_files == 0:
            return 0.0
            
        complete_files = sum(
            1 for file_metrics in metrics['files'].values()
            if all(field in file_metrics for field in required_fields)
        )
        
        return complete_files / total_files

    def _identify_reviewers(
        self,
        metrics: Dict[str, Any],
        critical_files: List[str]
    ) -> List[str]:
        """Identify team members with relevant expertise."""
        contributors = set()
        for file_path in critical_files:
            if file_path in metrics['files']:
                file_metrics = metrics['files'][file_path]
                if 'change_probability' in file_metrics:
                    contributors.update(
                        file_metrics['change_probability'].contributors
                    )
        return sorted(list(contributors))[:3]  # Top 3 contributors

    def _calculate_coupling(self, metrics: Dict[str, Any]) -> float:
        """Calculate average component coupling."""
        try:
            couplings = []
            for file_metrics in metrics['files'].values():
                architecture = file_metrics.get('architecture', {})
                if isinstance(architecture, dict):
                    coupling = architecture.get('component_coupling', 0.0)
                    if isinstance(coupling, (int, float)):
                        couplings.append(coupling)
            return np.mean(couplings) if couplings else 0.0
        except (AttributeError, TypeError):
            return 0.0

    def _calculate_recent_changes(self, metrics: Dict[str, Any]) -> float:
        """Calculate recent changes score."""
        try:
            current_time = datetime.now()
            recent_changes = sum(
                1 for file_metrics in metrics['files'].values()
                if (file_metrics.get('change_probability', {}).get('last_modified', current_time) - 
                    current_time) <= timedelta(days=7)
            )
            return min(1.0, recent_changes / max(1, len(metrics['files'])))
        except (AttributeError, TypeError):
            return 0.0

    def _calculate_test_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate test coverage score."""
        try:
            total_files = len(metrics['files'])
            if total_files == 0:
                return 0.0
                
            covered_files = sum(
                1 for file_metrics in metrics['files'].values()
                if file_metrics.get('test_coverage_files')
            )
            return covered_files / total_files
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_documentation_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate documentation coverage score."""
        try:
            total_code = 0
            total_comments = 0
            
            for file_metrics in metrics['files'].values():
                metrics_obj = file_metrics.get('metrics', {})
                total_code += metrics_obj.get('lines_code', 0)
                total_comments += metrics_obj.get('lines_comment', 0)
                
            return total_comments / total_code if total_code > 0 else 0.0
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.0

    def _calculate_churn_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate average code churn rate."""
        try:
            churn_rates = []
            for file_metrics in metrics['files'].values():
                if file_metrics.get('change_probability'):
                    churn_rates.append(
                        file_metrics['change_probability'].churn_rate
                    )
            return np.mean(churn_rates) if churn_rates else 0.0
        except (AttributeError, TypeError):
            return 0.0

    def _analyze_risk_factors(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk factors from explanation."""
        risk_factors = {
            'complexity': 0.0,
            'security': 0.0,
            'testing': 0.0,
            'dependencies': 0.0,
            'timing': 0.0
        }
        
        for contributor in explanation.get('top_contributors', []):
            feature = contributor.get('feature', '')
            impact = abs(contributor.get('impact', 0))
            
            if any(word in feature for word in ['complexity', 'cognitive']):
                risk_factors['complexity'] += impact
            elif any(word in feature for word in ['security', 'vulnerability']):
                risk_factors['security'] += impact
            elif 'test' in feature:
                risk_factors['testing'] += impact
            elif any(word in feature for word in ['coupling', 'dependencies']):
                risk_factors['dependencies'] += impact
            elif any(word in feature for word in ['window', 'availability']):
                risk_factors['timing'] += impact

        total = sum(risk_factors.values())
        return {k: v/total for k, v in risk_factors.items()} if total > 0 else risk_factors

    def _determine_severity_level(self, probability: float, explanation: Dict[str, Any]) -> str:
        """Determine severity level based on probability and impact."""
        if probability < 0.3:
            return "low"
            
        # Count high impact features
        high_impact_features = sum(
            1 for contrib in explanation.get('top_contributors', [])
            if abs(contrib.get('impact', 0)) > 0.2
        )
        
        if probability >= 0.7 or high_impact_features >= 3:
            return "high"
        return "medium"

    def _generate_mitigation_suggestions(self, explanation: Dict[str, Any]) -> List[str]:
        """Generate mitigation suggestions based on explanation."""
        suggestions = []
        for contributor in explanation.get('top_contributors', []):
            if abs(contributor.get('deviation', 0)) > 1.5:
                feature = contributor.get('feature', '')
                deviation = contributor.get('deviation', 0)
                
                if 'complexity' in feature:
                    suggestions.append(
                        f"Consider breaking down complex components - {feature} "
                        f"is {abs(deviation):.1f} standard deviations above normal"
                    )
                elif 'security' in feature:
                    suggestions.append(
                        f"Address security concerns - {feature} shows elevated risk"
                    )
                elif 'test' in feature:
                    suggestions.append(
                        f"Improve test coverage - current coverage is "
                        f"{abs(deviation):.1f} standard deviations below typical"
                    )
                elif 'coupling' in feature:
                    suggestions.append(
                        f"Reduce component coupling - {feature} indicates "
                        f"tight coupling {abs(deviation):.1f} standard deviations above normal"
                    )
        return suggestions

    def _update_models_if_needed(self) -> None:
        """Update prediction model with new deployment data."""
        if len(self.deployment_history) < self.min_samples_for_training:
            return
            
        try:
            X = []
            y = []
            
            for feedback in self.deployment_history:
                try:
                    features = self._extract_features(feedback.metrics)
                    X.append(features)
                    y.append(1 if feedback.rollback_occurred else 0)
                except Exception:
                    continue
                    
            if len(X) >= self.min_samples_for_training:
                X = np.array(X)
                y = np.array(y)
                self.model.fit(X, y)
                self._save_model()
                
        except Exception as e:
            print(f"Error updating models: {str(e)}")

    def _save_model(self) -> None:
        """Save trained model to disk."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)