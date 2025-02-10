from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from ..models.metrics import DeploymentFeedback

class BasePredictor(ABC):
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        self.deployment_history: List[DeploymentFeedback] = []

    def _calculate_confidence_score(self, relevant_samples: int) -> float:
        """Calculate confidence score based on historical data richness."""
        base_confidence = min(1.0, relevant_samples / 100)
        
        if relevant_samples >= 10:
            variance = np.var([f.actual_deployment_time for f in self.deployment_history[-relevant_samples:]])
            consistency_score = 1.0 / (1.0 + variance)
            return (base_confidence * 0.7 + consistency_score * 0.3)
        
        return base_confidence

    def record_deployment_feedback(self, feedback: DeploymentFeedback) -> None:
        """Record deployment feedback for continuous learning."""
        self.deployment_history.append(feedback)
        self._update_models_if_needed()

    @abstractmethod
    def _update_models_if_needed(self) -> None:
        """Update ML models with new data when appropriate."""
        pass

    @abstractmethod
    def _extract_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Extract features from metrics for model input."""
        pass

    def _find_similar_deployments(self, metrics: Dict[str, Any]) -> List[DeploymentFeedback]:
        """Find historical deployments with similar characteristics."""
        current_features = self._extract_features(metrics)
        similar_deployments = []
        
        for deployment in self.deployment_history:
            similarity = np.sum(np.abs(current_features - self._extract_features(deployment)))
            if similarity < 0.5:
                similar_deployments.append(deployment)
        
        return similar_deployments