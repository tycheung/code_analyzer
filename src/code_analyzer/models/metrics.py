from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional, Any
from datetime import datetime, time

@dataclass
class SecurityMetrics:
    potential_sql_injections: int = 0
    hardcoded_secrets: int = 0
    unsafe_regex: int = 0
    vulnerable_imports: List[str] = field(default_factory=list)
    insecure_patterns: Dict[str, int] = field(default_factory=dict)

@dataclass
class ArchitectureMetrics:
    circular_dependencies: List[List[str]] = field(default_factory=list)
    abstraction_level: float = 0.0
    component_coupling: float = 0.0
    interface_count: int = 0
    abstract_class_count: int = 0
    layering_violations: int = 0

@dataclass
class ChangeProbability:
    file_path: str
    change_frequency: int
    last_modified: datetime
    contributors: Set[str]
    churn_rate: float

@dataclass
class ComplexityMetrics:
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    cognitive_complexity: int = 0
    halstead_metrics: Dict[str, float] = field(default_factory=dict)
    maintainability_index: float = 0.0
    change_risk: float = 0.0

@dataclass
class CodeMetrics:
    lines_code: int = 0
    lines_comment: int = 0
    lines_blank: int = 0
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    architecture: ArchitectureMetrics = field(default_factory=ArchitectureMetrics)
    security: SecurityMetrics = field(default_factory=SecurityMetrics)
    file_size: int = 0
    avg_function_length: float = 0
    max_function_length: int = 0
    todo_count: int = 0
    test_coverage_files: Set[str] = field(default_factory=set)
    code_patterns: Dict[str, int] = field(default_factory=dict)
    change_probability: Optional[ChangeProbability] = None
    
@dataclass
class DeploymentWindow:
    """Represents a deployment time window with associated metrics and explanations."""
    
    # Time window
    start_time: time
    end_time: time
    
    # Risk metrics
    risk_score: float
    team_availability: float
    historical_success_rate: float
    
    # Deployment constraints
    required_team_size: int = 2
    required_skills: List[str] = None
    estimated_duration: float = 2.0  # hours
    
    # Environment factors
    system_load: Dict[str, float] = None  # Service load metrics
    concurrent_deployments: List[str] = None  # Other planned deployments
    
    # Explainability
    explanation: str = ""
    feature_importance: Dict[str, float] = None
    top_contributors: List[Dict[str, Any]] = None
    feature_interactions: List[Dict[str, Any]] = None
    
    # Quality metrics
    confidence_score: float = 1.0
    data_completeness: float = 1.0
    prediction_quality: Dict[str, bool] = None

@dataclass
class ResourceAllocation:
    recommended_team_size: int
    required_skills: Set[str]
    estimated_support_duration: float
    confidence_score: float
    explanation: str
    feature_importance: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    feature_interactions: List[Dict[str, Any]]

@dataclass
class RollbackPrediction:
    """Prediction results for deployment rollback likelihood with detailed explanations."""
    
    # Core prediction
    probability: float  # Probability of rollback occurring
    
    # Risk analysis
    risk_factors: Dict[str, float]  # Contribution of each risk category
    severity_level: str  # 'low', 'medium', or 'high' based on probability and impact
    critical_files: List[str]  # Files with highest risk factors
    
    # Mitigation planning
    mitigation_suggestions: List[str]  # Concrete suggestions to reduce risk
    recommended_reviewers: List[str]  # Team members with relevant expertise
    
    # Model confidence
    confidence_score: float  # Overall confidence in prediction
    prediction_quality: Dict[str, bool]  # Quality factors affecting prediction
    data_completeness: float  # Score indicating input data completeness
    
    # Explainability
    explanation: str  # Natural language explanation of prediction
    feature_importance: Dict[str, float]  # Impact of each feature
    top_contributors: List[Dict[str, Any]]  # Most influential factors
    feature_interactions: List[Dict[str, Any]]  # Important feature relationships
    
    # Historical context
    similar_cases: List[str]  # IDs of similar historical deployments
    historical_pattern: Dict[str, float]  # Relevant historical metrics

@dataclass
class IncidentPrediction:
    probability: float
    potential_areas: List[str]
    severity_level: str
    estimated_resolution_time: float
    confidence_score: float
    explanation: str
    feature_importance: Dict[str, float]
    top_contributors: List[Dict[str, Any]]
    feature_interactions: List[Dict[str, Any]]

@dataclass
class DeploymentFeedback:
    deployment_id: str
    actual_deployment_time: float
    actual_support_hours: float
    success: bool
    rollback_occurred: bool
    issues_encountered: List[str]
    support_tickets: List[str]
    team_size: int
    start_time: datetime
    end_time: datetime
    affected_services: List[str]

@dataclass
class DeploymentAnalysis:
    """Comprehensive deployment analysis results with detailed explanations."""
    
    # Deployment windows
    optimal_windows: List['DeploymentWindow']
    
    # Predictions
    rollback_prediction: 'RollbackPrediction'
    resource_prediction: 'ResourceAllocation'
    incident_prediction: 'IncidentPrediction'
    
    # Confidence metrics
    overall_confidence: float
    confidence_breakdown: Dict[str, float]
    
    # Explainability
    feature_importances: Dict[str, Dict[str, float]]
    system_explanation: str
    key_insights: List[Dict[str, Any]]
    
    # Analysis metadata
    timestamp: float = None
    analysis_version: str = "1.0"
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    # Analysis quality
    data_completeness: float = 1.0
    prediction_quality: Dict[str, bool] = field(default_factory=dict)
    error_message: str = None

@dataclass
class MLSystemMetrics:
    """System-wide ML performance metrics."""
    
    # Model performance
    predictor_accuracies: Dict[str, float]
    confidence_scores: Dict[str, float]
    data_quality_scores: Dict[str, float]
    
    # Historical performance
    historical_success_rate: float
    prediction_stability: float
    calibration_score: float
    
    # Data metrics
    training_samples: Dict[str, int]
    feature_coverage: Dict[str, float]
    recent_accuracy: Dict[str, List[float]]
    
    # System health
    last_update_timestamp: float
    model_health_checks: Dict[str, bool]
    data_drift_detected: bool = False
    requires_retraining: bool = False