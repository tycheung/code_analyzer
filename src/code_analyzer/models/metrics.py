from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional
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
    start_time: time
    end_time: time
    risk_score: float
    team_availability: float
    historical_success_rate: float

@dataclass
class ResourceAllocation:
    recommended_team_size: int
    required_skills: Set[str]
    estimated_support_duration: float
    confidence_score: float

@dataclass
class RollbackPrediction:
    probability: float
    risk_factors: Dict[str, float]
    mitigation_suggestions: List[str]
    confidence_score: float

@dataclass
class IncidentPrediction:
    probability: float
    potential_areas: List[str]
    severity_level: str
    estimated_resolution_time: float
    confidence_score: float

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