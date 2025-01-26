from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional
from datetime import datetime

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