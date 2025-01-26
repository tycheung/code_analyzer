from abc import ABC, abstractmethod
from typing import Dict, Any
from ..models.metrics import CodeMetrics

class BaseAnalyzer(ABC):
    """Abstract base class for code analyzers."""
    
    @abstractmethod
    def analyze_file(self, file_path: str) -> CodeMetrics:
        """Analyze a single file and return metrics."""
        pass
    
    @abstractmethod
    def scan_directory(self, directory: str) -> None:
        """Scan a directory recursively and analyze all relevant files."""
        pass
    
    @abstractmethod
    def print_stats(self) -> None:
        """Print analysis statistics."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Return all collected metrics."""
        pass