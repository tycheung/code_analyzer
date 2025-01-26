import os
import re
import math
from pathlib import Path
from typing import Dict, Set, Any, Optional
from datetime import datetime

from ..models.metrics import (
    CodeMetrics, SecurityMetrics, ArchitectureMetrics, 
    ComplexityMetrics, ChangeProbability
)
from ..utils.git_utils import GitAnalyzer
from .base_analyzer import BaseAnalyzer
from .duplication import CodeDuplication

class CodebaseAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.rb', '.php', '.go', '.rs', '.swift'
        }
        
        self.ignore_dirs = {
            'node_modules', 'venv', '.git', '__pycache__',
            'build', 'dist', 'target', 'bin', 'obj'
        }
        
        # Security pattern detection
        self.security_patterns = {
            'sql_injection': r'execute\s*\(.*?\+.*?\)|raw_input\s*\(.*?\)|f["\']SELECT.*?WHERE.*?\{.*?\}["\']',
            'hardcoded_secret': r'password\s*=\s*[\'"][^\'"]+[\'""]|api_key\s*=\s*[\'"][^\'"]+[\'""]',
            'unsafe_regex': r'eval\(.*\)|exec\(.*\)',
            'vulnerable_imports': {
                'telnetlib', 'pickle', 'marshal', 'subprocess.call', 'subprocess.Popen'
            }
        }
        
        # Architecture patterns
        self.architecture_patterns = {
            'interface': r'interface\s+\w+|protocol\s+\w+|abstract\s+class',
            'abstract_class': r'abstract\s+class|@abstractmethod',
            'layering': r'from\s+\.\..*import|from\s+../../.*import'
        }
        
        # Code quality patterns
        self.code_quality_patterns = {
            'magic_numbers': r'\b\d+\b(?!\.0*\b)',
            'large_try_blocks': r'try:(?:[^}]*){30,}catch',
            'boolean_traps': r'\w+\s*\([^)]*bool\s+\w+[^)]*\)',
            'long_parameter_list': r'\([^)]{120,}\)'
        }

        # Initialize components
        self.git_analyzer = GitAnalyzer()
        self.code_duplication = CodeDuplication()
        
        # Storage for analysis results
        self.stats: Dict[str, CodeMetrics] = {}

    def clone_repo(self, repo_url: str, target_dir: str) -> str:
        """Clone the repository and prepare for analysis."""
        return self.git_analyzer.clone_repository(repo_url, target_dir)

    def cleanup_repo(self, target_dir: str) -> None:
        """Clean up the cloned repository."""
        self.git_analyzer.cleanup_repository(target_dir)

    def is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary by attempting to read it as text."""
        chunk_size = 1024
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                return b'\x00' in chunk
        except Exception:
            return False

    def analyze_security(self, content: str) -> SecurityMetrics:
        """Analyze security patterns and potential vulnerabilities."""
        metrics = SecurityMetrics()
        
        # Detect potential security issues
        metrics.potential_sql_injections = len(re.findall(self.security_patterns['sql_injection'], content))
        metrics.hardcoded_secrets = len(re.findall(self.security_patterns['hardcoded_secret'], content))
        metrics.unsafe_regex = len(re.findall(self.security_patterns['unsafe_regex'], content))
        
        # Check for vulnerable imports
        for vuln_import in self.security_patterns['vulnerable_imports']:
            if vuln_import in content:
                metrics.vulnerable_imports.append(vuln_import)
        
        # Check for additional insecure patterns
        metrics.insecure_patterns = {
            'shell_injection': len(re.findall(r'os\.system\(|subprocess\.call\(', content)),
            'temp_file': len(re.findall(r'tempfile\.mk(temp|stemp)\(', content)),
            'unsafe_deserialization': len(re.findall(r'pickle\.loads?\(|yaml\.load\(', content))
        }
        
        return metrics

    def analyze_architecture(self, content: str) -> ArchitectureMetrics:
        """Analyze architectural patterns and violations."""
        metrics = ArchitectureMetrics()
        
        # Detect interfaces and abstract classes
        metrics.interface_count = len(re.findall(self.architecture_patterns['interface'], content))
        metrics.abstract_class_count = len(re.findall(self.architecture_patterns['abstract_class'], content))
        
        # Detect layering violations
        metrics.layering_violations = len(re.findall(self.architecture_patterns['layering'], content))
        
        # Calculate abstraction level
        total_types = metrics.interface_count + metrics.abstract_class_count
        if total_types > 0:
            metrics.abstraction_level = metrics.interface_count / total_types
            
        # Analyze component coupling
        import_matches = re.findall(r'(?:from|import)\s+([.\w]+)(?:\s+import|\s*$)', content)
        if import_matches:
            # Calculate component coupling as ratio of external to total imports
            total_imports = len(import_matches)
            external_imports = sum(1 for imp in import_matches if not imp.startswith('.'))
            metrics.component_coupling = external_imports / total_imports if total_imports > 0 else 0
            
        # Detect potential circular dependencies
        module_imports = re.findall(r'from\s+([\w.]+)\s+import', content)
        if module_imports:
            current_module = os.path.dirname(content)
            circular_candidates = [
                [current_module, imp] 
                for imp in module_imports 
                if imp.startswith(current_module)
            ]
            metrics.circular_dependencies.extend(circular_candidates)
        
        return metrics

    def analyze_complexity(self, content: str) -> ComplexityMetrics:
        """Analyze code complexity metrics."""
        metrics = ComplexityMetrics()
        
        # Calculate cyclomatic complexity (McCabe)
        # Count control flow statements and structural elements
        patterns = [
            # Control flow - each worth 1 point
            r'\bif\b',
            r'\belse\b', 
            r'\belif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\btry\b',
            r'\bcatch\b',
            r'\bexcept\b',
            # Logical operators - each worth 1 point
            r'\band\b',
            r'\bor\b',
            # Function and class definitions - each worth 1 point
            r'\bdef\b',
            r'\bclass\b',
            # Dangerous functions - each worth 1 point
            r'\beval\b',
            r'\bexec\b',
            # Return statements with conditions
            r'return\s+(?:[\w\s]*[=<>!]+)',
            # List comprehensions and generator expressions
            r'\[.*\bfor\b.*\]',
            r'\(.*\bfor\b.*\)',
        ]
        
        # Base complexity is 1
        complexity = 1
        
        # Add 1 for each pattern match
        for pattern in patterns:
            matches = re.findall(pattern, content)
            complexity += len(matches)
        
        metrics.cyclomatic_complexity = complexity
            
        # Calculate maximum nesting depth
        lines = content.splitlines()
        current_depth = 0
        max_depth = 0
        for line in lines:
            indent = len(line) - len(line.lstrip())
            current_depth = indent // 4  # Assuming 4 spaces per indent level
            max_depth = max(max_depth, current_depth)
        metrics.max_nesting_depth = max_depth
        
        # Calculate cognitive complexity
        cognitive_patterns = {
            'nesting': r'^\s*(?:if|for|while|def|class)',
            'recursion': r'def\s+(\w+).*\1\s*\(',
            'boolean_operators': r'\band\b|\bor\b|\bnot\b',
            'catch_blocks': r'\bcatch\b|\bexcept\b',
            'jumps': r'\bbreak\b|\bcontinue\b|\breturn\b|\braise\b'
        }
        
        cognitive_score = 0
        for line in lines:
            nesting_level = (len(line) - len(line.lstrip())) // 4
            for pattern_type, pattern in cognitive_patterns.items():
                if re.search(pattern, line):
                    if pattern_type == 'nesting':
                        cognitive_score += 1 + nesting_level
                    else:
                        cognitive_score += 1
        metrics.cognitive_complexity = cognitive_score
        
        # Calculate Halstead metrics
        operators_pattern = r'[+\-*/=<>!&|^~%]|<=|>=|==|!=|&&|\|\||<<|>>|\+=|-=|\*=|/=|//|\*\*'
        operands_pattern = r'\b\w+\b|\b\d+\b|\'.*?\'|".*?"'
        
        operators = set(re.findall(operators_pattern, content))
        operands = set(re.findall(operands_pattern, content))
        
        n1 = len(operators)  # unique operators
        n2 = len(operands)   # unique operands
        N1 = len(re.findall(operators_pattern, content))  # total operators
        N2 = len(re.findall(operands_pattern, content))  # total operands
        
        try:
            program_length = N1 + N2
            vocabulary = n1 + n2
            volume = program_length * (math.log2(vocabulary) if vocabulary > 0 else 0)
            difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
            effort = volume * difficulty
            
            metrics.halstead_metrics = {
                'volume': volume,
                'difficulty': difficulty,
                'effort': effort,
                'vocabulary': vocabulary,
                'length': program_length
            }
        except:
            # In case of math errors, set default values
            metrics.halstead_metrics = {
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'vocabulary': 0,
                'length': 0
            }
        
        # Calculate maintainability index
        if metrics.halstead_metrics['volume'] > 0:
            loc = len(lines)
            comments = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*')))
            
            maintainability = (
                171 - 
                5.2 * math.log(metrics.halstead_metrics['volume']) -
                0.23 * metrics.cyclomatic_complexity -
                16.2 * math.log(loc) +
                50 * math.sin(math.sqrt(2.4 * (comments/loc if loc > 0 else 0)))
            )
            
            metrics.maintainability_index = max(0, min(100, maintainability))
        
        return metrics

    def analyze_file(self, file_path: str) -> CodeMetrics:
        """Analyze a single file for metrics."""
        if self.is_binary_file(file_path):
            return CodeMetrics()

        metrics = CodeMetrics()
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.splitlines()

            # Basic line counting
            for line in lines:
                line = line.strip()
                if not line:
                    metrics.lines_blank += 1
                elif line.startswith(('#', '//')):
                    metrics.lines_comment += 1
                else:
                    metrics.lines_code += 1

            # Security analysis
            metrics.security = self.analyze_security(content)
            
            # Architecture analysis
            metrics.architecture = self.analyze_architecture(content)
            
            # Complexity analysis
            metrics.complexity = self.analyze_complexity(content)
            
            # Change probability analysis
            metrics.change_probability = self.analyze_change_probability(file_path)
            
            # Update change risk score based on change probability
            if metrics.change_probability:
                days_since_change = (datetime.now() - metrics.change_probability.last_modified).days
                recency_factor = math.exp(-days_since_change / 365)  # Exponential decay over a year
                
                risk_score = (
                    0.3 * min(1.0, metrics.change_probability.change_frequency / 100) +  # Change frequency
                    0.3 * recency_factor +  # Recent changes
                    0.2 * min(1.0, len(metrics.change_probability.contributors) / 10) +  # Number of contributors
                    0.2 * min(1.0, metrics.change_probability.churn_rate / 1000)  # Churn rate
                )
                metrics.complexity.change_risk = risk_score * 100
            
            # File metrics
            metrics.file_size = os.path.getsize(file_path)
            metrics.todo_count = len(re.findall(r'\bTODO\b', content, re.IGNORECASE))
            
            # Add to duplication checker
            self.code_duplication.add_file(file_path, lines)

        return metrics
    
    def analyze_change_probability(self, file_path: str) -> ChangeProbability:
        """Analyze the probability and risk of changes for a file."""
        history = self.git_analyzer.change_history.get(file_path, {})
        
        if not history:
            return ChangeProbability(
                file_path=file_path,
                change_frequency=0,
                last_modified=datetime.now(),
                contributors=set(),
                churn_rate=0.0
            )
        
        return ChangeProbability(
            file_path=file_path,
            change_frequency=history.get('frequency', 0),
            last_modified=history.get('last_modified', datetime.now()),
            contributors=history.get('contributors', set()),
            churn_rate=history.get('churn_rate', 0.0)
        )

    def scan_directory(self, directory: str) -> None:
        """Scan directory recursively and analyze files."""
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = Path(file).suffix
                
                if ext in self.code_extensions:
                    try:
                        self.stats[file_path] = self.analyze_file(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    def print_stats(self) -> None:
        """Print analysis statistics."""
        # Print detailed statistics
        print("\nCode Analysis Results")
        print("=" * 80)
        
        # File counts
        extensions = {}
        for file_path in self.stats:
            ext = Path(file_path).suffix
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print("\nFile Distribution:")
        for ext, count in sorted(extensions.items()):
            print(f"{ext:>10}: {count:>5} files")
        
        # Line counts
        total_code = sum(m.lines_code for m in self.stats.values())
        total_comments = sum(m.lines_comment for m in self.stats.values())
        total_blank = sum(m.lines_blank for m in self.stats.values())
        
        print("\nLine Counts:")
        print(f"{'Code lines':>15}: {total_code:>8}")
        print(f"{'Comment lines':>15}: {total_comments:>8}")
        print(f"{'Blank lines':>15}: {total_blank:>8}")
        print(f"{'Total lines':>15}: {total_code + total_comments + total_blank:>8}")
        
        # Security issues
        total_security_issues = sum(
            m.security.potential_sql_injections + 
            m.security.hardcoded_secrets + 
            len(m.security.vulnerable_imports)
            for m in self.stats.values()
        )
        
        if total_security_issues > 0:
            print("\nSecurity Issues:")
            print(f"Found {total_security_issues} potential security issues")
        
        # Code duplication
        duplicates = self.code_duplication.find_duplicates()
        if duplicates:
            print("\nCode Duplication:")
            print(f"Found {len(duplicates)} duplicate code blocks")
            
        # High complexity files
        high_complexity = [
            (file, m.complexity.cyclomatic_complexity)
            for file, m in self.stats.items()
            if m.complexity.cyclomatic_complexity > 15
        ]
        
        if high_complexity:
            print("\nHigh Complexity Files:")
            for file, complexity in sorted(high_complexity, key=lambda x: x[1], reverse=True)[:5]:
                print(f"{os.path.basename(file):>30}: Complexity = {complexity}")

    def get_metrics(self) -> Dict[str, Any]:
        """Return all collected metrics."""
        return {
            'files': {
                file: {
                    'lines_code': metrics.lines_code,
                    'lines_comment': metrics.lines_comment,
                    'lines_blank': metrics.lines_blank,
                    'complexity': metrics.complexity.__dict__,
                    'security': metrics.security.__dict__,
                    'architecture': metrics.architecture.__dict__
                }
                for file, metrics in self.stats.items()
            },
            'summary': {
                'total_files': len(self.stats),
                'total_lines': sum(m.lines_code + m.lines_comment + m.lines_blank 
                                 for m in self.stats.values()),
                'security_issues': sum(
                    m.security.potential_sql_injections + 
                    m.security.hardcoded_secrets + 
                    len(m.security.vulnerable_imports)
                    for m in self.stats.values()
                ),
                'duplication': self.code_duplication.get_duplicate_stats()
            }
        }