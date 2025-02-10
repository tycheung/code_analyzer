import os
import subprocess
import shutil
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime, time
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from collections import Counter
from ..models.metrics import (
    DeploymentWindow, ResourceAllocation, RollbackPrediction, 
    IncidentPrediction, CodeMetrics
)

class PDFReportGenerator:
    """Generates comprehensive PDF reports from code analysis metrics."""
    
    def __init__(self, template_dir: str = None, ml_system = None):
        """Initialize the PDF report generator.
        
        Args:
            template_dir: Optional directory containing LaTeX templates.
                        Defaults to 'templates' subdirectory.
            ml_system: Optional ML system for deployment predictions.
        """
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        template_path = os.path.join(template_dir, 'report.tex')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
        
        self.ml_system = ml_system

    def prepare_metrics_data(self, stats: Dict[str, CodeMetrics]) -> Dict[str, Any]:
        """Prepare metrics data for the report."""
        # Prepare basic metrics
        data = {
            'TOTAL_LOC': str(sum(m.lines_code for m in stats.values())),  # Convert to string
            'TOTAL_FILES': str(len(stats)),  # Convert to string
            'PRIMARY_LANGUAGES': self._get_primary_languages(stats),
            'MAINTAINABILITY_SCORE': f"{self._calculate_maintainability_score(stats):.1f}",  # Format as string
            'KEY_FINDINGS': self._generate_key_findings(stats),
            'SIZE_METRICS_TABLE': self._generate_size_table(stats),
            'SECURITY_FINDINGS': self._generate_security_findings(stats),
            'DUPLICATION_ANALYSIS': self._generate_duplication_analysis(stats),
            'RISK_ANALYSIS_TABLE': self._generate_risk_table(stats),
            'RECOMMENDATIONS': self._generate_recommendations(stats),
            'DETAILED_METRICS': self._generate_detailed_metrics(stats)
        }
        
        # Add ML analysis if available
        if self.ml_system:
            team_availability = self._get_team_availability()
            try:
                ml_analysis = self.ml_system.analyze_deployment(stats, team_availability)
                ml_data = self._prepare_ml_data(stats)
                data.update(ml_data)
            except Exception as e:
                print(f"Warning: ML analysis failed: {str(e)}")
                data.update(self._prepare_ml_error_data({}))
        
        return data

    def _get_primary_languages(self, stats: Dict[str, CodeMetrics]) -> str:
        """Get primary languages based on file extensions."""
        extensions = Counter(Path(file).suffix.lstrip('.').upper() 
                           for file in stats.keys())
        top_langs = [f"{lang} ({count})" 
                    for lang, count in extensions.most_common(3)]
        return ', '.join(top_langs)

    def _calculate_maintainability_score(self, stats: Dict[str, CodeMetrics]) -> float:
        """Calculate overall maintainability score."""
        if not stats:
            return 0.0
        return sum(m.complexity.maintainability_index 
                  for m in stats.values()) / len(stats)

    def _generate_size_table(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate the size metrics table content."""
        total_code = sum(m.lines_code for m in stats.values())
        total_comments = sum(m.lines_comment for m in stats.values())
        total_blank = sum(m.lines_blank for m in stats.values())
        total_lines = total_code + total_comments + total_blank
        
        if total_lines == 0:
            return "No Code & 0 & 0\\%"
        
        rows = [
            f"Code Lines & {total_code:,} & {(total_code/total_lines*100):.1f}\\%",
            f"Comment Lines & {total_comments:,} & {(total_comments/total_lines*100):.1f}\\%",
            f"Blank Lines & {total_blank:,} & {(total_blank/total_lines*100):.1f}\\%"
        ]
        
        return '\\\\\n        '.join(rows)

    def _generate_key_findings(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate key findings section."""
        findings = []
        
        # Security issues
        security_issues = sum(
            m.security.potential_sql_injections + 
            m.security.hardcoded_secrets +
            len(m.security.vulnerable_imports)
            for m in stats.values()
        )
        if security_issues > 0:
            findings.append(f"\\item Found {security_issues} potential security issues")
        
        # Complexity
        high_complexity = sum(
            1 for m in stats.values() 
            if m.complexity.cyclomatic_complexity > 15
        )
        if high_complexity > 0:
            findings.append(
                f"\\item {high_complexity} files have high cyclomatic complexity (>15)"
            )
        
        # Maintainability
        poor_maintainability = sum(
            1 for m in stats.values()
            if m.complexity.maintainability_index < 40
        )
        if poor_maintainability > 0:
            findings.append(
                f"\\item {poor_maintainability} files have poor maintainability (MI < 40)"
            )
        
        if not findings:
            findings.append("\\item No critical issues found")
        
        return '\n'.join(findings)

    def _generate_security_findings(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate security findings section."""
        findings = []
        
        # Aggregate security issues by type
        sql_injections = sum(m.security.potential_sql_injections for m in stats.values())
        secrets = sum(m.security.hardcoded_secrets for m in stats.values())
        unsafe_patterns = sum(m.security.unsafe_regex for m in stats.values())
        all_imports = set().union(*(m.security.vulnerable_imports for m in stats.values()))
        
        # Add SQL injection findings
        if sql_injections > 0:
            findings.append(
                f"\\item {{\\color{{warning}} SQL Injection Risks}}: "
                f"Found {sql_injections} potential vulnerabilities"
            )
            
        # Add hardcoded secrets findings
        if secrets > 0:
            findings.append(
                f"\\item {{\\color{{critical}} Hardcoded Secrets}}: "
                f"Detected {secrets} instances of hardcoded credentials"
            )
            
        # Add unsafe pattern findings
        if unsafe_patterns > 0:
            findings.append(
                f"\\item {{\\color{{warning}} Unsafe Code Patterns}}: "
                f"Found {unsafe_patterns} instances of potentially unsafe code"
            )
            
        # Add vulnerable import findings
        if all_imports:
            findings.append(
                f"\\item {{\\color{{critical}} Vulnerable Dependencies}}: "
                f"Found {len(all_imports)} potentially vulnerable imports:"
            )
            for imp in sorted(all_imports):
                findings.append(f"  \\subitem {self._escape_latex(imp)}")
        
        # Add insecure patterns if any exist
        insecure_patterns = Counter()
        for m in stats.values():
            for pattern, count in m.security.insecure_patterns.items():
                insecure_patterns[pattern] += count
                
        if insecure_patterns:
            findings.append(
                f"\\item {{\\color{{warning}} Additional Security Concerns}}:"
            )
            for pattern, count in insecure_patterns.most_common():
                findings.append(
                    f"  \\subitem {pattern.replace('_', ' ').title()}: {count} instances"
                )
        
        if not findings:
            findings.append("\\item No significant security issues detected")
            
        return '\n'.join(findings)

    def _generate_duplication_analysis(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate duplication analysis section."""
        total_lines = sum(m.lines_code for m in stats.values())
        duplicate_blocks = sum(
            1 for m in stats.values() 
            for pattern in m.code_patterns.values() 
            if pattern > 5
        )
        
        if duplicate_blocks == 0:
            return "No significant code duplication detected."
        
        return (
            f"Found {duplicate_blocks} blocks of duplicated code. "
            f"Consider refactoring these sections to improve maintainability."
        )

    def _generate_risk_table(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate risk analysis table content."""
        risk_files = [
            (file, m.complexity.change_risk, 
             m.change_probability.change_frequency if m.change_probability else 0)
            for file, m in stats.items()
            if m.complexity.change_risk > 0
        ]
        
        if not risk_files:
            return "No files & 0 & 0"
        
        rows = []
        for file, risk, freq in sorted(risk_files, key=lambda x: x[1], reverse=True)[:5]:
            short_path = Path(file).name
            rows.append(
                f"{self._escape_latex(short_path)} & {risk:.1f} & {freq}"
            )
        
        return '\\\\\n        '.join(rows)

    def _generate_recommendations(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Security recommendations
        security_issues = sum(
            m.security.potential_sql_injections + 
            m.security.hardcoded_secrets +
            len(m.security.vulnerable_imports)
            for m in stats.values()
        )
        if security_issues > 0:
            recommendations.extend([
                "\\subsection{Security Recommendations}",
                "\\begin{itemize}",
                "\\item Review and fix SQL injection vulnerabilities",
                "\\item Remove hardcoded secrets and use environment variables",
                "\\item Update or replace vulnerable package imports",
                "\\end{itemize}"
            ])

        # Code quality recommendations
        quality_issues = []
        if any(m.complexity.cyclomatic_complexity > 15 for m in stats.values()):
            quality_issues.append("\\item Refactor complex functions (cyclomatic complexity > 15)")
        if any(m.complexity.maintainability_index < 40 for m in stats.values()):
            quality_issues.append("\\item Improve maintainability of poorly maintained files")
        if any(len(m.code_patterns) > 2 for m in stats.values()):
            quality_issues.append("\\item Address code duplication in identified files")
        
        if quality_issues:
            recommendations.extend([
                "\\subsection{Code Quality Recommendations}",
                "\\begin{itemize}",
                *quality_issues,
                "\\end{itemize}"
            ])

        # Architecture recommendations
        arch_issues = []
        if any(m.architecture.layering_violations > 0 for m in stats.values()):
            arch_issues.append("\\item Fix layer violations to improve architecture")
        if any(m.architecture.component_coupling > 0.7 for m in stats.values()):
            arch_issues.append("\\item Reduce high coupling between components")
        
        if arch_issues:
            recommendations.extend([
                "\\subsection{Architecture Recommendations}",
                "\\begin{itemize}",
                *arch_issues,
                "\\end{itemize}"
            ])

        if not recommendations:
            return "No critical issues requiring immediate attention were found."
        
        return '\n'.join(recommendations)

    def _generate_detailed_metrics(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate detailed metrics section."""
        sections = []
        
        # Complexity metrics
        avg_complexity = (
            sum(m.complexity.cyclomatic_complexity for m in stats.values()) / 
            len(stats) if stats else 0
        )
        sections.extend([
            "\\subsection{Complexity Metrics}",
            f"Average Cyclomatic Complexity: {avg_complexity:.2f}",
            "\\vspace{0.5cm}"
        ])
        
        # Most complex files
        complex_files = sorted(
            stats.items(),
            key=lambda x: x[1].complexity.cyclomatic_complexity,
            reverse=True
        )[:5]
        
        if complex_files:
            sections.extend([
                "Most Complex Files:",
                "\\begin{itemize}"
            ])
            for file, metrics in complex_files:
                sections.append(
                    f"\\item {self._escape_latex(Path(file).name)}: "
                    f"Complexity = {metrics.complexity.cyclomatic_complexity}"
                )
            sections.append("\\end{itemize}")

        # Code quality metrics
        sections.extend([
            "\\subsection{Code Quality Metrics}",
            "\\begin{itemize}"
        ])

        # Average maintainability
        avg_maintainability = sum(m.complexity.maintainability_index for m in stats.values()) / len(stats) if stats else 0
        sections.append(f"\\item Average Maintainability Index: {avg_maintainability:.2f}/100")

        # Documentation coverage
        total_code = sum(m.lines_code for m in stats.values())
        total_comments = sum(m.lines_comment for m in stats.values())
        doc_ratio = (total_comments / total_code * 100) if total_code > 0 else 0
        sections.append(f"\\item Documentation Coverage: {doc_ratio:.1f}\\%")

        sections.append("\\end{itemize}")

        # Change Probability Analysis
        sections.extend([
            "\\subsection{Change Probability Analysis}",
            "\\begin{itemize}"
        ])

        # High churn files
        high_churn_files = sorted(
            [(file, m) for file, m in stats.items() if m.change_probability],
            key=lambda x: x[1].change_probability.churn_rate,
            reverse=True
        )[:5]

        if high_churn_files:
            sections.append("High Churn Files:")
            for file, metrics in high_churn_files:
                sections.append(
                    f"\\item {self._escape_latex(Path(file).name)}: "
                    f"Churn Rate = {metrics.change_probability.churn_rate:.1f}, "
                    f"Change Frequency = {metrics.change_probability.change_frequency}"
                )

        sections.append("\\end{itemize}")
        
        return '\n'.join(sections)

    def _format_ml_explanations(self, ml_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Format ML analysis explanations for LaTeX."""
        # Format feature importance table
        importance_rows = []
        for feature, impact in ml_analysis['feature_importances']['cross_model'].items():
            max_deviation = max(
                abs(contributor['deviation'])
                for pred in ['rollback_prediction', 'resource_prediction', 'incident_prediction']
                if pred in ml_analysis
                for contributor in ml_analysis[pred].top_contributors
                if contributor['feature'] == feature
            )
            importance_rows.append(
                f"{feature.replace('_', ' ').title()} & "
                f"{impact*100:.1f}\\% & "
                f"{max_deviation:.1f}\\sigma"
            )
        feature_importance_table = '\\\\\n        '.join(importance_rows[:5])

        return {
            'FEATURE_IMPORTANCE_TABLE': feature_importance_table,
            'WINDOW_EXPLANATION': self._escape_latex(ml_analysis['window_explanation']),
            'ROLLBACK_EXPLANATION': self._escape_latex(ml_analysis['rollback_explanation']),
            'RESOURCE_EXPLANATION': self._escape_latex(ml_analysis['resource_explanation']),
            'INCIDENT_EXPLANATION': self._escape_latex(ml_analysis['incident_explanation']),
            'FEATURE_INTERACTIONS': self._format_feature_interactions(ml_analysis),
            'CONFIDENCE_TABLE': self._format_confidence_table(ml_analysis),
            'CONFIDENCE_FACTORS': self._format_confidence_factors(ml_analysis)
        }

    def _prepare_ml_error_data(self, analysis: Dict) -> Dict[str, str]:
        """Prepare error message for ML section when analysis fails."""
        return {
            'DEPLOYMENT_CONFIDENCE': "N/A",
            'DEPLOYMENT_WINDOWS_TABLE': "Insufficient data & N/A & N/A & N/A",
            'RESOURCE_REQUIREMENTS': "\\item Insufficient historical data for accurate prediction",
            'ROLLBACK_RISK_TABLE': "N/A & N/A",
            'ROLLBACK_MITIGATION': "\\item Not available due to insufficient data",
            'INCIDENT_PROBABILITY': "N/A",
            'INCIDENT_SEVERITY': "Unknown",
            'INCIDENT_RESOLUTION_TIME': "N/A",
            'INCIDENT_AREAS': "\\item Unable to determine potential incident areas"
        }

    def _format_feature_interactions(self, ml_analysis: Dict[str, Any]) -> str:
        """Format feature interactions for LaTeX."""
        interactions = []
        for model_name, pred in [
            ('Deployment Window', ml_analysis.get('optimal_windows', [None])[0]),
            ('Rollback', ml_analysis.get('rollback_prediction')),
            ('Resource', ml_analysis.get('resource_prediction')),
            ('Incident', ml_analysis.get('incident_prediction'))
        ]:
            if pred and hasattr(pred, 'feature_interactions'):
                for interaction in pred.feature_interactions[:2]:  # Top 2 per model
                    interactions.append(
                        f"\\item {model_name}: "
                        f"{interaction['features'][0].replace('_', ' ').title()} and "
                        f"{interaction['features'][1].replace('_', ' ').title()} "
                        f"(strength: {interaction['strength']:.2f})"
                    )
        return '\n'.join(interactions) if interactions else "\\item No significant feature interactions detected"

    def _format_confidence_table(self, ml_analysis: Dict[str, Any]) -> str:
        """Format confidence metrics table for LaTeX."""
        confidence_rows = []
        for aspect, score in ml_analysis.get('confidence_breakdown', {}).items():
            confidence_rows.append(
                f"{aspect.replace('_', ' ').title()} & {score*100:.1f}\\%"
            )
        return '\\\\\n        '.join(confidence_rows) if confidence_rows else "No Data & 0\\%"

    def _format_confidence_factors(self, ml_analysis: Dict[str, Any]) -> str:
        """Format confidence factors for LaTeX."""
        factors = []
        for aspect, pred in [
            ('Window', ml_analysis.get('optimal_windows', [None])[0]),
            ('Rollback', ml_analysis.get('rollback_prediction')),
            ('Resource', ml_analysis.get('resource_prediction')),
            ('Incident', ml_analysis.get('incident_prediction'))
        ]:
            if pred and hasattr(pred, 'confidence_factors'):
                for factor, present in pred.confidence_factors.get('confidence_factors', {}).items():
                    if present:
                        factors.append(
                            f"\\item {factor.replace('_', ' ').title()} "
                            f"in {aspect} prediction"
                        )
        return '\n'.join(factors) if factors else "\\item No significant confidence factors identified"

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        if not isinstance(text, str):
            return str(text)
            
        chars = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
            '\\': '\\textbackslash{}'
        }
        
        # Don't escape if it looks like a LaTeX command
        if text.startswith('\\') or '\\item' in text or '\\begin' in text or '\\end' in text:
            return text
            
        return ''.join(chars.get(c, c) for c in text)

    def generate_charts(self, stats: Dict[str, CodeMetrics], output_dir: str) -> Dict[str, str]:
        """Generate charts and visualizations for the report."""
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Use default style instead of seaborn
        plt.style.use('default')
        
        # Complexity distribution chart
        plt.figure(figsize=(10, 6))
        complexities = [m.complexity.cyclomatic_complexity for m in stats.values()]
        plt.hist(complexities, bins=20, edgecolor='black')
        plt.title('Complexity Distribution')
        plt.xlabel('Cyclomatic Complexity')
        plt.ylabel('Number of Files')
        complexity_chart = 'charts/complexity_dist.png'
        plt.savefig(os.path.join(output_dir, complexity_chart), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Maintainability chart
        plt.figure(figsize=(8, 8))
        maintainability_scores = [m.complexity.maintainability_index for m in stats.values()]
        categories = ['Poor (<40)', 'Fair (40-65)', 'Good (65-85)', 'Excellent (>85)']
        counts = [
            sum(1 for m in maintainability_scores if m < 40),
            sum(1 for m in maintainability_scores if 40 <= m < 65),
            sum(1 for m in maintainability_scores if 65 <= m < 85),
            sum(1 for m in maintainability_scores if m >= 85)
        ]
        colors = ['#DC143C', '#FF8C00', '#4682B4', '#2E8B57']
        plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors)
        plt.title('Maintainability Distribution')
        maintainability_chart = 'charts/maintainability_pie.png'
        plt.savefig(os.path.join(output_dir, maintainability_chart), bbox_inches='tight', dpi=300)
        plt.close()
        
        # File distribution chart
        plt.figure(figsize=(10, 6))
        extensions = Counter(Path(file).suffix for file in stats.keys())
        plt.bar(list(extensions.keys()), list(extensions.values()), color='#4682B4', edgecolor='black')
        plt.title('Files by Extension')
        plt.xticks(rotation=45)
        plt.tight_layout()
        distribution_chart = 'charts/file_distribution.png'
        plt.savefig(os.path.join(output_dir, distribution_chart), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Convert paths for LaTeX
        return {
            'COMPLEXITY_CHART': complexity_chart.replace('\\', '/'),
            'MAINTAINABILITY_CHART': maintainability_chart.replace('\\', '/'),
            'DISTRIBUTION_CHART': distribution_chart.replace('\\', '/')
        }

    def _get_team_availability(self) -> Dict[str, List[time]]:
        """Get team availability data. Override this method to provide actual data."""
        return {
            'team_member_1': [(time(9, 0), time(17, 0))],
            'team_member_2': [(time(10, 0), time(18, 0))],
            'team_member_3': [(time(8, 0), time(16, 0))]
        }

    def generate_pdf(self, stats: Dict[str, CodeMetrics], repo_url: str, output_dir: str) -> None:
        """Generate the PDF report.
        
        Args:
            stats: Dictionary mapping filenames to their CodeMetrics
            repo_url: URL of the repository being analyzed
            output_dir: Directory where the PDF and supporting files will be generated
        """
        # Create output directory and ensure it exists
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate charts first
        print("Generating charts...")
        charts = self.generate_charts(stats, output_dir)
        
        # Prepare report data
        print("Preparing report data...")
        data = self.prepare_metrics_data(stats)
        data.update(charts)  # Add chart paths
        data.update({
            'REPOSITORY_URL': repo_url,
            'REPOSITORY_NAME': Path(repo_url).name
        })
        
        # Escape all string values for LaTeX
        escaped_data = {
            k: (self._escape_latex(v) if isinstance(v, str) else v)
            for k, v in data.items()
        }
        
        # Replace variables in template
        print("Generating LaTeX content...")
        latex_content = self.template
        for key, value in escaped_data.items():
            placeholder = f"${{{key}}}"
            latex_content = latex_content.replace(placeholder, str(value))
        
        # Remove any markdown-style bold markers
        latex_content = latex_content.replace('**', '')
        
        # Write LaTeX file
        tex_file = os.path.join(output_dir, 'report.tex')
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Compile PDF
        try:
            print("Compiling PDF...")
            env = os.environ.copy()
            env['TEXINPUTS'] = f"{output_dir}:"  # Help TeX find local files
            
            # Run pdflatex twice for proper TOC generation
            for _ in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_file],
                    cwd=output_dir,
                    capture_output=True,
                    text=True,
                    env=env
                )
                if result.returncode != 0:
                    print("LaTeX compilation error:")
                    print(result.stdout)
                    print(result.stderr)
                    raise Exception("PDF compilation failed")
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.toc', '.out']:
                aux_file = os.path.join(output_dir, f'report{ext}')
                if os.path.exists(aux_file):
                    os.remove(aux_file)
            
            print(f"\nPDF report generated: {output_dir}/report.pdf")
            
        except FileNotFoundError:
            print("Error: pdflatex not found. Please install TeX Live or MiKTeX.")
            print("Windows: https://miktex.org/download")
            print("Linux: sudo apt-get install texlive-full")
            print("macOS: brew install basictex")
            raise

    def _prepare_ml_data(self, stats: Dict[str, CodeMetrics]) -> Dict[str, Any]:
        """Prepare ML-related data for the report."""
        team_availability = self._get_team_availability()
        analysis = self.ml_system.analyze_deployment(stats, team_availability)
        
        if 'error' in analysis:
            return self._prepare_ml_error_data(analysis)
        
        return {
            'DEPLOYMENT_CONFIDENCE': f"{analysis['overall_confidence']*100:.1f}\\%",
            'DEPLOYMENT_WINDOWS_TABLE': self._format_deployment_windows(
                analysis['optimal_windows']
            ),
            'RESOURCE_REQUIREMENTS': self._format_resource_requirements(
                analysis['resource_prediction']
            ),
            'ROLLBACK_RISK_TABLE': self._format_rollback_risks(
                analysis['rollback_prediction']
            ),
            'ROLLBACK_MITIGATION': self._format_mitigation_suggestions(
                analysis['rollback_prediction'].mitigation_suggestions
            ),
            'INCIDENT_PROBABILITY': f"{analysis['incident_prediction'].probability*100:.1f}\\%",
            'INCIDENT_SEVERITY': analysis['incident_prediction'].severity_level.title(),
            'INCIDENT_RESOLUTION_TIME': f"{analysis['incident_prediction'].estimated_resolution_time:.1f}",
            'INCIDENT_AREAS': self._format_incident_areas(
                analysis['incident_prediction'].potential_areas
            )
        }

    def _format_deployment_windows(self, windows: List[DeploymentWindow]) -> str:
        """Format deployment windows for LaTeX table."""
        rows = []
        for window in windows:
            start_time = window.start_time.strftime('%H:%M')
            end_time = window.end_time.strftime('%H:%M')
            rows.append(
                f"{start_time}-{end_time} & "
                f"{window.historical_success_rate*100:.1f}\\% & "
                f"{window.team_availability*100:.1f}\\% & "
                f"{window.risk_score*100:.1f}\\%"
            )
        return '\\\\\n        '.join(rows) if rows else "No suitable windows & 0\\% & 0\\% & 0\\%"

    def _format_resource_requirements(self, resource_pred: ResourceAllocation) -> str:
        """Format resource requirements for LaTeX."""
        if not resource_pred:
            return "\\item No resource prediction available"
            
        items = [
            f"\\item Required Team Size: {resource_pred.recommended_team_size} developers",
            f"\\item Estimated Support Duration: {resource_pred.estimated_support_duration:.1f} hours",
            "\\item Required Skills:",
            "\\begin{itemize}"
        ]
        
        for skill in sorted(resource_pred.required_skills):
            items.append(f"  \\item {skill.replace('_', ' ').title()}")
        
        items.append("\\end{itemize}")
        items.append(f"\\item Prediction Confidence: {resource_pred.confidence_score*100:.1f}\\%")
        
        return '\n'.join(items)

    def _format_rollback_risks(self, rollback_pred: RollbackPrediction) -> str:
        """Format rollback risks for LaTeX table."""
        if not rollback_pred:
            return "No Data & 0\\%"
            
        rows = []
        for factor, score in rollback_pred.risk_factors.items():
            rows.append(
                f"{factor.replace('_', ' ').title()} & "
                f"{score*100:.1f}\\%"
            )
        return '\\\\\n        '.join(rows)

    def _format_mitigation_suggestions(self, suggestions: List[str]) -> str:
        """Format mitigation suggestions for LaTeX."""
        if not suggestions:
            return "\\item No specific mitigation steps required"
        return '\n'.join(f"\\item {self._escape_latex(suggestion)}" for suggestion in suggestions)

    def _format_incident_areas(self, areas: List[str]) -> str:
        """Format incident areas for LaTeX."""
        if not areas:
            return "\\item No specific areas of concern identified"
        return '\n'.join(f"\\item {self._escape_latex(area)}" for area in areas)

    def _format_latex_path(self, path: str) -> str:
        """Convert a path to a LaTeX-friendly format."""
        # Convert backslashes to forward slashes
        path = path.replace('\\', '/')
        # Escape any special LaTeX characters
        special_chars = {
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '&': '\\&',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        return ''.join(special_chars.get(c, c) for c in path)