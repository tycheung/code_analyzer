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
from ..models.metrics import DeploymentWindow, ResourceAllocation, RollbackPrediction, IncidentPrediction

class PDFReportGenerator:
    def __init__(self, template_dir: str = None, ml_system = None):
        """Initialize the PDF report generator."""
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        
        template_path = os.path.join(template_dir, 'report.tex')
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
        
        self.ml_system = ml_system

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
    
    def generate_charts(self, stats: Dict, output_dir: str) -> Dict[str, str]:
        """Generate charts and visualizations for the report."""
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Complexity distribution chart
        plt.figure(figsize=(10, 6))
        complexities = [m.complexity.cyclomatic_complexity for m in stats.values()]
        sns.histplot(complexities, bins=20)
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
        plt.bar(extensions.keys(), extensions.values())
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
    
    def prepare_metrics_data(self, stats: Dict) -> Dict[str, Any]:
        """Prepare metrics data for the report."""
        # Get base metrics
        data = super().prepare_metrics_data(stats)
        
        # Add ML analysis if available
        if self.ml_system:
            team_availability = self._get_team_availability()  # You need to implement this
            ml_analysis = self.ml_system.analyze_deployment(stats, team_availability)
            
            if 'error' not in ml_analysis:
                # Format ML explanations
                ml_explanations = self._format_ml_explanations(ml_analysis)
                data.update(ml_explanations)
        
        return data

    def _generate_size_table(self, stats: Dict) -> str:
        """Generate the size metrics table content."""
        total_code = sum(m.lines_code for m in stats.values())
        total_comments = sum(m.lines_comment for m in stats.values())
        total_blank = sum(m.lines_blank for m in stats.values())
        total_lines = total_code + total_comments + total_blank
        
        # Format with proper LaTeX escaping
        rows = [
            f"Code Lines & {total_code:,} & {(total_code/total_lines*100):.1f}\\%",
            f"Comment Lines & {total_comments:,} & {(total_comments/total_lines*100):.1f}\\%",
            f"Blank Lines & {total_blank:,} & {(total_blank/total_lines*100):.1f}\\%"
        ]
        
        return '\\\\\n        '.join(rows)
    
    def _generate_key_findings(self, stats: Dict) -> str:
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
        
        return '\n'.join(findings)

    def _generate_security_findings(self, stats: Dict) -> str:
        """Generate security findings section."""
        findings = []
        
        # Aggregate security issues by type
        sql_injections = sum(m.security.potential_sql_injections for m in stats.values())
        secrets = sum(m.security.hardcoded_secrets for m in stats.values())
        unsafe_patterns = sum(m.security.unsafe_regex for m in stats.values())
        vulnerable_imports = set().union(*(m.security.vulnerable_imports for m in stats.values()))
        
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
        if vulnerable_imports:
            findings.append(
                f"\\item {{\\color{{critical}} Vulnerable Dependencies}}: "
                f"Found {len(vulnerable_imports)} potentially vulnerable imports:"
            )
            for imp in sorted(vulnerable_imports):
                findings.append(f"  \\subitem {imp}")
        
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
    
    def _generate_duplication_analysis(self, stats: Dict) -> str:
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

    def _generate_risk_table(self, stats: Dict) -> str:
        """Generate risk analysis table content."""
        # Collect all files with their risk scores
        risk_files = [
            (file, m.complexity.change_risk, m.change_probability.change_frequency if m.change_probability else 0)
            for file, m in stats.items()
            if m.complexity.change_risk > 0  # Show all files with any risk
        ]
        
        if not risk_files:
            return "No files & 0 & 0"
        
        # Sort by risk score and take top 5
        rows = []
        for file, risk, freq in sorted(risk_files, key=lambda x: x[1], reverse=True)[:5]:
            short_path = Path(file).name
            rows.append(f"{short_path} & {risk:.1f} & {freq}")
        
        return '\\\\\n        '.join(rows)

    def _generate_recommendations(self, stats: Dict) -> str:
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

    def _generate_detailed_metrics(self, stats: Dict) -> str:
        """Generate detailed metrics section."""
        sections = []
        
        # Complexity metrics
        avg_complexity = sum(m.complexity.cyclomatic_complexity for m in stats.values()) / len(stats) if stats else 0
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
                    f"\\item {Path(file).name}: "
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

        # TODO count
        total_todos = sum(m.todo_count for m in stats.values())
        if total_todos > 0:
            sections.append(f"\\item Total TODO Comments: {total_todos}")

        sections.append("\\end{itemize}")

        # Architecture metrics
        sections.extend([
            "\\subsection{Architecture Metrics}",
            "\\begin{itemize}"
        ])

        total_interfaces = sum(m.architecture.interface_count for m in stats.values())
        total_abstract_classes = sum(m.architecture.abstract_class_count for m in stats.values())
        total_violations = sum(m.architecture.layering_violations for m in stats.values())

        sections.extend([
            f"\\item Interfaces/Protocols: {total_interfaces}",
            f"\\item Abstract Classes: {total_abstract_classes}",
            f"\\item Architecture Violations: {total_violations}"
        ])

        # Calculate average component coupling
        avg_coupling = sum(
            m.architecture.component_coupling 
            for m in stats.values() 
            if hasattr(m.architecture, 'component_coupling')
        ) / len(stats) if stats else 0
        sections.append(f"\\item Average Component Coupling: {avg_coupling:.2f}")
        sections.append("\\end{itemize}")

        return '\n'.join(sections)
    
    def _prepare_ml_data(self, stats: Dict) -> Dict[str, Any]:
        """Prepare ML-related data for the report."""
        # Get deployment analysis
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

    def _prepare_ml_error_data(self, analysis: Dict) -> Dict[str, Any]:
        """Prepare error message for ML section."""
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
        return '\\\\\n        '.join(rows)

    def _format_resource_requirements(self, resource_pred: ResourceAllocation) -> str:
        """Format resource requirements for LaTeX."""
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
        return '\n'.join(f"\\item {suggestion}" for suggestion in suggestions)

    def _format_incident_areas(self, areas: List[str]) -> str:
        """Format incident areas for LaTeX."""
        if not areas:
            return "\\item No specific areas of concern identified"
        return '\n'.join(f"\\item {area}" for area in areas)

    def _get_team_availability(self) -> Dict[str, List[time]]:
        """Get team availability data."""
        # This is a placeholder - implement based on your actual data source
        return {
            'team_member_1': [(time(9, 0), time(17, 0))],
            'team_member_2': [(time(10, 0), time(18, 0))],
            'team_member_3': [(time(8, 0), time(16, 0))]
        }
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        if not isinstance(text, str):
            return text
            
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

    def generate_pdf(self, stats: Dict, repo_url: str, output_dir: str) -> None:
        """Generate the PDF report."""
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
            # Set up environment for LaTeX
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

    def _format_ml_explanations(self, ml_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Format ML analysis explanations for LaTeX."""
        # Format feature importance table
        importance_rows = []
        for feature, impact in ml_analysis['feature_importances']['cross_model'].items():
            # Find maximum deviation across models for this feature
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
        feature_importance_table = '\\\\\n        '.join(importance_rows[:5])  # Top 5 features

        # Format individual model explanations
        window_explanation = ml_analysis['optimal_windows'][0].explanation.replace('\n', '\\\\\n')
        rollback_explanation = ml_analysis['rollback_prediction'].explanation.replace('\n', '\\\\\n')
        resource_explanation = ml_analysis['resource_prediction'].explanation.replace('\n', '\\\\\n')
        incident_explanation = ml_analysis['incident_prediction'].explanation.replace('\n', '\\\\\n')

        # Format feature interactions
        interactions = []
        for model_name, pred in [
            ('Deployment Window', ml_analysis['optimal_windows'][0]),
            ('Rollback', ml_analysis['rollback_prediction']),
            ('Resource', ml_analysis['resource_prediction']),
            ('Incident', ml_analysis['incident_prediction'])
        ]:
            if hasattr(pred, 'feature_interactions'):
                for interaction in pred.feature_interactions[:2]:  # Top 2 per model
                    interactions.append(
                        f"\\item {model_name}: "
                        f"{interaction['features'][0].replace('_', ' ').title()} and "
                        f"{interaction['features'][1].replace('_', ' ').title()} "
                        f"(strength: {interaction['strength']:.2f})"
                    )

        # Format confidence analysis
        confidence_rows = []
        confidence_factors = []
        for aspect, score in ml_analysis['confidence_breakdown'].items():
            confidence_rows.append(
                f"{aspect.replace('_', ' ').title()} & {score*100:.1f}\\%"
            )
            
            # Add relevant confidence factors
            if aspect in ml_analysis and hasattr(ml_analysis[aspect], 'confidence_factors'):
                factors = ml_analysis[aspect].confidence_factors['confidence_factors']
                for factor, present in factors.items():
                    if present:
                        confidence_factors.append(
                            f"\\item {factor.replace('_', ' ').title()} "
                            f"in {aspect.replace('_', ' ').title()}"
                        )

        return {
            'FEATURE_IMPORTANCE_TABLE': feature_importance_table,
            'WINDOW_EXPLANATION': window_explanation,
            'ROLLBACK_EXPLANATION': rollback_explanation,
            'RESOURCE_EXPLANATION': resource_explanation,
            'INCIDENT_EXPLANATION': incident_explanation,
            'FEATURE_INTERACTIONS': '\n'.join(interactions),
            'CONFIDENCE_TABLE': '\\\\\n        '.join(confidence_rows),
            'CONFIDENCE_FACTORS': '\n'.join(confidence_factors)
        }