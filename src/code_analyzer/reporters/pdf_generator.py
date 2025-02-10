from typing import Dict, Any, List
import os
from datetime import datetime, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pylatex import (
    Document, Section, Subsection, Tabular, Math, TikZ, Axis, Plot, Figure,
    Package, Command, NoEscape, Center
)
from pylatex.utils import bold, italic
from ..models.metrics import (
    DeploymentWindow, ResourceAllocation, RollbackPrediction, 
    IncidentPrediction, CodeMetrics
)

class PDFReportGenerator:
    """Generates comprehensive PDF reports from code analysis metrics using PyLaTeX."""
    
    def __init__(self, template_dir: str = None, ml_system = None):
        """Initialize the PDF report generator.
        
        Args:
            template_dir: Optional directory containing additional resources.
                        Defaults to 'templates' subdirectory.
            ml_system: Optional ML system for deployment predictions.
        """
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            
        self.template_dir = template_dir
        self.ml_system = ml_system
        
        # Document settings
        self.geometry_options = {
            "margin": "1in",
            "headheight": "40pt",
            "footskip": "30pt"
        }
        
        # Color definitions
        self.colors = {
            'light-gray': 'gray!95',
            'medium-gray': 'gray!85',
            'primary': 'rgb:70,130,180',
            'warning': 'rgb:255,140,0',
            'critical': 'rgb:220,20,60',
            'success': 'rgb:46,139,87'
        }
        
        # Additional packages needed
        self.packages = [
            'inputenc',
            'geometry',
            'graphicx',
            'booktabs',
            'xcolor',
            'tikz',
            'pgfplots',
            'listings',
            'hyperref',
            'fancyhdr'
        ]

    def prepare_metrics_data(self, stats: Dict[str, CodeMetrics]) -> Dict[str, Any]:
        """Prepare metrics data for the report."""
        data = {
            'TOTAL_LOC': str(sum(m.lines_code for m in stats.values())),
            'TOTAL_FILES': str(len(stats)),
            'PRIMARY_LANGUAGES': self._get_primary_languages(stats),
            'MAINTAINABILITY_SCORE': f"{self._calculate_maintainability_score(stats):.1f}",
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

    def generate_pdf(self, stats: Dict[str, CodeMetrics], repo_url: str, output_dir: str) -> None:
        """Generate the PDF report using PyLaTeX."""
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize document
        doc = Document(documentclass='article')
        
        # Add packages and setup before any content
        self._setup_document(doc)
        
        # Generate charts before content
        print("Generating charts...")
        chart_paths = self.generate_charts(stats, output_dir)
        
        # Prepare metrics data
        print("Preparing metrics data...")
        data = self.prepare_metrics_data(stats)
        data.update(chart_paths)
        data['REPOSITORY_URL'] = repo_url
        data['REPOSITORY_NAME'] = Path(repo_url).name

        # Start the document
        print("Generating report content...")
        
        # Title page
        with doc.create(Section('Code Analysis Report')):
            doc.append(NoEscape(r'\large\textbf{Repository URL:} ' + repo_url))
            doc.append(NoEscape(r'\vspace{1cm}'))
            
            if 'COMPLEXITY_CHART' in data:
                with doc.create(Center()):
                    with doc.create(Figure()) as fig:
                        fig.add_image(data['COMPLEXITY_CHART'], width=NoEscape('0.4\\textwidth'))
            
            doc.append(NoEscape(r'\vspace{1cm}'))
            
            doc.append(NoEscape(r'\large Generated on \today'))
        
        # Add content sections
        self._add_executive_summary(doc, data)
        self._add_detailed_analysis(doc, data)
        self._add_deployment_impact_analysis(doc, data)
        self._add_recommendations_section(doc, data)
        self._add_detailed_metrics_section(doc, data)
        self._add_appendix(doc)
        
        # Generate PDF
        print("Compiling PDF...")
        try:
            os.chdir(output_dir)  # Change to output directory for compilation
            doc.generate_pdf(
                'report',
                clean_tex=False,
                compiler='pdflatex',
                compiler_args=[
                    '-interaction=nonstopmode',
                    '-halt-on-error'
                ]
            )
            print(f"\nPDF report generated: {output_dir}/report.pdf")
            
        except Exception as e:
            print("\nError generating PDF:", str(e))
            
            # Try to read the log file for more details
            log_file = os.path.join(output_dir, 'report.log')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    print("\nLaTeX compilation log:")
                    print(f.read())
                    
            print("\nPlease ensure you have LaTeX installed:")
            print("Windows: https://miktex.org/download")
            print("Linux: sudo apt-get install texlive-full")
            print("macOS: brew install basictex")
            raise

    def _setup_document(self, doc: Document) -> None:
        """Configure document packages and styling."""
        # Define the document class with options
        doc.documentclass = Command('documentclass', options=['a4paper'], arguments=['article'])
        
        # Add packages in correct order
        doc.packages.append(Package('fontenc', options=['T1']))
        doc.packages.append(Package('inputenc', options=['utf8']))
        doc.packages.append(Package('lmodern'))
        doc.packages.append(Package('textcomp'))
        doc.packages.append(Package('geometry', options=['margin=1in']))
        doc.packages.append(Package('xcolor', options=['table']))  # Add table option
        doc.packages.append(Package('graphicx'))
        doc.packages.append(Package('booktabs'))
        doc.packages.append(Package('tikz'))
        doc.packages.append(Package('pgfplots'))
        doc.packages.append(Package('listings'))
        
        # Add hyperref last to avoid conflicts
        doc.packages.append(Package('hyperref'))
        doc.packages.append(Package('fancyhdr'))
        
        # Define colors before using them
        doc.preamble.append(Command('definecolor', ['primary', 'RGB', '70,130,180']))
        doc.preamble.append(Command('definecolor', ['warning', 'RGB', '255,140,0']))
        doc.preamble.append(Command('definecolor', ['critical', 'RGB', '220,20,60']))
        doc.preamble.append(Command('definecolor', ['success', 'RGB', '46,139,87']))
        doc.preamble.append(Command('definecolor', ['light-gray', 'gray', '0.95']))
        doc.preamble.append(Command('definecolor', ['medium-gray', 'gray', '0.85']))
        
        # Configure pgfplots
        doc.preamble.append(Command('pgfplotsset', 'compat=newest'))
        
        # Configure listings
        doc.preamble.append(NoEscape(
            '\\lstset{'
            'backgroundcolor=\\color{light-gray},'
            'basicstyle=\\small\\ttfamily,'
            'breaklines=true,'
            'captionpos=b,'
            'commentstyle=\\color{gray},'
            'frame=single,'
            'numbers=left,'
            'numberstyle=\\tiny\\color{gray},'
            'showstringspaces=false,'
            'keywordstyle=\\color{primary}'
            '}'
        ))
        
        # Configure page style
        doc.preamble.append(Command('pagestyle', 'fancy'))
        doc.preamble.append(Command('fancyhf', ''))
        doc.preamble.append(Command('rhead', 'Code Analysis Report'))
        doc.preamble.append(Command('lhead', NoEscape('\\today')))
        doc.preamble.append(Command('rfoot', NoEscape('Page \\thepage')))

    def _add_title_page(self, doc: Document, data: Dict[str, Any]) -> None:
        """Create the report title page."""
        doc.append(NoEscape(r'\begin{titlepage}'))
        with doc.create(Center()) as centered:
            doc.append(NoEscape(r'\vspace*{2cm}'))
            
            # Title
            doc.append(NoEscape(r'\Huge'))
            doc.append(NoEscape(r'\textbf{Code Analysis Report}'))
            
            doc.append(NoEscape(r'\vspace{1.5cm}'))
            
            # Repository name
            doc.append(NoEscape(r'\Large'))
            doc.append(data['REPOSITORY_NAME'])
            
            doc.append(NoEscape(r'\vspace{1.5cm}'))
            
            # Add complexity chart if available
            if 'COMPLEXITY_CHART' in data:
                doc.append(NoEscape(
                    r'\includegraphics[width=0.4\textwidth]{' + 
                    data['COMPLEXITY_CHART'] + 
                    r'}'
                ))
            
            doc.append(NoEscape(r'\vspace{1.5cm}'))
            
            # Generation date
            doc.append(NoEscape(r'\large'))
            doc.append(NoEscape(r'Generated on \today'))
            
            doc.append(NoEscape(r'\vfill'))
            
            # Repository URL
            doc.append(NoEscape(r'\textbf{Repository URL:}\\'))
            doc.append(NoEscape(r'\url{' + data['REPOSITORY_URL'] + r'}'))
            
        doc.append(NoEscape(r'\end{titlepage}'))
        doc.append(NoEscape(r'\newpage'))

    def _add_executive_summary(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add executive summary section."""
        with doc.create(Section('Executive Summary')):
            # Key Metrics subsection
            with doc.create(Subsection('Key Metrics')):
                # Create table for key metrics
                with doc.create(Center()):
                    with doc.create(Tabular('ll')) as table:
                        table.add_hline()
                        table.add_row((NoEscape(r'\textbf{Metric}'), NoEscape(r'\textbf{Value}')))
                        table.add_hline()
                        metrics = [
                            ('Total Lines of Code:', data['TOTAL_LOC']),
                            ('Number of Files:', data['TOTAL_FILES']),
                            ('Primary Languages:', data['PRIMARY_LANGUAGES']),
                            ('Overall Maintainability Score:', f"{data['MAINTAINABILITY_SCORE']}/100")
                        ]
                        for label, value in metrics:
                            table.add_row((NoEscape(label), NoEscape(value)))
                        table.add_hline()
            
            # Key Findings subsection
            with doc.create(Subsection('Key Findings')):
                doc.append(NoEscape(r'\begin{itemize}'))
                doc.append(NoEscape(data['KEY_FINDINGS']))
                doc.append(NoEscape(r'\end{itemize}'))

    def _add_detailed_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add detailed analysis section."""
        with doc.create(Section('Detailed Analysis')):
            # Code Size and Structure subsection
            with doc.create(Subsection('Code Size and Structure')):
                with doc.create(Center()):
                    with doc.create(Tabular('lrr')) as table:
                        table.add_hline()
                        table.add_row((
                            NoEscape(r'\textbf{Metric}'),
                            NoEscape(r'\textbf{Count}'),
                            NoEscape(r'\textbf{Percentage}')
                        ))
                        table.add_hline()
                        for row in data['SIZE_METRICS_TABLE'].split(r'\\'):
                            if row.strip():
                                cols = [NoEscape(col.strip()) for col in row.split('&')]
                                table.add_row(cols)
                        table.add_hline()

            # Complexity Analysis subsection
            with doc.create(Subsection('Complexity Analysis')):
                with doc.create(Figure(position='h')) as fig:
                    fig.add_image(data['COMPLEXITY_CHART'], width=NoEscape(r'0.8\textwidth'))
                    fig.add_caption('Distribution of Cyclomatic Complexity Across Files')

            # Maintainability Distribution subsection
            with doc.create(Subsection('Maintainability Distribution')):
                if 'MAINTAINABILITY_CHART' in data:
                    with doc.create(Figure(position='h')) as fig:
                        fig.add_image(
                            data['MAINTAINABILITY_CHART'],
                            width=NoEscape(r'0.6\textwidth')
                        )
                        fig.add_caption('Distribution of Maintainability Index')

            # Security Analysis subsection
            with doc.create(Subsection('Security Analysis')):
                doc.append(NoEscape(r'\begin{itemize}'))
                doc.append(NoEscape(data['SECURITY_FINDINGS']))
                doc.append(NoEscape(r'\end{itemize}'))

            # Code Duplication subsection
            with doc.create(Subsection('Code Duplication')):
                doc.append(data['DUPLICATION_ANALYSIS'])

            # Change Risk Analysis subsection
            with doc.create(Subsection('Change Risk Analysis')):
                with doc.create(Center()):
                    with doc.create(Tabular('lrr')) as table:
                        table.add_hline()
                        table.add_row((
                            NoEscape(r'\textbf{File}'),
                            NoEscape(r'\textbf{Risk Score}'),
                            NoEscape(r'\textbf{Change Frequency}')
                        ))
                        table.add_hline()
                        table.append(NoEscape(data['RISK_ANALYSIS_TABLE']))
                        table.add_hline()

    def _generate_security_findings(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate security findings section with LaTeX formatting."""
        findings = []
        
        # Aggregate security issues by type
        sql_injections = sum(m.security.potential_sql_injections for m in stats.values())
        secrets = sum(m.security.hardcoded_secrets for m in stats.values())
        unsafe_patterns = sum(m.security.unsafe_regex for m in stats.values())
        all_imports = set().union(*(m.security.vulnerable_imports for m in stats.values()))
        
        # Add SQL injection findings
        if sql_injections > 0:
            findings.append(
                r'\item {\color{warning} SQL Injection Risks}: ' +
                f"Found {sql_injections} potential vulnerabilities"
            )
            
        # Add hardcoded secrets findings
        if secrets > 0:
            findings.append(
                r'\item {\color{critical} Hardcoded Secrets}: ' +
                f"Detected {secrets} instances of hardcoded credentials"
            )
            
        # Add unsafe pattern findings
        if unsafe_patterns > 0:
            findings.append(
                r'\item {\color{warning} Unsafe Code Patterns}: ' +
                f"Found {unsafe_patterns} instances of potentially unsafe code"
            )
            
        # Add vulnerable import findings
        if all_imports:
            findings.append(
                r'\item {\color{critical} Vulnerable Dependencies}: ' +
                f"Found {len(all_imports)} potentially vulnerable imports:"
            )
            for imp in sorted(all_imports):
                findings.append(f"  \\subitem {self._escape_latex(imp)}")
        
        if not findings:
            findings.append(r'\item No significant security issues detected')
            
        return '\n'.join(findings)
    
    def _calculate_maintainability_score(self, stats: Dict[str, CodeMetrics]) -> float:
        """Calculate overall maintainability score."""
        if not stats:
            return 0.0
        return sum(m.complexity.maintainability_index 
                  for m in stats.values()) / len(stats)

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
    
    def _add_deployment_impact_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add deployment impact analysis section."""
        with doc.create(Section('Deployment Impact Analysis')):
            # Deployment Confidence subsection
            with doc.create(Subsection('Deployment Confidence')):
                with doc.create(Center()):
                    doc.append(NoEscape(r'\Large\textbf{' + data['DEPLOYMENT_CONFIDENCE'] + r'}'))
                doc.append(NoEscape(r'\vspace{0.5cm}'))

            # Factor Analysis subsection
            with doc.create(Subsection('Factor Analysis')):
                with doc.create(Center()):
                    with doc.create(Tabular('lrr')) as table:
                        table.add_hline()
                        table.add_row((
                            NoEscape(r'\textbf{Feature}'),
                            NoEscape(r'\textbf{Cross-Model Impact}'),
                            NoEscape(r'\textbf{Deviation}')
                        ))
                        table.add_hline()
                        table.append(NoEscape(data['FEATURE_IMPORTANCE_TABLE']))
                        table.add_hline()

            # Model Explanations subsection
            with doc.create(Subsection('Model Explanations')):
                # Window Selection
                doc.append(NoEscape(r'\subsubsection{Deployment Window Selection}'))
                doc.append(NoEscape(data['WINDOW_EXPLANATION']))

                # Rollback Risk
                doc.append(NoEscape(r'\subsubsection{Rollback Risk Factors}'))
                doc.append(NoEscape(data['ROLLBACK_EXPLANATION']))

                # Resource Requirements
                doc.append(NoEscape(r'\subsubsection{Resource Requirements Rationale}'))
                doc.append(NoEscape(data['RESOURCE_EXPLANATION']))

                # Incident Risk
                doc.append(NoEscape(r'\subsubsection{Incident Risk Analysis}'))
                doc.append(NoEscape(data['INCIDENT_EXPLANATION']))

            # Feature Interactions subsection
            with doc.create(Subsection('Feature Interactions')):
                doc.append(NoEscape(r'\begin{itemize}'))
                doc.append(NoEscape(data['FEATURE_INTERACTIONS']))
                doc.append(NoEscape(r'\end{itemize}'))

            # Deployment Windows subsection
            with doc.create(Subsection('Optimal Deployment Windows')):
                with doc.create(Center()):
                    with doc.create(Tabular('lrrr')) as table:
                        table.add_hline()
                        table.add_row((
                            NoEscape(r'\textbf{Time Window}'),
                            NoEscape(r'\textbf{Success Rate}'),
                            NoEscape(r'\textbf{Team Availability}'),
                            NoEscape(r'\textbf{Risk Score}')
                        ))
                        table.add_hline()
                        table.append(NoEscape(data['DEPLOYMENT_WINDOWS_TABLE']))
                        table.add_hline()

            # Resource Requirements subsection
            with doc.create(Subsection('Resource Requirements')):
                doc.append(NoEscape(r'\begin{itemize}'))
                doc.append(NoEscape(data['RESOURCE_REQUIREMENTS']))
                doc.append(NoEscape(r'\end{itemize}'))

            # Incident Analysis subsection
            with doc.create(Subsection('Incident Analysis')):
                with doc.create(Center()):
                    with doc.create(Tabular('lr')) as table:
                        table.add_row((
                            NoEscape(r'\textbf{Incident Probability:}'),
                            NoEscape(r'{\large\color{warning}' + data['INCIDENT_PROBABILITY'] + r'}')
                        ))
                        table.add_row((
                            NoEscape(r'\textbf{Severity Level:}'),
                            NoEscape(r'{\large\color{warning}' + data['INCIDENT_SEVERITY'] + r'}')
                        ))
                        table.add_row((
                            NoEscape(r'\textbf{Est. Resolution Time:}'),
                            NoEscape(r'{\large ' + data['INCIDENT_RESOLUTION_TIME'] + r' hours}')
                        ))

                doc.append(NoEscape(r'\textbf{Potential Areas of Concern:}'))
                doc.append(NoEscape(r'\begin{itemize}'))
                doc.append(NoEscape(data['INCIDENT_AREAS']))
                doc.append(NoEscape(r'\end{itemize}'))

    def _add_recommendations_section(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add recommendations section."""
        with doc.create(Section('Recommendations')):
            doc.append(NoEscape(data['RECOMMENDATIONS']))

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
                r'\subsection{Security Recommendations}',
                r'\begin{itemize}',
                r'\item Review and fix SQL injection vulnerabilities',
                r'\item Remove hardcoded secrets and use environment variables',
                r'\item Update or replace vulnerable package imports',
                r'\end{itemize}'
            ])

        # Code quality recommendations
        quality_issues = []
        if any(m.complexity.cyclomatic_complexity > 15 for m in stats.values()):
            quality_issues.append(r'\item Refactor complex functions (cyclomatic complexity > 15)')
        if any(m.complexity.maintainability_index < 40 for m in stats.values()):
            quality_issues.append(r'\item Improve maintainability of poorly maintained files')
        if any(len(m.code_patterns) > 2 for m in stats.values()):
            quality_issues.append(r'\item Address code duplication in identified files')
        
        if quality_issues:
            recommendations.extend([
                r'\subsection{Code Quality Recommendations}',
                r'\begin{itemize}',
                *quality_issues,
                r'\end{itemize}'
            ])

        if not recommendations:
            return "No critical issues requiring immediate attention were found."
        
        return '\n'.join(recommendations)
    
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
        extensions = self._count_file_extensions(stats)
        plt.bar(list(extensions.keys()), list(extensions.values()), color='#4682B4', edgecolor='black')
        plt.title('Files by Extension')
        plt.xticks(rotation=45)
        plt.tight_layout()
        distribution_chart = 'charts/file_distribution.png'
        plt.savefig(os.path.join(output_dir, distribution_chart), bbox_inches='tight', dpi=300)
        plt.close()
        
        return {
            'COMPLEXITY_CHART': complexity_chart,
            'MAINTAINABILITY_CHART': maintainability_chart,
            'DISTRIBUTION_CHART': distribution_chart
        }
    
    def _generate_size_table(self, stats: Dict[str, CodeMetrics]) -> str:
        """Generate the size metrics table content."""
        total_code = sum(m.lines_code for m in stats.values())
        total_comments = sum(m.lines_comment for m in stats.values())
        total_blank = sum(m.lines_blank for m in stats.values())
        total_lines = total_code + total_comments + total_blank
        
        if total_lines == 0:
            return r"No Code & 0 & 0\%"
        
        rows = []
        for label, count in [
            ('Code Lines', total_code),
            ('Comment Lines', total_comments),
            ('Blank Lines', total_blank)
        ]:
            rows.append(
                f"{label} & {count:,} & {(count/total_lines*100):.1f}\\%"
            )
        
        return ' \\\\\n'.join(rows)
    
    def _add_detailed_metrics_section(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add detailed metrics section."""
        with doc.create(Section('Detailed Metrics')):
            doc.append(NoEscape(data['DETAILED_METRICS']))

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

        return '\n'.join(sections)
    
    def _add_appendix(self, doc: Document) -> None:
        """Add appendix section."""
        with doc.create(Section('Appendix')):
            with doc.create(Subsection('Analysis Methodology')):
                doc.append('This report was generated using automated code analysis tools that examine:')
                metrics = [
                    'Static code analysis',
                    'Complexity metrics',
                    'Security patterns',
                    'Git history',
                    'Dependency graphs',
                    'Code duplication',
                    'Architecture patterns',
                    'Historical deployment data',
                    'Team availability patterns',
                    'Incident history'
                ]
                doc.append(NoEscape(r'\begin{itemize}'))
                for metric in metrics:
                    doc.append(NoEscape(f'\\item {metric}'))
                doc.append(NoEscape(r'\end{itemize}'))

            if self.ml_system:
                with doc.create(Subsection('ML Model Information')):
                    doc.append('The machine learning predictions in this report are based on:')
                    factors = [
                        'Historical deployment patterns',
                        'Team performance metrics',
                        'Code change patterns',
                        'Incident history',
                        'System architecture evolution'
                    ]
                    doc.append(NoEscape(r'\begin{itemize}'))
                    for factor in factors:
                        doc.append(NoEscape(f'\\item {factor}'))
                    doc.append(NoEscape(r'\end{itemize}'))

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
        
        return ' \\\\\n        '.join(rows)
    
    def _format_risk_score(self, score: float) -> str:
        """Format risk score with appropriate coloring based on severity."""
        if score >= 75:
            return f"{{\\color{{critical}}{score:.1f}}}"
        elif score >= 50:
            return f"{{\\color{{warning}}{score:.1f}}}"
        else:
            return f"{{\\color{{success}}{score:.1f}}}"
    
    def _get_primary_languages(self, stats: Dict[str, CodeMetrics]) -> str:
        """Get primary languages based on file extensions."""
        from collections import Counter
        extensions = Counter(Path(file).suffix.lstrip('.').upper() 
                           for file in stats.keys())
        top_langs = [f"{lang} ({count})" 
                    for lang, count in extensions.most_common(3)]
        return ', '.join(top_langs)

    def _count_file_extensions(self, stats: Dict[str, CodeMetrics]) -> Dict[str, int]:
        """Count files by extension."""
        extensions = {}
        for file_path in stats.keys():
            ext = Path(file_path).suffix
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
        return extensions
    
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
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        if not isinstance(text, str):
            return str(text)
            
        # Don't escape if it looks like a LaTeX command
        if (text.startswith('\\') or '\\item' in text or 
            '\\begin' in text or '\\end' in text or 
            '\\section' in text or '\\subsection' in text):
            return text
            
        # Handle math mode subscripts specially
        if '_' in text and '{' in text and '}' in text:
            # Check if it matches the pattern for math mode subscripts (e.g., a_{b})
            import re
            if re.match(r'^.*_\{[^}]+\}.*$', text):
                return text

        # Process using character map
        chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}'
        }
        
        result = []
        i = 0
        while i < len(text):
            if text[i] in chars:
                if i > 0 and text[i-1] == '\\':  # Already escaped
                    result.append(text[i])
                else:
                    result.append(chars[text[i]])
            else:
                result.append(text[i])
            i += 1
            
        return ''.join(result)
    
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
            ),
            'WINDOW_EXPLANATION': self._escape_latex(analysis['window_explanation']),
            'ROLLBACK_EXPLANATION': self._escape_latex(analysis['rollback_explanation']),
            'RESOURCE_EXPLANATION': self._escape_latex(analysis['resource_explanation']),
            'INCIDENT_EXPLANATION': self._escape_latex(analysis['incident_explanation']),
            'FEATURE_INTERACTIONS': self._format_feature_interactions(analysis),
            'FEATURE_IMPORTANCE_TABLE': self._format_feature_importance_table(analysis),
            'CONFIDENCE_TABLE': self._format_confidence_table(analysis),
            'CONFIDENCE_FACTORS': self._format_confidence_factors(analysis)
        }

    def _prepare_ml_error_data(self, analysis: Dict) -> Dict[str, str]:
        """Prepare error message for ML section when analysis fails."""
        return {
            'DEPLOYMENT_CONFIDENCE': "N/A",
            'DEPLOYMENT_WINDOWS_TABLE': r"Insufficient data & N/A & N/A & N/A",
            'RESOURCE_REQUIREMENTS': r"\item Insufficient historical data for accurate prediction",
            'ROLLBACK_RISK_TABLE': "N/A & N/A",
            'ROLLBACK_MITIGATION': r"\item Not available due to insufficient data",
            'INCIDENT_PROBABILITY': "N/A",
            'INCIDENT_SEVERITY': "Unknown",
            'INCIDENT_RESOLUTION_TIME': "N/A",
            'INCIDENT_AREAS': r"\item Unable to determine potential incident areas",
            'WINDOW_EXPLANATION': "Analysis failed",
            'ROLLBACK_EXPLANATION': "Analysis failed",
            'RESOURCE_EXPLANATION': "Analysis failed",
            'INCIDENT_EXPLANATION': "Analysis failed",
            'FEATURE_INTERACTIONS': r"\item No significant feature interactions detected",
            'FEATURE_IMPORTANCE_TABLE': "No Data & 0\\% & 0",
            'CONFIDENCE_TABLE': "No Data & 0\\%",
            'CONFIDENCE_FACTORS': r"\item No significant confidence factors identified"
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
        return ' \\\\\n        '.join(rows) if rows else "No suitable windows & 0\\% & 0\\% & 0\\%"

    def _format_resource_requirements(self, resource_pred: ResourceAllocation) -> str:
        """Format resource requirements for LaTeX."""
        if not resource_pred:
            return r"\item No resource prediction available"
            
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
        return ' \\\\\n        '.join(rows)

    def _format_mitigation_suggestions(self, suggestions: List[str]) -> str:
        """Format mitigation suggestions for LaTeX."""
        if not suggestions:
            return r"\item No specific mitigation steps required"
        return '\n'.join(f"\\item {self._escape_latex(suggestion)}" for suggestion in suggestions)

    def _format_incident_areas(self, areas: List[str]) -> str:
        """Format incident areas for LaTeX."""
        if not areas:
            return r"\item No specific areas of concern identified"
        return '\n'.join(f"\\item {self._escape_latex(area)}" for area in areas)

    def _format_feature_interactions(self, analysis: Dict[str, Any]) -> str:
        """Format feature interactions for LaTeX."""
        interactions = []
        for model_name, pred in [
            ('Deployment Window', analysis.get('optimal_windows', [None])[0]),
            ('Rollback', analysis.get('rollback_prediction')),
            ('Resource', analysis.get('resource_prediction')),
            ('Incident', analysis.get('incident_prediction'))
        ]:
            if pred and hasattr(pred, 'feature_interactions') and pred.feature_interactions:
                for interaction in list(pred.feature_interactions)[:2]:
                    interactions.append(
                        f"\\item {model_name}: "
                        f"{interaction['features'][0].replace('_', ' ').title()} and "
                        f"{interaction['features'][1].replace('_', ' ').title()} "
                        f"(strength: {interaction['strength']:.2f})"
                    )
        return '\n'.join(interactions) if interactions else r"\item No significant feature interactions detected"

    def _format_feature_importance_table(self, analysis: Dict[str, Any]) -> str:
        """Format feature importance table for LaTeX."""
        importance_rows = []
        for feature, impact in analysis['feature_importances']['cross_model'].items():
            # Calculate max deviation safely
            max_dev = 0.0
            for pred_name in ['rollback_prediction', 'resource_prediction', 'incident_prediction']:
                pred = analysis.get(pred_name)
                if not pred or not hasattr(pred, 'top_contributors'):
                    continue
                for contrib in getattr(pred, 'top_contributors', []):
                    if contrib.get('feature') == feature:
                        dev = abs(contrib.get('deviation', contrib.get('impact', 0.0)))
                        max_dev = max(max_dev, dev)
            
            formatted_row = (
                f"{feature.replace('_', ' ').title()} & "
                f"{impact*100:.1f}\\% & "
                f"{max_dev:.1f}\\sigma"
            )
            importance_rows.append(formatted_row)
        
        if not importance_rows:
            return "No Data & 0\\% & 0"
        
        return ' \\\\\n'.join(importance_rows[:5])

    def _format_confidence_table(self, analysis: Dict[str, Any]) -> str:
        """Format confidence metrics table for LaTeX."""
        confidence_rows = []
        for aspect, score in analysis.get('confidence_breakdown', {}).items():
            confidence_rows.append(
                f"{aspect.replace('_', ' ').title()} & {score*100:.1f}\\%"
            )
        return ' \\\\\n        '.join(confidence_rows) if confidence_rows else "No Data & 0\\%"

    def _format_confidence_factors(self, analysis: Dict[str, Any]) -> str:
        """Format confidence factors for LaTeX."""
        factors = []
        for aspect, pred in [
            ('Window', analysis.get('optimal_windows', [None])[0]),
            ('Rollback', analysis.get('rollback_prediction')),
            ('Resource', analysis.get('resource_prediction')),
            ('Incident', analysis.get('incident_prediction'))
        ]:
            if pred and hasattr(pred, 'confidence_factors'):
                for factor, present in pred.confidence_factors.get('confidence_factors', {}).items():
                    if present:
                        factors.append(
                            f"\\item {factor.replace('_', ' ').title()} "
                            f"in {aspect} prediction"
                        )
        return '\n'.join(factors) if factors else r"\item No significant confidence factors identified"

    def _get_team_availability(self) -> Dict[str, List[time]]:
        """Get team availability data. Override this method to provide actual data."""
        return {
            'team_member_1': [(time(9, 0), time(17, 0))],
            'team_member_2': [(time(10, 0), time(18, 0))],
            'team_member_3': [(time(8, 0), time(16, 0))]
        }
    
    def _format_table_content(self, rows: List[List[str]], has_header: bool = True) -> List[List[str]]:
        """Format table content ensuring proper LaTeX formatting."""
        formatted_rows = []
        
        if has_header:
            # Format header row
            header = [NoEscape(r'\textbf{' + cell.strip() + '}') for cell in rows[0]]
            formatted_rows.append(header)
            rows = rows[1:]
        
        # Format data rows
        for row in rows:
            formatted_row = []
            for cell in row:
                # Handle percentage signs
                if '%' in cell:
                    cell = cell.replace('%', r'\%')
                # Handle math mode
                if any(c in cell for c in ['_', '^', '$']):
                    cell = f"${cell}$"
                formatted_row.append(NoEscape(cell.strip()))
            formatted_rows.append(formatted_row)
        
        return formatted_rows

    def _create_table(self, doc: Document, headers: List[str], rows: List[List[str]], 
                     alignment: str = None) -> None:
        """Create a table with proper formatting."""
        if alignment is None:
            alignment = 'l' * len(headers)
            
        with doc.create(Center()):
            with doc.create(Tabular(alignment)) as table:
                # Add header
                table.add_hline()
                table.add_row([NoEscape(r'\textbf{' + h + '}') for h in headers])
                table.add_hline()
                
                # Add rows
                for row in rows:
                    table.add_row([NoEscape(cell) for cell in row])
                
                # Add final line
                table.add_hline()

    def _add_size_metrics_table(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add size metrics table with proper formatting."""
        headers = ['Metric', 'Count', 'Percentage']
        rows = []
        
        total_code = sum(m.lines_code for m in data['stats'].values())
        total_comments = sum(m.lines_comment for m in data['stats'].values())
        total_blank = sum(m.lines_blank for m in data['stats'].values())
        total_lines = total_code + total_comments + total_blank
        
        if total_lines > 0:
            for label, count in [
                ('Code Lines', total_code),
                ('Comment Lines', total_comments),
                ('Blank Lines', total_blank)
            ]:
                percentage = f"{(count/total_lines*100):.1f}\\%"
                rows.append([label, f"{count:,}", percentage])
        else:
            rows.append(['No Code', '0', '0\\%'])
            
        self._create_table(doc, headers, rows, 'lrr')

    def _add_risk_table(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add risk analysis table with proper formatting."""
        risk_files = [
            (file, m.complexity.change_risk, 
             m.change_probability.change_frequency if m.change_probability else 0)
            for file, m in data['stats'].items()
            if m.complexity.change_risk > 0
        ]
        
        headers = ['File', 'Risk Score', 'Change Frequency']
        if not risk_files:
            rows = [['No files', '0', '0']]
        else:
            rows = [
                [Path(file).name, f"{risk:.1f}", str(freq)]
                for file, risk, freq in sorted(risk_files, key=lambda x: x[1], reverse=True)[:5]
            ]
        
        self._create_table(doc, headers, rows, 'lrr')