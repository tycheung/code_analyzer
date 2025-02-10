from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, Image, ListFlowable, ListItem
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
import datetime
import math

class CodeMetricsPDFGenerator:
    """Generates comprehensive PDF reports for code analysis metrics."""
    
    def __init__(self, output_path: str, pagesize=A4):
        self.output_path = output_path
        self.pagesize = pagesize
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Initialize document
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=pagesize,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Story will contain all flowables
        self.story = []
        
        # Define colors for charts and tables
        self.colors = {
            'primary': colors.HexColor('#1f77b4'),
            'secondary': colors.HexColor('#ff7f0e'),
            'danger': colors.HexColor('#d62728'),
            'success': colors.HexColor('#2ca02c'),
            'warning': colors.HexColor('#ffeb3b'),
            'info': colors.HexColor('#17a2b8'),
            'background': colors.HexColor('#f8f9fa'),
            'text': colors.HexColor('#212529')
        }
        
    def _setup_custom_styles(self):
        """Set up custom paragraph and table styles."""
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=self.colors['text']
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=self.colors['text']
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=15,
            textColor=self.colors['text']
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=self.colors['text']
        ))
        
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=self.colors['primary']
        ))
        
        # Table styles
        self.table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['background']),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['text']),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])
        
        # Style for alternating table rows
        self.alternating_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['background']),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['text']),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, self.colors['background']])
        ])
    
    def _add_title(self, title: str, style='MainTitle'):
        """Add a title to the document."""
        self.story.append(Paragraph(title, self.styles[style]))
        self.story.append(Spacer(1, 20))
        
    def _add_paragraph(self, text: str, style='BodyText'):
        """Add a paragraph to the document."""
        self.story.append(Paragraph(text, self.styles[style]))
        self.story.append(Spacer(1, 10))
        
    def _add_table(self, data, colWidths=None, style=None):
        """Add a table to the document."""
        if style is None:
            style = self.table_style
        table = Table(data, colWidths=colWidths)
        table.setStyle(style)
        self.story.append(table)
        self.story.append(Spacer(1, 20))
        
    def _add_page_break(self):
        """Add a page break to the document."""
        self.story.append(PageBreak())
        
    def generate_pdf(self, code_metrics, deployment_analysis=None):
        """Generate the complete PDF report."""
        # Cover page
        self._generate_cover_page(code_metrics)
        self._add_page_break()
        
        # Table of contents placeholder (will be filled by reportlab)
        self.toc = []
        self._add_title("Table of Contents")
        self._add_paragraph("[[TOC]]")  # ReportLab will replace this
        self._add_page_break()
        
        # Generate each section
        self._generate_summary_section(code_metrics)
        self._add_page_break()
        
        self._generate_security_section(code_metrics)
        self._add_page_break()
        
        self._generate_architecture_section(code_metrics)
        self._add_page_break()
        
        self._generate_complexity_section(code_metrics)
        self._add_page_break()
        
        self._generate_testing_section(code_metrics)
        self._add_page_break()
        
        self._generate_trends_section(code_metrics)
        self._add_page_break()
        
        if deployment_analysis:
            self._generate_deployment_section(deployment_analysis)
            self._add_page_break()
        
        self._generate_appendix(code_metrics)
        
        # Build the PDF with table of contents
        self.doc.build(
            self.story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer
        )
        
    def _generate_cover_page(self, code_metrics):
        """Generate the report cover page."""
        # Logo or header image could be added here
        self._add_title("Code Analysis Report", "MainTitle")
        self._add_paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick stats
        total_files = len(code_metrics)
        total_lines = sum(m.lines_code + m.lines_comment + m.lines_blank for m in code_metrics.values())
        
        stats = [
            f"Files Analyzed: {total_files}",
            f"Total Lines: {total_lines:,}",
            f"Generated by: CodeMetricsPDFGenerator v1.0"
        ]
        
        for stat in stats:
            self._add_paragraph(stat)
            
    def _header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(72, doc.pagesize[1] - 36, "Code Analysis Report")
        
        # Footer
        canvas.setFont('Helvetica', 8)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawString(doc.pagesize[0] - 108, 36, text)
        
        canvas.restoreState()
        
    def _generate_appendix(self, code_metrics):
        """Generate appendix with additional details."""
        self._add_title("Appendix", "SectionTitle")
        
        # Methodology
        self._add_title("Analysis Methodology", "SubsectionTitle")
        methodology = """
        This report was generated using static code analysis techniques, examining various aspects
        of the codebase including complexity, security, architecture, and testing metrics. The
        analysis includes both automated tools and pattern recognition to identify potential
        issues and areas for improvement.
        """
        self._add_paragraph(methodology)
        
        # Metric Definitions
        self._add_title("Metric Definitions", "SubsectionTitle")
        metrics = [
            ("Cyclomatic Complexity", 
             "Measures the number of linearly independent paths through code"),
            ("Cognitive Complexity",
             "Measures how difficult the code is to understand"),
            ("Maintainability Index",
             "Indicates how maintainable the code is (0-100 scale)"),
            ("Component Coupling",
             "Measures how interconnected different components are"),
            ("Change Risk",
             "Probability of issues when modifying the code")
        ]
        
        for title, description in metrics:
            self._add_paragraph(f"• {title}: {description}")
            
        # Version Information
        self._add_title("Analysis Information", "SubsectionTitle")
        info = [
            f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Generator Version: 1.0",
            "Analysis Tools:",
            "  - Static Code Analysis",
            "  - Security Pattern Detection",
            "  - Architecture Analysis",
            "  - Testing Coverage Analysis"
        ]
        
        for line in info:
            self._add_paragraph(line)

    def _generate_summary_section(self, code_metrics):
        """Generate the summary section of the report."""
        self._add_title("1. Executive Summary", "SectionTitle")
        
        # Overall metrics summary
        total_files = len(code_metrics)
        total_lines = sum(m.lines_code + m.lines_comment + m.lines_blank for m in code_metrics.values())
        total_code_lines = sum(m.lines_code for m in code_metrics.values())
        
        summary_text = f"""
        This report analyzes {total_files} source code files containing a total of {total_lines:,} lines,
        of which {total_code_lines:,} are lines of code. The analysis covers code quality, security,
        architecture, and complexity metrics.
        """
        self._add_paragraph(summary_text)
        
        # Key metrics table
        self._add_title("Key Metrics", "SubsectionTitle")
        
        # Calculate aggregate metrics
        avg_complexity = sum(m.complexity.cyclomatic_complexity for m in code_metrics.values()) / total_files
        security_issues = sum(
            m.security.potential_sql_injections + 
            m.security.hardcoded_secrets + 
            len(m.security.vulnerable_imports)
            for m in code_metrics.values()
        )
        arch_issues = sum(
            len(m.architecture.circular_dependencies) + 
            m.architecture.layering_violations
            for m in code_metrics.values()
        )
        
        key_metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Files Analyzed', str(total_files), 'INFO'],
            ['Total Lines', f"{total_lines:,}", 'INFO'],
            ['Average Complexity', f"{avg_complexity:.2f}", 'WARNING' if avg_complexity > 10 else 'SUCCESS'],
            ['Security Issues', str(security_issues), 'DANGER' if security_issues > 0 else 'SUCCESS'],
            ['Architecture Issues', str(arch_issues), 'WARNING' if arch_issues > 0 else 'SUCCESS']
        ]
        
        # Apply color coding based on status
        status_colors = {
            'DANGER': self.colors['danger'],
            'WARNING': self.colors['warning'],
            'SUCCESS': self.colors['success'],
            'INFO': self.colors['info']
        }
        
        table_style = self.table_style
        for i, row in enumerate(key_metrics_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (1, i), (1, i), color)
        
        self._add_table(key_metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # File distribution pie chart
        self._add_title("File Distribution", "SubsectionTitle")
        
        # Group files by extension
        extension_counts = {}
        for file_path in code_metrics:
            ext = file_path.split('.')[-1]
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
        
        # Create pie chart
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 25
        pie.width = 150
        pie.height = 150
        
        pie_data = list(extension_counts.values())
        pie_labels = list(extension_counts.keys())
        
        pie.data = pie_data
        pie.labels = pie_labels
        pie.slices.strokeWidth = 0.5
        
        # Use different colors for each slice
        for i, (ext, count) in enumerate(extension_counts.items()):
            pie.slices[i].fillColor = colors.HexColor(f"#{hash(ext) % 0xFFFFFF:06x}")
        
        drawing.add(pie)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Add legend
        legend_data = [['Extension', 'Files', 'Percentage']]
        total = sum(pie_data)
        for ext, count in extension_counts.items():
            percentage = (count / total) * 100
            legend_data.append([ext, str(count), f"{percentage:.1f}%"])
        
        self._add_table(legend_data, colWidths=[1.5*inch, inch, inch])
        
        # Code quality metrics
        self._add_title("Code Quality Overview", "SubsectionTitle")
        
        quality_data = [
            ['Metric', 'Value', 'Threshold'],
            ['Comment Ratio', f"{sum(m.lines_comment for m in code_metrics.values()) / total_lines * 100:.1f}%", "20%"],
            ['Average Function Length', f"{sum(m.avg_function_length for m in code_metrics.values()) / total_files:.1f}", "15"],
            ['TODOs', str(sum(m.todo_count for m in code_metrics.values())), "N/A"]
        ]
        
        self._add_table(quality_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        
        # Add summary insights
        self._add_title("Key Insights", "SubsectionTitle")
        
        # Calculate insights
        high_complexity_files = [
            f for f, m in code_metrics.items() 
            if m.complexity.cyclomatic_complexity > 15
        ]
        
        security_files = [
            f for f, m in code_metrics.items()
            if (m.security.potential_sql_injections + 
                m.security.hardcoded_secrets + 
                len(m.security.vulnerable_imports)) > 0
        ]
        
        insights = [
            f"• {len(high_complexity_files)} files exceed recommended complexity thresholds",
            f"• {len(security_files)} files contain potential security issues",
            f"• Average maintainability index: {sum(m.complexity.maintainability_index for m in code_metrics.values()) / total_files:.1f}/100"
        ]
        
        for insight in insights:
            self._add_paragraph(insight)
            
        # Add recommendations if there are issues
        if high_complexity_files or security_files:
            self._add_title("Recommendations", "SubsectionTitle")
            recommendations = [
                "• Consider refactoring complex files to improve maintainability",
                "• Address security issues in identified files as a priority",
                "• Increase test coverage for high-risk components",
                "• Review and resolve TODO items in the codebase",
                "• Consider adding more documentation for complex components"
            ]
            
            for recommendation in recommendations:
                self._add_paragraph(recommendation)
                
        # Add a trend chart if historical data is available
        if hasattr(code_metrics, 'history'):
            self._add_title("Historical Trends", "SubsectionTitle")
            
            # Create line chart for historical metrics
            drawing = Drawing(400, 200)
            chart = HorizontalLineChart()
            chart.x = 50
            chart.y = 50
            chart.height = 125
            chart.width = 300
            
            # Extract historical data
            dates = [h['date'] for h in code_metrics.history]
            complexities = [h['avg_complexity'] for h in code_metrics.history]
            issues = [h['total_issues'] for h in code_metrics.history]
            
            chart.data = [complexities, issues]
            chart.lines[0].strokeColor = self.colors['primary']
            chart.lines[1].strokeColor = self.colors['danger']
            
            chart.categoryAxis.categoryNames = dates
            chart.categoryAxis.labels.boxAnchor = 'ne'
            chart.categoryAxis.labels.dx = -8
            chart.categoryAxis.labels.dy = -2
            chart.categoryAxis.labels.angle = 30
            
            drawing.add(chart)
            self.story.append(drawing)
            self.story.append(Spacer(1, 20))

    def _generate_security_section(self, code_metrics):
        """Generate the security analysis section of the report."""
        self._add_title("2. Security Analysis", "SectionTitle")
        
        # Overview paragraph
        total_security_issues = sum(
            m.security.potential_sql_injections + 
            m.security.hardcoded_secrets + 
            m.security.unsafe_regex +
            len(m.security.vulnerable_imports)
            for m in code_metrics.values()
        )
        
        overview = f"""
        Security analysis identified {total_security_issues} potential security issues across the codebase.
        This section details the types of vulnerabilities found and their locations.
        """
        self._add_paragraph(overview)
        
        # Security issues breakdown
        self._add_title("Security Issues Breakdown", "SubsectionTitle")
        
        # Aggregate security metrics
        total_sql_injections = sum(m.security.potential_sql_injections for m in code_metrics.values())
        total_hardcoded = sum(m.security.hardcoded_secrets for m in code_metrics.values())
        total_unsafe_regex = sum(m.security.unsafe_regex for m in code_metrics.values())
        total_vuln_imports = sum(len(m.security.vulnerable_imports) for m in code_metrics.values())
        
        security_data = [
            ['Vulnerability Type', 'Count', 'Severity'],
            ['SQL Injection Risks', str(total_sql_injections), 'HIGH'],
            ['Hardcoded Secrets', str(total_hardcoded), 'HIGH'],
            ['Unsafe Regex Usage', str(total_unsafe_regex), 'MEDIUM'],
            ['Vulnerable Imports', str(total_vuln_imports), 'MEDIUM']
        ]
        
        # Add severity colors
        severity_colors = {
            'HIGH': self.colors['danger'],
            'MEDIUM': self.colors['warning'],
            'LOW': self.colors['info']
        }
        
        table_style = self.table_style
        for i, row in enumerate(security_data[1:], 1):
            severity = row[2]
            color = severity_colors.get(severity, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(security_data, colWidths=[2.5*inch, inch, 1.5*inch], style=table_style)
        
        # Security issues distribution chart
        drawing = Drawing(400, 200)
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        chart.data = [[
            total_sql_injections,
            total_hardcoded,
            total_unsafe_regex,
            total_vuln_imports
        ]]
        
        chart.categoryAxis.categoryNames = ['SQL Injection', 'Secrets', 'Regex', 'Imports']
        chart.bars[0].fillColor = self.colors['danger']
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Files with security issues
        self._add_title("Files with Security Issues", "SubsectionTitle")
        
        vulnerable_files = [
            (file_path, metrics.security)
            for file_path, metrics in code_metrics.items()
            if (metrics.security.potential_sql_injections + 
                metrics.security.hardcoded_secrets + 
                metrics.security.unsafe_regex +
                len(metrics.security.vulnerable_imports)) > 0
        ]
        
        if vulnerable_files:
            vuln_data = [['File', 'Issues Found', 'Risk Level']]
            
            for file_path, security in vulnerable_files:
                total_issues = (
                    security.potential_sql_injections +
                    security.hardcoded_secrets +
                    security.unsafe_regex +
                    len(security.vulnerable_imports)
                )
                
                risk_level = 'HIGH' if total_issues > 2 else 'MEDIUM' if total_issues > 1 else 'LOW'
                
                vuln_data.append([
                    file_path.split('/')[-1],  # Just the filename
                    str(total_issues),
                    risk_level
                ])
            
            # Apply color coding for risk levels
            table_style = self.alternating_table_style
            for i, row in enumerate(vuln_data[1:], 1):
                risk = row[2]
                color = severity_colors.get(risk, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(vuln_data, colWidths=[3*inch, inch, inch], style=table_style)
        else:
            self._add_paragraph("No files with security issues were found.")
        
        # Insecure patterns analysis
        self._add_title("Insecure Pattern Analysis", "SubsectionTitle")
        
        # Aggregate insecure patterns
        pattern_counts = {}
        for metrics in code_metrics.values():
            for pattern, count in metrics.security.insecure_patterns.items():
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + count
        
        if pattern_counts:
            pattern_data = [['Pattern Type', 'Occurrences', 'Risk Level']]
            
            for pattern, count in pattern_counts.items():
                risk_level = 'HIGH' if count > 5 else 'MEDIUM' if count > 2 else 'LOW'
                pattern_data.append([
                    pattern.replace('_', ' ').title(),
                    str(count),
                    risk_level
                ])
            
            # Apply color coding for risk levels
            table_style = self.alternating_table_style
            for i, row in enumerate(pattern_data[1:], 1):
                risk = row[2]
                color = severity_colors.get(risk, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(pattern_data, colWidths=[2.5*inch, inch, 1.5*inch], style=table_style)
        else:
            self._add_paragraph("No insecure patterns were detected.")
        
        # Recommendations
        self._add_title("Security Recommendations", "SubsectionTitle")
        
        recommendations = []
        
        if total_sql_injections > 0:
            recommendations.append(
                "• Use parameterized queries or ORM to prevent SQL injection vulnerabilities"
            )
        if total_hardcoded > 0:
            recommendations.append(
                "• Move secrets to environment variables or secure secret management systems"
            )
        if total_unsafe_regex > 0:
            recommendations.append(
                "• Review and secure regex patterns, especially in user input processing"
            )
        if total_vuln_imports > 0:
            recommendations.append(
                "• Replace vulnerable imports with secure alternatives"
            )
        
        if recommendations:
            for rec in recommendations:
                self._add_paragraph(rec)
        else:
            self._add_paragraph("No specific security recommendations at this time.")

    def _generate_architecture_section(self, code_metrics):
        """Generate the architecture analysis section of the report."""
        self._add_title("3. Architecture Analysis", "SectionTitle")
        
        # Overview
        total_files = len(code_metrics)
        total_interfaces = sum(m.architecture.interface_count for m in code_metrics.values())
        total_abstract_classes = sum(m.architecture.abstract_class_count for m in code_metrics.values())
        total_circular_deps = sum(len(m.architecture.circular_dependencies) for m in code_metrics.values())
        
        overview = f"""
        Architectural analysis examined {total_files} files, identifying {total_interfaces} interfaces
        and {total_abstract_classes} abstract classes. The analysis revealed {total_circular_deps}
        potential circular dependencies.
        """
        self._add_paragraph(overview)
        
        # Architecture Metrics Overview
        self._add_title("Architecture Metrics Overview", "SubsectionTitle")
        
        # Calculate average coupling and abstraction
        avg_coupling = sum(m.architecture.component_coupling for m in code_metrics.values()) / total_files
        avg_abstraction = sum(m.architecture.abstraction_level for m in code_metrics.values()) / total_files
        total_layering = sum(m.architecture.layering_violations for m in code_metrics.values())
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Component Coupling', f"{avg_coupling:.2f}", 'HIGH' if avg_coupling > 0.7 else 'MEDIUM' if avg_coupling > 0.4 else 'LOW'],
            ['Abstraction Level', f"{avg_abstraction:.2f}", 'LOW' if avg_abstraction < 0.2 else 'MEDIUM' if avg_abstraction < 0.4 else 'HIGH'],
            ['Layering Violations', str(total_layering), 'HIGH' if total_layering > 10 else 'MEDIUM' if total_layering > 5 else 'LOW'],
            ['Circular Dependencies', str(total_circular_deps), 'HIGH' if total_circular_deps > 0 else 'LOW']
        ]
        
        # Apply status colors
        status_colors = {
            'HIGH': self.colors['danger'],
            'MEDIUM': self.colors['warning'],
            'LOW': self.colors['success']
        }
        
        table_style = self.table_style
        for i, row in enumerate(metrics_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # Component Coupling Distribution
        self._add_title("Component Coupling Analysis", "SubsectionTitle")
        
        # Create coupling distribution chart
        coupling_ranges = {
            'Low (0-0.3)': 0,
            'Medium (0.3-0.7)': 0,
            'High (0.7-1.0)': 0
        }
        
        for metrics in code_metrics.values():
            coupling = metrics.architecture.component_coupling
            if coupling <= 0.3:
                coupling_ranges['Low (0-0.3)'] += 1
            elif coupling <= 0.7:
                coupling_ranges['Medium (0.3-0.7)'] += 1
            else:
                coupling_ranges['High (0.7-1.0)'] += 1
        
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 25
        pie.width = 150
        pie.height = 150
        
        pie.data = list(coupling_ranges.values())
        pie.labels = list(coupling_ranges.keys())
        
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = self.colors['success']
        pie.slices[1].fillColor = self.colors['warning']
        pie.slices[2].fillColor = self.colors['danger']
        
        drawing.add(pie)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Dependency Analysis
        self._add_title("Dependency Analysis", "SubsectionTitle")
        
        # Find files with circular dependencies
        circular_deps = [
            (file_path, metrics.architecture.circular_dependencies)
            for file_path, metrics in code_metrics.items()
            if metrics.architecture.circular_dependencies
        ]
        
        if circular_deps:
            deps_data = [['File', 'Circular Dependencies', 'Severity']]
            
            for file_path, dependencies in circular_deps:
                severity = 'HIGH' if len(dependencies) > 2 else 'MEDIUM'
                deps_data.append([
                    file_path.split('/')[-1],
                    str(len(dependencies)),
                    severity
                ])
            
            # Apply severity colors
            table_style = self.alternating_table_style
            for i, row in enumerate(deps_data[1:], 1):
                severity = row[2]
                color = status_colors.get(severity, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(deps_data, colWidths=[3*inch, 2*inch, inch], style=table_style)
            
            # Detailed circular dependencies
            self._add_title("Circular Dependency Details", "SubsectionTitle")
            for file_path, dependencies in circular_deps:
                self._add_paragraph(f"File: {file_path}", "SubsectionTitle")
                for dep in dependencies:
                    self._add_paragraph(f"• Circular dependency with: {' -> '.join(dep)}")
        else:
            self._add_paragraph("No circular dependencies detected in the codebase.")
        
        # Layering Analysis
        self._add_title("Layering Analysis", "SubsectionTitle")
        
        # Find files with layering violations
        layering_violations = [
            (file_path, metrics.architecture.layering_violations)
            for file_path, metrics in code_metrics.items()
            if metrics.architecture.layering_violations > 0
        ]
        
        if layering_violations:
            violations_data = [['File', 'Violations', 'Severity']]
            
            for file_path, violation_count in layering_violations:
                severity = 'HIGH' if violation_count > 5 else 'MEDIUM' if violation_count > 2 else 'LOW'
                violations_data.append([
                    file_path.split('/')[-1],
                    str(violation_count),
                    severity
                ])
            
            # Apply severity colors
            table_style = self.alternating_table_style
            for i, row in enumerate(violations_data[1:], 1):
                severity = row[2]
                color = status_colors.get(severity, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(violations_data, colWidths=[3*inch, inch, inch], style=table_style)
        else:
            self._add_paragraph("No layering violations detected in the codebase.")
        
        # Abstraction Analysis
        self._add_title("Abstraction Analysis", "SubsectionTitle")
        
        # Create abstraction metrics chart
        drawing = Drawing(400, 200)
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        # Collect abstraction metrics
        high_abstraction = sum(1 for m in code_metrics.values() if m.architecture.abstraction_level > 0.7)
        medium_abstraction = sum(1 for m in code_metrics.values() if 0.3 < m.architecture.abstraction_level <= 0.7)
        low_abstraction = sum(1 for m in code_metrics.values() if m.architecture.abstraction_level <= 0.3)
        
        chart.data = [[high_abstraction, medium_abstraction, low_abstraction]]
        chart.categoryAxis.categoryNames = ['High', 'Medium', 'Low']
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.valueAxis.valueMin = 0
        
        # Color coding for abstraction levels
        chart.bars[0].fillColor = self.colors['primary']
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Architecture Recommendations
        self._add_title("Architecture Recommendations", "SubsectionTitle")
        
        recommendations = []
        
        if avg_coupling > 0.4:
            recommendations.append(
                "• Consider reducing component coupling through better encapsulation and interface-based design"
            )
        if avg_abstraction < 0.3:
            recommendations.append(
                "• Increase abstraction level by introducing more interfaces and abstract classes"
            )
        if total_layering > 0:
            recommendations.append(
                "• Address layering violations to maintain proper architectural boundaries"
            )
        if total_circular_deps > 0:
            recommendations.append(
                "• Resolve circular dependencies to improve maintainability and reduce complexity"
            )
        
        if recommendations:
            for rec in recommendations:
                self._add_paragraph(rec)
        else:
            self._add_paragraph("No specific architecture recommendations at this time.")

    def _generate_complexity_section(self, code_metrics):
        """Generate the complexity analysis section of the report."""
        self._add_title("4. Complexity Analysis", "SectionTitle")
        
        # Overview
        avg_cyclomatic = sum(m.complexity.cyclomatic_complexity for m in code_metrics.values()) / len(code_metrics)
        avg_cognitive = sum(m.complexity.cognitive_complexity for m in code_metrics.values()) / len(code_metrics)
        avg_maintainability = sum(m.complexity.maintainability_index for m in code_metrics.values()) / len(code_metrics)
        
        overview = f"""
        Complexity analysis examined various aspects of code complexity across the codebase. 
        The average cyclomatic complexity is {avg_cyclomatic:.2f}, with an average cognitive complexity 
        of {avg_cognitive:.2f}. The overall maintainability index is {avg_maintainability:.2f}/100.
        """
        self._add_paragraph(overview)
        
        # Complexity Metrics Overview
        self._add_title("Complexity Metrics Overview", "SubsectionTitle")
        
        max_nesting = max(m.complexity.max_nesting_depth for m in code_metrics.values())
        avg_change_risk = sum(m.complexity.change_risk for m in code_metrics.values()) / len(code_metrics)
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Cyclomatic Complexity', f"{avg_cyclomatic:.2f}", 'HIGH' if avg_cyclomatic > 15 else 'MEDIUM' if avg_cyclomatic > 10 else 'LOW'],
            ['Cognitive Complexity', f"{avg_cognitive:.2f}", 'HIGH' if avg_cognitive > 20 else 'MEDIUM' if avg_cognitive > 15 else 'LOW'],
            ['Max Nesting Depth', str(max_nesting), 'HIGH' if max_nesting > 5 else 'MEDIUM' if max_nesting > 3 else 'LOW'],
            ['Maintainability Index', f"{avg_maintainability:.2f}", 'LOW' if avg_maintainability < 50 else 'MEDIUM' if avg_maintainability < 75 else 'HIGH'],
            ['Change Risk', f"{avg_change_risk:.2f}%", 'HIGH' if avg_change_risk > 75 else 'MEDIUM' if avg_change_risk > 50 else 'LOW']
        ]
        
        # Apply status colors
        status_colors = {
            'HIGH': self.colors['danger'],
            'MEDIUM': self.colors['warning'],
            'LOW': self.colors['success']
        }
        
        table_style = self.table_style
        for i, row in enumerate(metrics_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # Complexity Distribution
        self._add_title("Complexity Distribution", "SubsectionTitle")
        
        # Create complexity distribution chart
        drawing = Drawing(400, 200)
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        # Group files by complexity ranges
        complexity_ranges = {
            'Low (0-10)': 0,
            'Medium (11-20)': 0,
            'High (21-30)': 0,
            'Very High (>30)': 0
        }
        
        for metrics in code_metrics.values():
            complexity = metrics.complexity.cyclomatic_complexity
            if complexity <= 10:
                complexity_ranges['Low (0-10)'] += 1
            elif complexity <= 20:
                complexity_ranges['Medium (11-20)'] += 1
            elif complexity <= 30:
                complexity_ranges['High (21-30)'] += 1
            else:
                complexity_ranges['Very High (>30)'] += 1
        
        chart.data = [list(complexity_ranges.values())]
        chart.categoryAxis.categoryNames = list(complexity_ranges.keys())
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.categoryAxis.labels.angle = 30
        chart.valueAxis.valueMin = 0
        
        # Color coding for complexity
        chart.bars[0].fillColor = self.colors['primary']
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Files with High Complexity
        self._add_title("High Complexity Files", "SubsectionTitle")
        
        high_complexity_files = [
            (file_path, metrics.complexity)
            for file_path, metrics in code_metrics.items()
            if metrics.complexity.cyclomatic_complexity > 15
        ]
        
        if high_complexity_files:
            complexity_data = [['File', 'Cyclomatic', 'Cognitive', 'Maintainability', 'Risk Level']]
            
            for file_path, complexity in high_complexity_files:
                risk_level = 'HIGH' if complexity.cyclomatic_complexity > 30 else 'MEDIUM'
                complexity_data.append([
                    file_path.split('/')[-1],
                    str(complexity.cyclomatic_complexity),
                    str(complexity.cognitive_complexity),
                    f"{complexity.maintainability_index:.1f}",
                    risk_level
                ])
            
            # Sort by cyclomatic complexity
            complexity_data[1:] = sorted(
                complexity_data[1:],
                key=lambda x: int(x[1]),
                reverse=True
            )
            
            # Apply risk level colors
            table_style = self.alternating_table_style
            for i, row in enumerate(complexity_data[1:], 1):
                risk = row[4]
                color = status_colors.get(risk, self.colors['info'])
                table_style.add('TEXTCOLOR', (4, i), (4, i), color)
            
            self._add_table(complexity_data, colWidths=[2*inch, inch, inch, inch, inch], style=table_style)
        else:
            self._add_paragraph("No files with high complexity were found.")
        
        # Halstead Metrics Analysis
        self._add_title("Halstead Metrics Analysis", "SubsectionTitle")
        
        # Calculate average Halstead metrics
        avg_halstead = {
            'volume': 0,
            'difficulty': 0,
            'effort': 0,
            'vocabulary': 0,
            'length': 0
        }
        
        for metrics in code_metrics.values():
            for key in avg_halstead:
                avg_halstead[key] += metrics.complexity.halstead_metrics.get(key, 0)
        
        for key in avg_halstead:
            avg_halstead[key] /= len(code_metrics)
        
        halstead_data = [
            ['Metric', 'Average Value', 'Status'],
            ['Program Length', f"{avg_halstead['length']:.2f}", 'MEDIUM'],
            ['Vocabulary Size', f"{avg_halstead['vocabulary']:.2f}", 'INFO'],
            ['Program Volume', f"{avg_halstead['volume']:.2f}", 'MEDIUM'],
            ['Difficulty Level', f"{avg_halstead['difficulty']:.2f}", 'HIGH' if avg_halstead['difficulty'] > 30 else 'MEDIUM'],
            ['Programming Effort', f"{avg_halstead['effort']:.2f}", 'HIGH' if avg_halstead['effort'] > 1000000 else 'MEDIUM']
        ]
        
        # Apply status colors
        table_style = self.table_style
        for i, row in enumerate(halstead_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(halstead_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # Change Risk Analysis
        self._add_title("Change Risk Analysis", "SubsectionTitle")
        
        high_risk_files = [
            (file_path, metrics.complexity.change_risk)
            for file_path, metrics in code_metrics.items()
            if metrics.complexity.change_risk > 75
        ]
        
        if high_risk_files:
            risk_data = [['File', 'Risk Score', 'Risk Level']]
            
            for file_path, risk in high_risk_files:
                risk_level = 'HIGH' if risk > 90 else 'MEDIUM'
                risk_data.append([
                    file_path.split('/')[-1],
                    f"{risk:.1f}%",
                    risk_level
                ])
            
            # Sort by risk score
            risk_data[1:] = sorted(
                risk_data[1:],
                key=lambda x: float(x[1].rstrip('%')),
                reverse=True
            )
            
            # Apply risk level colors
            table_style = self.alternating_table_style
            for i, row in enumerate(risk_data[1:], 1):
                risk = row[2]
                color = status_colors.get(risk, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(risk_data, colWidths=[3*inch, inch, inch], style=table_style)
        
        # Recommendations
        self._add_title("Complexity Recommendations", "SubsectionTitle")
        
        recommendations = []
        
        if avg_cyclomatic > 10:
            recommendations.append(
                "• Consider breaking down complex methods to reduce cyclomatic complexity below 10"
            )
        if avg_cognitive > 15:
            recommendations.append(
                "• Simplify complex logic flows to reduce cognitive load on developers"
            )
        if max_nesting > 3:
            recommendations.append(
                "• Reduce nesting depth through early returns and guard clauses"
            )
        if avg_maintainability < 75:
            recommendations.append(
                "• Improve code maintainability through better documentation and simpler designs"
            )
        if avg_change_risk > 50:
            recommendations.append(
                "• Focus on high-risk files during code reviews and testing"
            )
        if avg_halstead['difficulty'] > 30:
            recommendations.append(
                "• Simplify complex algorithms and reduce code volume where possible"
            )
        
        if recommendations:
            for rec in recommendations:
                self._add_paragraph(rec)
        else:
            self._add_paragraph("No specific complexity recommendations at this time.")

    def _generate_deployment_section(self, deployment_analysis):
        """Generate the deployment analysis section of the report."""
        self._add_title("5. Deployment Analysis", "SectionTitle")
        
        # Overview with confidence scores
        overview = f"""
        Deployment analysis was performed with an overall confidence of {deployment_analysis.overall_confidence * 100:.1f}%.
        The analysis includes optimal deployment windows, rollback risk assessment, resource requirements,
        and potential incident predictions.
        """
        self._add_paragraph(overview)
        
        # Confidence Breakdown
        self._add_title("Analysis Confidence", "SubsectionTitle")
        
        confidence_data = [['Component', 'Confidence Score', 'Status']]
        for component, score in deployment_analysis.confidence_breakdown.items():
            status = 'HIGH' if score > 0.8 else 'MEDIUM' if score > 0.6 else 'LOW'
            confidence_data.append([
                component.title(),
                f"{score * 100:.1f}%",
                status
            ])
        
        # Apply status colors
        status_colors = {
            'HIGH': self.colors['success'],
            'MEDIUM': self.colors['warning'],
            'LOW': self.colors['danger']
        }
        
        table_style = self.table_style
        for i, row in enumerate(confidence_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(confidence_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # Optimal Deployment Windows
        if deployment_analysis.optimal_windows:
            self._add_title("Optimal Deployment Windows", "SubsectionTitle")
            
            windows_data = [['Time Window', 'Risk Score', 'Team Availability', 'Status']]
            for window in deployment_analysis.optimal_windows:
                risk_status = 'LOW' if window.risk_score < 0.3 else 'MEDIUM' if window.risk_score < 0.7 else 'HIGH'
                windows_data.append([
                    f"{window.start_time.strftime('%H:%M')} - {window.end_time.strftime('%H:%M')}",
                    f"{window.risk_score * 100:.1f}%",
                    f"{window.team_availability * 100:.1f}%",
                    risk_status
                ])
            
            # Apply risk colors
            table_style = self.alternating_table_style
            for i, row in enumerate(windows_data[1:], 1):
                status = row[3]
                color = status_colors.get(status, self.colors['info'])
                table_style.add('TEXTCOLOR', (3, i), (3, i), color)
            
            self._add_table(windows_data, colWidths=[2*inch, inch, inch, inch], style=table_style)
            
            # Add window explanations
            self._add_paragraph("Window Selection Rationale:", "SubsectionTitle")
            self._add_paragraph(deployment_analysis.optimal_windows[0].explanation)
        
        # Rollback Risk Analysis
        if deployment_analysis.rollback_prediction:
            self._add_title("Rollback Risk Analysis", "SubsectionTitle")
            
            rollback = deployment_analysis.rollback_prediction
            risk_data = [
                ['Risk Factor', 'Impact', 'Severity'],
                ['Overall Probability', f"{rollback.probability * 100:.1f}%", rollback.severity_level.upper()]
            ]
            
            for factor, impact in rollback.risk_factors.items():
                severity = 'HIGH' if impact > 0.7 else 'MEDIUM' if impact > 0.3 else 'LOW'
                risk_data.append([
                    factor.replace('_', ' ').title(),
                    f"{impact * 100:.1f}%",
                    severity
                ])
            
            # Apply severity colors
            table_style = self.alternating_table_style
            for i, row in enumerate(risk_data[1:], 1):
                severity = row[2]
                color = status_colors.get(severity, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(risk_data, colWidths=[2.5*inch, 1.5*inch, inch], style=table_style)
            
            # Critical files
            if rollback.critical_files:
                self._add_title("Critical Files", "SubsectionTitle")
                critical_files_text = "The following files require special attention:\n" + \
                    "\n".join(f"• {file}" for file in rollback.critical_files)
                self._add_paragraph(critical_files_text)
        
        # Resource Requirements
        if deployment_analysis.resource_prediction:
            self._add_title("Resource Requirements", "SubsectionTitle")
            
            resource = deployment_analysis.resource_prediction
            resource_data = [
                ['Resource Metric', 'Requirement', 'Notes'],
                ['Team Size', str(resource.recommended_team_size), 'Required'],
                ['Support Duration', f"{resource.estimated_support_duration:.1f} hours", 'Estimated'],
            ]
            
            self._add_table(resource_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            
            # Required skills
            if resource.required_skills:
                self._add_paragraph("Required Skills:", "SubsectionTitle")
                skills_text = "\n".join(f"• {skill}" for skill in resource.required_skills)
                self._add_paragraph(skills_text)
        
        # Incident Prediction
        if deployment_analysis.incident_prediction:
            self._add_title("Incident Prediction", "SubsectionTitle")
            
            incident = deployment_analysis.incident_prediction
            incident_data = [
                ['Metric', 'Value', 'Severity'],
                ['Incident Probability', f"{incident.probability * 100:.1f}%", incident.severity_level.upper()],
                ['Est. Resolution Time', f"{incident.estimated_resolution_time:.1f} hours", 'INFO']
            ]
            
            # Apply severity colors
            table_style = self.table_style
            severity = incident.severity_level.upper()
            color = status_colors.get(severity, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, 1), (2, 1), color)
            
            self._add_table(incident_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
            
            # Potential problem areas
            if incident.potential_areas:
                self._add_paragraph("Potential Problem Areas:", "SubsectionTitle")
                areas_text = "\n".join(f"• {area}" for area in incident.potential_areas)
                self._add_paragraph(areas_text)
        
        # Key Insights
        if deployment_analysis.key_insights:
            self._add_title("Key Insights", "SubsectionTitle")
            
            for insight in deployment_analysis.key_insights:
                impact_color = self.colors['danger'] if insight['impact'] > 0.7 else \
                            self.colors['warning'] if insight['impact'] > 0.3 else \
                            self.colors['success']
                
                insight_text = (
                    f"• {insight['aspect'].replace('_', ' ').title()}: "
                    f"{insight['feature']} (Impact: {insight['impact'] * 100:.1f}%)"
                )
                
                p = Paragraph(insight_text, self.styles['BodyText'])
                p.textColor = impact_color
                self.story.append(p)
                self.story.append(Spacer(1, 5))
        
        # Feature Importance Analysis
        if deployment_analysis.feature_importances.get('cross_model'):
            self._add_title("Feature Importance Analysis", "SubsectionTitle")
            
            # Create feature importance chart
            drawing = Drawing(400, 200)
            chart = HorizontalLineChart()
            chart.x = 50
            chart.y = 50
            chart.height = 125
            chart.width = 300
            
            # Sort features by importance
            sorted_features = sorted(
                deployment_analysis.feature_importances['cross_model'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 features
            
            features = [f[0] for f in sorted_features]
            importances = [f[1] for f in sorted_features]
            
            chart.data = [importances]
            chart.categoryAxis.categoryNames = features
            chart.categoryAxis.labels.boxAnchor = 'ne'
            chart.categoryAxis.labels.angle = 30
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max(importances) * 1.1
            
            chart.lines[0].strokeColor = self.colors['primary']
            
            drawing.add(chart)
            self.story.append(drawing)
            self.story.append(Spacer(1, 20))
        
        # System Explanation
        if deployment_analysis.system_explanation:
            self._add_title("Detailed Analysis", "SubsectionTitle")
            self._add_paragraph(deployment_analysis.system_explanation)

    def _generate_trends_section(self, code_metrics):
        """Generate historical trends analysis section."""
        self._add_title("6. Historical Trends", "SectionTitle")
        
        # Overview
        self._add_paragraph(
            "This section analyzes how code metrics have changed over time, identifying "
            "trends and patterns in code quality, complexity, and maintenance."
        )
        
        # Get historical metrics
        history = getattr(code_metrics, 'history', None)
        if not history:
            self._add_paragraph(
                "No historical data available for trend analysis. Consider enabling metric "
                "history tracking for future reports."
            )
            return
            
        # Complexity Trends
        self._add_title("Complexity Evolution", "SubsectionTitle")
        
        drawing = Drawing(400, 200)
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        # Extract trend data
        dates = [h['date'] for h in history]
        avg_complexity = [h['avg_complexity'] for h in history]
        avg_cognitive = [h['avg_cognitive_complexity'] for h in history]
        
        chart.data = [avg_complexity, avg_cognitive]
        chart.lines[0].strokeColor = self.colors['primary']
        chart.lines[1].strokeColor = self.colors['secondary']
        
        chart.categoryAxis.categoryNames = dates
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.categoryAxis.labels.angle = 30
        
        # Add legend
        legend = Legend()
        legend.x = 380
        legend.y = 150
        legend.alignment = 'right'
        legend.columnMaximum = 1
        legend.colorNamePairs = [
            (self.colors['primary'], 'Cyclomatic Complexity'),
            (self.colors['secondary'], 'Cognitive Complexity')
        ]
        drawing.add(legend)
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Code Size Trends
        self._add_title("Code Base Growth", "SubsectionTitle")
        
        drawing = Drawing(400, 200)
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        total_lines = [h['total_lines'] for h in history]
        code_lines = [h['code_lines'] for h in history]
        
        chart.data = [total_lines, code_lines]
        chart.lines[0].strokeColor = self.colors['primary']
        chart.lines[1].strokeColor = self.colors['success']
        
        chart.categoryAxis.categoryNames = dates
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.categoryAxis.labels.angle = 30
        
        legend = Legend()
        legend.x = 380
        legend.y = 150
        legend.alignment = 'right'
        legend.columnMaximum = 1
        legend.colorNamePairs = [
            (self.colors['primary'], 'Total Lines'),
            (self.colors['success'], 'Code Lines')
        ]
        drawing.add(legend)
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Issue Trends
        self._add_title("Issue Trends", "SubsectionTitle")
        
        drawing = Drawing(400, 200)
        chart = HorizontalLineChart()
        chart.x = 50
        chart.y = 50
        chart.height = 125
        chart.width = 300
        
        security_issues = [h['security_issues'] for h in history]
        architecture_issues = [h['architecture_issues'] for h in history]
        
        chart.data = [security_issues, architecture_issues]
        chart.lines[0].strokeColor = self.colors['danger']
        chart.lines[1].strokeColor = self.colors['warning']
        
        chart.categoryAxis.categoryNames = dates
        chart.categoryAxis.labels.boxAnchor = 'ne'
        chart.categoryAxis.labels.angle = 30
        
        legend = Legend()
        legend.x = 380
        legend.y = 150
        legend.alignment = 'right'
        legend.columnMaximum = 1
        legend.colorNamePairs = [
            (self.colors['danger'], 'Security Issues'),
            (self.colors['warning'], 'Architecture Issues')
        ]
        drawing.add(legend)
        
        drawing.add(chart)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Trend Analysis
        self._add_title("Trend Analysis", "SubsectionTitle")
        
        # Calculate trend indicators
        latest_idx = -1
        month_ago_idx = -4 if len(history) >= 4 else 0
        
        metrics_trends = {
            'Complexity': (
                avg_complexity[latest_idx] - avg_complexity[month_ago_idx],
                avg_complexity[latest_idx]
            ),
            'Code Size': (
                (total_lines[latest_idx] - total_lines[month_ago_idx]) / total_lines[month_ago_idx] * 100,
                total_lines[latest_idx]
            ),
            'Security Issues': (
                security_issues[latest_idx] - security_issues[month_ago_idx],
                security_issues[latest_idx]
            ),
            'Architecture Issues': (
                architecture_issues[latest_idx] - architecture_issues[month_ago_idx],
                architecture_issues[latest_idx]
            )
        }
        
        trends_data = [['Metric', 'Current Value', 'Change', 'Trend']]
        
        for metric, (change, current) in metrics_trends.items():
            if metric in ['Complexity', 'Security Issues', 'Architecture Issues']:
                trend = 'POSITIVE' if change <= 0 else 'NEGATIVE'
            else:  # Code Size
                trend = 'POSITIVE' if change > 0 else 'NEUTRAL'
                
            trends_data.append([
                metric,
                f"{current:.1f}",
                f"{change:+.1f}" + ("%" if metric == 'Code Size' else ""),
                trend
            ])
        
        # Apply trend colors
        trend_colors = {
            'POSITIVE': self.colors['success'],
            'NEGATIVE': self.colors['danger'],
            'NEUTRAL': self.colors['info']
        }
        
        table_style = self.alternating_table_style
        for i, row in enumerate(trends_data[1:], 1):
            trend = row[3]
            color = trend_colors.get(trend, self.colors['info'])
            table_style.add('TEXTCOLOR', (3, i), (3, i), color)
        
        self._add_table(trends_data, colWidths=[2*inch, 1.5*inch, inch, inch], style=table_style)
        
        # Trend Insights
        self._add_title("Trend Insights", "SubsectionTitle")
        
        insights = []
        
        # Generate insights based on trends
        if metrics_trends['Complexity'][0] > 0:
            insights.append(
                "• Code complexity is increasing - consider refactoring complex components"
            )
        else:
            insights.append(
                "• Code complexity is stable or decreasing - good maintenance practices"
            )
            
        if metrics_trends['Code Size'][0] > 10:
            insights.append(
                "• Codebase is growing rapidly - ensure documentation and testing keep pace"
            )
            
        if metrics_trends['Security Issues'][0] > 0:
            insights.append(
                "• Security issues are increasing - prioritize security fixes"
            )
            
        if metrics_trends['Architecture Issues'][0] > 0:
            insights.append(
                "• Architecture issues are increasing - review architectural decisions"
            )
            
        for insight in insights:
            self._add_paragraph(insight)

    def _generate_testing_section(self, code_metrics):
        """Generate testing and quality metrics section."""
        self._add_title("7. Testing & Quality Metrics", "SectionTitle")
        
        # Overview
        total_files = len(code_metrics)
        files_with_tests = sum(1 for m in code_metrics.values() if m.test_coverage_files)
        test_coverage_ratio = files_with_tests / total_files if total_files > 0 else 0
        
        overview = f"""
        Analysis of testing metrics across {total_files} files shows {files_with_tests} files 
        ({test_coverage_ratio*100:.1f}%) have associated tests. This section provides detailed insights 
        into test coverage, quality metrics, and testing patterns.
        """
        self._add_paragraph(overview)
        
        # Test Coverage Overview
        self._add_title("Test Coverage Analysis", "SubsectionTitle")
        
        coverage_data = [
            ['Metric', 'Value', 'Status'],
            ['Files with Tests', f"{files_with_tests}/{total_files}", 
            'HIGH' if test_coverage_ratio > 0.8 else 'MEDIUM' if test_coverage_ratio > 0.6 else 'LOW'],
            ['Coverage Ratio', f"{test_coverage_ratio*100:.1f}%",
            'HIGH' if test_coverage_ratio > 0.8 else 'MEDIUM' if test_coverage_ratio > 0.6 else 'LOW']
        ]
        
        # Apply status colors
        status_colors = {
            'HIGH': self.colors['success'],
            'MEDIUM': self.colors['warning'],
            'LOW': self.colors['danger']
        }
        
        table_style = self.table_style
        for i, row in enumerate(coverage_data[1:], 1):
            status = row[2]
            color = status_colors.get(status, self.colors['info'])
            table_style.add('TEXTCOLOR', (2, i), (2, i), color)
        
        self._add_table(coverage_data, colWidths=[2*inch, 1.5*inch, 1.5*inch], style=table_style)
        
        # Test Coverage Distribution
        self._add_title("Coverage Distribution", "SubsectionTitle")
        
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 25
        pie.width = 150
        pie.height = 150
        
        coverage_dist = {
            'No Tests': total_files - files_with_tests,
            'With Tests': files_with_tests
        }
        
        pie.data = list(coverage_dist.values())
        pie.labels = list(coverage_dist.keys())
        
        pie.slices[0].fillColor = self.colors['danger']
        pie.slices[1].fillColor = self.colors['success']
        
        drawing.add(pie)
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))
        
        # Files Lacking Tests
        if total_files - files_with_tests > 0:
            self._add_title("Critical Files Lacking Tests", "SubsectionTitle")
            
            untested_files = [
                (file_path, metrics)
                for file_path, metrics in code_metrics.items()
                if not metrics.test_coverage_files and 
                metrics.complexity.cyclomatic_complexity > 10
            ]
            
            if untested_files:
                untested_data = [['File', 'Complexity', 'Risk Level']]
                
                for file_path, metrics in sorted(
                    untested_files,
                    key=lambda x: x[1].complexity.cyclomatic_complexity,
                    reverse=True
                )[:10]:  # Show top 10 most complex untested files
                    complexity = metrics.complexity.cyclomatic_complexity
                    risk_level = 'HIGH' if complexity > 20 else 'MEDIUM' if complexity > 15 else 'LOW'
                    
                    untested_data.append([
                        file_path.split('/')[-1],
                        str(complexity),
                        risk_level
                    ])
                
                # Apply risk colors
                table_style = self.alternating_table_style
                for i, row in enumerate(untested_data[1:], 1):
                    risk = row[2]
                    color = status_colors.get(risk, self.colors['info'])
                    table_style.add('TEXTCOLOR', (2, i), (2, i), color)
                
                self._add_table(untested_data, colWidths=[3*inch, inch, inch], style=table_style)
        
        # Code Quality Patterns
        self._add_title("Code Quality Patterns", "SubsectionTitle")
        
        patterns_data = [['Pattern', 'Occurrences', 'Impact']]
        total_patterns = sum(
            sum(metrics.code_patterns.values())
            for metrics in code_metrics.values()
        )
        
        if total_patterns > 0:
            # Aggregate patterns across files
            all_patterns = {}
            for metrics in code_metrics.values():
                for pattern, count in metrics.code_patterns.items():
                    all_patterns[pattern] = all_patterns.get(pattern, 0) + count
            
            # Sort patterns by occurrence
            sorted_patterns = sorted(
                all_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for pattern, count in sorted_patterns:
                impact = 'HIGH' if count > total_files * 0.5 else \
                        'MEDIUM' if count > total_files * 0.2 else 'LOW'
                
                patterns_data.append([
                    pattern.replace('_', ' ').title(),
                    str(count),
                    impact
                ])
            
            # Apply impact colors
            table_style = self.alternating_table_style
            for i, row in enumerate(patterns_data[1:], 1):
                impact = row[2]
                color = status_colors.get(impact, self.colors['info'])
                table_style.add('TEXTCOLOR', (2, i), (2, i), color)
            
            self._add_table(patterns_data, colWidths=[2.5*inch, inch, 1.5*inch], style=table_style)
        
        # Testing Recommendations
        self._add_title("Testing Recommendations", "SubsectionTitle")
        
        recommendations = []
        
        if test_coverage_ratio < 0.8:
            recommendations.append(
                "• Increase overall test coverage - aim for at least 80% of files having tests"
            )
        
        # Check for complex untested files
        complex_untested = [
            metrics for metrics in code_metrics.values()
            if not metrics.test_coverage_files and 
            metrics.complexity.cyclomatic_complexity > 15
        ]
        if complex_untested:
            recommendations.append(
                "• Prioritize adding tests for complex files with high cyclomatic complexity"
            )
        
        # Check for frequently changing files without tests
        high_churn_untested = [
            metrics for metrics in code_metrics.values()
            if not metrics.test_coverage_files and 
            getattr(metrics.change_probability, 'churn_rate', 0) > 0.7
        ]
        if high_churn_untested:
            recommendations.append(
                "• Add tests for frequently modified files to prevent regression issues"
            )
        
        # Check for security-critical files without tests
        security_untested = [
            metrics for metrics in code_metrics.values()
            if not metrics.test_coverage_files and (
                metrics.security.potential_sql_injections > 0 or
                metrics.security.hardcoded_secrets > 0 or
                len(metrics.security.vulnerable_imports) > 0
            )
        ]
        if security_untested:
            recommendations.append(
                "• Add security-focused tests for files with potential vulnerabilities"
            )
        
        # Quality pattern recommendations
        if any(m.code_patterns.get('magic_numbers', 0) > 5 for m in code_metrics.values()):
            recommendations.append(
                "• Replace magic numbers with named constants to improve maintainability"
            )
        if any(m.code_patterns.get('large_try_blocks', 0) > 0 for m in code_metrics.values()):
            recommendations.append(
                "• Break down large try-catch blocks into smaller, more focused error handling"
            )
        if any(m.code_patterns.get('boolean_traps', 0) > 3 for m in code_metrics.values()):
            recommendations.append(
                "• Refactor methods with boolean parameters to improve clarity and testability"
            )
        
        if recommendations:
            for rec in recommendations:
                self._add_paragraph(rec)
        else:
            self._add_paragraph("No specific testing recommendations at this time.")