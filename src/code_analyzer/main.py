#!/usr/bin/env python3
import sys
import os
import argparse
import json
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from .analyzers import CodebaseAnalyzer
from .reporters import PDFReportGenerator, CodeMetricsPDFGenerator

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Code analysis for GitHub repositories'
    )
    
    parser.add_argument(
        'repo_url',
        help='URL of the GitHub repository'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./repo_clone',
        help='Directory to clone the repository into'
    )
    
    parser.add_argument(
        '--min-duplicate-lines',
        type=int,
        default=6,
        help='Minimum number of lines to consider as duplicate code'
    )
    
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export metrics to JSON file'
    )
    
    parser.add_argument(
        '--generate-pdf',
        action='store_true',
        help='Generate PDF report'
    )
    
    parser.add_argument(
        '--pdf-output',
        default='./report',
        help='Directory for PDF report output'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # New arguments for enhanced analysis
    parser.add_argument(
        '--include-test-analysis',
        action='store_true',
        default=True,
        help='Include test coverage analysis'
    )
    
    parser.add_argument(
        '--include-trends',
        action='store_true',
        default=True,
        help='Include historical trend analysis'
    )
    
    parser.add_argument(
        '--trend-window',
        default='6months',
        choices=['1month', '3months', '6months', '1year'],
        help='Time window for trend analysis'
    )
    
    parser.add_argument(
        '--trend-granularity',
        default='weekly',
        choices=['daily', 'weekly', 'monthly'],
        help='Data granularity for trend analysis'
    )
    
    return parser.parse_args()

def export_json_metrics(metrics: Dict[str, Any], output_file: str = 'metrics.json') -> None:
    """Export metrics to a JSON file."""
    try:
        # Convert metrics to JSON-serializable format
        json_metrics = {}
        for file_path, file_metrics in metrics.items():
            json_metrics[file_path] = {
                'lines': {
                    'code': file_metrics.lines_code,
                    'comment': file_metrics.lines_comment,
                    'blank': file_metrics.lines_blank
                },
                'complexity': {
                    'cyclomatic': file_metrics.complexity.cyclomatic_complexity,
                    'cognitive': file_metrics.complexity.cognitive_complexity,
                    'maintainability': file_metrics.complexity.maintainability_index,
                    'change_risk': file_metrics.complexity.change_risk,
                    'halstead_metrics': file_metrics.complexity.halstead_metrics
                },
                'security': {
                    'sql_injections': file_metrics.security.potential_sql_injections,
                    'hardcoded_secrets': file_metrics.security.hardcoded_secrets,
                    'unsafe_regex': file_metrics.security.unsafe_regex,
                    'vulnerable_imports': list(file_metrics.security.vulnerable_imports),
                    'insecure_patterns': file_metrics.security.insecure_patterns
                },
                'architecture': {
                    'interfaces': file_metrics.architecture.interface_count,
                    'abstract_classes': file_metrics.architecture.abstract_class_count,
                    'layering_violations': file_metrics.architecture.layering_violations,
                    'circular_dependencies': list(file_metrics.architecture.circular_dependencies),
                    'component_coupling': file_metrics.architecture.component_coupling,
                    'abstraction_level': file_metrics.architecture.abstraction_level
                },
                'test_coverage': {
                    'has_tests': bool(file_metrics.test_coverage_files),
                    'coverage_files': list(file_metrics.test_coverage_files)
                },
                'code_patterns': file_metrics.code_patterns
            }
        
        # Add summary metrics
        json_metrics['summary'] = {
            'total_files': len(metrics),
            'total_lines': sum(m.lines_code + m.lines_comment + m.lines_blank for m in metrics.values()),
            'avg_complexity': sum(m.complexity.cyclomatic_complexity for m in metrics.values()) / len(metrics) if metrics else 0,
            'security_issues': sum(
                m.security.potential_sql_injections + 
                m.security.hardcoded_secrets + 
                len(m.security.vulnerable_imports) 
                for m in metrics.values()
            ),
            'architecture_issues': sum(
                len(m.architecture.circular_dependencies) + 
                m.architecture.layering_violations 
                for m in metrics.values()
            ),
            'test_coverage': sum(1 for m in metrics.values() if m.test_coverage_files) / len(metrics) if metrics else 0
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2)
            
        logging.info(f"Metrics exported to {output_file}")
        
    except Exception as e:
        logging.error(f"Failed to export metrics to JSON: {e}")
        raise

def main() -> int:
    """Main entry point for the code analyzer."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Initialize analyzer
        analyzer = CodebaseAnalyzer()
        analyzer.code_duplication.min_lines = args.min_duplicate_lines
        
        # Clone and analyze repository
        repo_dir = analyzer.clone_repo(args.repo_url, args.output_dir)
        logging.info(f"Scanning repository in {repo_dir}...")
        
        # Analyze git history first
        analyzer.analyze_git_history(repo_dir)
        
        # Perform code analysis
        analyzer.scan_directory(repo_dir)
        analyzer.print_stats()
        
        # Export metrics if requested
        if args.export_json:
            export_json_metrics(analyzer.stats)
        
        # Generate PDF report if requested
        if args.generate_pdf:
            logging.info("Generating enhanced PDF report...")
            
            # Create output directory if it doesn't exist
            pdf_dir = Path(args.pdf_output)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare metrics data for PDF generation
            metrics_data = {
                'code_metrics': analyzer.stats,
                'test_coverage': analyzer.get_test_coverage() if args.include_test_analysis else None,
                'trends': analyzer.get_historical_trends() if args.include_trends else None,
                'deployment_analysis': None  # Only used in web interface
            }
            
            # Generate the PDF using the new generator
            pdf_generator = CodeMetricsPDFGenerator(
                str(pdf_dir / 'report.pdf')
            )
            
            try:
                pdf_generator.generate_pdf(metrics_data)
                logging.info(f"PDF report generated at {pdf_dir / 'report.pdf'}")
            except Exception as e:
                logging.error(f"Failed to generate PDF report: {e}")
                if args.verbose:
                    logging.exception("Detailed error information:")
                return 1
        
        # Cleanup
        analyzer.cleanup_repo(repo_dir)
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        if args.verbose:
            logging.exception("Detailed error information:")
        return 1

if __name__ == "__main__":
    sys.exit(main())