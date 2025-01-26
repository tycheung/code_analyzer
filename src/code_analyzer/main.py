#!/usr/bin/env python3
import sys
import os
import argparse
import json
import logging
from typing import Dict, Any
from pathlib import Path

from .analyzers import CodebaseAnalyzer
from .reporters import PDFReportGenerator

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
                    'maintainability': file_metrics.complexity.maintainability_index,
                    'change_risk': file_metrics.complexity.change_risk
                },
                'security': {
                    'sql_injections': file_metrics.security.potential_sql_injections,
                    'hardcoded_secrets': file_metrics.security.hardcoded_secrets,
                    'unsafe_regex': file_metrics.security.unsafe_regex,
                    'vulnerable_imports': list(file_metrics.security.vulnerable_imports)
                },
                'architecture': {
                    'interfaces': file_metrics.architecture.interface_count,
                    'abstract_classes': file_metrics.architecture.abstract_class_count,
                    'layering_violations': file_metrics.architecture.layering_violations
                }
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
            logging.info("Generating PDF report...")
            report_generator = PDFReportGenerator()
            try:
                report_generator.generate_pdf(
                    analyzer.stats,
                    args.repo_url,
                    args.pdf_output
                )
            except Exception as e:
                logging.error(f"Failed to generate PDF report: {e}")
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