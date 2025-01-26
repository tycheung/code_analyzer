#!/usr/bin/env python3
"""
Example script demonstrating how to use the code analyzer.
"""
import os
import sys
from code_analyzer.analyzers import CodebaseAnalyzer
from code_analyzer.reporters import PDFReportGenerator

def analyze_repository(repo_url: str, output_dir: str = "./output"):
    """
    Analyze a GitHub repository and generate reports.
    
    Args:
        repo_url: URL of the GitHub repository
        output_dir: Directory for output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    repo_dir = os.path.join(output_dir, "repo")
    
    try:
        # Initialize analyzer
        analyzer = CodebaseAnalyzer()
        
        # Clone and analyze repository
        analyzer.clone_repo(repo_url, repo_dir)
        print(f"Analyzing repository: {repo_url}")
        
        # Analyze git history
        analyzer.analyze_git_history(repo_dir)
        
        # Perform code analysis
        analyzer.scan_directory(repo_dir)
        
        # Print analysis results
        analyzer.print_stats()
        
        # Generate PDF report
        print("\nGenerating PDF report...")
        report_dir = os.path.join(output_dir, "report")
        report_generator = PDFReportGenerator()
        report_generator.generate_pdf(
            stats=analyzer.stats,
            repo_url=repo_url,
            output_dir=report_dir
        )
        
        print(f"\nAnalysis complete! Check {report_dir} for the PDF report.")
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1
    finally:
        # Cleanup
        analyzer.cleanup_repo(repo_dir)
    
    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <github_repo_url> [output_dir]")
        print("\nExample:")
        print("  python example_usage.py https://github.com/username/repo ./analysis_output")
        return 1
    
    repo_url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    return analyze_repository(repo_url, output_dir)

if __name__ == "__main__":
    sys.exit(main())