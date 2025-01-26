from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile
import logging
import os
import json
import uvicorn
from pathlib import Path

from ..analyzers import CodebaseAnalyzer
from ..reporters import PDFReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Code Quality Analyzer",
    description="A comprehensive tool for analyzing code quality and complexity",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class ComparisonOptions(BaseModel):
    enabled: bool = False
    type: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)

class AnalysisOptions(BaseModel):
    verbose: bool = False
    generatePdf: bool = False
    exportJson: bool = False
    minLines: Optional[int] = None
    comparison: ComparisonOptions = Field(default_factory=ComparisonOptions)

class AnalysisRequest(BaseModel):
    repoUrl: str
    options: AnalysisOptions

class ComparisonAnalyzer:
    """Handles different types of comparative analysis."""

    @staticmethod
    async def compare_directories(analyzer: CodebaseAnalyzer, dir1: str, dir2: str, 
                                metrics_to_compare: List[str]) -> Dict:
        """Compare metrics between two directories."""
        dir1_metrics = await analyzer.analyze_directory(dir1)
        dir2_metrics = await analyzer.analyze_directory(dir2)
        
        comparison = {
            'dir1': {'path': dir1, 'metrics': dir1_metrics},
            'dir2': {'path': dir2, 'metrics': dir2_metrics},
            'differences': {}
        }

        for metric in metrics_to_compare:
            if metric == 'complexity':
                comparison['differences']['complexity'] = {
                    'cyclomatic': dir2_metrics.complexity.cyclomatic_complexity - dir1_metrics.complexity.cyclomatic_complexity,
                    'cognitive': dir2_metrics.complexity.cognitive_complexity - dir1_metrics.complexity.cognitive_complexity,
                    'maintainability': dir2_metrics.complexity.maintainability_index - dir1_metrics.complexity.maintainability_index
                }
            elif metric == 'security':
                comparison['differences']['security'] = {
                    'vulnerabilities': dir2_metrics.security.total_vulnerabilities - dir1_metrics.security.total_vulnerabilities,
                    'severity_score': dir2_metrics.security.severity_score - dir1_metrics.security.severity_score
                }

        return comparison

    @staticmethod
    async def compare_snapshots(analyzer: CodebaseAnalyzer, repo_dir: str, 
                              baseline: str, target: str, component_path: str = None) -> Dict:
        """Compare metrics between two Git references."""
        current_branch = analyzer.repo.active_branch.name
        
        comparison = {
            'baseline': {'ref': baseline},
            'target': {'ref': target},
            'differences': {}
        }

        try:
            analyzer.repo.git.checkout(baseline)
            baseline_metrics = await analyzer.analyze_directory(
                os.path.join(repo_dir, component_path) if component_path else repo_dir
            )
            comparison['baseline']['metrics'] = baseline_metrics

            analyzer.repo.git.checkout(target)
            target_metrics = await analyzer.analyze_directory(
                os.path.join(repo_dir, component_path) if component_path else repo_dir
            )
            comparison['target']['metrics'] = target_metrics

            comparison['differences'] = await analyzer.calculate_metric_differences(
                baseline_metrics, target_metrics
            )

        finally:
            analyzer.repo.git.checkout(current_branch)

        return comparison

    @staticmethod
    async def analyze_historical_trends(analyzer: CodebaseAnalyzer, repo_dir: str, 
                                      start_date: str, end_date: str, interval: str) -> Dict:
        """Analyze metrics over time."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        commits = list(analyzer.repo.iter_commits(
            since=start.isoformat(),
            until=end.isoformat()
        ))

        trends = {
            'timeline': [],
            'metrics': {
                'complexity': [],
                'security': [],
                'maintainability': []
            }
        }

        for commit in commits:
            analyzer.repo.git.checkout(commit.hexsha)
            metrics = await analyzer.analyze_directory(repo_dir)
            
            trends['timeline'].append(commit.committed_datetime.isoformat())
            trends['metrics']['complexity'].append(metrics.complexity.cyclomatic_complexity)
            trends['metrics']['security'].append(metrics.security.total_vulnerabilities)
            trends['metrics']['maintainability'].append(metrics.complexity.maintainability_index)

        return trends

async def analyze_repository(request_data: AnalysisRequest, temp_dir: str) -> Dict[str, Any]:
    """Analyze the repository and return results."""
    log_file = os.path.join(temp_dir, 'analysis.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    try:
        logger.setLevel(logging.DEBUG if request_data.options.verbose else logging.INFO)
        analyzer = CodebaseAnalyzer()
        
        repo_dir = os.path.join(temp_dir, "repo")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting analysis of {request_data.repoUrl}")
        logger.debug(f"Options: {request_data.options.model_dump_json(indent=2)}")

        await analyzer.clone_repo(request_data.repoUrl, repo_dir)
        await analyzer.analyze_git_history(repo_dir)
        
        if request_data.options.minLines:
            analyzer.code_duplication.min_lines = request_data.options.minLines
        
        comparison_results = None
        if request_data.options.comparison.enabled:
            comparison_type = request_data.options.comparison.type
            comparison_options = request_data.options.comparison.options
            metrics_to_compare = request_data.options.comparison.metrics
            
            if comparison_type == 'directory':
                comparison_results = await ComparisonAnalyzer.compare_directories(
                    analyzer,
                    os.path.join(repo_dir, comparison_options['dir1']),
                    os.path.join(repo_dir, comparison_options['dir2']),
                    metrics_to_compare
                )
            elif comparison_type == 'snapshot':
                comparison_results = await ComparisonAnalyzer.compare_snapshots(
                    analyzer,
                    repo_dir,
                    comparison_options['baseline'],
                    comparison_options['target'],
                    comparison_options.get('component')
                )
            elif comparison_type == 'historical':
                comparison_results = await ComparisonAnalyzer.analyze_historical_trends(
                    analyzer,
                    repo_dir,
                    comparison_options['startDate'],
                    comparison_options['endDate'],
                    comparison_options['interval']
                )
        
        await analyzer.scan_directory(repo_dir)
        
        results = {
            'success': True,
            'outputDir': output_dir,
            'files': await analyzer.get_metrics()
        }

        if comparison_results:
            results['comparison'] = comparison_results
            comparison_report_path = os.path.join(output_dir, 'comparison_report.json')
            with open(comparison_report_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            results['comparisonReportPath'] = comparison_report_path

        if request_data.options.generatePdf:
            pdf_dir = os.path.join(output_dir, "pdf")
            os.makedirs(pdf_dir, exist_ok=True)
            
            logger.info("Generating PDF report")
            report_generator = PDFReportGenerator()
            await report_generator.generate_pdf(
                analyzer.stats,
                request_data.repoUrl,
                pdf_dir,
                comparison_results
            )
            
            pdf_path = os.path.join(pdf_dir, "report.pdf")
            if os.path.exists(pdf_path):
                results['pdfPath'] = pdf_path
                logger.info(f"PDF report generated: {pdf_path}")

        if request_data.options.exportJson:
            json_path = os.path.join(output_dir, "metrics.json")
            metrics = await analyzer.get_metrics()
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            results['jsonPath'] = json_path
            logger.info(f"JSON metrics exported: {json_path}")

        results['logPath'] = log_file
        logger.info("Analysis completed successfully")
        return results

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.removeHandler(file_handler)

@app.post("/analyze")
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Handle repository analysis request."""
    temp_dir = tempfile.mkdtemp(prefix='code_analyzer_')
    try:
        results = await analyze_repository(request, temp_dir)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Handle file download requests."""
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(
            file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
    raise HTTPException(status_code=404, detail="File not found")

def run_server(port: int = 8000, open_browser: bool = True):
    """Run the FastAPI server."""
    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_server()