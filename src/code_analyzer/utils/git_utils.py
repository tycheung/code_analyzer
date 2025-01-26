import os
import shutil
import git
from typing import Dict, Set, Optional
from datetime import datetime
from collections import defaultdict

class GitAnalyzer:
    def __init__(self):
        self.repo = None
        self.change_history = defaultdict(lambda: {
            'frequency': 0,
            'last_modified': None,
            'contributors': set(),
            'churn_rate': 0.0
        })

    def clone_repository(self, repo_url: str, target_dir: str) -> str:
        """Clone a git repository to the target directory."""
        try:
            if not os.path.exists(target_dir):
                print(f"Cloning repository from {repo_url}...")
                git.Repo.clone_from(repo_url, target_dir)
            return target_dir
        except git.exc.GitCommandError as e:
            raise Exception(f"Failed to clone repository: {e}")

    def cleanup_repository(self, target_dir: str) -> None:
        """Remove the cloned repository directory."""
        try:
            if os.path.exists(target_dir):
                print(f"Cleaning up repository in {target_dir}...")
                shutil.rmtree(target_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up repository: {e}")

    def analyze_history(self, repo_path: str) -> Dict:
        """Analyze git history for change patterns and risks."""
        try:
            self.repo = git.Repo(repo_path)
            
            # Analyze commits
            for commit in self.repo.iter_commits():
                for file in commit.stats.files:
                    self.change_history[file]['frequency'] += 1
                    self.change_history[file]['last_modified'] = commit.committed_datetime
                    self.change_history[file]['contributors'].add(commit.author.name)
                    
                    # Calculate churn rate (changes per day)
                    changes = commit.stats.files[file]['lines']
                    self.change_history[file]['churn_rate'] = (
                        self.change_history[file].get('churn_rate', 0) + abs(changes)
                    )
            
            return dict(self.change_history)
            
        except Exception as e:
            print(f"Warning: Git history analysis failed: {e}")
            return {}

    def get_recent_contributors(self, days: int = 30) -> Set[str]:
        """Get contributors who made changes in the last N days."""
        contributors = set()
        if not self.repo:
            return contributors

        cutoff_date = datetime.now().timestamp() - (days * 86400)  # days to seconds
        for commit in self.repo.iter_commits():
            if commit.committed_date < cutoff_date:
                break
            contributors.add(commit.author.name)
        
        return contributors

    def get_file_blame_info(self, file_path: str) -> Optional[Dict]:
        """Get blame information for a specific file."""
        if not self.repo or not os.path.exists(file_path):
            return None

        try:
            blame = self.repo.blame('HEAD', file_path)
            blame_info = {
                'total_lines': 0,
                'contributors': defaultdict(int),
                'oldest_commit': None,
                'newest_commit': None
            }
            
            for commit, lines in blame:
                blame_info['total_lines'] += len(lines)
                blame_info['contributors'][commit.author.name] += len(lines)
                
                if (not blame_info['oldest_commit'] or 
                    commit.committed_datetime < blame_info['oldest_commit']):
                    blame_info['oldest_commit'] = commit.committed_datetime
                
                if (not blame_info['newest_commit'] or 
                    commit.committed_datetime > blame_info['newest_commit']):
                    blame_info['newest_commit'] = commit.committed_datetime
            
            return dict(blame_info)
            
        except Exception as e:
            print(f"Warning: Failed to get blame info for {file_path}: {e}")
            return None