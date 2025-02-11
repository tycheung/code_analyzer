import pytest
import os
import shutil
import git
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from collections import defaultdict
from code_analyzer.utils.git_utils import GitAnalyzer

@pytest.fixture
def git_analyzer():
    return GitAnalyzer()

@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

@pytest.fixture
def mock_repo():
    mock = Mock(spec=git.Repo)
    # Set up basic commit structure
    mock_stats = Mock()
    mock_stats.files = {'file1.py': {'lines': 10}}
    
    mock_commit = Mock(spec=git.Commit)
    mock_commit.stats = mock_stats
    mock_commit.committed_datetime = datetime.now()
    mock_commit.committed_date = datetime.now().timestamp()
    mock_commit.author.name = "Test Author"
    
    mock.iter_commits.return_value = [mock_commit]
    return mock

def test_clone_repository(git_analyzer, temp_dir):
    repo_url = "https://github.com/test/repo.git"
    
    print("\nDEBUG: Starting clone repository test")
    # Ensure directory doesn't exist
    if os.path.exists(temp_dir):
        print("DEBUG: Removing existing temp dir")
        shutil.rmtree(temp_dir)
    print(f"DEBUG: temp_dir exists after cleanup: {os.path.exists(temp_dir)}")
    
    with patch('git.repo.base.Repo.clone_from') as mock_clone:
        print("DEBUG: Mock set up complete")
        mock_clone.return_value = Mock()
        
        print("DEBUG: Calling clone_repository")
        result = git_analyzer.clone_repository(repo_url, temp_dir)
        
        print(f"DEBUG: Mock call count: {mock_clone.call_count}")
        print(f"DEBUG: Mock call args: {mock_clone.call_args_list}")
        print(f"DEBUG: Result directory exists: {os.path.exists(result)}")
        
        assert result == temp_dir
        mock_clone.assert_called_once_with(repo_url, temp_dir)

def test_clone_repository_error(git_analyzer, temp_dir):
    repo_url = "https://github.com/test/repo.git"
    
    print("\nDEBUG: Starting clone repository error test")
    # Ensure directory doesn't exist
    if os.path.exists(temp_dir):
        print("DEBUG: Removing existing temp dir")
        shutil.rmtree(temp_dir)
    print(f"DEBUG: temp_dir exists after cleanup: {os.path.exists(temp_dir)}")
    
    with patch('git.repo.base.Repo.clone_from') as mock_clone:
        print("DEBUG: Mock set up complete")
        mock_clone.side_effect = git.exc.GitCommandError('clone', 'error')
        
        with pytest.raises(Exception) as exc_info:
            result = git_analyzer.clone_repository(repo_url, temp_dir)
            print("DEBUG: No exception was raised!")
            print(f"DEBUG: Result was: {result}")
            
        print(f"DEBUG: Exception was raised as expected: {str(exc_info.value)}")
        print(f"DEBUG: Mock was called: {mock_clone.call_count} times")
        
        assert "Failed to clone repository" in str(exc_info.value)
        assert mock_clone.call_count == 1  # Verify mock was called

def test_cleanup_repository(git_analyzer, temp_dir):
    # Create a test file in temp directory
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, 'w') as f:
        f.write("test content")
    
    git_analyzer.cleanup_repository(temp_dir)
    assert not os.path.exists(temp_dir)

def test_cleanup_repository_nonexistent(git_analyzer):
    # Should not raise error for non-existent directory
    nonexistent_dir = "/path/does/not/exist"
    git_analyzer.cleanup_repository(nonexistent_dir)

def test_analyze_history(git_analyzer):
    with patch('git.Repo') as mock_repo_class:
        # Set up mock repo
        mock_stats = Mock()
        mock_stats.files = {'file1.py': {'lines': 10}}
        
        mock_commit = Mock()
        mock_commit.stats = mock_stats
        mock_commit.committed_datetime = datetime.now()
        mock_commit.author.name = "Test Author"
        
        mock_repo = mock_repo_class.return_value
        mock_repo.iter_commits.return_value = [mock_commit]
        
        # Run analysis
        history = git_analyzer.analyze_history("/fake/path")
        
        # Verify results
        assert "file1.py" in history
        assert history["file1.py"]["frequency"] == 1
        assert "Test Author" in history["file1.py"]["contributors"]
        assert isinstance(history["file1.py"]["churn_rate"], float)

def test_analyze_history_empty_repo(git_analyzer):
    with patch('git.Repo') as mock_repo_class:
        mock_repo = mock_repo_class.return_value
        mock_repo.iter_commits.return_value = []
        
        history = git_analyzer.analyze_history("/fake/path")
        assert history == {}

def test_get_recent_contributors(git_analyzer):
    with patch('code_analyzer.utils.git_utils.git.Repo') as mock_repo_class:
        now = datetime.now()
        
        # Create author objects with name attribute properly set
        recent_author = Mock()
        recent_author.name = "Recent Author"  # Set as attribute, not Mock
        old_author = Mock()
        old_author.name = "Old Author"  # Set as attribute, not Mock
        
        mock_commits = [
            Mock(
                committed_date=(now - timedelta(days=1)).timestamp(),
                author=recent_author
            ),
            Mock(
                committed_date=(now - timedelta(days=40)).timestamp(),
                author=old_author
            )
        ]
        
        mock_repo = mock_repo_class.return_value
        mock_repo.iter_commits.return_value = mock_commits
        git_analyzer.repo = mock_repo
        
        contributors = git_analyzer.get_recent_contributors(days=30)
        assert "Recent Author" in contributors

def test_get_recent_contributors_no_repo(git_analyzer):
    git_analyzer.repo = None
    contributors = git_analyzer.get_recent_contributors()
    assert contributors == set()

def test_get_file_blame_info(git_analyzer):
    with patch('code_analyzer.utils.git_utils.git.Repo') as mock_repo_class, \
         patch('os.path.exists', return_value=True):
        now = datetime.now()
        
        # Create author objects with name attribute properly set
        author1 = Mock()
        author1.name = "Author 1"  # Set as attribute, not Mock
        author2 = Mock()
        author2.name = "Author 2"  # Set as attribute, not Mock
        
        commit1 = Mock(
            author=author1,
            committed_datetime=now
        )
        commit2 = Mock(
            author=author2,
            committed_datetime=now - timedelta(days=1)
        )
        
        mock_blame = [
            (commit1, ["line1", "line2"]),
            (commit2, ["line3"])
        ]
        
        mock_repo = mock_repo_class.return_value
        mock_repo.blame.return_value = mock_blame
        git_analyzer.repo = mock_repo
        
        blame_info = git_analyzer.get_file_blame_info("test_file.py")
        
        assert blame_info is not None
        assert blame_info["total_lines"] == 3
        assert blame_info["contributors"]["Author 1"] == 2

def test_get_file_blame_info_nonexistent_file(git_analyzer):
    with patch('os.path.exists', return_value=False):
        git_analyzer.repo = Mock(spec=git.Repo)
        blame_info = git_analyzer.get_file_blame_info("nonexistent.py")
        assert blame_info is None

def test_get_file_blame_info_error(git_analyzer):
    with patch('os.path.exists', return_value=True):
        mock_repo = Mock(spec=git.Repo)
        mock_repo.blame.side_effect = Exception("Blame failed")
        git_analyzer.repo = mock_repo
        
        blame_info = git_analyzer.get_file_blame_info("test.py")
        assert blame_info is None