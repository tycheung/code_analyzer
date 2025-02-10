import pytest
from pathlib import Path
from code_analyzer.analyzers.duplication import CodeDuplication

@pytest.fixture
def duplication_checker():
    return CodeDuplication(min_lines=3)

@pytest.fixture
def test_files_dir(tmp_path):
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    return test_dir

def test_duplicate_detection(duplication_checker, test_files_dir):
    # Create files with duplicate content
    file1_content = [
        "def duplicate_function():",
        "    print('hello')",
        "    return True",
        "    # More code here"
    ]
    
    file2_content = [
        "def another_function():",
        "def duplicate_function():",  # Duplicate starts here
        "    print('hello')",
        "    return True",
        "    # Different content"
    ]
    
    file1 = test_files_dir / "file1.py"
    file2 = test_files_dir / "file2.py"
    
    file1.write_text("\n".join(file1_content))
    file2.write_text("\n".join(file2_content))
    
    duplication_checker.add_file(str(file1), file1_content)
    duplication_checker.add_file(str(file2), file2_content)
    
    duplicates = duplication_checker.find_duplicates()
    assert len(duplicates) > 0
    
    # Verify duplicate content
    file1_path, file2_path, duplicate_lines = duplicates[0]
    assert len(duplicate_lines) >= 3
    assert "print('hello')" in duplicate_lines
    assert "return True" in duplicate_lines

def test_hash_sequence(duplication_checker):
    sequence1 = ["def test():", "    return True"]
    sequence2 = ["def test():", "    return True"]
    sequence3 = ["def test():", "    return False"]
    
    hash1 = duplication_checker.hash_sequence(sequence1)
    hash2 = duplication_checker.hash_sequence(sequence2)
    hash3 = duplication_checker.hash_sequence(sequence3)
    
    assert hash1 == hash2  # Same content should have same hash
    assert hash1 != hash3  # Different content should have different hash

def test_minimum_lines(test_files_dir):
    # Test with different minimum line settings
    checker_small = CodeDuplication(min_lines=2)
    checker_large = CodeDuplication(min_lines=5)
    
    content = [
        "line 1",
        "line 2",
        "line 3",
        "line 4"
    ]
    
    file1 = test_files_dir / "test1.py"
    file2 = test_files_dir / "test2.py"
    file1.write_text("\n".join(content))
    file2.write_text("\n".join(content))
    
    # Small minimum should find duplicates
    checker_small.add_file(str(file1), content)
    checker_small.add_file(str(file2), content)
    assert len(checker_small.find_duplicates()) > 0
    
    # Large minimum should not find duplicates
    checker_large.add_file(str(file1), content)
    checker_large.add_file(str(file2), content)
    assert len(checker_large.find_duplicates()) == 0

def test_get_duplicate_stats(duplication_checker, test_files_dir):
    # Create files with known duplication
    file1_content = ["duplicate line 1", "duplicate line 2", "unique line"]
    file2_content = ["different line", "duplicate line 1", "duplicate line 2"]
    
    file1 = test_files_dir / "stats1.py"
    file2 = test_files_dir / "stats2.py"
    
    file1.write_text("\n".join(file1_content))
    file2.write_text("\n".join(file2_content))
    
    duplication_checker.add_file(str(file1), file1_content)
    duplication_checker.add_file(str(file2), file2_content)
    duplication_checker.find_duplicates()
    
    stats = duplication_checker.get_duplicate_stats()
    assert isinstance(stats, dict)
    assert 'total_duplicates' in stats
    assert 'files_with_duplicates' in stats
    assert 'total_duplicate_lines' in stats