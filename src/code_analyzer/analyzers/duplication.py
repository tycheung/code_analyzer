import hashlib
from typing import List, Tuple, Dict
from collections import defaultdict

class CodeDuplication:
    def __init__(self, min_lines: int = 6):
        self.min_lines = min_lines
        self.hash_to_lines = defaultdict(list)
        self.duplicates = []

    def normalize_line(self, line: str, preserve_indent: bool = True) -> str:
        """Normalize a line of code by stripping whitespace and comments."""
        # Remove trailing comments
        line = line.split('#')[0].rstrip()
        if preserve_indent:
            # Normalize whitespace while preserving indentation
            return line
        else:
            # Remove all leading whitespace
            return line.lstrip()

    def hash_sequence(self, lines: List[str]) -> str:
        """Create a hash for a sequence of lines."""
        # Use preserved indentation for hashing to maintain structure
        normalized = '\n'.join(self.normalize_line(line, preserve_indent=True) 
                             for line in lines)
        return hashlib.md5(normalized.encode()).hexdigest()

    def add_file(self, filepath: str, lines: List[str]) -> None:
        """Process a file for duplicate code detection."""
        if not lines or len(lines) < self.min_lines:
            return

        for i in range(len(lines) - self.min_lines + 1):
            sequence = lines[i:i + self.min_lines]
            # Skip sequences that are all empty or whitespace
            if all(not line.strip() for line in sequence):
                continue
            
            seq_hash = self.hash_sequence(sequence)
            self.hash_to_lines[seq_hash].append((filepath, i, sequence))

    def extend_duplicate(self, file1: str, file2: str, start1: int, start2: int, 
                        lines1: List[str], lines2: List[str]) -> List[str]:
        """Extend the duplicate sequence as far as possible."""
        max_length = min(len(lines1) - start1, len(lines2) - start2)
        end = self.min_lines
        
        while end < max_length:
            line1 = self.normalize_line(lines1[start1 + end], preserve_indent=True)
            line2 = self.normalize_line(lines2[start2 + end], preserve_indent=True)
            if line1 != line2:
                break
            end += 1
            
        # Return normalized lines without indentation for comparison
        return [self.normalize_line(line, preserve_indent=False) 
                for line in lines1[start1:start1 + end]]

    def find_duplicates(self) -> List[Tuple[str, str, List[str]]]:
        """Identify duplicate code sequences."""
        self.duplicates = []
        seen_pairs = set()

        for locations in self.hash_to_lines.values():
            if len(locations) > 1:
                for i in range(len(locations)):
                    for j in range(i + 1, len(locations)):
                        file1, start1, seq1 = locations[i]
                        file2, start2, seq2 = locations[j]
                        
                        if file1 != file2:
                            pair_key = tuple(sorted([file1, file2]))
                            if pair_key in seen_pairs:
                                continue
                            
                            extended_sequence = self.extend_duplicate(
                                file1, file2, start1, start2, seq1, seq2
                            )
                            
                            if len(extended_sequence) >= self.min_lines:
                                self.duplicates.append((file1, file2, extended_sequence))
                                seen_pairs.add(pair_key)

        return self.duplicates

    def get_duplicate_stats(self) -> Dict[str, int]:
        """Return statistics about code duplication."""
        if not self.duplicates:
            self.find_duplicates()

        return {
            'total_duplicates': len(self.duplicates),
            'files_with_duplicates': len(set(
                file for dup in self.duplicates
                for file in [dup[0], dup[1]]
            )),
            'total_duplicate_lines': sum(len(dup[2]) for dup in self.duplicates)
        }