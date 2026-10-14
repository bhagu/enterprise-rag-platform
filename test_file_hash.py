
import tempfile
from pathlib import Path

from scripts.run_ingestion import get_file_hash


with tempfile.TemporaryDirectory() as temp_dir:
    test_file = Path(temp_dir) / "sample.txt"

    test_file.write_text("Original content", encoding="utf-8")
    original_hash = get_file_hash(str(test_file))

    # Identical content should produce the same hash.
    same_hash = get_file_hash(str(test_file))
    assert original_hash == same_hash

    # Changed content should produce a different hash.
    test_file.write_text("Updated content", encoding="utf-8")
    updated_hash = get_file_hash(str(test_file))
    assert original_hash != updated_hash

    # Empty files should also be handled.
    test_file.write_bytes(b"")
    empty_hash = get_file_hash(str(test_file))
    assert len(empty_hash) == 64

print("PASS: identical content produces the same hash")
print("PASS: changed content produces a different hash")
print("PASS: empty files are handled")
