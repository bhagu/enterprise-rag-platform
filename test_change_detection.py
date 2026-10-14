
import tempfile
from pathlib import Path

from scripts.run_ingestion import detect_changes, get_file_hash


with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)

    doc1 = root / "doc1.pdf"
    doc2 = root / "doc2.pdf"

    doc1.write_bytes(b"Document one - original content")
    doc2.write_bytes(b"Document two - new content")

    path1 = str(doc1)
    path2 = str(doc2)

    # 1. An unchanged document should not trigger a rebuild.
    manifest = {path1: get_file_hash(path1)}
    changed, deleted = detect_changes(
        [{"path": path1}], manifest
    )
    assert changed is False
    assert not deleted
    print("PASS: unchanged document is not flagged")

    # 2. Modified content should be detected.
    doc1.write_bytes(b"Document one - modified content")
    changed, deleted = detect_changes(
        [{"path": path1}], manifest
    )
    assert changed is True
    print("PASS: modified document is detected")

    # 3. A new document should be detected.
    changed, deleted = detect_changes(
        [{"path": path2}], {}
    )
    assert changed is True
    print("PASS: new document is detected")

    # 4. A legacy timestamp entry should be flagged for migration.
    changed, deleted = detect_changes(
        [{"path": path1}], {path1: 1720000000.0}
    )
    assert changed is True
    print("PASS: legacy timestamp entry is flagged")


print("\nAll change-detection tests passed.")
