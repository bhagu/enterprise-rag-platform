
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil


def test_directory_promotion_and_rollback():
    with TemporaryDirectory() as temp:
        root = Path(temp)

        active = root / "vector_store"
        staging = root / "vector_store_staging"
        backup = root / "vector_store_backup"

        # Simulate the existing index.
        active.mkdir()
        (active / "version.txt").write_text("old")

        # Simulate a successfully built staging directory.
        staging.mkdir()
        (staging / "version.txt").write_text("new")

        # Move old active directory to backup.
        active.rename(backup)

        # Promote staging.
        staging.rename(active)

        assert (active / "version.txt").read_text() == "new"
        assert (backup / "version.txt").read_text() == "old"

        # Simulate a failed promotion and restore the backup.
        failed = root / "vector_store_failed"
        active.rename(failed)
        backup.rename(active)

        assert (active / "version.txt").read_text() == "old"
        assert (failed / "version.txt").read_text() == "new"

        print("PASS: promotion and rollback preserved both versions")


if __name__ == "__main__":
    test_directory_promotion_and_rollback()
