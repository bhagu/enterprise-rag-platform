
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_ingestion


def create_staging_files(staging_path: Path) -> None:
    staging_path.mkdir(parents=True)

    # Placeholder files are sufficient because FAISS loading is mocked.
    for filename in (
        "faiss.index",
        "chunks.pkl",
        "processed_files.json",
    ):
        (staging_path / filename).write_text("test placeholder")

    (staging_path / "version.txt").write_text("new")


def fake_valid_index():
    return SimpleNamespace(
        index=SimpleNamespace(ntotal=1),
        chunks=[{"text": "test chunk"}],
    )


def test_successful_promotion():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        active = root / "vector_store"
        staging = root / "vector_store_staging"
        backup = root / "vector_store_backup"

        active.mkdir()
        (active / "version.txt").write_text("old")
        create_staging_files(staging)

        with patch.object(
            run_ingestion.FAISSVectorStore,
            "load",
            side_effect=[fake_valid_index(), fake_valid_index()],
        ):
            run_ingestion.promote_staged_index(
                active, staging, backup
            )

        assert (active / "version.txt").read_text() == "new"
        assert (backup / "version.txt").read_text() == "old"

        print("PASS: successful promotion preserves the old backup")


def test_failed_promotion_restores_backup():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        active = root / "vector_store"
        staging = root / "vector_store_staging"
        backup = root / "vector_store_backup"

        active.mkdir()
        (active / "version.txt").write_text("old")
        create_staging_files(staging)

        with patch.object(
            run_ingestion.FAISSVectorStore,
            "load",
            side_effect=[
                fake_valid_index(),
                RuntimeError("simulated verification failure"),
            ],
        ):
            try:
                run_ingestion.promote_staged_index(
                    active, staging, backup
                )
            except RuntimeError as exc:
                assert "simulated verification failure" in str(exc)
            else:
                raise AssertionError("Expected promotion to fail")

        # The original active index should have been restored.
        assert (active / "version.txt").read_text() == "old"

        # The failed promoted directory should be preserved for diagnosis.
        failed_dirs = list(root.glob("vector_store_failed_*"))
        assert len(failed_dirs) == 1
        assert (failed_dirs[0] / "version.txt").read_text() == "new"

        print("PASS: failed promotion restores the original index")
        print("PASS: failed promotion is preserved for inspection")


if __name__ == "__main__":
    test_successful_promotion()
    test_failed_promotion_restores_backup()
    print("\nAll promotion tests passed.")
