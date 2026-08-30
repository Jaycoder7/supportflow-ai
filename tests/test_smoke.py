"""Fast checks for the initial project scaffold."""

from src.config import DEPARTMENTS, PROJECT_ROOT


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.is_dir()


def test_department_taxonomy_has_five_unique_labels() -> None:
    assert len(DEPARTMENTS) == 5
    assert len(set(DEPARTMENTS)) == 5

