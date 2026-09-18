"""Session 13: the CSV batch importer that replaced the "Raise Block
Request" UI form -- must apply exactly the same validation as
POST /requests (they share railblock.api.request_builder.build_request_row)
and must never silently skip or guess at an invalid row.

Session 32: imported rows land in the shared, unscoped demo_task_templates
table (store.add_template_task/template_tasks), not any particular
visitor's own `requests` -- see store.py's module docstring and
import_tasks.py's own Session 32 note for why."""

import csv

import pytest

from railblock.api.store import get_store, reset_store_for_tests
from railblock.integrations.import_tasks import check_for_new_rows, import_csv
from railblock.paths import TRAIN_DETAILS_CSV

pytestmark = pytest.mark.skipif(not TRAIN_DETAILS_CSV.exists(), reason="real dataset not present")

VALID_ROW = {
    "department": "Signalling",
    "section_id": "MAS-BBQ",
    "defect_type": "Signal lamp failure",
    "requester_priority": "Critical",
    "estimated_block_hours": "0.75",
    "due_date": "2026-12-31",
    "raised_date": "",
}


@pytest.fixture(autouse=True)
def store():
    reset_store_for_tests()
    return get_store()


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(VALID_ROW.keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_valid_row_imports_with_file_imported_source(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [VALID_ROW])

    imported, failed = import_csv(str(path))

    assert imported == 1
    assert failed == 0
    rows = store.template_tasks()
    assert len(rows) == 1
    assert rows[0]["data_source"] == "FILE_IMPORTED"
    assert rows[0]["status"] == "pending"
    assert rows[0]["department"] == "Signalling"


def test_invalid_department_rejected_not_skipped(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [{**VALID_ROW, "department": "Catering"}])

    imported, failed = import_csv(str(path))

    assert imported == 0
    assert failed == 1
    assert store.template_tasks() == []


def test_invalid_defect_type_for_department_rejected(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [{**VALID_ROW, "department": "Engineering", "defect_type": "Signal lamp failure"}])

    imported, failed = import_csv(str(path))

    assert imported == 0
    assert failed == 1


def test_invalid_section_id_rejected(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [{**VALID_ROW, "section_id": "NOT-A-REAL-SECTION"}])

    imported, failed = import_csv(str(path))

    assert imported == 0
    assert failed == 1


def test_bad_date_format_rejected(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [{**VALID_ROW, "due_date": "2026/12/31"}])  # neither YYYY-MM-DD nor DD-MM-YYYY

    imported, failed = import_csv(str(path))

    assert imported == 0
    assert failed == 1


def test_mixed_valid_and_invalid_rows_partial_import(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [VALID_ROW, {**VALID_ROW, "department": "Catering"}, VALID_ROW])

    imported, failed = import_csv(str(path))

    assert imported == 2
    assert failed == 1
    assert len(store.template_tasks()) == 2


def test_raised_date_defaults_when_blank(tmp_path, store):
    from datetime import date

    path = tmp_path / "tasks.csv"
    _write_csv(path, [VALID_ROW])

    import_csv(str(path))

    row = store.template_tasks()[0]
    assert row["raised_date"] == date.today().isoformat()


def test_each_imported_row_gets_a_unique_task_id(tmp_path, store):
    path = tmp_path / "tasks.csv"
    _write_csv(path, [VALID_ROW, VALID_ROW])

    import_csv(str(path))

    task_ids = [r["task_id"] for r in store.template_tasks()]
    assert len(set(task_ids)) == 2


def test_ddmmyyyy_date_format_also_accepted(tmp_path, store):
    """A real user actually entered dates as DD-MM-YYYY, not the
    documented YYYY-MM-DD -- both must work, not just the documented one."""
    path = tmp_path / "tasks.csv"
    _write_csv(path, [{**VALID_ROW, "due_date": "31-12-2026", "raised_date": "06-09-2026"}])

    imported, failed = import_csv(str(path))

    assert imported == 1
    assert failed == 0
    row = store.template_tasks()[0]
    assert row["due_date"] == "2026-12-31"
    assert row["raised_date"] == "2026-09-06"


def test_check_for_new_rows_only_imports_rows_past_the_saved_state(tmp_path, store, monkeypatch):
    """The live-watch entry point: appending a row to the SAME file and
    calling this again must import only the new row, never re-import
    everything already seen -- otherwise every edit would duplicate the
    whole batch."""
    import railblock.integrations.import_tasks as import_tasks_mod

    path = tmp_path / "tasks.csv"
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(import_tasks_mod, "TASKS_CSV_IMPORT_STATE_JSON", state_path)

    _write_csv(path, [VALID_ROW, VALID_ROW])
    imported, failed = check_for_new_rows(path)
    assert (imported, failed) == (2, 0)
    assert len(store.template_tasks()) == 2

    # Calling again with no changes must import nothing new.
    imported, failed = check_for_new_rows(path)
    assert (imported, failed) == (0, 0)
    assert len(store.template_tasks()) == 2

    # Append one more row (simulating the user editing and saving the file).
    _write_csv(path, [VALID_ROW, VALID_ROW, VALID_ROW])
    imported, failed = check_for_new_rows(path)
    assert (imported, failed) == (1, 0)
    assert len(store.template_tasks()) == 3


def test_check_for_new_rows_missing_file_is_a_noop(tmp_path):
    result = check_for_new_rows(tmp_path / "does_not_exist.csv")
    assert result == (0, 0)
