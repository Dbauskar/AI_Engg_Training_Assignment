import json

from app import kpi
from scripts import kpi_gate


def test_spec_has_36_unique_kpis_in_order():
    ids = [s[0] for s in kpi.SPEC]
    assert ids == [f"KPI-{i:02d}" for i in range(1, 37)]


def test_missing_results_are_not_measured_never_pass(tmp_path):
    rows = kpi.evaluate(tmp_path)
    assert len(rows) == 36 and {r.status for r in rows} == {"NOT MEASURED"}


def _write(tmp_path, load):
    d = tmp_path / "evals" / "results"
    d.mkdir(parents=True)
    (d / "load_k6.json").write_text(json.dumps({"kpis": load}))


def test_load_kpis_only_pass_at_full_scale(tmp_path):
    _write(tmp_path, {"KPI-31": {"value": 100, "scale": "500 VUs (local)"}, "KPI-32": {"value": 100, "scale": "50000 VUs (cluster)"},
                      "KPI-33": {"value": 9999, "scale": "50000 VUs (cluster)"}})
    by = {r.kpi: r.status for r in kpi.evaluate(tmp_path)}
    assert by["KPI-31"] == "UNVERIFIED (reduced scale)"   # meets threshold, but not at the target scale
    assert by["KPI-32"] == "PASS"
    assert by["KPI-33"] == "FAIL"


def test_reduced_scale_pass_is_labelled(tmp_path):
    d = tmp_path / "evals" / "results"
    d.mkdir(parents=True)
    (d / "kpi_extra.json").write_text(json.dumps({"kpis": {"KPI-01": {"value": 5, "scale": "local"}}}))
    assert {r.kpi: r.status for r in kpi.evaluate(tmp_path)}["KPI-01"] == "PASS (reduced scale)"


def test_gate_exits_nonzero_on_current_repo_and_prints_36_rows(capsys):
    assert kpi_gate.main([]) == 1
    out = capsys.readouterr().out
    assert sum(1 for line in out.splitlines() if line.startswith("KPI-")) == 36
