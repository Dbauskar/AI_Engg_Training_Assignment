import pytest

from app.harness import repair_amount
from app.pii import PiiDetector
from evals import redteam as R
from evals import redteam_set as S


def test_set_sizes_are_as_specified():
    assert sum(S.counts()[k] for k in ("direct", "indirect", "crescendo", "pii", "overflow", "sqli")) == 100 and S.counts()["benign"] == 100


@pytest.mark.parametrize("url", ["http://example.com", "http://10.0.0.5:8080", "https://api.prod.internal"])
def test_non_loopback_targets_are_refused(url):
    with pytest.raises(SystemExit):
        R.require_loopback(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080"])
def test_loopback_targets_are_allowed(url):
    R.require_loopback(url)


def test_forty_digit_amount_does_not_crash_auto_repair():  # found by the red team: used to raise decimal.InvalidOperation
    assert repair_amount("9" * 40) == "9" * 40  # left unrepaired; the contract regex then rejects it


async def test_tool_argument_attacks_are_all_rejected_before_any_handler_runs(spans):
    t = R.Target(PiiDetector(use_ner=False))
    rows = await R._tool_attacks(t, S.OVERFLOW, "overflow") + await R._tool_attacks(t, S.SQLI, "sqli")
    assert all(r["contained"] for r in rows), [r["id"] for r in rows if not r["contained"]]
    assert t.reached == {}


async def test_the_runner_emits_a_span_per_category_and_gates_are_evaluated(spans):
    t = R.Target(PiiDetector(use_ner=False))
    res = await R.run_inproc(t)
    assert [s.attributes["redteam.category"] for s in spans.get_finished_spans() if s.name == "redteam.category"] == list(res)
    summary = R.summarise(res)
    assert {g["gate"][:6] for g in summary["gates"]} == {"KPI-03", "KPI-06"} and summary["overall_attacks"]["n"] == 100
