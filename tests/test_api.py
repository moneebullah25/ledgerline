import base64
import os
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session
from test_report_generate import seed_report

from ledgerline.api import main as api
from ledgerline.api.main import app, get_session
from ledgerline.config import settings
from ledgerline.models import Claim, Company, EvalRun, Report, Verdict

KEY = "test-key"


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setenv("LEDGERLINE_API_KEY", KEY)
    settings.cache_clear()
    app.dependency_overrides[get_session] = lambda: db
    yield TestClient(app, headers={"X-API-Key": KEY})
    app.dependency_overrides.clear()
    settings.cache_clear()


def _is_422_list(r) -> bool:
    return r.status_code == 422 and isinstance(r.json()["detail"], list)


@pytest.mark.db
def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.db
def test_post_enqueues_without_running(client, db):
    seed_report(db, n_claims=0)
    r = client.post(
        "/v1/reports", json={"ticker": "aapl", "fiscal_year": 2025, "fiscal_period": "Q4"}
    )
    assert r.status_code == 202
    report = db.get(Report, uuid.UUID(r.json()["job_id"]))
    assert report.status == "queued"
    assert r.headers["Location"] == f"/v1/reports/{report.id}"
    assert client.get(f"/v1/reports/{report.id}").json()["claims"] == []


@pytest.mark.db
def test_post_dedupes_per_quarter_and_reruns_only_failed(client, db):
    done = seed_report(db, n_claims=0)  # AAPL FY2025 Q3, status done
    body = {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q3"}
    assert client.post("/v1/reports", json=body).json()["job_id"] == str(done.id)
    done.status = "failed"
    db.flush()
    again = client.post("/v1/reports", json=body).json()["job_id"]
    assert again != str(done.id)
    assert client.post("/v1/reports", json=body).json()["job_id"] == again  # now queued


@pytest.mark.db
def test_post_rerun_recomputes_a_done_report(client, db):
    done = seed_report(db, n_claims=0)
    body = {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q3", "rerun": True}
    fresh = client.post("/v1/reports", json=body).json()["job_id"]
    assert fresh != str(done.id)
    assert client.post("/v1/reports", json=body).json()["job_id"] == fresh  # queued: reused
    client.headers.pop("X-API-Key")
    assert client.post("/v1/reports", json=body).status_code == 401  # still key-gated


@pytest.mark.db
def test_post_caps_the_queue(client, db, monkeypatch):
    seed_report(db, n_claims=0)
    monkeypatch.setattr(settings(), "max_pending_reports", 1)
    ok = client.post(
        "/v1/reports", json={"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q1"}
    )
    assert ok.status_code == 202
    r = client.post(
        "/v1/reports", json={"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q2"}
    )
    assert r.status_code == 429 and r.headers["Retry-After"]


@pytest.mark.db
def test_post_caps_pending_per_key(client, db, monkeypatch):
    seed_report(db, n_claims=0)
    monkeypatch.setattr(settings(), "max_pending_reports", 1)
    # One queued report now exists (status=done from seed_report — add a queued one).
    db.add(
        Report(
            company_id=db.query(Company.id).scalar(),
            fiscal_year=2025,
            fiscal_period="Q1",
            status="queued",
        )
    )
    db.flush()
    r = client.post(
        "/v1/reports", json={"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q2"}
    )
    assert r.status_code == 429
    assert r.json()["detail"] == "too many pending reports"
    assert r.headers["Retry-After"]


@pytest.mark.db
def test_post_requires_api_key(client, db, monkeypatch):
    seed_report(db, n_claims=0)
    body = {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q1"}
    assert client.post("/v1/reports", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/reports", json=body, headers={"X-API-Key": ""}).status_code == 401
    monkeypatch.delenv("LEDGERLINE_API_KEY")
    monkeypatch.setattr(
        settings(), "api_key", None
    )  # a key in the developer's .env must not leak in
    assert client.post("/v1/reports", json=body).status_code == 503  # fail closed
    assert client.get("/v1/reports").status_code == 200  # reads stay public


@pytest.mark.db
def test_post_rejects_fy_with_a_reason(client, db):
    seed_report(db, n_claims=0)
    r = client.post(
        "/v1/reports", json={"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "FY"}
    )
    assert _is_422_list(r) and "request Q4" in r.json()["detail"][0]["msg"]


@pytest.mark.db
@pytest.mark.parametrize(
    "body, code",
    [
        ({"ticker": "NOPE", "fiscal_year": 2025, "fiscal_period": "Q1"}, 404),
        ({"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q5"}, 422),
        (
            {"ticker": "AAPL'; DROP TABLE report;--", "fiscal_year": 2025, "fiscal_period": "Q1"},
            422,
        ),
        ({"ticker": "AAPL", "fiscal_period": "Q1"}, 422),
    ],
)
def test_post_validation(client, body, code):
    assert client.post("/v1/reports", json=body).status_code == code


@pytest.mark.db
def test_get_report_ranked(client, db):
    report = seed_report(db)
    body = client.get(f"/v1/reports/{report.id}").json()
    assert body["status"] == "done" and body["ticker"] == "AAPL"
    assert [c["label"] for c in body["claims"]] == ["contradicted", "supported", "supported"]
    assert body["claims"][0]["sources"][0]["kind"] == "xbrl_fact"
    assert client.get(f"/v1/reports/{uuid.uuid4()}").status_code == 404
    assert client.get("/v1/reports/not-a-uuid").status_code == 422


@pytest.mark.db
def test_list_reports_keyset_pagination(client, db):
    base = seed_report(db, n_claims=0)
    for fp in ("Q1", "Q2", "Q4"):
        db.add(Report(company_id=base.company_id, fiscal_year=2025, fiscal_period=fp))
    db.flush()
    seen, cursor = [], None
    while True:
        page = client.get(
            "/v1/reports", params={"limit": 3, **({"cursor": cursor} if cursor else {})}
        ).json()
        seen += [r["id"] for r in page["items"]]
        if not (cursor := page["next_cursor"]):
            break
    assert len(seen) == len(set(seen)) == 4
    assert client.get("/v1/reports", params={"limit": 101}).status_code == 422
    assert _is_422_list(client.get("/v1/reports", params={"cursor": "garbage"}))
    assert _is_422_list(client.get("/v1/reports", params={"cursor": ""}))


def _cursor(id_: str) -> str:
    return base64.urlsafe_b64encode(f"2025-01-01T00:00:00+00:00|{id_}".encode()).decode()


@pytest.mark.db
@pytest.mark.parametrize("id_", ["\u00b2", "9" * 5000, "", "-1", "1e3"])
def test_bad_claim_cursors_are_422_not_500(client, db, id_):
    seed_report(db, n_claims=0)
    r = client.get("/v1/companies/AAPL/claims", params={"cursor": _cursor(id_)})
    assert _is_422_list(r), r.text


@pytest.mark.db
def test_claim_detail_and_company_claims(client, db):
    report = seed_report(db, n_claims=3)
    claim_id = db.query(Claim.id).order_by(Claim.id).first()[0]
    body = client.get(f"/v1/claims/{claim_id}").json()
    assert body["claim"]["value"] == 97.0e9
    assert body["verdict"]["label"] == "contradicted"
    assert body["verdict"]["report_id"] == str(report.id)
    assert [e["agent"] for e in body["evidence"]] == ["xbrl_table"]
    assert [s["name"] for s in body["spans"]] == ["verify_claim"]
    assert client.get("/v1/claims/999999999").status_code == 404

    page = client.get("/v1/companies/aapl/claims", params={"limit": 2}).json()
    assert len(page["items"]) == 2 and page["next_cursor"]
    rest = client.get("/v1/companies/AAPL/claims", params={"cursor": page["next_cursor"]}).json()
    assert len(rest["items"]) == 1 and rest["next_cursor"] is None
    assert client.get("/v1/companies/ZZZZ/claims").status_code == 404


@pytest.mark.db
def test_evals_latest_one_per_layer_and_name(client, db):
    from datetime import UTC, datetime, timedelta

    t = datetime.now(UTC)
    db.add_all(
        [
            EvalRun(
                layer="L2", metrics={"contradiction_precision": 0.61}, created_at=t - timedelta(1)
            ),
            EvalRun(layer="L2", metrics={"contradiction_precision": 0.72}, created_at=t),
            EvalRun(layer="L1", name="wer", metrics={"wer": 0.09}, created_at=t),
            EvalRun(layer="L1", name="ner", metrics={"f1": 0.8}, created_at=t - timedelta(1)),
            EvalRun(layer="L1", name="ner", metrics={"f1": 0.9}, created_at=t),
        ]
    )
    db.flush()
    runs = {(r["layer"], r["name"]): r["metrics"] for r in client.get("/v1/evals/latest").json()}
    assert runs == {
        ("L1", "ner"): {"f1": 0.9},
        ("L1", "wer"): {"wer": 0.09},
        ("L2", "L2"): {"contradiction_precision": 0.72},
    }  # name defaults to layer


@pytest.mark.db
@pytest.mark.perf
@pytest.mark.skipif(
    not os.environ.get("LEDGERLINE_PERF"),
    reason="perf: timing-sensitive; run with LEDGERLINE_PERF=1 pytest -m perf",
)
def test_nfr2_report_p95(client, db):
    """NFR-2: GET /v1/reports/{id} p95 <= 300 ms on a populated report (40 claims), measured
    in-process through the full ASGI stack against the local Postgres."""
    report = seed_report(db, n_claims=40)
    url = f"/v1/reports/{report.id}"
    client.get(url)  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        assert client.get(url).status_code == 200
        times.append((time.perf_counter() - t0) * 1000)
    p95 = statistics.quantiles(times, n=100)[94]
    print(
        f"\nNFR-2 GET /v1/reports/{{id}} 40 claims, n=200: "
        f"p50={statistics.median(times):.1f} ms p95={p95:.1f} ms"
    )
    assert p95 <= 300


@pytest.mark.db
def test_healthz_is_json_503_when_the_db_is_down(client):
    dead = create_engine(
        "postgresql+psycopg://nobody@127.0.0.1:1/none", connect_args={"connect_timeout": 1}
    )
    app.dependency_overrides[get_session] = lambda: Session(dead)
    r = client.get("/healthz")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"
    dead.dispose()


@pytest.mark.db
@pytest.mark.parametrize("claim_id", ["2147483648", "9" * 36, "0", "-1"])
def test_claim_id_outside_int4_is_422(client, claim_id):
    assert _is_422_list(client.get(f"/v1/claims/{claim_id}"))


@pytest.mark.db
@pytest.mark.parametrize("id_", ["2147483648", "9" * 36, "0"])
def test_claim_cursor_ids_outside_int4_are_422(client, db, id_):
    seed_report(db, n_claims=0)
    assert _is_422_list(client.get("/v1/companies/AAPL/claims", params={"cursor": _cursor(id_)}))


@pytest.mark.db
@pytest.mark.parametrize("limit", [0, 101])
def test_company_claims_limit_bounds(client, db, limit):
    seed_report(db, n_claims=0)
    assert _is_422_list(client.get("/v1/companies/AAPL/claims", params={"limit": limit}))


@pytest.mark.db
def test_claim_detail_shows_only_a_done_reports_verdict(client, db):
    """K2: a newer running (or failed) report's partial verdict is not the claim's verdict;
    a claim no done report verified has none."""
    done = seed_report(db, n_claims=1)
    claim = db.query(Claim).one()
    running = Report(
        company_id=done.company_id, fiscal_year=2025, fiscal_period="Q3", status="running"
    )
    db.add(running)
    db.flush()
    db.add(
        Verdict(
            report_id=running.id,
            claim_id=claim.id,
            label="supported",
            confidence=0.5,
            rationale="partial",
            citations=[],
            prompt_version="v1",
            trace_id=uuid.uuid4(),
        )
    )
    db.flush()
    body = client.get(f"/v1/claims/{claim.id}").json()
    assert body["verdict"]["report_id"] == str(done.id)

    done.status = "failed"
    db.flush()
    body = client.get(f"/v1/claims/{claim.id}").json()
    assert body["claim"]["id"] == claim.id
    assert (body["verdict"], body["evidence"], body["spans"]) == (None, [], [])


@pytest.mark.db
def test_failed_report_shows_its_error(client, db):
    report = seed_report(db, n_claims=0)
    report.status, report.error = "failed", "RateLimited: claude-opus-5: rate-limited"
    db.flush()
    body = client.get(f"/v1/reports/{report.id}").json()
    assert (body["status"], body["error"]) == ("failed", report.error)


@pytest.mark.db
def test_401_names_the_auth_scheme(client, db):
    seed_report(db, n_claims=0)
    r = client.post(
        "/v1/reports",
        json={"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q1"},
        headers={"X-API-Key": "no"},
    )
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "X-API-Key"


@pytest.mark.db
@pytest.mark.parametrize(
    "body",
    [
        {"ticker": "AAPL", "fiscal_year": "2025", "fiscal_period": "Q1"},  # strict: no coercion
        {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q1", "rerun": "true"},
        {"ticker": "AAPL", "fiscal_year": 2025.0, "fiscal_period": "Q1"},
    ],
)
def test_post_is_strict(client, db, body):
    seed_report(db, n_claims=0)
    assert _is_422_list(client.post("/v1/reports", json=body))


@pytest.mark.parametrize(
    "fy, fp, ok",
    [
        # FY2027 can end in January 2027, so its Q2 may have ended by July 2026; Q3 not before October.
        (2025, "Q4", True),
        (2027, "Q2", True),
        (2027, "Q3", False),
        (2028, "Q1", False),
    ],
)
def test_future_quarters_are_rejected(monkeypatch, fy, fp, ok):
    class Today(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 25, tzinfo=tz)

    monkeypatch.setattr(api, "datetime", Today)
    body = {"ticker": "AAPL", "fiscal_year": fy, "fiscal_period": fp}
    if ok:
        api.ReportRequest.model_validate(body)
    else:
        with pytest.raises(ValidationError, match="has not ended yet"):
            api.ReportRequest.model_validate(body)


@pytest.fixture
def committed_company(db_url, monkeypatch):
    """A committed company and a Session per request, so two POSTs really run on two
    connections; everything created is deleted again."""
    engine = create_engine(db_url)
    with Session(engine) as s:
        co = Company(cik="0000000042", ticker="ZZTOP", name="Race Test")
        s.add(co)
        s.commit()
        company_id = co.id

    def sessions():
        with Session(engine, expire_on_commit=False) as s:
            yield s

    monkeypatch.setenv("LEDGERLINE_API_KEY", KEY)
    settings.cache_clear()
    app.dependency_overrides[get_session] = sessions
    try:
        yield company_id
    finally:
        app.dependency_overrides.clear()
        settings.cache_clear()
        with engine.begin() as c:
            c.execute(delete(Report).where(Report.company_id == company_id))
            c.execute(delete(Company).where(Company.id == company_id))
        engine.dispose()


@pytest.mark.db
def test_concurrent_posts_for_one_quarter_create_one_report(committed_company, monkeypatch):
    """K10: both requests pass the company lookup before either takes the quarter's lock;
    without the advisory lock both would find no report and insert one each."""
    barrier = threading.Barrier(2, timeout=10)
    lookup = api._company

    def company_then_wait(s, ticker):
        co = lookup(s, ticker)
        barrier.wait()
        return co

    monkeypatch.setattr(api, "_company", company_then_wait)
    body = {"ticker": "ZZTOP", "fiscal_year": 2025, "fiscal_period": "Q2"}

    def post(_):
        with TestClient(app, headers={"X-API-Key": KEY}) as c:
            return c.post("/v1/reports", json=body)

    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(post, range(2)))
    assert [r.status_code for r in responses] == [202, 202]
    assert responses[0].json()["job_id"] == responses[1].json()["job_id"]
