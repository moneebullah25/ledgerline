import statistics
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from test_report_generate import seed_report

from ledgerline.api.main import app, get_session
from ledgerline.models import Claim, EvalRun, Report


@pytest.fixture
def client(db):
    app.dependency_overrides[get_session] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.db
def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.db
def test_post_enqueues_without_running(client, db):
    seed_report(db, n_claims=0)
    r = client.post("/v1/reports", json={"ticker": "aapl", "fiscal_year": 2025,
                                         "fiscal_period": "Q4"})
    assert r.status_code == 202
    report = db.get(Report, uuid.UUID(r.json()["job_id"]))
    assert report.status == "queued"
    assert client.get(f"/v1/reports/{report.id}").json()["claims"] == []


@pytest.mark.db
@pytest.mark.parametrize("body, code", [
    ({"ticker": "NOPE", "fiscal_year": 2025, "fiscal_period": "Q1"}, 404),
    ({"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q5"}, 422),
    ({"ticker": "AAPL'; DROP TABLE report;--", "fiscal_year": 2025, "fiscal_period": "Q1"}, 422),
    ({"ticker": "AAPL", "fiscal_period": "Q1"}, 422),
])
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
        page = client.get("/v1/reports", params={"limit": 3, **({"cursor": cursor} if cursor
                                                                 else {})}).json()
        seen += [r["id"] for r in page["items"]]
        if not (cursor := page["next_cursor"]):
            break
    assert len(seen) == len(set(seen)) == 4
    assert client.get("/v1/reports", params={"limit": 101}).status_code == 422
    assert client.get("/v1/reports", params={"cursor": "garbage"}).status_code == 422


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
def test_evals_latest_one_per_layer(client, db):
    from datetime import UTC, datetime, timedelta

    t = datetime.now(UTC)
    db.add_all([
        EvalRun(layer="L2", metrics={"contradiction_precision": 0.61}, created_at=t - timedelta(1)),
        EvalRun(layer="L2", metrics={"contradiction_precision": 0.72}, created_at=t),
        EvalRun(layer="L1", metrics={"wer": 0.09}, created_at=t),
    ])
    db.flush()
    runs = {r["layer"]: r["metrics"] for r in client.get("/v1/evals/latest").json()}
    assert runs == {"L1": {"wer": 0.09}, "L2": {"contradiction_precision": 0.72}}


@pytest.mark.db
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
    print(f"\nNFR-2 GET /v1/reports/{{id}} 40 claims, n=200: "
          f"p50={statistics.median(times):.1f} ms p95={p95:.1f} ms")
    assert p95 <= 300
