"""Object store, HTML->text, HTTP classification, Yahoo parsing: no network, no database."""

import json

import httpx
import pytest
import respx

from ledgerline.config import settings
from ledgerline.ingest import http
from ledgerline.ingest.documents import html_to_text
from ledgerline.ingest.market import parse_chart
from ledgerline.ingest.objects import put_object, read_object


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings(), "data_dir", tmp_path)
    return tmp_path


def test_object_store_is_content_addressed_and_idempotent(data_dir):
    sha, rel = put_object(b"hello", ".htm")
    assert rel == f"objects/{sha[:2]}/{sha}.htm" and read_object(rel) == b"hello"
    mtime = (data_dir / rel).stat().st_mtime_ns
    assert put_object(b"hello", ".htm") == (sha, rel)
    assert (data_dir / rel).stat().st_mtime_ns == mtime  # second write did not touch it
    assert len(list(data_dir.rglob("*.htm"))) == 1


HTML = b"""<html><head><title>t</title><style>p{}</style></head><body>
<div style="display:none"><ix:header><ix:hidden>HIDDEN</ix:hidden></ix:header></div>
<p>Total   net&nbsp;sales were <b>$94.9</b>
billion.</p><div>Next<br>line</div>
<table><tr><td>Net sales</td><td></td><td>$</td><td>94,930</td><td>(</td><td>1.2</td><td>)%</td></tr>
<tr><th>Gross margin</th><td>46.3</td><td>%</td></tr></table><script>x()</script></body></html>"""


def test_html_to_text_layout():
    assert html_to_text(HTML) == (
        "Total net sales were $94.9 billion.\nNext\nline\n"
        "Net sales | $94,930 | (1.2)%\nGross margin | 46.3%")


def test_html_to_text_is_stable():
    assert len({html_to_text(HTML) for _ in range(3)}) == 1
    assert html_to_text(b"a  b\r\n\r\n c", ".txt") == "a b\nc"


@respx.mock
def test_sec_get_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(http.get.retry, "sleep", lambda _: None)
    route = respx.get("https://sec.example/x").mock(side_effect=[
        httpx.Response(429), httpx.Response(503), httpx.Response(200, content=b"ok")])
    assert http.sec_get("https://sec.example/x") == b"ok"
    assert route.call_count == 3
    assert route.calls[0].request.headers["User-Agent"] == settings().sec_user_agent


@respx.mock
def test_sec_get_404_is_permanent_and_classified(monkeypatch):
    monkeypatch.setattr(http.get.retry, "sleep", lambda _: None)
    route = respx.get("https://sec.example/gone").mock(return_value=httpx.Response(404))
    with pytest.raises(http.PermanentFetchError) as e:
        http.sec_get("https://sec.example/gone")
    assert route.call_count == 1
    assert http.error_class(e.value) == "http_4xx"
    assert http.error_class(KeyError("facts")) == "schema_drift"
    assert http.error_class(json.JSONDecodeError("x", "", 0)) == "parse"


def test_parse_chart_uses_exchange_local_day_and_skips_null_bars():
    payload = {"chart": {"result": [{
        "meta": {"gmtoffset": -14400},
        "timestamp": [1756992600, 1757079000],  # 2025-09-04 and 09-05, 09:30 New York
        "indicators": {"quote": [{"close": [239.78, None], "volume": [100, 200]}],
                       "adjclose": [{"adjclose": [238.9, None]}]}}]}}
    rows = parse_chart(payload, "AAPL", "u")
    assert [(str(r["day"]), r["close"], r["adj_close"]) for r in rows] == [
        ("2025-09-04", 239.78, 238.9)]
