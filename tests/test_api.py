"""API contract tests.

The service is trained on a deliberately tiny corpus here so the whole file
runs in a few seconds; the point is the request/response contract, not model
quality (that is covered in ``test_pipeline.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from reputation.pipeline.train import run  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a throwaway bundle, point the app at it, and yield a test client."""
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    run(
        argparse.Namespace(
            source="synthetic",
            data_dir="data",
            artifact_dir=str(artifact_dir),
            city=None,
            limit=None,
            n_businesses=25,
            months=26,
            skip_importance=True,   # permutation importance is the slow part
        )
    )

    import os

    os.environ["ARTIFACT_DIR"] = str(artifact_dir)
    from reputation.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_businesses_are_ranked_by_risk(client):
    body = client.get("/businesses?limit=10").json()
    assert body
    risks = [row["decline_risk"] for row in body]
    assert risks == sorted(risks, reverse=True)
    assert all(0.0 <= r <= 1.0 for r in risks)


def test_report_endpoint_returns_actionable_findings(client):
    business_id = client.get("/businesses?limit=1").json()[0]["business_id"]
    report = client.get(f"/businesses/{business_id}/report").json()
    assert report["business_id"] == business_id
    assert report["summary"]
    assert report["findings"][0]["suggested_action"]


def test_unknown_business_returns_404(client):
    assert client.get("/businesses/does-not-exist/report").status_code == 404


def test_analyse_endpoint_routes_free_text(client):
    response = client.post(
        "/analyse",
        json={
            "texts": [
                "the restroom was disgusting and the floor was sticky",
                "we queued for an hour before being seated",
            ],
            "stars": [1, 2],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["top_aspect"] == "cleanliness"
    assert results[1]["top_aspect"] == "wait_time"
    assert all(r["is_complaint"] for r in results)


def test_metrics_summary_is_served(client):
    metrics = client.get("/metrics-summary").json()
    assert metrics["data"]["source"] == "synthetic"
    assert "decline_model" in metrics
