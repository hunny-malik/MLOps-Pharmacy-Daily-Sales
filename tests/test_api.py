import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app


class DummyModel:
    def predict(self, df):
        return [42.0]


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: DummyModel())


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "flu_cases_reported": 120,
        "temp_c": 20.0,
        "nearby_clinics": 4,
        "discount_pct": 10.0,
        "stock_lines": 3000,
        "distance_to_hospital_km": 2.5,
        "shop_type": "retail",
        "locality": "Old Town",
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    assert r.json()["predicted_units_sold"] == 42.0


def test_boundaries_accepted(client, valid_payload):
    valid_payload["stock_lines"] = 5500
    valid_payload["temp_c"] = 2.5
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_stock_lines_too_high(client, valid_payload):
    valid_payload["stock_lines"] = 5501
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_temp_too_low(client, valid_payload):
    valid_payload["temp_c"] = 2.4
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_sentinel_clinics_rejected(client, valid_payload):
    valid_payload["nearby_clinics"] = 9999
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_missing_field(client, valid_payload):
    del valid_payload["locality"]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_extra_field_rejected(client, valid_payload):
    valid_payload["bogus"] = 1
    assert client.post("/predict", json=valid_payload).status_code == 422