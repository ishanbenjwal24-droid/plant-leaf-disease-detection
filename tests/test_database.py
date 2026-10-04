import json

from backend.app.db import PredictionHistory, create_session_factory


def test_history_create_read_and_clear(tmp_path):
    session_factory = create_session_factory(tmp_path / "history.sqlite3")
    prediction = [{"model": "ml", "predicted_class": "Apple___healthy"}]
    with session_factory() as session:
        row = PredictionHistory(
            mode="ml", predictions_json=json.dumps(prediction), latency_ms=3.5
        )
        session.add(row)
        session.commit()
        assert row.id is not None

    with session_factory() as session:
        row = session.query(PredictionHistory).one()
        assert row.mode == "ml"
        assert row.predictions == prediction
        assert row.latency_ms == 3.5
        assert "image" not in row.predictions_json.lower()
        assert session.query(PredictionHistory).delete() == 1
        session.commit()
        assert session.query(PredictionHistory).count() == 0
