"""Sanity checks for Streamlit, CLI and FastAPI interfaces."""
import pathlib
import sys

# Ensure project root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from typer.testing import CliRunner
from streamlit.testing.v1 import AppTest

from api.main import app as fastapi_app
from cli.main import app as cli_app


def test_fastapi_price_endpoint():
    client = TestClient(fastapi_app)
    payload = {
        "option_type": "Call",
        "strike": 1.0,
        "maturity": 1.0,
        "s0": 1.0,
        "rate": 0.05,
        "sigma": 0.2,
        "s_steps": 20,
        "t_steps": 20,
    }
    response = client.post("/price", json=payload)
    assert response.status_code == 200
    assert "price" in response.json()


def test_cli_price_command():
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        [
            "price",
            "--option-type",
            "Call",
            "--strike",
            "1.0",
            "--maturity",
            "1.0",
            "--s0",
            "1.0",
            "--rate",
            "0.05",
            "--sigma",
            "0.2",
            "--s-steps",
            "20",
            "--t-steps",
            "20",
        ],
    )
    assert result.exit_code == 0
    assert "Price:" in result.stdout


def test_streamlit_app_runs():
    at = AppTest.from_file("apps/streamlit_app.py")
    at.run()
    at.number_input[5].set_value(5)
    at.number_input[6].set_value(5)
    at.button[0].click().run()
    assert not at.exception
