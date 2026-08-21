from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
VENV_PYTHON = "/opt/forexbot-orca/venv/bin/python"
PANEL = "/srv/forexbot-backtest/orca/prices.parquet"
ENV_ROOT = "/etc/forexbot-orca"


def _text(name: str) -> str:
    return (DEPLOY / name).read_text(encoding="utf-8")


def test_orca_units_use_isolated_native_runtime_and_external_env() -> None:
    monitor = _text("orca-monitor.service")
    prediction = _text("orca-prediction.service")
    eodhd = _text("orca-eodhd.service")

    assert f"ExecStart={VENV_PYTHON} -m monitoring.orca_monitor" in monitor
    assert f"ExecStart={VENV_PYTHON} -m monitoring.orca_prediction" in prediction
    assert f"ExecStart={VENV_PYTHON} -m monitoring.orca_eodhd_ingest" in eodhd
    assert f"EnvironmentFile={ENV_ROOT}/orca-monitor.env" in monitor
    assert f"EnvironmentFile={ENV_ROOT}/orca-prediction.env" in prediction
    assert f"EnvironmentFile={ENV_ROOT}/orca-eodhd.env" in eodhd
    assert "forexbot.service" not in monitor + prediction + eodhd


def test_orca_units_fail_closed_on_missing_inputs() -> None:
    monitor = _text("orca-monitor.service")
    prediction = _text("orca-prediction.service")

    assert f"ConditionPathExists={PANEL}" in monitor
    assert f"ConditionPathExists={PANEL}" in prediction
    assert (
        "ConditionPathExists=/opt/forexbot-orca/models/orca-rf.json"
    ) in prediction
    assert "StateDirectory=forexbot-orca" in monitor
    assert "StateDirectory=forexbot-orca" in prediction
    assert "PrivateNetwork=true" in prediction


def test_orca_refresh_order_and_paths_are_consistent() -> None:
    eodhd = _text("orca-eodhd.service")
    eodhd_env = _text("orca-eodhd.env.example")
    monitor_env = _text("orca-monitor.env.example")
    prediction_env = _text("orca-prediction.env.example")
    prediction_timer = _text("orca-prediction.timer")

    assert f"ORCA_EODHD_OUTPUT_PATH={PANEL}" in eodhd_env
    assert f"ORCA_PRICES_PATH={PANEL}" in monitor_env
    assert f"ORCA_PRICES_PATH={PANEL}" in prediction_env
    assert "OnSuccess=orca-prediction.service" in eodhd
    assert "ORCA_REFRESH_SECONDS=300" in monitor_env
    assert "00,12:30:00 UTC" in prediction_timer


def test_runtime_requirements_do_not_install_training_stack() -> None:
    runtime = (ROOT / "requirements-orca-runtime.txt").read_text(encoding="utf-8")
    assert "numpy==" in runtime
    assert "pandas==" in runtime
    assert "pyarrow==" in runtime
    assert "scikit-learn" not in runtime
    assert "lse-data" not in runtime
