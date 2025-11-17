# tests/test_etl.py
import pandas as pd
from pipeline.etl import clean_tickets, compute_sla_metrics, manager_operator_performance

def test_clean_tickets_parses_datetimes_and_fills_operator():
    raw = pd.DataFrame({
        "Report ID": ["AXA-20201231-1101-WLESS"],
        "Ticket Open Time": ["2020/12/31 17:07:04"],
        "Ticket Resp Time": ["12/31/2020 17:10"],
        "Issue Res Time": ["12/31/2020 20:44"],
        "Operator": [""],
        "Report Channel": ["CH01"]
    })
    cleaned = clean_tickets(raw)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["ticket_open_time"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["ticket_resp_time"])
    assert cleaned.loc[0, "operator"] == "UNKNOWN"
    assert "service_code" in cleaned.columns

def test_compute_sla_basic():
    df = pd.DataFrame({
        "report_id": ["r1"],
        "ticket_open_time": [pd.to_datetime("2020-01-01 10:00:00")],
        "ticket_resp_time": [pd.to_datetime("2020-01-01 10:00:05")], # 5s
        "issue_res_time": [pd.to_datetime("2020-01-01 11:00:05")]  # 60 minutes after response
    })
    res = compute_sla_metrics(df)
    assert res.loc[0, "response_seconds"] == 5
    assert res.loc[0, "resolution_minutes"] == 60.0
    assert res.loc[0, "response_sla_pass"] == True
    assert res.loc[0, "resolution_sla_pass"] == True
    assert res.loc[0, "resolution_category"] == "1 hour - 3 hours"

def test_manager_operator_performance_empty():
    res = manager_operator_performance(pd.DataFrame())
    assert isinstance(res, dict)
    assert "operators" in res and "managers" in res
