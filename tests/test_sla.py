# tests/test_sla.py
import pandas as pd
from zentel_pipeline.pipeline.etl import compute_sla_metrics

def test_escalation_flag():
    df = pd.DataFrame({
        "report_id": ["r1"],
        "ticket_open_time": [pd.to_datetime("2020-01-01 10:00:00")],
        "ticket_resp_time": [pd.to_datetime("2020-01-01 10:00:00")],
        "issue_res_time": [pd.to_datetime("2020-01-01 14:01:00")],  # 241 minutes -> escalation
        "ticket_status": ["Completed"]
    })
    res = compute_sla_metrics(df)
    assert res.loc[0, "escalation"] == True
