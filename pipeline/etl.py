# zentel_pipeline/pipeline/etl.py
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np


def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads CSV files from the data directory into DataFrames.
    Expects files: service_data.csv, employees.csv, service_type.csv, channel.csv, fault_type.csv, location.csv
    Returns a dict of DataFrames.
    """
    p = Path(data_dir)
    def _read(name):
        path = p / name
        if not path.exists():
            return pd.DataFrame()
        # Try reading with engine fallback and ignore leading/trailing whitespace
        return pd.read_csv(path, dtype=str).applymap(lambda v: v.strip() if isinstance(v, str) else v)

    tables = {
        "service_data": _read("service_data.csv"),
        "employees": _read("employees.csv"),
        "service_type": _read("service_type.csv"),
        "channel": _read("channel.csv"),
        "fault_type": _read("fault_type.csv"),
        "location": _read("location.csv"),
    }
    return tables


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse datetimes, normalize column names, handle missing values.
    - Parses Ticket Open/Resp/Issue Res/Close times to datetime (try multiple formats).
    - Fills missing Operator with 'UNKNOWN'
    - Standardizes column names to snake_case for easier processing
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize column names to snake_case (basic)
    df = df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))

    # helper to parse datetimes robustly
    def parse_dt(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.NaT
        for fmt in ("%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                return pd.to_datetime(val, format=fmt)
            except Exception:
                continue
        # fallback to pandas parser
        try:
            return pd.to_datetime(val, infer_datetime_format=True, errors="coerce")
        except Exception:
            return pd.NaT

    # parse the key time columns (if present)
    for col in ["ticket_open_time", "ticket_resp_time", "issue_res_time", "ticket_close_time"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_dt)

    # fill missing operator
    if "operator" in df.columns:
        df["operator"] = df["operator"].replace("", np.nan).fillna("UNKNOWN")
    else:
        df["operator"] = "UNKNOWN"

    # trim Report ID and extract service code if possible
    if "report_id" in df.columns:
        df["report_id"] = df["report_id"].astype(str).str.strip()
        # try to extract the last hyphen piece as code (e.g., AXA-20201231-1101-WLESS)
        df["service_code"] = df["report_id"].str.split("-", expand=True).iloc[:, -1].fillna("")
    else:
        df["service_code"] = ""

    return df


def enrich_tickets(tickets_df: pd.DataFrame, employees_df: pd.DataFrame, channel_df: pd.DataFrame = None,
                   service_type_df: pd.DataFrame = None, fault_type_df: pd.DataFrame = None,
                   location_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Left-join lookup tables onto tickets_df.
    Joins performed:
      - employees by Operator -> Employee_name or by matching Employee_name column
      - channel by Report Channel -> Channel Key
      - service_type by service_code -> Service Code
      - fault_type by Fault Type text -> Fault
      - location by State Key -> State Key
    """
    if tickets_df is None or tickets_df.empty:
        return pd.DataFrame()

    df = tickets_df.copy()

    # standardize employees: ensure employee id/name columns exist
    if employees_df is not None and not employees_df.empty:
        emp = employees_df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
        # join by operator name (employee_name) matching Operator in tickets (case-insensitive)
        emp["employee_name_lc"] = emp.get("employee_name", emp.get("employee_name", "")).astype(str).str.lower()
        df["operator_lc"] = df["operator"].astype(str).str.lower()
        df = df.merge(emp, left_on="operator_lc", right_on="employee_name_lc", how="left", suffixes=("", "_emp"))
    else:
        df["employee_id"] = np.nan
        df["manager_id"] = np.nan
        df["designation"] = np.nan
        df["manager"] = np.nan

    # channel
    if channel_df is not None and not channel_df.empty:
        ch = channel_df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
        df = df.merge(ch, left_on="report_channel", right_on="channel_key", how="left")

    # service_type
    if service_type_df is not None and not service_type_df.empty:
        st = service_type_df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
        df = df.merge(st, left_on="service_code", right_on="service_code", how="left", suffixes=("", "_service"))

    # fault_type based on fault text (normalize)
    if fault_type_df is not None and not fault_type_df.empty:
        ft = fault_type_df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
        # simple text match: left join on fault type names
        df = df.merge(ft, left_on=df.get("fault_type", "").str.strip(), right_on="fault", how="left")

    # location
    if location_df is not None and not location_df.empty:
        loc = location_df.rename(columns=lambda s: s.strip().lower().replace(" ", "_"))
        df = df.merge(loc, left_on="state_key", right_on="state_key", how="left", suffixes=("", "_loc"))

    return df


def compute_sla_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SLA metrics:
      - response_seconds: (ticket_resp_time - ticket_open_time).total_seconds()
      - resolution_minutes: (issue_res_time - ticket_resp_time).total_seconds() / 60
      - response_sla_pass: response_seconds <= 10
      - resolution_sla_pass: resolution_minutes <= 180
      - escalation: resolution_minutes > 180 or issue_res_time is NaT when ticket closed
      - resolution_category: Less Than 30 Mins / 30Mins - 1 hour / 1 hour - 3 hours / Greater than 3 hours
    Returns the DataFrame with new columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # compute durations, careful with NaT
    def secs(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return (a - b).total_seconds()

    d["response_seconds"] = d.apply(lambda r: secs(r.get("ticket_resp_time"), r.get("ticket_open_time")), axis=1)
    d["resolution_seconds"] = d.apply(lambda r: secs(r.get("issue_res_time"), r.get("ticket_resp_time")), axis=1)
    d["resolution_minutes"] = d["resolution_seconds"] / 60.0

    # pass/fail flags
    d["response_sla_pass"] = d["response_seconds"].apply(lambda x: bool(x <= 10) if pd.notna(x) else False)
    d["resolution_sla_pass"] = d["resolution_minutes"].apply(lambda x: bool(x <= 180) if pd.notna(x) else False)

    # escalation: unresolved within 3 hours (resolution_minutes > 180) OR closed but no issue_res_time
    def escalation(row):
        if pd.notna(row.get("resolution_minutes")) and row["resolution_minutes"] > 180:
            return True
        # if ticket_status is Closed/Completed but we don't have issue_res_time -> escalate
        status = str(row.get("ticket_status", "")).lower()
        if status in ("completed", "closed") and pd.isna(row.get("issue_res_time")):
            return True
        return False

    d["escalation"] = d.apply(escalation, axis=1)

    # resolution category
    def category(mins):
        if pd.isna(mins):
            return "Unknown"
        if mins < 30:
            return "Less Than 30 Mins"
        if 30 <= mins < 60:
            return "30Mins - 1 hour"
        if 60 <= mins <= 180:
            return "1 hour - 3 hours"
        return "Greater than 3 hours"

    d["resolution_category"] = d["resolution_minutes"].apply(category)

    return d


def manager_operator_performance(df: pd.DataFrame) -> Dict:
    """
    Produce manager and operator ranking and basic KPIs:
     - per operator: total_tickets, avg_response_seconds, avg_resolution_minutes, sla_pass_rate
     - per manager: aggregate the operator metrics
    Returns nested dict: {'operators': {...}, 'managers': {...}}
    """
    if df is None or df.empty:
        return {"operators": {}, "managers": {}}

    d = df.copy()

    # operator metrics
    ops = d.groupby("operator").agg(
        total_tickets=pd.NamedAgg(column="report_id", aggfunc="count"),
        avg_response_seconds=pd.NamedAgg(column="response_seconds", aggfunc="mean"),
        avg_resolution_minutes=pd.NamedAgg(column="resolution_minutes", aggfunc="mean"),
        sla_pass_rate=pd.NamedAgg(column="response_sla_pass", aggfunc=lambda x: float(x.sum()) / len(x) if len(x) > 0 else 0.0)
    ).reset_index()

    operators = {row["operator"]: {
        "total_tickets": int(row["total_tickets"]),
        "avg_response_seconds": float(round(row["avg_response_seconds"] or 0.0, 2)),
        "avg_resolution_minutes": float(round(row["avg_resolution_minutes"] or 0.0, 2)),
        "sla_pass_rate": float(round(row["sla_pass_rate"] or 0.0, 3))
    } for _, row in ops.iterrows()}

    # manager metrics (use manager column if exists; else try manager from employees)
    if "manager" in d.columns:
        mgr_group = d.groupby("manager").agg(
            total_tickets=pd.NamedAgg(column="report_id", aggfunc="count"),
            avg_response_seconds=pd.NamedAgg(column="response_seconds", aggfunc="mean"),
            avg_resolution_minutes=pd.NamedAgg(column="resolution_minutes", aggfunc="mean"),
        ).reset_index()

        managers = {row["manager"]: {
            "total_tickets": int(row["total_tickets"]),
            "avg_response_seconds": float(round(row["avg_response_seconds"] or 0.0, 2)),
            "avg_resolution_minutes": float(round(row["avg_resolution_minutes"] or 0.0, 2)),
        } for _, row in mgr_group.iterrows()}
    else:
        managers = {}

    return {"operators": operators, "managers": managers}


# load_tables("data")