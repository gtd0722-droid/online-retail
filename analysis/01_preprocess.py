"""온라인 소매 원본을 일별 매출 테이블로 전처리한다.

기준: report/전처리_기준.md (1~5단계)
입력: data/raw/Online Retail.xlsx
출력: data/processed/daily_sales.csv
      outputs/tables/preprocessing_exclusion_log.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "Online Retail.xlsx"
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
LOG_PATH = ROOT / "outputs" / "tables" / "preprocessing_exclusion_log.csv"

PERIOD_START = pd.Timestamp("2010-12-01")
PERIOD_END = pd.Timestamp("2011-12-08")  # 양 끝 포함
CUTOFF = PERIOD_END + pd.Timedelta(days=1)  # 2011-12-09 00:00 이후 제외

ADJUSTMENT_CODES = {"AMAZONFEE", "BANK CHARGES", "CRUK", "B"}
MANUAL_CODES = {"M", "D", "S"}
POSTAGE_CODES = {"POST", "DOT", "C2"}
GIFT_PREFIX = "GIFT_"

UK_HOLIDAYS = pd.to_datetime([
    "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25",
    "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
])

OUTLIER_Z = 3.5
MAD_SCALE = 0.6745


# 1. 불러오기와 자료형
def load_raw() -> pd.DataFrame:
    df = pd.read_excel(
        RAW_PATH,
        sheet_name="Online Retail",
        dtype={"InvoiceNo": str, "StockCode": str, "Description": str},
    )
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["Description"] = df["Description"].str.strip()
    df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")
    return df


# 2. 행 제외 규칙
def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_columns = list(df.columns)
    df = df.copy()
    df["amount"] = df["Quantity"] * df["UnitPrice"]

    rules = [
        ("2-1", "완전 중복 행", lambda d: d.duplicated(subset=raw_columns, keep="first")),
        ("2-2", "회계 조정 코드", lambda d: d["StockCode"].str.upper().isin(ADJUSTMENT_CODES)),
        ("2-3", "수기 정정·할인·샘플", lambda d: d["StockCode"].str.upper().isin(MANUAL_CODES)),
        ("2-4", "배송비", lambda d: d["StockCode"].str.upper().isin(POSTAGE_CODES)),
        ("2-5", "상품권 판매", lambda d: d["StockCode"].str.upper().str.startswith(GIFT_PREFIX)),
        ("2-6", "단가 0 이하", lambda d: d["UnitPrice"] <= 0),
        ("2-7", "취소 아닌 음수 수량",
         lambda d: (d["Quantity"] < 0) & ~d["InvoiceNo"].str.startswith("C")),
        ("2-8", "불완전한 마지막 날", lambda d: d["InvoiceDate"] >= CUTOFF),
    ]

    records = [{
        "step": "0", "rule": "원본", "excluded_rows": 0, "excluded_amount": 0.0,
        "remaining_rows": len(df), "remaining_amount": df["amount"].sum(),
    }]
    for step, name, rule in rules:
        mask = rule(df)
        records.append({
            "step": step,
            "rule": name,
            "excluded_rows": int(mask.sum()),
            "excluded_amount": df.loc[mask, "amount"].sum(),
            "remaining_rows": int((~mask).sum()),
            "remaining_amount": df.loc[~mask, "amount"].sum(),
        })
        df = df.loc[~mask]

    return df.reset_index(drop=True), pd.DataFrame(records)


# 3. 일별 집계
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        date=df["InvoiceDate"].dt.normalize(),
        is_cancel=df["InvoiceNo"].str.startswith("C"),
    )
    sale = df[~df["is_cancel"]]
    cancel = df[df["is_cancel"]]

    daily = pd.DataFrame({
        "net_sales": df.groupby("date")["amount"].sum(),
        "gross_sales": sale.groupby("date")["amount"].sum(),
        "cancel_amount": cancel.groupby("date")["amount"].sum(),
        "n_invoices": sale.groupby("date")["InvoiceNo"].nunique(),
        "n_cancel_invoices": cancel.groupby("date")["InvoiceNo"].nunique(),
        "n_lines": sale.groupby("date").size(),
        "quantity": sale.groupby("date")["Quantity"].sum(),
        "n_customers": sale.groupby("date")["CustomerID"].nunique(dropna=True),
    })
    return daily


# 4. 달력 채우기와 날짜 변수
def build_calendar(daily: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(PERIOD_START, PERIOD_END, freq="D", name="date")
    out = daily.reindex(dates).fillna(0)
    count_cols = ["n_invoices", "n_cancel_invoices", "n_lines", "quantity", "n_customers"]
    out[count_cols] = out[count_cols].astype("int64")

    raw_dates = set(raw["InvoiceDate"].dt.normalize())
    out["is_closed"] = (~out.index.isin(raw_dates)).astype("int64")
    out["dow"] = out.index.dayofweek
    out["is_sunday"] = (out["dow"] == 6).astype("int64")
    out["is_uk_holiday"] = out.index.isin(UK_HOLIDAYS).astype("int64")
    out["year_month"] = out.index.strftime("%Y-%m")

    days_in_period = out.groupby("year_month").size()
    days_in_month = out.index.to_series().groupby(out["year_month"]).first().dt.days_in_month
    partial = days_in_period < days_in_month
    out["is_partial_month"] = out["year_month"].map(partial).astype("int64")
    return out


# 5. 이상치 표시
def flag_outliers(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    open_days = out["is_closed"] == 0
    if (out.loc[open_days, "net_sales"] <= 0).any():
        raise ValueError("영업일 net_sales에 0 이하 값이 있어 로그를 계산할 수 없다.")

    y = np.log(out.loc[open_days, "net_sales"])
    dow = out.loc[open_days, "dow"]
    med = y.groupby(dow).transform("median")
    mad = (y - med).abs().groupby(dow).transform("median")
    z = MAD_SCALE * (y - med) / mad

    out["outlier_z"] = np.nan
    out.loc[open_days, "outlier_z"] = z
    out["is_outlier"] = (out["outlier_z"].abs() > OUTLIER_Z).astype("int64")
    return out


def main() -> None:
    raw = load_raw()
    clean, log = apply_exclusions(raw)
    daily = aggregate_daily(clean)
    out = build_calendar(daily, raw)
    out = flag_outliers(out)

    # 검산
    assert np.isclose(out["net_sales"].sum(), clean["amount"].sum(), rtol=0, atol=1e-6)
    assert np.allclose(out["net_sales"], out["gross_sales"] + out["cancel_amount"], rtol=0, atol=1e-6)
    assert len(out) == (PERIOD_END - PERIOD_START).days + 1

    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(log.to_string(index=False))
    print(f"\n일별 테이블: {len(out)}행, 영업일 {int((out['is_closed'] == 0).sum())}일, "
          f"휴무일 {int(out['is_closed'].sum())}일, 이상치 {int(out['is_outlier'].sum())}일")
    print(f"net_sales 합계: {out['net_sales'].sum():,.2f}")


if __name__ == "__main__":
    main()
