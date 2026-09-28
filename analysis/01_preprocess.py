"""
협업자 전처리 기준(report/전처리_기준.md) 1~5단계를 그대로 구현한다.
실행: 프로젝트 루트에서 `python analysis/01_preprocess.py`
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/Online Retail.xlsx"
DAILY_OUT = "data/processed/daily_sales.csv"
LOG_OUT = "outputs/tables/preprocessing_exclusion_log.csv"

USAGE_START = pd.Timestamp("2010-12-01")
USAGE_END = pd.Timestamp("2011-12-08")  # 양 끝 포함

UK_HOLIDAYS = {
    "2010-12-27", "2010-12-28", "2011-01-03",
    "2011-04-22", "2011-04-25", "2011-04-29",
    "2011-05-02", "2011-05-30", "2011-08-29",
}

# ---------------------------------------------------------------
# 1. 불러오기와 자료형
# ---------------------------------------------------------------
raw = pd.read_excel(RAW_PATH, sheet_name="Online Retail")
n_raw = len(raw)

df = raw.copy()
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Description"] = df["Description"].astype(str).str.strip().where(df["Description"].notna(), df["Description"])
df["Quantity"] = df["Quantity"].astype("int64")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["UnitPrice"] = df["UnitPrice"].astype("float64")
df["CustomerID"] = df["CustomerID"].apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)
df["Country"] = df["Country"].astype(str).str.strip()

# 원본(제외 규칙 적용 전) 날짜 목록: 4-2 휴무일 판정에 사용
raw_dates_in_range = set(
    df.loc[
        (df["InvoiceDate"].dt.normalize() >= USAGE_START) & (df["InvoiceDate"].dt.normalize() <= USAGE_END),
        "InvoiceDate",
    ].dt.normalize()
)

df["amount"] = df["Quantity"] * df["UnitPrice"]
df["is_cancel"] = df["InvoiceNo"].str.upper().str.startswith("C")
df["date"] = df["InvoiceDate"].dt.normalize()
df["StockCode_upper"] = df["StockCode"].str.upper()

original_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]

# ---------------------------------------------------------------
# 2. 행 제외 규칙 (순서대로, 앞 단계에서 남은 행에만 적용)
# ---------------------------------------------------------------
log_rows = []


def apply_step(step, rule, mask_exclude, frame):
    """mask_exclude: frame 내에서 제외할 행을 True로 표시. 제외 후 남은 frame을 반환."""
    n_excl = int(mask_exclude.sum())
    amt_excl = float(frame.loc[mask_exclude, "amount"].sum())
    kept = frame.loc[~mask_exclude].copy()
    log_rows.append(
        {
            "step": step,
            "rule": rule,
            "excluded_count": n_excl,
            "excluded_amount": round(amt_excl, 2),
            "remaining_rows": len(kept),
            "remaining_amount": round(float(kept["amount"].sum()), 2),
        }
    )
    print(f"[{step}] {rule}: 제외 {n_excl:,}건, 제외 금액 {amt_excl:,.2f} -> 남은 행 {len(kept):,}")
    return kept


cur = df

# 2-1 완전 중복 행
mask = cur.duplicated(subset=original_cols, keep="first")
cur = apply_step("2-1", "완전 중복 행", mask, cur)

# 2-2 회계 조정 코드
adj_codes = {"AMAZONFEE", "BANK CHARGES", "CRUK", "B"}
mask = cur["StockCode_upper"].isin(adj_codes)
cur = apply_step("2-2", "회계 조정 코드", mask, cur)

# 2-3 수기 정정·할인·샘플
manual_codes = {"M", "D", "S"}
mask = cur["StockCode_upper"].isin(manual_codes)
cur = apply_step("2-3", "수기 정정·할인·샘플", mask, cur)

# 2-4 배송비
post_codes = {"POST", "DOT", "C2"}
mask = cur["StockCode_upper"].isin(post_codes)
cur = apply_step("2-4", "배송비", mask, cur)

# 2-5 상품권 판매
mask = cur["StockCode_upper"].str.startswith("GIFT_")
cur = apply_step("2-5", "상품권 판매", mask, cur)

# 2-6 단가 0 이하
mask = cur["UnitPrice"] <= 0
cur = apply_step("2-6", "단가 0 이하", mask, cur)

# 2-7 취소 아닌 음수 수량 (안전장치)
mask = (cur["Quantity"] < 0) & (~cur["is_cancel"])
cur = apply_step("2-7", "취소 아닌 음수 수량", mask, cur)

# 2-8 불완전한 마지막 날
mask = cur["InvoiceDate"] >= pd.Timestamp("2011-12-09")
cur = apply_step("2-8", "불완전한 마지막 날", mask, cur)

n_final = len(cur)
amt_final = float(cur["amount"].sum())
print(f"\n최종 남은 행: {n_final:,} (원본 {n_raw:,}건 대비 {n_final / n_raw:.1%}), 금액 합계 {amt_final:,.2f}")

pd.DataFrame(log_rows).to_csv(LOG_OUT, index=False, encoding="utf-8-sig")
print(f"제외 기록 저장: {LOG_OUT}")

# 제외하지 않는 항목 확인 (기록용 출력)
n_missing_customer = int(cur["CustomerID"].isna().sum())
n_cancel_kept = int(cur["is_cancel"].sum())
cancel_amt_kept = float(cur.loc[cur["is_cancel"], "amount"].sum())
n_dcgs = int(cur["StockCode_upper"].str.startswith("DCGS").sum())
print(f"참고: CustomerID 결측 유지 {n_missing_customer:,}건 / 취소 행 유지 {n_cancel_kept:,}건(금액 {cancel_amt_kept:,.2f}) / DCGS 유지 {n_dcgs:,}건")

# ---------------------------------------------------------------
# 3. 일별 집계
# ---------------------------------------------------------------
normal = cur.loc[~cur["is_cancel"]]
cancel = cur.loc[cur["is_cancel"]]

daily = pd.DataFrame(index=pd.Index([], name="date"))

gross = normal.groupby("date")["amount"].sum().rename("gross_sales")
cancel_amt = cancel.groupby("date")["amount"].sum().rename("cancel_amount")
net = cur.groupby("date")["amount"].sum().rename("net_sales")
n_invoices = normal.groupby("date")["InvoiceNo"].nunique().rename("n_invoices")
n_cancel_invoices = cancel.groupby("date")["InvoiceNo"].nunique().rename("n_cancel_invoices")
n_lines = normal.groupby("date").size().rename("n_lines")
quantity = normal.groupby("date")["Quantity"].sum().rename("quantity")
n_customers = normal.groupby("date")["CustomerID"].apply(lambda s: s.dropna().nunique()).rename("n_customers")

daily = pd.concat([net, gross, cancel_amt, n_invoices, n_cancel_invoices, n_lines, quantity, n_customers], axis=1)

# 검산: 일별 net_sales 합 = 2단계 후 남은 행의 amount 합
check_diff = abs(daily["net_sales"].sum() - amt_final)
assert check_diff < 1e-6, f"검산 실패: 차이 {check_diff}"
print(f"\n검산 통과: 일별 net_sales 합({daily['net_sales'].sum():,.2f}) = 2단계 후 amount 합({amt_final:,.2f})")

# ---------------------------------------------------------------
# 4. 달력 채우기와 날짜 변수
# ---------------------------------------------------------------
full_range = pd.date_range(USAGE_START, USAGE_END, freq="D")
count_cols = ["net_sales", "gross_sales", "cancel_amount", "n_invoices", "n_cancel_invoices", "n_lines", "quantity", "n_customers"]
daily = daily.reindex(full_range)
daily.index.name = "date"
daily[count_cols] = daily[count_cols].fillna(0.0)
for c in ["n_invoices", "n_cancel_invoices", "n_lines", "quantity", "n_customers"]:
    daily[c] = daily[c].astype("int64")

# 4-2 휴무일 표시 (원본 기준)
daily["is_closed"] = (~daily.index.isin(raw_dates_in_range)).astype(int)
n_open = int((daily["is_closed"] == 0).sum())
n_closed = int((daily["is_closed"] == 1).sum())
print(f"영업일 {n_open}일, 휴무일 {n_closed}일")

# 4-3 달력 변수
daily["dow"] = daily.index.dayofweek
daily["is_sunday"] = (daily["dow"] == 6).astype(int)
date_str = daily.index.strftime("%Y-%m-%d")
daily["is_uk_holiday"] = date_str.isin(UK_HOLIDAYS).astype(int)
daily["year_month"] = daily.index.strftime("%Y-%m")

month_start = daily.index.to_period("M").start_time
month_end = daily.index.to_period("M").end_time
daily["is_partial_month"] = ((month_start < USAGE_START) | (month_end > USAGE_END)).astype(int)

# ---------------------------------------------------------------
# 5. 이상치 표시 (지우지 않음, 영업일에만 적용)
# ---------------------------------------------------------------
daily["is_outlier"] = 0
open_mask = daily["is_closed"] == 0
y = np.log(daily.loc[open_mask, "net_sales"])
grp = daily.loc[open_mask, "dow"]

med = y.groupby(grp).transform("median")
mad = (y - med).abs().groupby(grp).transform("median")
z = 0.6745 * (y - med) / mad
daily.loc[open_mask, "is_outlier"] = (z.abs() > 3.5).astype(int)

n_outlier = int(daily["is_outlier"].sum())
print(f"이상치로 표시된 영업일: {n_outlier}일")

daily.to_csv(DAILY_OUT, encoding="utf-8-sig")
print(f"\n저장 완료: {DAILY_OUT} ({len(daily)}행)")
print("\n상위 5행:")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(daily.head(5))
