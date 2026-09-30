"""매출 분석 보고서 (Streamlit).

실행: streamlit run analysis/07_streamlit_report.py
입력: data/processed/daily_sales.csv, outputs/tables/*.csv, outputs/figures/decomposition_by_month/*.png
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#a9b8cc"
DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

st.set_page_config(page_title="온라인 소매 매출 분석 보고서", layout="wide")


@st.cache_data
def load():
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_sales.csv", parse_dates=["date"])
    decomp = pd.read_csv(TABLES / "daily_decomposition.csv", parse_dates=["date"])
    aov = pd.read_csv(TABLES / "aov_decomposition.csv", parse_dates=["date"])
    excl = pd.read_csv(TABLES / "preprocessing_exclusion_log.csv")
    return daily, decomp, aov, excl


def monthly_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("year_month")
    m = pd.DataFrame({
        "net_sales_total": g["net_sales"].sum(),
        "n_days": g.size(),
        "n_open_days": g["is_closed"].apply(lambda s: int((s == 0).sum())),
        "is_partial_month": g["date"].apply(
            lambda s: int(len(s) < s.iloc[0].days_in_month)
        ),
    })
    m["avg_per_day"] = m["net_sales_total"] / m["n_days"]
    m["avg_per_open_day"] = m["net_sales_total"] / m["n_open_days"]
    return m.reset_index()


def weekday_table(d: pd.DataFrame) -> pd.DataFrame:
    o = d[d["is_closed"] == 0]
    g = o.groupby("dow")
    w = pd.DataFrame({
        "n_open_days": g.size(),
        "mean_sales": g["net_sales"].mean(),
        "median_sales": g["net_sales"].median(),
        "std_sales": g["net_sales"].std(),
        "mean_invoices": g["n_invoices"].mean(),
        "mean_customers": g["n_customers"].mean(),
        "avg_invoice_value": g["gross_sales"].sum() / g["n_invoices"].sum(),
    })
    w["share_of_total"] = g["net_sales"].sum() / o["net_sales"].sum()
    w["cv"] = w["std_sales"] / w["mean_sales"]
    w["index_vs_overall"] = w["mean_sales"] / o["net_sales"].mean()
    split = pd.Timestamp("2011-09-01")
    w["mean_sales_jan_aug"] = o[o["date"] < split].groupby("dow")["net_sales"].mean()
    w["mean_sales_sep_dec"] = o[o["date"] >= split].groupby("dow")["net_sales"].mean()
    w = w.reindex(range(7)).reset_index()
    w["weekday"] = [DOW_ORDER[i] for i in w["dow"]]
    return w


def weekday_effect_table(comp: pd.DataFrame) -> pd.DataFrame:
    o = comp[comp["is_closed"] == 0]
    e = (o["observed"] - o["trend"]).groupby(o["dow"]).mean()
    e = e - e.mean()
    e = e.reindex(range(7))
    return pd.DataFrame({"weekday": DOW_ORDER, "effect_decomposition": e.values})


daily_all, decomp_all, aov_all, excl = load()

# ---------------------------------------------------------------- 기간 선택
dmin, dmax = daily_all["date"].min().date(), daily_all["date"].max().date()
with st.sidebar:
    st.header("분석 기간")
    picked = st.date_input("시작일 ~ 종료일", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="date_range")
    if len(picked) != 2:
        st.info("종료일까지 선택하세요.")
        st.stop()
    start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    st.button("전체 기간으로 되돌리기", on_click=lambda: st.session_state.update(date_range=(dmin, dmax)))
    st.caption(
        "추세·요일 분해는 전체 기간으로 계산한 값을 선택 기간만큼 잘라 보여줍니다. "
        "전처리 표는 항상 전체 기간 기준입니다."
    )

in_range = lambda df: df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
daily, decomp, aov = in_range(daily_all), in_range(decomp_all), in_range(aov_all)
open_days = daily[daily["is_closed"] == 0]
if open_days.empty:
    st.warning("선택한 기간에 영업일이 없습니다. 기간을 넓혀 주세요.")
    st.stop()
monthly = monthly_table(daily)
weekday = weekday_table(daily)
effect = weekday_effect_table(decomp)


def won(v):  # 통화 표기 (원본 통화 단위: 파운드)
    return f"£{v:,.0f}"


# ---------------------------------------------------------------- 헤더
st.title("온라인 소매 매출 분석 보고서")
st.caption(
    f"분석 기간 {daily['date'].min():%Y-%m-%d} ~ {daily['date'].max():%Y-%m-%d} ({len(daily)}일) · "
    "데이터: UCI Online Retail (CC BY 4.0) · 금액 단위는 원본 기준(£)"
)

total = daily["net_sales"].sum()
gross = daily["gross_sales"].sum()
cancel = daily["cancel_amount"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("순매출 합계", won(total))
c2.metric("영업일 평균 일매출", won(open_days["net_sales"].mean()))
c3.metric("영업일 / 휴무일", f"{len(open_days)}일 / {int(daily['is_closed'].sum())}일")
c4.metric("취소 차감률", f"{-cancel / gross:.1%}", help="취소 금액 ÷ 정상 판매 금액")

# ---------------------------------------------------------------- 요약
full = monthly[monthly["is_partial_month"] == 0]
wk = weekday.dropna(subset=["mean_sales"]).set_index("weekday")

st.header("핵심 요약")
lines = []
if len(full) >= 2:
    first, last = full.iloc[0], full.iloc[-1]
    peak = full.loc[full["net_sales_total"].idxmax()]
    lines.append(
        f"- **월별 흐름**: 영업일 평균 일매출이 {first['year_month']} {won(first['avg_per_open_day'])}에서 "
        f"{last['year_month']} {won(last['avg_per_open_day'])}로 **{last['avg_per_open_day'] / first['avg_per_open_day'] - 1:+.0%}** 변했습니다. "
        f"월 매출은 {peak['year_month']}가 {won(peak['net_sales_total'])}로 가장 큽니다. (완전한 달만 비교)"
    )
else:
    lines.append("- 완전한 달이 2개 미만이라 월별 흐름 요약은 생략했습니다. 기간을 넓혀 보세요.")
if len(wk) >= 2:
    best, worst = wk["mean_sales"].idxmax(), wk["mean_sales"].idxmin()
    lines.append(
        f"- **요일 패턴**: 평균 매출은 **{best}요일** ({won(wk.loc[best, 'mean_sales'])})이 가장 높고, "
        f"**{worst}요일** ({won(wk.loc[worst, 'mean_sales'])})이 가장 낮습니다. "
        f"{worst}요일은 전체 평균의 약 {wk.loc[worst, 'index_vs_overall']:.0%} 수준입니다."
    )
if "cv" in wk and wk["cv"].notna().any():
    lines.append(
        f"- **변동성**: 일매출의 변동계수는 요일별 {wk['cv'].min():.2f}–{wk['cv'].max():.2f}입니다."
    )
lines.append("- 비교는 휴무일 수 차이를 없애기 위해 **영업일당 평균**이 적절합니다.")
st.markdown("\n".join(lines))

tab_pre, tab_trend, tab_month, tab_week, tab_dec, tab_aov, tab_fc, tab_note = st.tabs(
    ["데이터 준비", "일별 추이", "월별 매출", "요일별 매출", "추세·요일 분해", "주문당 평균 금액", "예측 모형 비교", "해석과 한계"]
)

# ---------------------------------------------------------------- 데이터 준비
with tab_pre:
    st.subheader("전처리 결과")
    st.write(
        "원본 541,909행에서 아래 순서대로 제외해 529,591행을 남겼습니다. "
        "취소 행은 제외하지 않고 취소가 일어난 날짜의 매출에서 차감했습니다."
    )
    view = excl.rename(
        columns={
            "step": "단계", "rule": "규칙", "excluded_rows": "제외 행수", "excluded_amount": "제외 금액",
            "remaining_rows": "남은 행수", "remaining_amount": "남은 금액",
        }
    )
    st.dataframe(
        view.style.format({"제외 행수": "{:,.0f}", "제외 금액": "{:,.2f}", "남은 행수": "{:,.0f}", "남은 금액": "{:,.2f}"}),
        hide_index=True, width='stretch',
    )
    st.markdown(
        "- 일별 테이블은 373일 전체를 채웠고, 거래가 없는 날은 0으로 두었습니다.\n"
        "- 휴무일(원본에 행이 없는 날): 모든 토요일, 연말연시, 영국 공휴일 등 69일\n"
        "- 마지막 날(2011-12-09)은 12:50까지만 기록돼 제외했습니다."
    )

# ---------------------------------------------------------------- 일별 추이
with tab_trend:
    st.subheader("일별 순매출")
    show_closed = st.checkbox("휴무일 표시", value=False)
    d = daily.copy()
    d["구분"] = d["is_outlier"].map({0: "일반", 1: "이상치"})
    d["구분"] = d["구분"].where(d["is_closed"] == 0, "휴무")
    d["MA7"] = d["net_sales"].where(d["is_closed"] == 0).rolling(7, center=True, min_periods=1).mean()
    base = d if show_closed else d[d["is_closed"] == 0]
    line = alt.Chart(base).mark_line(color=GRAY).encode(
        x=alt.X("date:T", title="날짜"), y=alt.Y("net_sales:Q", title="순매출 (£)")
    )
    pts = alt.Chart(base).mark_circle(size=35).encode(
        x="date:T", y="net_sales:Q",
        color=alt.Color("구분:N", scale=alt.Scale(domain=["일반", "이상치", "휴무"], range=[BLUE, ORANGE, "#999"]), title=None),
        tooltip=[alt.Tooltip("date:T", title="날짜"), alt.Tooltip("net_sales:Q", title="순매출", format=",.0f"), "구분:N"],
    )
    ma = alt.Chart(d).mark_line(color=BLUE, strokeWidth=2.5).encode(x="date:T", y="MA7:Q")
    st.altair_chart((line + pts + ma).properties(height=380).interactive(), width='stretch')
    st.caption("파란 굵은 선: 영업일 기준 7일 이동평균. 이상치는 요일별 로그 매출의 MAD 기준(|z| > 3.5)으로 표시했으며 삭제하지 않았습니다.")

    out = daily[daily["is_outlier"] == 1][["date", "dow", "net_sales", "outlier_z"]].copy()
    out["요일"] = out["dow"].map(dict(enumerate(DOW_ORDER)))
    st.markdown(f"**이상치 {len(out)}일**")
    st.dataframe(
        out[["date", "요일", "net_sales", "outlier_z"]].rename(columns={"date": "날짜", "net_sales": "순매출", "outlier_z": "z"})
        .style.format({"순매출": "{:,.0f}", "z": "{:.2f}", "날짜": "{:%Y-%m-%d}"}),
        hide_index=True, width='stretch',
    )

# ---------------------------------------------------------------- 월별
with tab_month:
    st.subheader("월별 매출")
    metric = st.radio("지표", ["영업일당 평균", "월 합계"], horizontal=True)
    col, label = ("avg_per_open_day", "영업일당 평균 매출 (£)") if metric == "영업일당 평균" else ("net_sales_total", "월 순매출 (£)")
    m = monthly.copy()
    m["상태"] = m["is_partial_month"].map({0: "완전한 달", 1: "부분 월(기간 일부만 포함)"})
    st.altair_chart(
        alt.Chart(m).mark_bar().encode(
            x=alt.X("year_month:O", title="월"), y=alt.Y(f"{col}:Q", title=label),
            color=alt.Color("상태:N", scale=alt.Scale(domain=["완전한 달", "부분 월(기간 일부만 포함)"], range=[BLUE, GRAY]), title=None),
            tooltip=["year_month", alt.Tooltip(f"{col}:Q", format=",.0f"), "n_open_days"],
        ).properties(height=360),
        width='stretch',
    )
    st.dataframe(
        m.rename(columns={
            "year_month": "월", "net_sales_total": "순매출 합계", "n_days": "일수", "n_open_days": "영업일",
            "avg_per_day": "일평균", "avg_per_open_day": "영업일평균",
        })[["월", "순매출 합계", "일수", "영업일", "일평균", "영업일평균", "상태"]]
        .style.format({"순매출 합계": "{:,.0f}", "일평균": "{:,.0f}", "영업일평균": "{:,.0f}"}),
        hide_index=True, width='stretch',
    )
    st.info("2011-12는 8일까지만 포함된 부분 월이므로 다른 달의 합계와 직접 비교하지 않습니다.")

# ---------------------------------------------------------------- 요일별
with tab_week:
    st.subheader("요일별 매출")
    w = weekday.dropna(subset=["mean_sales"]).copy()
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            alt.Chart(w).mark_bar(color=BLUE).encode(
                x=alt.X("weekday:N", sort=DOW_ORDER, title="요일", axis=alt.Axis(labelAngle=0)), y=alt.Y("mean_sales:Q", title="평균 일매출 (£)"),
                tooltip=["weekday", alt.Tooltip("mean_sales:Q", format=",.0f"), alt.Tooltip("median_sales:Q", format=",.0f")],
            ).properties(height=320, title="평균 일매출"),
            width='stretch',
        )
    with right:
        st.altair_chart(
            alt.Chart(w).mark_bar(color=ORANGE).encode(
                x=alt.X("weekday:N", sort=DOW_ORDER, title="요일", axis=alt.Axis(labelAngle=0)), y=alt.Y("share_of_total:Q", title="매출 비중", axis=alt.Axis(format="%")),
                tooltip=["weekday", alt.Tooltip("share_of_total:Q", format=".1%")],
            ).properties(height=320, title="전체 매출에서 차지하는 비중"),
            width='stretch',
        )
    period = w.melt(id_vars="weekday", value_vars=["mean_sales_jan_aug", "mean_sales_sep_dec"], var_name="기간", value_name="평균")
    period["기간"] = period["기간"].map({"mean_sales_jan_aug": "2011-09 이전", "mean_sales_sep_dec": "2011-09 이후"})
    st.altair_chart(
        alt.Chart(period).mark_bar().encode(
            x=alt.X("weekday:N", sort=DOW_ORDER, title="요일", axis=alt.Axis(labelAngle=0)), xOffset="기간:N", y=alt.Y("평균:Q", title="평균 일매출 (£)"),
            color=alt.Color("기간:N", scale=alt.Scale(range=[GRAY, BLUE])),
        ).properties(height=320, title="기간별 요일 평균 (전 요일이 함께 상승)"),
        width='stretch',
    )
    st.dataframe(
        w[["weekday", "n_open_days", "mean_sales", "median_sales", "std_sales", "mean_invoices", "mean_customers", "avg_invoice_value", "cv", "index_vs_overall"]]
        .rename(columns={
            "weekday": "요일", "n_open_days": "영업일수", "mean_sales": "평균", "median_sales": "중앙값", "std_sales": "표준편차",
            "mean_invoices": "평균 주문수", "mean_customers": "평균 고객수", "avg_invoice_value": "주문당 금액",
            "cv": "변동계수", "index_vs_overall": "전체 대비 지수",
        }).style.format({
            "평균": "{:,.0f}", "중앙값": "{:,.0f}", "표준편차": "{:,.0f}", "평균 주문수": "{:.1f}", "평균 고객수": "{:.1f}",
            "주문당 금액": "{:,.0f}", "변동계수": "{:.2f}", "전체 대비 지수": "{:.2f}",
        }),
        hide_index=True, width='stretch',
    )

# ---------------------------------------------------------------- 분해
with tab_dec:
    st.subheader("추세 + 요일 효과 + 잔차 (덧셈 모형)")
    st.write("영업일만 대상으로 7일 창 평균으로 추세를 구하고, 요일 효과는 추세를 뺀 값의 요일 평균으로 추정했습니다. 잔차는 나머지입니다.")
    comp = decomp[decomp["is_closed"] == 0].melt(
        id_vars="date", value_vars=["observed", "trend", "weekday_effect", "residual"], var_name="성분", value_name="값"
    )
    names = {"observed": "관측값", "trend": "추세", "weekday_effect": "요일 효과", "residual": "잔차"}
    comp["성분"] = comp["성분"].map(names)
    for name in names.values():
        st.altair_chart(
            alt.Chart(comp[comp["성분"] == name]).mark_line(color=BLUE).encode(
                x=alt.X("date:T", title="날짜"), y=alt.Y("값:Q", title=f"{name} (£)"),
            ).properties(height=150),
            width="stretch",
        )
    e = effect.dropna().copy()
    e["방향"] = (e["effect_decomposition"] >= 0).map({True: "평균보다 높음", False: "평균보다 낮음"})
    st.altair_chart(
        alt.Chart(e).mark_bar().encode(
            x=alt.X("weekday:N", sort=DOW_ORDER, title="요일", axis=alt.Axis(labelAngle=0)), y=alt.Y("effect_decomposition:Q", title="요일 효과 (£)"),
            color=alt.Color("방향:N", scale=alt.Scale(domain=["평균보다 높음", "평균보다 낮음"], range=[BLUE, ORANGE]), title=None),
            tooltip=["weekday", alt.Tooltip("effect_decomposition:Q", format=",.0f")],
        ).properties(height=300, title="요일 효과 (분해 기준)"),
        width='stretch',
    )
    if not e.empty:
        hi, lo = e.loc[e["effect_decomposition"].idxmax()], e.loc[e["effect_decomposition"].idxmin()]
        st.markdown(
            f"선택 기간에서 요일 효과는 **{hi['weekday']}요일이 {hi['effect_decomposition']:+,.0f}**로 가장 높고, "
            f"**{lo['weekday']}요일이 {lo['effect_decomposition']:+,.0f}**로 가장 낮습니다."
        )

    st.markdown("#### 월별 분해 그래프")
    in_months = set(daily["year_month"])
    files = [f for f in sorted((FIGS / "decomposition_by_month").glob("decomposition_*.png"))
             if f.stem.replace("decomposition_", "") in in_months]
    if files:
        pick = st.selectbox("월 선택", [f.stem.replace("decomposition_", "") for f in files])
        st.image(str(FIGS / "decomposition_by_month" / f"decomposition_{pick}.png"), width='stretch')
    else:
        st.caption("선택 기간에 해당하는 월별 그래프가 없습니다.")

# ---------------------------------------------------------------- 주문당 평균 금액
with tab_aov:
    st.subheader("주문당 평균 금액 분해")
    st.write("주문당 평균 금액 = 정상 판매 금액 ÷ 정상 판매 송장 수. 매출 분해와 같은 방식(추세 + 요일 효과 + 잔차)입니다.")
    ao = aov[aov["is_closed"] == 0]
    st.metric("선택 기간 주문당 평균 금액", won(daily["gross_sales"].sum() / daily["n_invoices"].sum()), help="기간 전체 정상 판매 금액 ÷ 송장 수")
    aname = {"observed": "관측값", "trend": "추세", "weekday_effect": "요일 효과", "residual": "잔차"}
    am = ao.melt(id_vars="date", value_vars=list(aname), var_name="성분", value_name="값")
    am["성분"] = am["성분"].map(aname)
    for name in aname.values():
        st.altair_chart(
            alt.Chart(am[am["성분"] == name]).mark_line(color=BLUE).encode(
                x=alt.X("date:T", title="날짜"), y=alt.Y("값:Q", title=f"{name} (£/주문)"),
            ).properties(height=150),
            width="stretch",
        )
    res = ao["residual"]
    mad = (aov_all.loc[aov_all["is_closed"] == 0, "residual"].pipe(lambda r: (r - r.median()).abs().median() * 1.4826))
    med = aov_all.loc[aov_all["is_closed"] == 0, "residual"].median()
    chk = ao.assign(z=(res - med) / mad)
    chk = chk[chk["z"].abs() > 2.5].merge(
        daily[["date", "n_invoices", "quantity", "gross_sales", "cancel_amount"]], on="date"
    ).sort_values("z", key=abs, ascending=False)
    chk["송장당 수량"] = chk["quantity"] / chk["n_invoices"]
    st.markdown(f"**잔차 점검일 (전체 기간 기준 |z| > 2.5): {len(chk)}일**")
    st.dataframe(
        chk[["date", "observed", "trend", "residual", "z", "n_invoices", "송장당 수량", "cancel_amount"]].rename(columns={
            "date": "날짜", "observed": "주문당 금액", "trend": "추세", "residual": "잔차", "z": "z",
            "n_invoices": "송장수", "cancel_amount": "취소 금액",
        }).style.format({"날짜": "{:%Y-%m-%d}", "주문당 금액": "{:,.0f}", "추세": "{:,.0f}", "잔차": "{:+,.0f}",
                         "z": "{:+.1f}", "송장수": "{:,.0f}", "송장당 수량": "{:,.0f}", "취소 금액": "{:,.0f}"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        "정상 판매 금액 기준이라 같은 날 취소된 주문도 포함됩니다. 예: 2011-01-18은 취소 −£77,297이 같은 날 발생해 "
        "주문당 금액이 £2,359로 튑니다(순매출 기준으로는 £17,067)."
    )

# ---------------------------------------------------------------- 예측 모형 비교
with tab_fc:
    st.subheader("예측 모형 비교: SES · Holt · Holt-Winters · SARIMA")

    def _tbl(name):
        return pd.read_csv(TABLES / name)

    def _param(df, key):
        v = df.loc[df["name"] == key, "value"]
        return v.iloc[0] if len(v) else None

    def _errors(actual, pred):
        e = actual - pred
        return {"MAE": e.abs().mean(), "RMSE": (e ** 2).mean() ** 0.5, "MAPE(%)": (e.abs() / actual).mean() * 100, "ME(실제-예측)": e.mean()}

    MODEL_COL = {"SES": "ses_forecast", "Holt": "holt_forecast", "Holt-Winters": "hw_forecast",
                 "SARIMA": "sarima_forecast", "계절 naive(기준선)": "seasonal_naive_forecast"}
    MODEL_COLOR = {"SES": ORANGE, "Holt": "#8a8984", "Holt-Winters": BLUE, "SARIMA": "#2f9e44", "계절 naive(기준선)": "#c9a227"}

    needed = ("ses_parameters.csv", "holt_parameters.csv", "holt_winters_parameters.csv", "sarima_parameters.csv",
              "holt_winters_forecast_november.csv", "sarima_forecast_november.csv",
              "december_forecast.csv", "december_forecast_sarima.csv", "december_forecast_parameters.csv")
    missing = [n for n in needed if not (TABLES / n).exists()]
    if missing:
        st.warning(f"예측 결과 파일이 없습니다: {', '.join(missing)}. analysis/09~15 스크립트를 먼저 실행하세요.")
    else:
        # ---- 공통 선택: 모형, SARIMA 예측구간
        c_a, c_b = st.columns([3, 1])
        with c_a:
            sel = st.multiselect("비교할 모형", list(MODEL_COL), default=["SES", "Holt", "Holt-Winters", "SARIMA"], key="fc_models")
        with c_b:
            show_band = st.checkbox("SARIMA 95% 예측구간 표시", value=True, key="fc_band")
        st.caption("모형과 기간 선택은 이 탭에만 적용됩니다(사이드바의 분석 기간과 별개). 학습·검증 구간 자체는 고정입니다.")

        # ---- 1. 학습·검증 구간
        d_all = daily_all
        split_rows = []
        for name, a, b in [("학습", "2011-01-01", "2011-10-31"), ("검증", "2011-11-01", "2011-11-30")]:
            x = d_all[(d_all["date"] >= a) & (d_all["date"] <= b)]
            split_rows.append({"구분": name, "시작": a, "종료": b, "달력 일수": len(x),
                               "영업일": int((x["is_closed"] == 0).sum()), "휴무일": int((x["is_closed"] == 1).sum()),
                               "영업일 평균 일매출": x.loc[x["is_closed"] == 0, "net_sales"].mean()})
        st.markdown("#### 1. 학습·검증 구간")
        st.dataframe(pd.DataFrame(split_rows).style.format({"영업일 평균 일매출": "£{:,.0f}"}), hide_index=True, width="stretch")
        st.markdown(
            "- 학습 구간은 검증 구간보다 앞에 있으며, 계수·초기값·차수·계절 형태는 학습 자료로만 정했습니다.\n"
            "- SES·Holt는 영업일만 이어 붙인 시계열에, Holt-Winters와 SARIMA는 토요일을 뺀 6일 주기 시계열에 적용했습니다. "
            "토요일 외 휴무일 9일은 같은 요일 전·후 주 평균으로 채워 적합에만 썼습니다.\n"
            "- 검증 구간은 학습 구간보다 영업일 평균 매출이 약 1.9배 높아 모든 모형의 예측이 낮게 나옵니다."
        )

        # ---- 2. 계수
        st.markdown("#### 2. 추정된 평활계수·초기값·차수")
        ps, ph, pw, pa = (_tbl("ses_parameters.csv"), _tbl("holt_parameters.csv"),
                          _tbl("holt_winters_parameters.csv"), _tbl("sarima_parameters.csv"))
        ptab = pd.DataFrame([
            {"모형": "SES", "α": _param(ps, "alpha"), "β": None, "γ": None, "초기 수준": _param(ps, "initial_level"), "초기 추세": None, "계절성": "-"},
            {"모형": "Holt", "α": _param(ph, "holt_alpha"), "β": _param(ph, "holt_beta"), "γ": None,
             "초기 수준": _param(ph, "holt_initial_level"), "초기 추세": _param(ph, "holt_initial_trend"), "계절성": "-"},
            {"모형": "Holt-Winters", "α": _param(pw, "alpha"), "β": _param(pw, "beta"), "γ": _param(pw, "gamma"),
             "초기 수준": _param(pw, "initial_level"), "초기 추세": _param(pw, "initial_trend"),
             "계절성": {"mul": "곱셈", "add": "덧셈"}.get(_param(pw, "selected_seasonal"), "-")},
        ])
        for c in ("α", "β", "γ", "초기 수준", "초기 추세"):
            ptab[c] = pd.to_numeric(ptab[c])
        st.dataframe(
            ptab.style.format({"α": "{:.4f}", "β": "{:.4f}", "γ": "{:.4f}", "초기 수준": "£{:,.0f}", "초기 추세": "{:+,.1f}"}, na_rep="-"),
            hide_index=True, width="stretch",
        )
        st.caption("β=0, γ=0은 추세·요일 지수를 학습 내내 초기값으로 고정하고 수준만 갱신했다는 뜻입니다.")
        log_on = int(float(_param(pa, "log_transform"))) == 1
        st.markdown(
            f"**SARIMA** {_param(pa, 'order')} × {_param(pa, 'seasonal_order')}"
            f"{' · 로그 변환 후 적합, 예측은 지수로 복원' if log_on else ''}  \n"
            f"차수는 학습 구간 안 내부 검증(2011-10 예측 MAE £{float(_param(pa, 'inner_MAE')):,.0f})으로 고른 뒤 학습 전체로 다시 적합했습니다. "
            f"잔차 Ljung-Box p값(lag 12) {float(_param(pa, 'ljung_box_p_lag12')):.2f}(크면 잔차 자기상관이 없다는 뜻), "
            f"검증 실제값의 {float(_param(pa, 'interval95_coverage')):.0%}가 95% 예측구간 안에 들었습니다."
        )
        coefs = pa[pa["name"].str.startswith("coef_")].assign(계수=lambda d: d["name"].str.replace("coef_", "", regex=False))[["계수", "value"]].rename(columns={"value": "추정값"})
        coefs["추정값"] = pd.to_numeric(coefs["추정값"])
        with st.expander("SARIMA 추정 계수"):
            st.dataframe(coefs.style.format({"추정값": "{:.4f}"}), hide_index=True, width="stretch")
            st.caption("계절 AR(≈0.99)과 계절 MA(≈−1.04)가 서로 거의 상쇄되는 형태라, 계수 자체를 해석하기보다 예측 결과로 평가하는 것이 안전합니다.")

        # ---- 3. 11월 예측 vs 실제
        st.markdown("#### 3. 11월 예측 vs 실제")
        nov = _tbl("holt_winters_forecast_november.csv").merge(_tbl("sarima_forecast_november.csv")[["date", "sarima_forecast", "lower95", "upper95"]], on="date")
        nov["date"] = pd.to_datetime(nov["date"])
        n_lo, n_hi = nov["date"].min().date(), nov["date"].max().date()
        n_rng = st.date_input("11월 비교 기간", value=(n_lo, n_hi), min_value=n_lo, max_value=n_hi, key="fc_nov_range")
        if len(n_rng) != 2:
            st.info("종료일까지 선택하세요.")
        elif not sel:
            st.info("비교할 모형을 하나 이상 선택하세요.")
        else:
            nv = nov[(nov["date"] >= pd.Timestamp(n_rng[0])) & (nov["date"] <= pd.Timestamp(n_rng[1]))]
            long = nv.melt(id_vars="date", value_vars=["actual"] + [MODEL_COL[m] for m in sel], var_name="key", value_name="값")
            inv = {"actual": "실제", **{MODEL_COL[m]: m for m in sel}}
            long["계열"] = long["key"].map(inv)
            domain = ["실제"] + sel
            rng_colors = ["#0b0b0b"] + [MODEL_COLOR[m] for m in sel]
            layers = []
            if show_band and "SARIMA" in sel:
                layers.append(alt.Chart(nv).mark_area(opacity=0.12, color=MODEL_COLOR["SARIMA"]).encode(
                    x="date:T", y="lower95:Q", y2="upper95:Q"))
            layers.append(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("date:T", title="날짜"), y=alt.Y("값:Q", title="일 순매출 (£)"),
                color=alt.Color("계열:N", scale=alt.Scale(domain=domain, range=rng_colors), title=None),
                tooltip=[alt.Tooltip("date:T", title="날짜"), "계열:N", alt.Tooltip("값:Q", format=",.0f")]))
            st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")

            # ---- 4. 검증 오차 비교 (선택한 모형·기간)
            st.markdown("#### 4. 검증 오차 비교")
            st.caption(f"선택한 기간 {n_rng[0]} ~ {n_rng[1]}의 영업일 {len(nv)}일 기준으로 다시 계산한 값입니다.")
            if len(nv) < 5:
                st.warning("영업일이 5일 미만이면 오차 지표가 불안정합니다. 기간을 넓혀 보세요.")
            rows = [{"모형": m, **_errors(nv["actual"], nv[MODEL_COL[m]])} for m in sel]
            met = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
            st.dataframe(
                met.style.format({"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "MAPE(%)": "{:.1f}%", "ME(실제-예측)": "{:+,.0f}"}),
                hide_index=True, width="stretch",
            )
            st.altair_chart(
                alt.Chart(met).mark_bar().encode(
                    y=alt.Y("모형:N", sort=list(met["모형"]), title=None),
                    x=alt.X("MAE:Q", title="MAE (£, 낮을수록 좋음)"),
                    color=alt.Color("모형:N", scale=alt.Scale(domain=list(MODEL_COLOR), range=list(MODEL_COLOR.values())), legend=None),
                    tooltip=["모형", alt.Tooltip("MAE:Q", format=",.0f"), alt.Tooltip("RMSE:Q", format=",.0f"), alt.Tooltip("MAPE(%):Q", format=".1f")],
                ).properties(height=40 + 38 * len(met)),
                width="stretch",
            )
            if len(met) >= 2:
                b = met.iloc[0]
                st.markdown(
                    f"선택한 모형 중 **{b['모형']}**의 MAE가 £{b['MAE']:,.0f}(MAPE {b['MAPE(%)']:.1f}%)로 가장 작습니다. "
                    "요일 계절성을 반영하는 Holt-Winters·SARIMA가 SES·Holt보다 낫고, 모든 모형이 11월의 수준 상승과 "
                    "대형 주문일(11-14 등)은 따라가지 못합니다."
                )
            with st.expander("일별 예측·오차 (선택 기간)"):
                e = nv[["date", "actual"] + [MODEL_COL[m] for m in sel]].rename(columns={"date": "날짜", "actual": "실제", **{MODEL_COL[m]: m for m in sel}})
                fmt = {"날짜": "{:%Y-%m-%d}", "실제": "{:,.0f}", **{m: "{:,.0f}" for m in sel}}
                st.dataframe(e.style.format(fmt), hide_index=True, width="stretch")

        # ---- 5. 12월 예측
        st.markdown("#### 5. 12월 예측 (1~11월 전체로 재학습)")
        dec = _tbl("december_forecast.csv").merge(_tbl("december_forecast_sarima.csv"), on="date")
        dec["date"] = pd.to_datetime(dec["date"])
        st.write(
            "1~11월 전체(영업일 277일)로 모형을 다시 적합해 12-01부터 12-23까지(토요일 제외 20일) 예측했습니다. "
            "SARIMA는 차수를 다시 고르지 않고 11월 비교에서 정한 차수를 그대로 쓰며 계수만 다시 추정했습니다. "
            "계절 naive 기준선은 12월에는 계산하지 않았습니다."
        )
        d_lo, d_hi = dec["date"].min().date(), dec["date"].max().date()
        d_rng = st.date_input("12월 예측 기간", value=(d_lo, d_hi), min_value=d_lo, max_value=d_hi, key="fc_dec_range")
        dsel = [m for m in sel if m != "계절 naive(기준선)"]
        if len(d_rng) != 2:
            st.info("종료일까지 선택하세요.")
        elif not dsel:
            st.info("비교할 모형을 하나 이상 선택하세요(계절 naive는 12월 제외).")
        else:
            dv = dec[(dec["date"] >= pd.Timestamp(d_rng[0])) & (dec["date"] <= pd.Timestamp(d_rng[1]))]
            hist = daily_all[(daily_all["date"] >= "2011-09-01") & (daily_all["is_closed"] == 0)][["date", "net_sales"]].rename(columns={"net_sales": "값"})
            hist["계열"] = "실제(9~11월)"
            dact = daily_all[(daily_all["date"] >= "2011-12-01") & (daily_all["is_closed"] == 0)][["date", "net_sales"]].rename(columns={"net_sales": "값"})
            dact["계열"] = "실제(12월)"
            dl = dv.melt(id_vars="date", value_vars=[MODEL_COL[m] for m in dsel], var_name="key", value_name="값")
            dl["계열"] = dl["key"].map({MODEL_COL[m]: m for m in dsel})
            allc = pd.concat([hist, dact, dl[["date", "값", "계열"]]])
            domain = ["실제(9~11월)", "실제(12월)"] + dsel
            rng_colors = [GRAY, "#0b0b0b"] + [MODEL_COLOR[m] for m in dsel]
            layers = []
            if show_band and "SARIMA" in dsel:
                layers.append(alt.Chart(dv).mark_area(opacity=0.12, color=MODEL_COLOR["SARIMA"]).encode(x="date:T", y="lower95:Q", y2="upper95:Q"))
            layers.append(alt.Chart(allc).mark_line(point=alt.OverlayMarkDef(size=25)).encode(
                x=alt.X("date:T", title="날짜"), y=alt.Y("값:Q", title="일 순매출 (£)"),
                color=alt.Color("계열:N", scale=alt.Scale(domain=domain, range=rng_colors), title=None),
                tooltip=[alt.Tooltip("date:T", title="날짜"), "계열:N", alt.Tooltip("값:Q", format=",.0f")]))
            st.altair_chart(alt.layer(*layers).properties(height=380), width="stretch")

            known = dv.dropna(subset=["actual"])
            if len(known):
                ref = pd.DataFrame([{"모형": m, **_errors(known["actual"], known[MODEL_COL[m]])} for m in dsel]).sort_values("MAE")
                st.markdown(f"**참고: 실제값이 있는 12월 영업일 {len(known)}일({known['date'].min():%m-%d}~{known['date'].max():%m-%d}) 비교**")
                st.dataframe(ref.style.format({"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "MAPE(%)": "{:.1f}%", "ME(실제-예측)": "{:+,.0f}"}),
                             hide_index=True, width="stretch")
            else:
                st.caption("선택한 기간에는 실제값이 있는 날이 없어 오차 표를 생략합니다(실제값은 12-08까지).")
            with st.expander("12월 일별 예측"):
                cols = ["date"] + [MODEL_COL[m] for m in dsel] + ["actual"]
                view = dv[cols].rename(columns={"date": "날짜", "actual": "실제", **{MODEL_COL[m]: m for m in dsel}})
                fmt = {"날짜": "{:%Y-%m-%d}", "실제": "{:,.0f}", **{m: "{:,.0f}" for m in dsel}}
                st.dataframe(view.style.format(fmt, na_rep="-"), hide_index=True, width="stretch")
        st.warning(
            "12월 예측은 12-24 이후를 포함하지 않습니다. 2010년에는 12-24부터 다음 해 01-03까지 연속 휴무였고, 2011년 휴무 일정은 알 수 없습니다. "
            "또 학습 자료가 1년 미만이라 연말 성수기·휴무 같은 연간 계절성은 모형이 알지 못합니다. "
            "실제값이 없는 구간의 예측은 검증되지 않았으므로 참고용으로만 보세요."
        )

# ---------------------------------------------------------------- 해석과 한계
with tab_note:
    st.subheader("해석")
    st.markdown(
        """
- 매출 수준은 2011년 상반기에 완만하다가 **9월부터 뚜렷이 올라 11월에 정점**입니다. 모든 요일이 함께 오른 것으로 보아 특정 요일이 아닌 전반적 수요 증가입니다.
- 주간 리듬(화·목 강세, 일요일 약세)은 기간과 무관하게 유지됩니다.
- 이상치는 대개 대형 주문이나 공휴일 전후 몰림 등으로 추정되나, 이 분석은 원인을 확인하지 않았습니다.
"""
    )
    st.subheader("한계")
    st.markdown(
        """
- 데이터가 약 1년이라 **연간 계절성(연말 성수기 반복 여부)은 판단할 수 없습니다.** 9–11월 증가가 계절 요인인지 성장인지 구분되지 않습니다.
- 취소는 원 주문과 짝짓지 않고 취소 발생일에 차감했으므로, 일별 값에는 이전 주문의 취소가 섞입니다.
- `CustomerID` 결측 행(130,543건)은 매출에 포함되지만 고객 수 지표에는 반영되지 않습니다.
- 예측은 지수평활 계열(SES·Holt·Holt-Winters)과 SARIMA만 비교했으며, 외부 요인(프로모션·재고 등)은 다루지 않았습니다.
"""
    )
