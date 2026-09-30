"""Holt 선형추세 모형(감쇠 없음)으로 2011-11 일 순매출을 예측하고 단순 지수평활(SES)과 비교한다.

학습: 2011-01-01 ~ 2011-10-31 (영업일 251일)
검증: 2011-11-01 ~ 2011-11-30 (영업일 26일)  -- 학습 단계에서 사용하지 않는다.

방법
  - 09_ses_forecast.py와 같이 영업일만 이어 붙인 시계열에 적용한다(휴무일 제외).
  - Holt: 수준 l_t = a*y_t + (1-a)(l_{t-1}+b_{t-1}), 추세 b_t = B*(l_t-l_{t-1}) + (1-B)*b_{t-1}
    h단계 예측 = l_T + h*b_T. 감쇠 추세는 쓰지 않는다.
  - 평활계수(alpha, beta)와 초기값(l0, b0)은 학습 자료의 제곱오차합(SSE)을 최소화해 자동 추정한다.

입력: data/processed/daily_sales.csv
출력: outputs/tables/holt_parameters.csv
      outputs/tables/holt_ses_forecast_november.csv
      outputs/tables/ses_holt_metrics.csv
      outputs/figures/holt_ses_forecast_november.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
PARAM_PATH = ROOT / "outputs" / "tables" / "holt_parameters.csv"
FCST_PATH = ROOT / "outputs" / "tables" / "holt_ses_forecast_november.csv"
METRIC_PATH = ROOT / "outputs" / "tables" / "ses_holt_metrics.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "holt_ses_forecast_november.png"

TRAIN = ("2011-01-01", "2011-10-31")
VALID = ("2011-11-01", "2011-11-30")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
TRAIN_COLOR = "#a9b8cc"
SES_COLOR = "#eb6834"
HOLT_COLOR = "#2a78d6"
BASELINE = "#8a8984"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def errors(actual: pd.Series, pred: pd.Series) -> dict:
    e = actual - pred
    return {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE(%)": float((e.abs() / actual).mean() * 100),
        "ME(실제-예측)": float(e.mean()),
    }


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    train = daily.loc[TRAIN[0]:TRAIN[1]]
    valid = daily.loc[VALID[0]:VALID[1]]
    y_train = train.loc[train["is_closed"] == 0, "net_sales"]
    y_valid = valid.loc[valid["is_closed"] == 0, "net_sales"]
    assert len(y_train) == 251 and len(y_valid) == 26
    h = len(y_valid)

    ses = SimpleExpSmoothing(y_train.to_numpy(), initialization_method="estimated").fit(optimized=True)
    holt = ExponentialSmoothing(
        y_train.to_numpy(), trend="add", damped_trend=False, seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    p = holt.params
    alpha, beta = float(p["smoothing_level"]), float(p["smoothing_trend"])
    l0, b0 = float(p["initial_level"]), float(p["initial_trend"])
    steps = np.arange(1, h + 1)
    # 학습 마지막 시점의 수준과 추세: 1단계 예측 = l_T + b_T 이므로 이를 이용해 분리한다.
    fc_holt = pd.Series(holt.forecast(h), index=y_valid.index)
    b_T = float(fc_holt.iloc[1] - fc_holt.iloc[0])
    l_T = float(fc_holt.iloc[0] - b_T)
    fc_ses = pd.Series(np.full(h, float(ses.forecast(1)[0])), index=y_valid.index)
    fit_holt = pd.Series(holt.fittedvalues, index=y_train.index)
    fit_ses = pd.Series(ses.fittedvalues, index=y_train.index)

    # 검산: h단계 예측 = l_T + h*b_T
    assert np.allclose(fc_holt.to_numpy(), l_T + steps * b_T)

    m_ses, m_holt = errors(y_valid, fc_ses), errors(y_valid, fc_holt)
    sse = lambda fit: float(((y_train - fit) ** 2).sum())

    params = pd.DataFrame([
        ("holt_alpha", alpha, "Holt 수준 평활계수"),
        ("holt_beta", beta, "Holt 추세 평활계수"),
        ("holt_initial_level", l0, "Holt 초기 수준 l0"),
        ("holt_initial_trend", b0, "Holt 초기 추세 b0 (영업일당 변화량)"),
        ("holt_last_level", l_T, "학습 마지막 영업일의 수준 l_T"),
        ("holt_last_trend", b_T, "학습 마지막 영업일의 추세 b_T (영업일당)"),
        ("holt_train_sse", sse(fit_holt), "학습 1단계 예측 제곱오차합"),
        ("ses_alpha", float(ses.params["smoothing_level"]), "비교용 SES 평활계수"),
        ("ses_initial_level", float(ses.params["initial_level"]), "비교용 SES 초기 수준"),
        ("ses_train_sse", sse(fit_ses), "비교용 SES 학습 제곱오차합"),
    ], columns=["name", "value", "note"])
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    params.to_csv(PARAM_PATH, index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame({"SES": m_ses, "Holt": m_holt}).T.reset_index(names="model")
    metrics.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")

    out = pd.DataFrame({
        "actual": y_valid, "ses_forecast": fc_ses, "holt_forecast": fc_holt,
        "ses_error": y_valid - fc_ses, "holt_error": y_valid - fc_holt,
    })
    out.reset_index().to_csv(FCST_PATH, index=False, encoding="utf-8-sig")

    plot(y_train, fit_holt, y_valid, fc_ses, fc_holt, alpha, beta, l0, b0, b_T, m_ses, m_holt)

    pd.set_option("display.width", 200)
    print(f"Holt alpha={alpha:.6f} beta={beta:.6f} l0={l0:,.2f} b0={b0:,.4f}")
    print(f"마지막 수준 l_T={l_T:,.2f}  추세 b_T={b_T:,.4f}/영업일")
    print(f"SES alpha={float(ses.params['smoothing_level']):.6f}")
    print(f"학습 SSE  SES={sse(fit_ses):,.0f}  Holt={sse(fit_holt):,.0f}")
    print(metrics.round(1).to_string(index=False))
    print(f"Holt 예측 첫날 {fc_holt.iloc[0]:,.0f} → 마지막날 {fc_holt.iloc[-1]:,.0f}")


def plot(y_train, fit_holt, y_valid, fc_ses, fc_holt, alpha, beta, l0, b0, b_T, m_ses, m_holt) -> None:
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(16, 6.5), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    def style(a):
        a.set_facecolor(SURFACE)
        a.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    style(ax)
    ax.plot(y_train.index, y_train.values, color=TRAIN_COLOR, lw=1.1, label="학습 구간 실제 (2011-01~10)")
    ax.plot(fit_holt.index, fit_holt.values, color=HOLT_COLOR, lw=1.4, alpha=0.6, label="Holt 학습 적합값(1단계)")
    ax.plot(y_valid.index, y_valid.values, color=TEXT_PRIMARY, lw=1.8, marker="o", ms=4, label="11월 실제")
    ax.plot(fc_ses.index, fc_ses.values, color=SES_COLOR, lw=2.4, marker="s", ms=3.5,
            label=f"SES 예측 (평평, £{fc_ses.iloc[0]:,.0f})")
    ax.plot(fc_holt.index, fc_holt.values, color=HOLT_COLOR, lw=2.4, marker="^", ms=4,
            label=f"Holt 예측 (£{fc_holt.iloc[0]:,.0f} → £{fc_holt.iloc[-1]:,.0f})")
    ax.axvline(pd.Timestamp("2011-11-01") - pd.Timedelta(hours=12), color=BASELINE, ls="--", lw=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(pd.Timestamp("2010-12-28"), pd.Timestamp("2011-12-02"))
    ax.set_ylabel("일 순매출 (£)", fontsize=10, color=TEXT_SECONDARY)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    # 오른쪽: 11월만 확대
    style(ax2)
    ax2.plot(y_valid.index, y_valid.values, color=TEXT_PRIMARY, lw=1.8, marker="o", ms=4, label="실제")
    ax2.plot(fc_ses.index, fc_ses.values, color=SES_COLOR, lw=2.2, label="SES")
    ax2.plot(fc_holt.index, fc_holt.values, color=HOLT_COLOR, lw=2.2, label="Holt")
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_title("11월 확대", loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle("Holt(추세, 감쇠 없음) vs 단순 지수평활: 11월 일 매출 예측", x=0.01, y=0.985, ha="left",
                 fontsize=15, color=TEXT_PRIMARY)
    fig.text(0.01, 0.905,
             f"Holt α={alpha:.4f}  β={beta:.4f}  l0=£{l0:,.0f}  b0={b0:+,.1f}/영업일  마지막 추세={b_T:+,.1f}/영업일   |   "
             f"검증 MAE  SES £{m_ses['MAE']:,.0f} / Holt £{m_holt['MAE']:,.0f}   "
             f"RMSE  £{m_ses['RMSE']:,.0f} / £{m_holt['RMSE']:,.0f}   "
             f"MAPE  {m_ses['MAPE(%)']:.1f}% / {m_holt['MAPE(%)']:.1f}%",
             fontsize=10, color=TEXT_SECONDARY)
    fig.text(0.01, 0.005,
             "영업일만 이어 붙인 시계열에 적용(휴무일 제외). 계수와 초기값은 학습 자료로만 자동 추정. "
             "Holt 예측 = 마지막 수준 + 단계 수 × 마지막 추세(직선).",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


if __name__ == "__main__":
    main()
