"""2011-12 일 순매출 예측: 1~11월 전체로 SES·Holt·Holt-Winters를 다시 적합해 12월을 예측한다.

학습: 2011-01-01 ~ 2011-11-30 (11월까지 실제값 모두 사용, 계수·초기값·계절 형태는 이 자료로 다시 자동 추정)
예측: 2011-12-01 ~ 2011-12-23 중 토요일이 아닌 날
  - 2010년에는 12-24 ~ 01-03이 연속 휴무였다. 2011년 연말 휴무일은 알 수 없으므로 12-24 이후는 예측하지 않는다.
  - 2011-12-01 ~ 12-08은 실제값이 있어 참고용으로 비교한다(영업일 7일). 12-09 이후는 실제값이 없다.
모형 구성은 11_holt_winters_forecast.py와 같다.
  - SES·Holt: 영업일만 이어 붙인 시계열, Holt-Winters: 토요일을 뺀 6일 주기 시계열(토요일 외 휴무일은 전·후 주 평균으로 채움)
  - 감쇠 추세 없음, 계절성 형태(덧셈/곱셈)는 학습 AICc로 선택

입력: data/processed/daily_sales.csv
출력: outputs/tables/december_forecast.csv
      outputs/tables/december_forecast_parameters.csv
      outputs/figures/december_forecast.png
"""
import importlib.util
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
FCST_PATH = ROOT / "outputs" / "tables" / "december_forecast.csv"
PARAM_PATH = ROOT / "outputs" / "tables" / "december_forecast_parameters.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "december_forecast.png"

_spec = importlib.util.spec_from_file_location("hw11", ROOT / "analysis" / "11_holt_winters_forecast.py")
hw11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hw11)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TRAIN = ("2011-01-01", "2011-11-30")
FC_START, FC_END = pd.Timestamp("2011-12-01"), pd.Timestamp("2011-12-23")
ACTUAL_END = pd.Timestamp("2011-12-08")


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    tr = daily.loc[TRAIN[0]:TRAIN[1]]
    y_open = tr.loc[tr["is_closed"] == 0, "net_sales"]
    y_slots, filled = hw11.build_series(daily, TRAIN)

    fdates = pd.date_range(FC_START, FC_END)
    fdates = fdates[fdates.dayofweek != 5]
    h = len(fdates)
    assert y_slots.index[-1] == pd.Timestamp("2011-11-30") and fdates[0].dayofweek == 3  # 11-30(수) 다음은 12-01(목)

    ses = SimpleExpSmoothing(y_open.to_numpy(), initialization_method="estimated").fit(optimized=True)
    holt = ExponentialSmoothing(y_open.to_numpy(), trend="add", damped_trend=False,
                                initialization_method="estimated").fit(optimized=True)
    fits, aic = {}, {}
    for seas in ("add", "mul"):
        f = ExponentialSmoothing(y_slots.to_numpy(), trend="add", damped_trend=False, seasonal=seas,
                                 seasonal_periods=hw11.PERIOD, initialization_method="estimated").fit(optimized=True)
        fits[seas], aic[seas] = f, float(f.aicc)
    best = min(aic, key=aic.get)
    hw = fits[best]

    out = pd.DataFrame({
        "ses_forecast": np.full(h, float(ses.forecast(1)[0])),
        "holt_forecast": np.asarray(holt.forecast(h)),
        "hw_forecast": np.asarray(hw.forecast(h)),
    }, index=fdates)
    out.index.name = "date"
    act = daily.loc[FC_START:ACTUAL_END]
    out["actual"] = act.loc[act["is_closed"] == 0, "net_sales"].reindex(out.index)
    out["dow"] = out.index.dayofweek

    known = out.dropna(subset=["actual"])
    metrics = {c: hw11.errors(known["actual"], known[c]) for c in ("ses_forecast", "holt_forecast", "hw_forecast")}

    p = hw.params
    rows = [
        ("train_range", f"{TRAIN[0]} ~ {TRAIN[1]}", "학습 구간"),
        ("n_train_open_days", len(y_open), "학습 영업일 수"),
        ("n_train_slots", len(y_slots), "Holt-Winters 학습 자리 수(토요일 제외 달력일)"),
        ("n_filled_slots", len(filled), "토요일 외 휴무일 채움 건수"),
        ("ses_alpha", float(ses.params["smoothing_level"]), "SES 평활계수"),
        ("ses_initial_level", float(ses.params["initial_level"]), "SES 초기 수준"),
        ("ses_forecast", float(ses.forecast(1)[0]), "SES 예측값(일정)"),
        ("holt_alpha", float(holt.params["smoothing_level"]), "Holt 수준 평활계수"),
        ("holt_beta", float(holt.params["smoothing_trend"]), "Holt 추세 평활계수"),
        ("holt_initial_level", float(holt.params["initial_level"]), "Holt 초기 수준"),
        ("holt_initial_trend", float(holt.params["initial_trend"]), "Holt 초기 추세"),
        ("hw_selected_seasonal", best, "Holt-Winters 계절성(학습 AICc 기준)"),
        ("hw_aicc_add", aic["add"], "덧셈 AICc"),
        ("hw_aicc_mul", aic["mul"], "곱셈 AICc"),
        ("hw_alpha", float(p["smoothing_level"]), "HW 수준 평활계수"),
        ("hw_beta", float(p["smoothing_trend"]), "HW 추세 평활계수"),
        ("hw_gamma", float(p["smoothing_seasonal"]), "HW 계절 평활계수"),
        ("hw_initial_level", float(p["initial_level"]), "HW 초기 수준"),
        ("hw_initial_trend", float(p["initial_trend"]), "HW 초기 추세"),
        ("n_forecast_days", h, "예측한 영업 가능일 수(토요일 제외, 12-23까지)"),
        ("n_known_days", len(known), "실제값이 있는 예측일(12-01~12-08 영업일)"),
    ]
    for m, d in metrics.items():
        for k, v in d.items():
            rows.append((f"ref_{m}_{k}", v, "12-01~12-08 참고 비교"))
    FCST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(FCST_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows, columns=["name", "value", "note"]).to_csv(PARAM_PATH, index=False, encoding="utf-8-sig")

    plot(y_open, daily, out, best)

    pd.set_option("display.width", 200)
    print("채움", len(filled), "AICc", {k: round(v, 1) for k, v in aic.items()}, "선택", best)
    print(f"HW α={p['smoothing_level']:.4f} β={p['smoothing_trend']:.4f} γ={p['smoothing_seasonal']:.4f}")
    print(f"SES α={ses.params['smoothing_level']:.4f} Holt α={holt.params['smoothing_level']:.4f} β={holt.params['smoothing_trend']:.4f}")
    print(pd.DataFrame(metrics).T.round(1).to_string())
    print(out.round(0).to_string())


def plot(y_open, daily, out, best) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    hist = y_open.loc["2011-09-01":]
    ax.plot(hist.index, hist.values, color="#a9b8cc", lw=1.3, label="실제 (2011-09~11)")
    act = daily.loc["2011-12-01":"2011-12-08"]
    act = act.loc[act["is_closed"] == 0, "net_sales"]
    ax.plot(act.index, act.values, color="#0b0b0b", lw=1.8, marker="o", ms=4, label="12월 실제 (12-01~12-08)")
    ax.plot(out.index, out["ses_forecast"], color="#eb6834", lw=1.8, label="SES")
    ax.plot(out.index, out["holt_forecast"], color="#8a8984", lw=1.8, ls="--", label="Holt")
    ax.plot(out.index, out["hw_forecast"], color="#2a78d6", lw=2.4, marker="^", ms=4, label=f"Holt-Winters({best})")
    ax.axvline(pd.Timestamp("2011-11-30 12:00"), color="#8a8984", ls="--", lw=1)
    ax.axvspan(pd.Timestamp("2011-12-24"), pd.Timestamp("2011-12-31"), color="#efeeea", lw=0)
    ax.text(pd.Timestamp("2011-12-24"), ax.get_ylim()[1] * 0.97, " 12-24 이후\n 예측 안 함\n (전년 연말 휴무)", fontsize=8.5,
            color="#52514e", va="top")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlim(pd.Timestamp("2011-09-01"), pd.Timestamp("2011-12-31"))
    ax.set_ylabel("일 순매출 (£)", fontsize=10, color="#52514e")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9, length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.suptitle("2011-12 일 매출 예측 (1~11월 전체로 재학습)", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
