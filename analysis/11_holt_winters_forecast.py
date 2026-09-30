"""Holt-Winters(추세 + 요일 계절성)로 2011-11 일 순매출을 예측하고 SES·Holt와 비교한다.

학습: 2011-01-01 ~ 2011-10-31
검증: 2011-11-01 ~ 2011-11-30  -- 학습 단계에서 사용하지 않는다.

시계열 구성 (계절 주기 6)
  - 토요일은 매주 휴무이므로 시계열에서 뺀다. 남은 6개 요일(일·월·화·수·목·금)이 한 주기(6)가 된다.
  - 토요일이 아닌 휴무일(공휴일 등)은 시계열에서 빼면 요일 위상이 어긋나므로 자리를 유지한다.
    이 날의 값은 학습 자료 안의 같은 요일 전·후 주 값의 평균으로 채운다(모형 적합용, 평가에는 쓰지 않음).
    학습 구간의 채움 건수는 실행 결과와 holt_winters_parameters.csv에 기록한다.
  - 검증 구간(11월)에는 토요일 외 휴무일이 없어 26개 자리가 모두 실제 영업일이다.

모형
  - 추세는 덧셈(감쇠 없음). 계절성은 덧셈/곱셈 두 가지를 적합하고,
    학습 자료의 AICc가 더 낮은 쪽을 선택한다(검증 결과로 고르지 않는다).
  - 평활계수(alpha, beta, gamma)와 초기값(수준, 추세, 계절 지수)은 학습 SSE 최소화로 자동 추정한다.
  - 기준선: 계절 naive(직전 같은 요일 값을 그대로 예측, 학습 마지막 6개 자리 값 사용).

입력: data/processed/daily_sales.csv, outputs/tables/holt_ses_forecast_november.csv (SES·Holt 예측)
출력: outputs/tables/holt_winters_parameters.csv
      outputs/tables/holt_winters_forecast_november.csv
      outputs/tables/model_comparison_metrics.csv
      outputs/figures/holt_winters_forecast_november.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
PRIOR_PATH = ROOT / "outputs" / "tables" / "holt_ses_forecast_november.csv"
PARAM_PATH = ROOT / "outputs" / "tables" / "holt_winters_parameters.csv"
FCST_PATH = ROOT / "outputs" / "tables" / "holt_winters_forecast_november.csv"
METRIC_PATH = ROOT / "outputs" / "tables" / "model_comparison_metrics.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "holt_winters_forecast_november.png"

TRAIN = ("2011-01-01", "2011-10-31")
VALID = ("2011-11-01", "2011-11-30")
PERIOD = 6
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
TRAIN_COLOR = "#a9b8cc"
SES_COLOR = "#eb6834"
HOLT_COLOR = "#8a8984"
HW_COLOR = "#2a78d6"
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


def build_series(daily: pd.DataFrame, span: tuple) -> tuple[pd.Series, pd.Index]:
    """토요일을 뺀 달력 시계열. 토요일 외 휴무일은 같은 요일 전·후 주 값 평균으로 채운다."""
    d = daily.loc[span[0]:span[1]]
    d = d[d["dow"] != 5]
    y = d["net_sales"].where(d["is_closed"] == 0)
    filled = y.index[y.isna()]
    for day in filled:
        vals = []
        for k in (1, 2, 3):
            for sign in (-1, 1):
                v = y.get(day + pd.Timedelta(days=7 * k * sign))
                if v is not None and not np.isnan(v):
                    vals.append(v)
            if vals:
                break
        y.loc[day] = float(np.mean(vals))
    return y, filled


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    y_train, filled = build_series(daily, TRAIN)
    valid = daily.loc[VALID[0]:VALID[1]]
    y_valid = valid.loc[valid["is_closed"] == 0, "net_sales"]
    valid_slots = valid[valid["dow"] != 5]
    assert (valid_slots["is_closed"] == 0).all() and len(y_valid) == 26 == len(valid_slots)
    # 학습 마지막 날(월)의 다음 자리가 11-01(화)인지 확인: 요일 위상이 이어져야 한다.
    assert y_train.index[-1].dayofweek == 0 and y_valid.index[0].dayofweek == 1
    h = len(y_valid)

    fits, aic = {}, {}
    for seas in ("add", "mul"):
        f = ExponentialSmoothing(
            y_train.to_numpy(), trend="add", damped_trend=False, seasonal=seas,
            seasonal_periods=PERIOD, initialization_method="estimated",
        ).fit(optimized=True)
        fits[seas], aic[seas] = f, float(f.aicc)
    best = min(aic, key=aic.get)
    fit = fits[best]

    pred = {s: pd.Series(fits[s].forecast(h), index=y_valid.index) for s in fits}
    fc_hw = pred[best]
    fitted_hw = pd.Series(fit.fittedvalues, index=y_train.index)

    p = fit.params
    seasons = np.asarray(p["initial_seasons"], dtype=float)
    first_dow = y_train.index[0].dayofweek
    season_labels = [DOW_NAMES[(first_dow + i) % 7] if (first_dow + i) % 7 != 5 else "?" for i in range(PERIOD)]
    # 6개 자리에 대응하는 요일 라벨: 시계열 첫 6일의 요일
    season_labels = [DOW_NAMES[d.dayofweek] for d in y_train.index[:PERIOD]]

    # 학습 마지막 시점의 수준·추세 (마지막 상태)
    level_T = float(fit.level[-1])
    trend_T = float(fit.trend[-1])

    prior = pd.read_csv(PRIOR_PATH, parse_dates=["date"]).set_index("date")
    assert prior.index.equals(y_valid.index) and np.allclose(prior["actual"], y_valid)
    fc_ses, fc_holt = prior["ses_forecast"], prior["holt_forecast"]

    # 계절 naive: 학습 마지막 6개 자리의 값을 반복해서 예측
    last6 = y_train.iloc[-PERIOD:].to_numpy()
    fc_naive = pd.Series([last6[i % PERIOD] for i in range(h)], index=y_valid.index)
    assert all(y_train.index[-PERIOD + (i % PERIOD)].dayofweek == y_valid.index[i].dayofweek for i in range(h))

    models = {
        "SES": fc_ses, "Holt": fc_holt,
        f"Holt-Winters({best})": fc_hw,
        f"Holt-Winters({'mul' if best == 'add' else 'add'}, 참고)": pred["mul" if best == "add" else "add"],
        "계절 naive(기준선)": fc_naive,
    }
    metrics = pd.DataFrame({k: errors(y_valid, v) for k, v in models.items()}).T.reset_index(names="model")
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")

    rows = [
        ("selected_seasonal", best, "학습 AICc가 낮은 계절성 형태"),
        ("aicc_add", aic["add"], "덧셈 계절성 학습 AICc"),
        ("aicc_mul", aic["mul"], "곱셈 계절성 학습 AICc"),
        ("alpha", float(p["smoothing_level"]), "수준 평활계수"),
        ("beta", float(p["smoothing_trend"]), "추세 평활계수"),
        ("gamma", float(p["smoothing_seasonal"]), "계절 평활계수"),
        ("initial_level", float(p["initial_level"]), "초기 수준"),
        ("initial_trend", float(p["initial_trend"]), "초기 추세(자리당 변화량)"),
        ("last_level", level_T, "학습 마지막 수준"),
        ("last_trend", trend_T, "학습 마지막 추세"),
        ("train_sse", float(fit.sse), "학습 1단계 예측 제곱오차합"),
        ("n_train_slots", len(y_train), "학습 자리 수(토요일 제외 달력일)"),
        ("n_filled_slots", len(filled), "토요일 외 휴무일 채움 건수(적합용)"),
        ("filled_dates", " ".join(f"{d:%Y-%m-%d}" for d in filled), "채운 날짜"),
    ]
    rows += [(f"initial_season_{lab}_{i}", float(v), "초기 계절 지수") for i, (lab, v) in enumerate(zip(season_labels, seasons))]
    pd.DataFrame(rows, columns=["name", "value", "note"]).to_csv(PARAM_PATH, index=False, encoding="utf-8-sig")

    out = pd.DataFrame({
        "actual": y_valid, "ses_forecast": fc_ses, "holt_forecast": fc_holt,
        "hw_forecast": fc_hw, "hw_error": y_valid - fc_hw, "hw_abs_pct_error": (y_valid - fc_hw).abs() / y_valid * 100,
        "seasonal_naive_forecast": fc_naive,
    })
    out.reset_index().to_csv(FCST_PATH, index=False, encoding="utf-8-sig")

    plot(y_train, fitted_hw, y_valid, fc_ses, fc_holt, fc_hw, best, p, metrics)

    pd.set_option("display.width", 220)
    print(f"채움 {len(filled)}건: {[d.strftime('%m-%d') for d in filled]}")
    print(f"AICc add={aic['add']:.1f} mul={aic['mul']:.1f} → 선택: {best}")
    print(f"alpha={p['smoothing_level']:.6f} beta={p['smoothing_trend']:.6f} gamma={p['smoothing_seasonal']:.6f}")
    print(f"l0={p['initial_level']:,.2f} b0={p['initial_trend']:,.4f} 마지막 수준={level_T:,.1f} 추세={trend_T:,.2f}")
    print("초기 계절:", dict(zip(season_labels, np.round(seasons, 3))))
    print(metrics.round(1).to_string(index=False))
    print(out[["actual", "ses_forecast", "holt_forecast", "hw_forecast"]].round(0).to_string())


def plot(y_train, fitted_hw, y_valid, fc_ses, fc_holt, fc_hw, best, p, metrics) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
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
    ax.plot(y_valid.index, y_valid.values, color=TEXT_PRIMARY, lw=1.8, marker="o", ms=4, label="11월 실제")
    ax.plot(fc_ses.index, fc_ses.values, color=SES_COLOR, lw=2, label="SES")
    ax.plot(fc_holt.index, fc_holt.values, color=HOLT_COLOR, lw=2, ls="--", label="Holt")
    ax.plot(fc_hw.index, fc_hw.values, color=HW_COLOR, lw=2.4, marker="^", ms=4, label=f"Holt-Winters({best})")
    ax.axvline(pd.Timestamp("2011-11-01") - pd.Timedelta(hours=12), color=BASELINE, ls="--", lw=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(pd.Timestamp("2010-12-28"), pd.Timestamp("2011-12-02"))
    ax.set_ylabel("일 순매출 (£)", fontsize=10, color=TEXT_SECONDARY)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    style(ax2)
    ax2.plot(y_valid.index, y_valid.values, color=TEXT_PRIMARY, lw=1.8, marker="o", ms=4, label="실제")
    ax2.plot(fc_ses.index, fc_ses.values, color=SES_COLOR, lw=1.8, label="SES")
    ax2.plot(fc_holt.index, fc_holt.values, color=HOLT_COLOR, lw=1.8, ls="--", label="Holt")
    ax2.plot(fc_hw.index, fc_hw.values, color=HW_COLOR, lw=2.4, marker="^", ms=4, label="Holt-Winters")
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.set_title("11월 확대", loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)

    m = metrics.set_index("model")
    hw_key = f"Holt-Winters({best})"
    fig.suptitle("Holt-Winters(추세 + 요일 계절성) vs SES · Holt: 11월 일 매출 예측", x=0.01, y=0.985,
                 ha="left", fontsize=15, color=TEXT_PRIMARY)
    fig.text(0.01, 0.905,
             f"HW({best}) α={p['smoothing_level']:.4f}  β={p['smoothing_trend']:.4f}  γ={p['smoothing_seasonal']:.4f}   |   "
             f"검증 MAE  SES £{m.loc['SES','MAE']:,.0f} / Holt £{m.loc['Holt','MAE']:,.0f} / HW £{m.loc[hw_key,'MAE']:,.0f}   "
             f"MAPE  {m.loc['SES','MAPE(%)']:.1f}% / {m.loc['Holt','MAPE(%)']:.1f}% / {m.loc[hw_key,'MAPE(%)']:.1f}%",
             fontsize=10, color=TEXT_SECONDARY)
    fig.text(0.01, 0.005,
             "토요일을 뺀 6일 주기 시계열에 적용. 토요일 외 휴무일은 같은 요일 전·후 주 평균으로 채워 적합(평가 제외). "
             "계수·초기값·계절 형태는 학습 자료로만 결정.",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


if __name__ == "__main__":
    main()
