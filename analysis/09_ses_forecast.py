"""단순 지수평활(SES)로 2011-11 일 순매출을 예측한다.

학습: 2011-01-01 ~ 2011-10-31 (영업일 251일)
검증: 2011-11-01 ~ 2011-11-30 (영업일 26일)  -- 학습 단계에서 전혀 사용하지 않는다.

방법
  - 영업일만 이어 붙인 시계열(휴무일 제외)에 SES를 적용한다. 예측 대상도 영업일이다.
  - 평활계수 alpha와 초기 수준 l0는 학습 자료의 제곱오차합(SSE)을 최소화하도록 자동 추정한다.
  - SES의 다기간 예측은 마지막 수준값으로 평평하다. 요일·추세는 반영하지 않는다.

입력: data/processed/daily_sales.csv
출력: outputs/tables/ses_parameters.csv
      outputs/tables/ses_forecast_november.csv
      outputs/figures/ses_forecast_november.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
PARAM_PATH = ROOT / "outputs" / "tables" / "ses_parameters.csv"
FCST_PATH = ROOT / "outputs" / "tables" / "ses_forecast_november.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "ses_forecast_november.png"

TRAIN = ("2011-01-01", "2011-10-31")
VALID = ("2011-11-01", "2011-11-30")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
ACTUAL = "#8a8984"
TRAIN_COLOR = "#a9b8cc"
FCST = "#eb6834"
FIT = "#2a78d6"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    train = daily.loc[TRAIN[0]:TRAIN[1]]
    valid = daily.loc[VALID[0]:VALID[1]]
    y_train = train.loc[train["is_closed"] == 0, "net_sales"]
    y_valid = valid.loc[valid["is_closed"] == 0, "net_sales"]
    assert len(y_train) == 251 and len(y_valid) == 26

    # 영업일 순서(1, 2, 3, ...)를 시간축으로 사용 (날짜 간격이 불규칙하므로 정수 인덱스)
    model = SimpleExpSmoothing(y_train.to_numpy(), initialization_method="estimated")
    fit = model.fit(optimized=True)
    alpha = float(fit.params["smoothing_level"])
    level0 = float(fit.params["initial_level"])
    fitted = pd.Series(fit.fittedvalues, index=y_train.index)  # 1단계 앞 예측
    forecast_value = float(fit.forecast(1)[0])

    pred = pd.Series(forecast_value, index=y_valid.index)
    err = y_valid - pred
    sse_train = float(((y_train - fitted) ** 2).sum())
    metrics = {
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE(%)": float((err.abs() / y_valid).mean() * 100),
        "ME(평균오차, 실제-예측)": float(err.mean()),
    }

    params = pd.DataFrame([
        ("alpha", alpha, "평활계수 (학습 SSE 최소화)"),
        ("initial_level", level0, "초기 수준 l0 (학습 SSE 최소화)"),
        ("last_level", forecast_value, "학습 마지막 영업일의 수준 = 11월 예측값(평평)"),
        ("train_sse", sse_train, "학습 구간 1단계 예측 제곱오차합"),
        ("n_train_open_days", len(y_train), "학습 영업일 수"),
        ("n_valid_open_days", len(y_valid), "검증 영업일 수"),
    ] + [(f"valid_{k}", v, "검증 구간(11월) 오차") for k, v in metrics.items()],
        columns=["name", "value", "note"])
    PARAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    params.to_csv(PARAM_PATH, index=False, encoding="utf-8-sig")

    out = pd.DataFrame({"actual": y_valid, "forecast": pred, "error": err,
                        "abs_pct_error": err.abs() / y_valid * 100})
    out.reset_index().to_csv(FCST_PATH, index=False, encoding="utf-8-sig")

    plot(daily, y_train, fitted, y_valid, pred, alpha, level0, forecast_value, metrics)

    pd.set_option("display.width", 200)
    print(f"alpha = {alpha:.6f}\ninitial_level = {level0:,.2f}\n예측값(11월 평평) = {forecast_value:,.2f}")
    print({k: round(v, 1) for k, v in metrics.items()})
    print(out.round(1).to_string())


def plot(daily, y_train, fitted, y_valid, pred, alpha, level0, fc, metrics) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # 연속 휴무(연말연시)가 학습 구간에는 없으므로 영업일 선을 그대로 잇는다.
    ax.plot(y_train.index, y_train.values, color=TRAIN_COLOR, lw=1.2, label="학습 구간 실제 (2011-01~10)")
    ax.plot(fitted.index, fitted.values, color=FIT, lw=1.6, alpha=0.9,
            label="학습 구간 1단계 적합값(수준)")
    ax.plot(y_valid.index, y_valid.values, color=TEXT_PRIMARY, lw=1.8, marker="o", ms=4,
            label="11월 실제")
    ax.plot(pred.index, pred.values, color=FCST, lw=2.4, marker="s", ms=4,
            label=f"11월 예측 (SES, £{fc:,.0f})")
    ax.axvline(pd.Timestamp("2011-11-01") - pd.Timedelta(hours=12), color=ACTUAL, ls="--", lw=1)
    ax.text(pd.Timestamp("2011-11-01"), ax.get_ylim()[1] * 0.97, " 검증 시작", fontsize=9,
            color=TEXT_SECONDARY, va="top")

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(pd.Timestamp("2010-12-28"), pd.Timestamp("2011-12-02"))
    ax.set_ylabel("일 순매출 (£)", fontsize=10, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    fig.suptitle("단순 지수평활(SES): 11월 일 매출 예측 vs 실제", x=0.01, y=0.985, ha="left",
                 fontsize=15, color=TEXT_PRIMARY)
    fig.text(0.01, 0.905,
             f"평활계수 α = {alpha:.4f}   초기 수준 l0 = £{level0:,.0f}   |   11월 검증: "
             f"MAE £{metrics['MAE']:,.0f}, RMSE £{metrics['RMSE']:,.0f}, MAPE {metrics['MAPE(%)']:.1f}%",
             fontsize=10.5, color=TEXT_SECONDARY)
    fig.text(0.01, 0.005,
             "영업일만 이어 붙인 시계열에 적용(휴무일 제외). α와 l0는 학습 자료로만 추정. "
             "SES 다기간 예측은 마지막 수준값으로 일정하다.",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


if __name__ == "__main__":
    main()
