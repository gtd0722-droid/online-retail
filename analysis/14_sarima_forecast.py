"""SARIMA로 2011-11 일 순매출을 예측하고 SES·Holt·Holt-Winters·계절 naive와 비교한다.

학습: 2011-01-01 ~ 2011-10-31    검증: 2011-11-01 ~ 2011-11-30 (학습·차수 선택에 사용하지 않음)

시계열 구성 (Holt-Winters와 동일, 11_holt_winters_forecast.py의 build_series 재사용)
  - 토요일을 뺀 6일 주기(계절 주기 s=6). 토요일 외 휴무일 9일은 같은 요일 전·후 주 평균으로 채워 적합에만 사용.

차수 선택 (검증 구간을 쓰지 않는다)
  - 학습 구간 안에서 내부 검증을 한다: 2011-01~09로 적합 → 2011-10(26자리)을 다기간 예측 → MAE가 가장 작은 차수를 선택.
    최종 예측과 같은 형태(한 달치 다기간 예측)로 고르기 위해서다. AIC는 차분 차수(d, D)가 다르면 비교할 수 없어 쓰지 않는다.
  - 후보: p,q ∈ {0,1,2}, d ∈ {0,1}, P,Q ∈ {0,1}, D ∈ {0,1}, 변환 ∈ {원자료, 로그}
    (d+D=0이면 상수항 포함, 아니면 제외). 로그 변환 예측은 exp로 되돌린다(중앙값 예측).
  - 선택한 차수로 학습 전체(1~10월)에 다시 적합해 11월 26자리를 예측한다.

입력: data/processed/daily_sales.csv, outputs/tables/model_comparison_metrics.csv,
      outputs/tables/holt_winters_forecast_november.csv (SES·Holt·HW·계절 naive 예측)
출력: outputs/tables/sarima_order_search.csv
      outputs/tables/sarima_parameters.csv
      outputs/tables/sarima_forecast_november.csv
      outputs/tables/model_comparison_with_sarima.csv
      outputs/figures/sarima_forecast_november.png
"""
import importlib.util
import itertools
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
FIG_PATH = ROOT / "outputs" / "figures" / "sarima_forecast_november.png"

_spec = importlib.util.spec_from_file_location("hw11", ROOT / "analysis" / "11_holt_winters_forecast.py")
hw11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hw11)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TRAIN = ("2011-01-01", "2011-10-31")
VALID = ("2011-11-01", "2011-11-30")
INNER_SPLIT = pd.Timestamp("2011-09-30")
S = hw11.PERIOD


def fit_sarima(y: pd.Series, order, sorder, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(y.to_numpy(), order=order, seasonal_order=(*sorder, S), trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(disp=False, maxiter=200)


def forecast(fit, h: int, log: bool):
    fc = fit.get_forecast(h)
    mean, ci = np.asarray(fc.predicted_mean), np.asarray(fc.conf_int(alpha=0.05))
    if log:
        mean, ci = np.exp(mean), np.exp(ci)
    return mean, ci


def main() -> None:
    daily = pd.read_csv(hw11.DAILY_PATH, parse_dates=["date"]).set_index("date")
    y_slots, filled = hw11.build_series(daily, TRAIN)
    valid = daily.loc[VALID[0]:VALID[1]]
    y_valid = valid.loc[valid["is_closed"] == 0, "net_sales"]
    assert len(y_valid) == 26 and y_slots.index[-1].dayofweek == 0 and y_valid.index[0].dayofweek == 1
    assert (y_slots > 0).all()
    h = len(y_valid)

    inner_train = y_slots.loc[:INNER_SPLIT]
    inner_test = y_slots.loc[INNER_SPLIT + pd.Timedelta(days=1):]
    assert len(inner_test) == 26

    rows = []
    grid = itertools.product((0, 1, 2), (0, 1), (0, 1, 2), (0, 1), (0, 1), (0, 1), (False, True))
    for p, d, q, P, D, Q, log in grid:
        trend = "c" if d + D == 0 else "n"
        try:
            yt = np.log(inner_train) if log else inner_train
            f = fit_sarima(yt, (p, d, q), (P, D, Q), trend)
            mean, _ = forecast(f, len(inner_test), log)
            mae = float(np.mean(np.abs(inner_test.to_numpy() - mean)))
            ok = bool(np.isfinite(mae))
        except Exception:
            mae, ok = np.nan, False
        rows.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": S, "log": int(log),
                     "inner_MAE": mae, "ok": ok})
    search = pd.DataFrame(rows).sort_values("inner_MAE").reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    search.to_csv(TABLES / "sarima_order_search.csv", index=False, encoding="utf-8-sig")

    best = search[search["ok"]].iloc[0]
    order, sorder = (int(best.p), int(best.d), int(best.q)), (int(best.P), int(best.D), int(best.Q))
    log = bool(best["log"])
    trend = "c" if order[1] + sorder[1] == 0 else "n"

    yt = np.log(y_slots) if log else y_slots
    fit = fit_sarima(yt, order, sorder, trend)
    mean, ci = forecast(fit, h, log)
    fc = pd.Series(mean, index=y_valid.index)
    lo, hi = pd.Series(ci[:, 0], index=y_valid.index), pd.Series(ci[:, 1], index=y_valid.index)

    resid = pd.Series(fit.resid).iloc[max(S, 12):]
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    covered = float(((y_valid >= lo) & (y_valid <= hi)).mean())

    prev = pd.read_csv(TABLES / "holt_winters_forecast_november.csv", parse_dates=["date"]).set_index("date")
    assert prev.index.equals(y_valid.index) and np.allclose(prev["actual"], y_valid)
    models = {
        "SES": prev["ses_forecast"], "Holt": prev["holt_forecast"], "Holt-Winters(mul)": prev["hw_forecast"],
        "계절 naive(기준선)": prev["seasonal_naive_forecast"], "SARIMA": fc,
    }
    metrics = pd.DataFrame({k: hw11.errors(y_valid, v) for k, v in models.items()}).T.reset_index(names="model")
    metrics.to_csv(TABLES / "model_comparison_with_sarima.csv", index=False, encoding="utf-8-sig")

    label = f"SARIMA{order}x{sorder}[{S}]" + (" 로그" if log else "")
    params = [("order", str(order), "(p,d,q)"), ("seasonal_order", f"{sorder} s={S}", "(P,D,Q) 주기"),
              ("log_transform", int(log), "1이면 로그 변환 후 적합, 예측은 exp로 복원"), ("trend", trend, "c=상수항, n=없음"),
              ("inner_MAE", float(best.inner_MAE), "차수 선택 내부 검증(2011-10) MAE"),
              ("train_AICc", float(fit.aicc), "최종 적합 AICc (참고)"), ("train_slots", len(y_slots), "학습 자리 수"),
              ("n_filled_slots", len(filled), "토요일 외 휴무일 채움 건수"),
              ("ljung_box_p_lag12", lb_p, "잔차 Ljung-Box p값(lag 12); 크면 잔차 자기상관 없음"),
              ("interval95_coverage", covered, "검증 실제값이 95% 예측구간에 든 비율")]
    params += [(f"coef_{n}", float(v), "추정 계수") for n, v in zip(fit.param_names, fit.params)]
    pd.DataFrame(params, columns=["name", "value", "note"]).to_csv(TABLES / "sarima_parameters.csv", index=False, encoding="utf-8-sig")

    out = pd.DataFrame({"actual": y_valid, "sarima_forecast": fc, "lower95": lo, "upper95": hi,
                        "error": y_valid - fc, "abs_pct_error": (y_valid - fc).abs() / y_valid * 100})
    out.reset_index().to_csv(TABLES / "sarima_forecast_november.csv", index=False, encoding="utf-8-sig")

    plot(y_slots, y_valid, models, lo, hi, label, metrics)

    pd.set_option("display.width", 220)
    print("탐색 후보:", len(search), "성공:", int(search["ok"].sum()))
    print(search.head(8).round(1).to_string(index=False))
    print("선택:", label, "inner_MAE", round(float(best.inner_MAE), 1))
    print("계수:", dict(zip(fit.param_names, np.round(fit.params, 4))))
    print(f"Ljung-Box p(lag12)={lb_p:.3f}  95%구간 포함비율={covered:.2f}")
    print(metrics.round(1).to_string(index=False))
    print(out.round(0).to_string())


def plot(y_slots, y_valid, models, lo, hi, label, metrics) -> None:
    colors = {"SES": "#eb6834", "Holt": "#8a8984", "Holt-Winters(mul)": "#2a78d6", "SARIMA": "#2f9e44"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150, gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    def style(a):
        a.set_facecolor("#fcfcfb")
        a.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        a.grid(axis="y", color="#e6e5e1", lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_visible(False)
        a.tick_params(colors="#52514e", labelsize=9, length=0)

    style(ax)
    hist = y_slots.loc["2011-08-01":]
    ax.plot(hist.index, hist.values, color="#a9b8cc", lw=1.2, label="학습 구간 실제 (2011-08~10)")
    ax.plot(y_valid.index, y_valid.values, color="#0b0b0b", lw=1.8, marker="o", ms=4, label="11월 실제")
    ax.fill_between(lo.index, lo.values, hi.values, color=colors["SARIMA"], alpha=0.12, lw=0, label="SARIMA 95% 예측구간")
    for name in ("SES", "Holt", "Holt-Winters(mul)", "SARIMA"):
        ax.plot(models[name].index, models[name].values, color=colors[name], lw=2.3 if name == "SARIMA" else 1.7,
                ls="--" if name == "Holt" else "-", marker="^" if name == "SARIMA" else None, ms=4, label=name)
    ax.axvline(pd.Timestamp("2011-11-01") - pd.Timedelta(hours=12), color="#8a8984", ls="--", lw=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("일 순매출 (£)", fontsize=10, color="#52514e")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    style(ax2)
    m = metrics.set_index("model")["MAE"]
    order = ["SES", "Holt", "계절 naive(기준선)", "Holt-Winters(mul)", "SARIMA"]
    bars = ax2.barh(order[::-1], [m[k] for k in order[::-1]],
                    color=[colors.get(k, "#a9b8cc") for k in order[::-1]], height=0.6)
    for b, k in zip(bars, order[::-1]):
        ax2.text(b.get_width() + 200, b.get_y() + b.get_height() / 2, f"£{m[k]:,.0f}", va="center", fontsize=9.5, color="#0b0b0b")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order[::-1])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: order[::-1][int(round(v))] if 0 <= int(round(v)) < len(order) else ""))
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_xlim(0, max(m) * 1.2)
    ax2.set_title("11월 검증 MAE (낮을수록 좋음)", loc="left", fontsize=11, color="#0b0b0b")
    ax2.grid(axis="y", visible=False)

    fig.suptitle(f"{label}: 11월 일 매출 예측과 기존 모형 비교", x=0.01, y=0.985, ha="left", fontsize=15)
    fig.text(0.01, 0.905, "  ".join(f"{k} MAPE {metrics.set_index('model').loc[k, 'MAPE(%)']:.1f}%"
                                     for k in ("SES", "Holt", "Holt-Winters(mul)", "SARIMA")), fontsize=10, color="#52514e")
    fig.text(0.01, 0.005, "토요일을 뺀 6일 주기 시계열. 차수는 학습 구간 안 내부 검증(2011-10)으로 선택. 검증 구간은 학습에 사용하지 않음.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
