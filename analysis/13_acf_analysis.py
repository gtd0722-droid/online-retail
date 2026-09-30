"""학습 기간 일별 판매금액 그래프와 시차 1~21일 자기상관 (ARIMA 비교 준비).

학습: 2011-01-01 ~ 2011-10-31 (달력 304일). 검증 구간(11월)은 사용하지 않는다.
대상: 일 순매출 net_sales. 휴무일(토요일·공휴일 등 53일)의 값은 0으로 기록되어 있다.

두 가지 자기상관을 함께 계산한다.
  A) 달력 일별 시계열 전체(휴무일 = 0 포함)의 자기상관. 표준 ACF(statsmodels.acf, 전체 평균으로 중심화).
     ARIMA 식별에서 쓰는 방식이지만, 매주 토요일이 0이라 7·14·21일 시차 상관이 휴무 패턴의 영향을 받는다.
  B) 두 날짜(t, t-k)가 모두 영업일인 쌍만으로 계산한 피어슨 상관. 휴무일 0의 영향을 뺀 값.
신뢰대(A): 무작위 시계열 가정에서 ±1.96/sqrt(n).

입력: data/processed/daily_sales.csv
출력: outputs/tables/acf_lags_1_21.csv
      outputs/figures/train_daily_sales_acf.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "acf_lags_1_21.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "train_daily_sales_acf.png"

TRAIN = ("2011-01-01", "2011-10-31")
MAX_LAG = 21
HIGHLIGHT = (7, 14, 21)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
LINE = "#2a78d6"
BAR = "#a9b8cc"
ACCENT = "#eb6834"
CLOSED = "#efeeea"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def open_pair_corr(y: pd.Series, is_open: pd.Series, k: int) -> tuple[float, int]:
    a, b = y.iloc[k:].to_numpy(), y.iloc[:-k].to_numpy()
    m = is_open.iloc[k:].to_numpy() & is_open.iloc[:-k].to_numpy()
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    tr = daily.loc[TRAIN[0]:TRAIN[1]]
    y = tr["net_sales"]
    is_open = tr["is_closed"] == 0
    assert len(y) == 304 and int((~is_open).sum()) == 53 and (y[~is_open] == 0).all()

    acf_a = acf(y.to_numpy(), nlags=MAX_LAG, fft=False)
    rows = []
    for k in range(1, MAX_LAG + 1):
        r_b, n_b = open_pair_corr(y, is_open, k)
        rows.append({"lag": k, "acf_calendar": float(acf_a[k]), "corr_open_pairs": r_b, "n_open_pairs": n_b})
    tab = pd.DataFrame(rows)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")

    conf = 1.96 / np.sqrt(len(y))
    plot(y, is_open, tab, conf)

    pd.set_option("display.width", 200)
    print(f"n={len(y)}  95% 신뢰대 ±{conf:.3f}")
    print(tab.round(3).to_string(index=False))
    print("\n7·14·21일 전:")
    print(tab[tab["lag"].isin(HIGHLIGHT)].round(3).to_string(index=False))


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def acf_panel(ax, tab: pd.DataFrame, col: str, title: str, conf: float | None) -> None:
    style(ax)
    colors = [ACCENT if k in HIGHLIGHT else BAR for k in tab["lag"]]
    ax.bar(tab["lag"], tab[col], color=colors, width=0.7, lw=0)
    ax.axhline(0, color="#8a8984", lw=0.8)
    if conf is not None:
        ax.axhspan(-conf, conf, color=LINE, alpha=0.10, lw=0)
    for k in HIGHLIGHT:
        v = float(tab.loc[tab["lag"] == k, col].iloc[0])
        ax.annotate(f"{v:+.2f}", xy=(k, v), xytext=(0, 4 if v >= 0 else -12), textcoords="offset points",
                    ha="center", fontsize=10, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_xticks(range(1, MAX_LAG + 1))
    ax.set_xlim(0.3, MAX_LAG + 0.7)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlabel("시차 (일 전)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("상관계수", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=8)


def plot(y: pd.Series, is_open: pd.Series, tab: pd.DataFrame, conf: float) -> None:
    fig = plt.figure(figsize=(15, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.38, wspace=0.16)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    style(ax0)
    ax0.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax0.plot(y.index, y.values, color=LINE, lw=1.2, label="일 순매출")
    closed = y[~is_open]
    ax0.scatter(closed.index, closed.values, s=14, color="#8a8984", marker="x", lw=0.9, zorder=3,
                label="휴무일 (값 0)")
    ax0.xaxis.set_major_locator(mdates.MonthLocator())
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.set_xlim(y.index.min() - pd.Timedelta(days=2), y.index.max() + pd.Timedelta(days=2))
    ax0.set_ylabel("순매출 (£)", fontsize=10, color=TEXT_SECONDARY)
    ax0.set_title("학습 기간 일별 판매금액 (2011-01-01 ~ 2011-10-31, 304일)", loc="left", fontsize=12.5,
                  color=TEXT_PRIMARY, pad=8)
    ax0.legend(loc="upper left", frameon=False, fontsize=9.5)

    acf_panel(ax1, tab, "acf_calendar", "A) 달력 일별 시계열 전체의 자기상관 (휴무일 0 포함)", conf)
    acf_panel(ax2, tab, "corr_open_pairs", "B) 두 날짜 모두 영업일인 쌍만의 상관 (휴무 영향 제외)", None)
    ax2.set_ylabel("")

    fig.suptitle("일 매출의 시차별 자기상관 (시차 1~21일)", x=0.01, y=0.985, ha="left", fontsize=15,
                 color=TEXT_PRIMARY)
    fig.text(0.01, 0.005,
             f"주황색 막대는 7·14·21일 전. A의 연한 파란 띠는 95% 신뢰대(±{conf:.3f}). "
             "A는 매주 토요일 0이 상관을 키우고, B는 영업일끼리만 비교한 값이다.",
             fontsize=9, color=TEXT_SECONDARY)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.06, right=0.985)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


if __name__ == "__main__":
    main()
