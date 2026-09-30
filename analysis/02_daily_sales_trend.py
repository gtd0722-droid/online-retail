"""일별 순매출 추이 그래프를 그린다.

입력: data/processed/daily_sales.csv (analysis/01_preprocess.py 산출물)
출력: outputs/figures/daily_sales_trend.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "daily_sales_trend.png"

MA_WINDOW = 7  # 달력 기준 7일, 가운데 정렬, 휴무일 제외 평균

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
DAILY_COLOR = "#a9b8cc"  # 일별 값: 흐린 보조색
MA_COLOR = "#2a78d6"  # 이동평균: 강조색
OUTLIER_COLOR = "#e34948"
CLOSED_SHADE = "#efeeea"

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")

    # 휴무일은 매출 0이 아니라 '영업 없음'이므로 선에서 빼고, 이동평균도 영업일만으로 계산한다.
    open_sales = daily["net_sales"].where(daily["is_closed"] == 0)
    ma = open_sales.rolling(MA_WINDOW, center=True, min_periods=1).mean()
    # 연속 휴무(연말연시)는 선을 끊고, 하루짜리 휴무는 앞뒤 영업일을 잇는다.
    open_days = open_sales.dropna()
    gap = open_days.index.to_series().diff() > pd.Timedelta(days=3)
    breaks = pd.Series(float("nan"), index=open_days.index[gap] - pd.Timedelta(days=1))
    daily_line = pd.concat([open_days, breaks]).sort_index()
    outliers = daily[daily["is_outlier"] == 1]

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # 연말연시 장기 휴무 구간 표시
    xmas_start, xmas_end = pd.Timestamp("2010-12-24"), pd.Timestamp("2011-01-03")
    ax.axvspan(xmas_start - pd.Timedelta(hours=12), xmas_end + pd.Timedelta(hours=12),
               color=CLOSED_SHADE, zorder=0, lw=0)
    ax.text(xmas_start + (xmas_end - xmas_start) / 2, 0.97, "연말연시\n휴무",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=TEXT_SECONDARY)

    ax.plot(daily_line.index, daily_line.values / 1000, color=DAILY_COLOR, lw=1.2,
            solid_joinstyle="round", label="일별 순매출 (영업일)", zorder=2)
    ax.plot(ma.index, ma.values / 1000, color=MA_COLOR, lw=2.2,
            solid_joinstyle="round", solid_capstyle="round",
            label=f"{MA_WINDOW}일 이동평균 (영업일 기준)", zorder=3)
    ax.scatter(outliers.index, outliers["net_sales"] / 1000, s=42, color=OUTLIER_COLOR,
               edgecolor=SURFACE, linewidth=1.5, zorder=4, label="이상치 표시일 (|z| > 3.5)")

    for date, row in outliers.iterrows():
        above = row["outlier_z"] > 0
        ax.annotate(f"{date:%m-%d}  {row['net_sales'] / 1000:,.1f}천",
                    xy=(date, row["net_sales"] / 1000),
                    xytext=(-8 if above else 8, 6 if above else -14),
                    textcoords="offset points", ha="right" if above else "left",
                    fontsize=9, color=TEXT_PRIMARY)

    ax.set_title("일별 순매출 추이 (2010-12-01 ~ 2011-12-08)", loc="left",
                 fontsize=14, color=TEXT_PRIMARY, pad=30)
    ax.set_ylabel("일 순매출 (천 파운드)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, None)
    ax.set_xlim(daily.index.min() - pd.Timedelta(days=2), daily.index.max() + pd.Timedelta(days=2))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=9, labelcolor=TEXT_PRIMARY, borderaxespad=0.2)
    fig.text(0.01, 0.01,
             "휴무일(토요일·공휴일·연말연시 69일)은 선에서 제외. "
             "순매출 = 상품 판매 금액 - 취소 금액. 2011-12-09는 부분일이라 제외.",
             fontsize=8, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)
    print(f"저장: {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
