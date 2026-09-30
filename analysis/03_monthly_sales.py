"""월별 매출 요약표와 월별 일매출·이동평균 그래프를 만든다.

입력: data/processed/daily_sales.csv (analysis/01_preprocess.py 산출물)
출력: outputs/tables/monthly_sales_summary.csv
      outputs/figures/monthly_daily_sales.png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "monthly_sales_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "monthly_daily_sales.png"

MA_WINDOW = 7  # 달력 기준 7일, 가운데 정렬, 휴무일 제외 평균 (02 그래프와 같은 정의)
N_COLS = 4

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BAR_COLOR = "#a9b8cc"
MA_COLOR = "#2a78d6"

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("year_month")
    summary = pd.DataFrame({
        "net_sales_total": g["net_sales"].sum(),
        "n_days": g.size(),
        "n_open_days": g["is_closed"].apply(lambda s: int((s == 0).sum())),
        "is_partial_month": g["is_partial_month"].first(),
    })
    summary["avg_per_day"] = summary["net_sales_total"] / summary["n_days"]
    summary["avg_per_open_day"] = summary["net_sales_total"] / summary["n_open_days"]
    return summary.reset_index()


def plot_months(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    # 이동평균은 전체 기간에서 계산한 뒤 월별로 잘라, 월 경계에서 값이 끊기지 않게 한다.
    open_sales = daily["net_sales"].where(daily["is_closed"] == 0)
    ma = open_sales.rolling(MA_WINDOW, center=True, min_periods=1).mean()

    months = summary["year_month"].tolist()
    n_rows = -(-len(months) // N_COLS)
    ymax = daily["net_sales"].max() / 1000 * 1.03

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, 3.1 * n_rows), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (_, row) in zip(axes.flat, summary.iterrows()):
        ym = row["year_month"]
        mask = daily["year_month"] == ym
        d = daily[mask]
        start = pd.Timestamp(f"{ym}-01")
        end = start + pd.offsets.MonthEnd(0)

        ax.set_facecolor(SURFACE)
        open_d = d[d["is_closed"] == 0]
        ax.bar(open_d.index, open_d["net_sales"] / 1000, width=0.7, color=BAR_COLOR,
               lw=0, zorder=2)
        ax.plot(ma[mask].index, ma[mask].values / 1000, color=MA_COLOR, lw=2,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)

        title = f"{ym}{' (1~8일)' if row['is_partial_month'] else ''}"
        ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=18)
        ax.text(0, 1.02, f"합계 {row['net_sales_total'] / 1000:,.1f}천 · "
                         f"일평균 {row['avg_per_day'] / 1000:,.1f}천",
                transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY, va="bottom")

        ax.set_xlim(start - pd.Timedelta(hours=18), end + pd.Timedelta(hours=18))
        ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d일"))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)

    for ax in axes.flat[len(months):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("일 순매출 (천 파운드)", fontsize=9, color=TEXT_SECONDARY)

    fig.suptitle("월별 일 순매출과 7일 이동평균", x=0.01, y=0.995, ha="left", fontsize=15,
                 color=TEXT_PRIMARY)
    fig.legend(handles=[Patch(color=BAR_COLOR, label="일별 순매출 (영업일)"),
                        Line2D([], [], color=MA_COLOR, lw=2,
                               label=f"{MA_WINDOW}일 이동평균 (영업일 기준, 전체 기간에서 계산)")],
               loc="upper left", bbox_to_anchor=(0.005, 0.975), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=TEXT_PRIMARY)
    fig.text(0.01, 0.005,
             "막대가 없는 날은 휴무일(토요일·공휴일·연말연시). 모든 패널의 세로축 범위는 같음. "
             "일평균 = 월 합계 / 해당 월의 기간 내 날짜 수(휴무일 포함).",
             fontsize=8.5, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.015, 1, 0.955), h_pad=2.2)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    summary = monthly_summary(daily)

    assert abs(summary["net_sales_total"].sum() - daily["net_sales"].sum()) < 1e-6

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    plot_months(daily, summary)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"저장: {TABLE_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
