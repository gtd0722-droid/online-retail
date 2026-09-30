"""요일별 매출 통계표와 그래프를 만든다.

입력: data/processed/daily_sales.csv (analysis/01_preprocess.py 산출물)
출력: outputs/tables/weekday_sales_summary.csv
      outputs/figures/weekday_sales.png

통계는 영업일(is_closed = 0)만 대상으로 한다. 휴무일의 0은 '영업 없음'이므로 평균에 넣지 않는다.
주 대비 지수: 같은 주(월~일)의 영업일 평균 대비 그날 매출의 비율. 계절 추세의 영향을 줄여 요일 효과만 본다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
TABLE_PATH = ROOT / "outputs" / "tables" / "weekday_sales_summary.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "weekday_sales.png"

DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
PEAK_START = pd.Timestamp("2011-09-01")  # 성수기 구분 (월별 그래프에서 9월부터 상승)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
POINT_COLOR = "#a9b8cc"
ACCENT = "#2a78d6"  # 계열 1
SECOND = "#eb6834"  # 계열 2

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def add_week_index(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["week"] = daily.index.to_period("W-SUN")
    open_mask = daily["is_closed"] == 0
    week_mean = daily[open_mask].groupby("week")["net_sales"].transform("mean")
    daily["week_index"] = np.nan
    daily.loc[open_mask, "week_index"] = daily.loc[open_mask, "net_sales"] / week_mean
    return daily


def weekday_summary(daily: pd.DataFrame) -> pd.DataFrame:
    open_d = daily[daily["is_closed"] == 0]
    g = open_d.groupby("dow")
    total = open_d["net_sales"].sum()
    overall_mean = open_d["net_sales"].mean()

    s = pd.DataFrame({
        "n_days_total": daily.groupby("dow").size(),
        "n_open_days": g.size(),
        "mean_sales": g["net_sales"].mean(),
        "median_sales": g["net_sales"].median(),
        "std_sales": g["net_sales"].std(),
        "min_sales": g["net_sales"].min(),
        "max_sales": g["net_sales"].max(),
        "share_of_total": g["net_sales"].sum() / total,
        "mean_invoices": g["n_invoices"].mean(),
        "mean_customers": g["n_customers"].mean(),
        "avg_invoice_value": g["gross_sales"].sum() / g["n_invoices"].sum(),
        "mean_week_index": g["week_index"].mean(),
        "mean_sales_jan_aug": open_d[open_d.index < PEAK_START].groupby("dow")["net_sales"].mean(),
        "mean_sales_sep_dec": open_d[open_d.index >= PEAK_START].groupby("dow")["net_sales"].mean(),
    }).reindex(range(7))
    s["n_open_days"] = s["n_open_days"].fillna(0).astype("int64")
    s["cv"] = s["std_sales"] / s["mean_sales"]
    s["index_vs_overall"] = s["mean_sales"] / overall_mean
    s.insert(0, "weekday", DOW_NAMES)
    s.index.name = "dow"
    return s.reset_index()


def plot(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    open_d = daily[daily["is_closed"] == 0]
    days = [d for d in range(7) if summary.loc[d, "n_open_days"] > 0]
    rng = np.random.default_rng(0)  # 점 흩뿌림 위치 고정

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150,
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURFACE)

    # 왼쪽: 요일별 일 매출 분포 + 평균·중앙값
    for x, d in enumerate(days):
        vals = open_d.loc[open_d["dow"] == d, "net_sales"] / 1000
        ax1.scatter(x + rng.uniform(-0.18, 0.18, len(vals)), vals, s=14, color=POINT_COLOR,
                    lw=0, zorder=2)
        mean, med = vals.mean(), vals.median()
        ax1.hlines(mean, x - 0.3, x + 0.3, color=ACCENT, lw=2.5, zorder=3)
        ax1.hlines(med, x - 0.3, x + 0.3, color=TEXT_PRIMARY, lw=1.2, ls=(0, (2, 2)), zorder=3)
        ax1.text(x + 0.33, mean, f"{mean:,.1f}", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels([f"{DOW_NAMES[d]}\n({summary.loc[d, 'n_open_days']}일)" for d in days])
    ax1.set_ylabel("일 순매출 (천 파운드)", fontsize=10, color=TEXT_SECONDARY)
    ax1.set_ylim(0, None)
    ax1.set_title("요일별 일 순매출 분포 (영업일, 토요일은 휴무)", loc="left", fontsize=12,
                  color=TEXT_PRIMARY, pad=30)
    ax1.legend(handles=[
        Line2D([], [], color=POINT_COLOR, marker="o", ls="", ms=5, label="하루"),
        Line2D([], [], color=ACCENT, lw=2.5, label="평균 (숫자)"),
        Line2D([], [], color=TEXT_PRIMARY, lw=1.2, ls=(0, (2, 2)), label="중앙값"),
    ], loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=3,
       borderaxespad=0.2)

    # 오른쪽: 기간별 요일 평균 (계절 차이에도 요일 모양이 유지되는지)
    x = np.arange(len(days))
    w = 0.36
    jan = summary.loc[days, "mean_sales_jan_aug"] / 1000
    sep = summary.loc[days, "mean_sales_sep_dec"] / 1000
    ax2.bar(x - w / 2 - 0.01, jan, width=w, color=ACCENT, label="2010-12 ~ 2011-08", zorder=2)
    ax2.bar(x + w / 2 + 0.01, sep, width=w, color=SECOND, label="2011-09 ~ 2011-12-08", zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DOW_NAMES[d] for d in days])
    ax2.set_ylabel("요일 평균 일 순매출 (천 파운드)", fontsize=10, color=TEXT_SECONDARY)
    ax2.set_title("기간별 요일 평균", loc="left", fontsize=12, color=TEXT_PRIMARY, pad=30)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=2,
               borderaxespad=0.2)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5, length=0)

    fig.text(0.01, 0.01, "대상: 2010-12-01 ~ 2011-12-08 영업일 304일. 휴무일(매출 0)은 평균에서 제외.",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=3)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    daily = add_week_index(daily)
    summary = weekday_summary(daily)

    assert summary["n_open_days"].sum() == int((daily["is_closed"] == 0).sum())
    assert abs(summary["share_of_total"].sum() - 1) < 1e-9

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    plot(daily, summary.set_index("dow"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))
    print(f"저장: {TABLE_PATH.relative_to(ROOT)}, {FIG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
