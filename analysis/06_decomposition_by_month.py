"""분해 결과(관측값·요일 효과·추세·잔차)를 월마다 별도 그림으로 그린다.

입력: outputs/tables/daily_decomposition.csv (analysis/05_decomposition.py 산출물)
출력: outputs/figures/decomposition_by_month/decomposition_YYYY-MM.png (13개)

분해는 전체 기간에서 한 번 계산한 값을 월별로 잘라 쓴다(월마다 다시 분해하지 않음).
월끼리 비교할 수 있도록 성분별 세로축 범위를 모든 월에서 같게 둔다.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMP_PATH = ROOT / "outputs" / "tables" / "daily_decomposition.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "decomposition_by_month"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BASELINE = "#8a8984"
OBS_COLOR = "#a9b8cc"
ACCENT = "#2a78d6"
POS_COLOR = "#2a78d6"
NEG_COLOR = "#eb6834"
CLOSED_SHADE = "#efeeea"

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

COLS = ["observed", "trend", "weekday_effect", "residual"]


def with_breaks(s: pd.Series) -> pd.Series:
    """연속 휴무(3일 초과)에서 선을 끊도록 빈 값을 끼워 넣는다."""
    s = s.dropna()
    gap_after = s.index.to_series().diff().shift(-1) > pd.Timedelta(days=3)
    breaks = s.index[gap_after.values] + pd.Timedelta(days=1)
    return s.reindex(s.index.union(breaks))


def padded_range(lo: float, hi: float, pad: float = 0.06) -> tuple[float, float]:
    span = hi - lo
    return lo - span * pad, hi + span * pad


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_ylabel("천 파운드", fontsize=9, color=TEXT_SECONDARY)


def plot_month(ym: str, m: pd.DataFrame, ylims: dict) -> Path:
    start = pd.Timestamp(f"{ym}-01")
    end = start + pd.offsets.MonthEnd(0)
    k = m[COLS] / 1000
    closed = m.index[m["is_closed"] == 1]
    open_m = m[m["is_closed"] == 0]

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 0.8, 1, 1]})
    fig.patch.set_facecolor(SURFACE)
    titles = ["① 관측값: 일 순매출", "② 요일 효과", "③ 추세", "④ 잔차 = 관측값 - 추세 - 요일 효과"]

    for ax, title in zip(axes, titles):
        style(ax)
        for d in closed:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=CLOSED_SHADE, lw=0, zorder=0)
        ax.set_title(title, loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=6)

    obs, trend = with_breaks(k["observed"]), with_breaks(k["trend"])
    axes[0].plot(obs.index, obs.values, color=OBS_COLOR, lw=1.5, marker="o", ms=4)
    axes[0].plot(trend.index, trend.values, color=ACCENT, lw=2)
    axes[0].text(1.0, 1.02, "회색: 관측값   파란색: 추세(③)", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY)

    for ax, col in ((axes[1], "weekday_effect"), (axes[3], "residual")):
        v = k[col].dropna()
        ax.bar(v.index, v.values, width=0.7, lw=0,
               color=[POS_COLOR if x >= 0 else NEG_COLOR for x in v.values])
        ax.axhline(0, color=BASELINE, lw=0.8)

    axes[2].plot(trend.index, trend.values, color=ACCENT, lw=2, marker="o", ms=3.5)

    r = k["residual"].dropna()
    if len(r):
        top = r.abs().idxmax()
        axes[3].annotate(f"{top:%m-%d}  {r[top]:+,.1f}", xy=(top, r[top]),
                         xytext=(6, 2 if r[top] > 0 else -10), textcoords="offset points",
                         fontsize=9, color=TEXT_PRIMARY)

    for ax, col in zip(axes, ["observed", "weekday_effect", "trend", "residual"]):
        ax.set_ylim(*ylims[col])

    axes[3].set_xlim(start - pd.Timedelta(hours=18), end + pd.Timedelta(hours=18))
    axes[3].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 10, 15, 20, 25, 30]))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%-d일"))

    partial = " (1~8일)" if end > m.index.max() + pd.Timedelta(days=1) else ""
    fig.suptitle(f"{ym}{partial} 일 순매출 분해", x=0.01, y=0.99, ha="left", va="top", fontsize=15,
                 color=TEXT_PRIMARY)
    fig.text(0.01, 0.952,
             f"월 합계 {open_m['observed'].sum() / 1000:,.1f}천 · "
             f"추세 평균 {open_m['trend'].mean() / 1000:,.1f}천 · "
             f"잔차 표준편차 {open_m['residual'].std() / 1000:,.1f}천 · "
             f"영업일 {len(open_m)}일",
             fontsize=10, color=TEXT_SECONDARY)
    fig.text(0.01, 0.005,
             "회색 세로 띠는 휴무일. ②·④ 파란색 +, 주황색 -. 분해는 전체 기간에서 계산한 값이며, "
             "성분별 세로축 범위는 모든 월에서 같음.",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.015, 1, 0.94), h_pad=1.2)

    path = FIG_DIR / f"decomposition_{ym}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    comp = pd.read_csv(COMP_PATH, parse_dates=["date"]).set_index("date")
    k = comp[COLS] / 1000
    ylims = {
        "observed": (0, k["observed"].max() * 1.05),
        "trend": (0, k["trend"].max() * 1.05),
        "weekday_effect": padded_range(k["weekday_effect"].min(), k["weekday_effect"].max()),
        "residual": padded_range(k["residual"].min(), k["residual"].max()),
    }

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comp["year_month"] = comp.index.strftime("%Y-%m")
    for ym, m in comp.groupby("year_month"):
        path = plot_month(ym, m, ylims)
        print(f"저장: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
