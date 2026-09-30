"""일별 순매출을 추세 + 요일 효과 + 잔차로 분해한다 (덧셈 모형).

입력: data/processed/daily_sales.csv (analysis/01_preprocess.py 산출물)
출력: outputs/tables/daily_decomposition.csv
      outputs/tables/decomposition_weekday_effect.csv
      outputs/figures/sales_decomposition.png

방법 (영업일만 대상, 휴무일은 모든 성분에서 비움):
  1) 추세 T0 = 달력 7일 가운데 정렬 창 안의 영업일 평균
  2) 요일 효과 S = 요일별 (y - T0) 평균, 6개 요일 평균이 0이 되도록 중심화
  3) 공휴일로 창 안의 요일 구성이 어긋나는 영향을 줄이기 위해, 요일 효과를 뺀 값(y - S)으로
     추세를 다시 계산(T)하고, S를 (y - T)로 다시 추정한다 (1회 반복)
  4) 잔차 R = y - T - S
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
COMP_PATH = ROOT / "outputs" / "tables" / "daily_decomposition.csv"
EFFECT_PATH = ROOT / "outputs" / "tables" / "decomposition_weekday_effect.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "sales_decomposition.png"

WINDOW = 7
DOW_NAMES = ["월", "화", "수", "목", "금", "토", "일"]

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


def rolling_open_mean(values: pd.Series) -> pd.Series:
    return values.rolling(WINDOW, center=True, min_periods=1).mean()


def weekday_effect(detrended: pd.Series, dow: pd.Series) -> pd.Series:
    effect = detrended.groupby(dow).mean()
    return effect - effect.mean()


def decompose(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = daily["net_sales"].where(daily["is_closed"] == 0)
    dow = daily["dow"]

    trend0 = rolling_open_mean(y)
    effect0 = weekday_effect(y - trend0, dow)

    trend = rolling_open_mean(y - dow.map(effect0).where(y.notna()))
    effect = weekday_effect(y - trend, dow)

    seasonal = dow.map(effect).where(y.notna())
    trend = trend.where(y.notna())
    comp = pd.DataFrame({
        "dow": dow,
        "is_closed": daily["is_closed"],
        "observed": y,
        "trend": trend,
        "weekday_effect": seasonal,
        "residual": y - trend - seasonal,
    })
    return comp, effect


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def plot(comp: pd.DataFrame) -> None:
    k = comp[["observed", "trend", "weekday_effect", "residual"]] / 1000
    # 연속 휴무(연말연시)는 선을 끊고, 하루짜리 휴무는 앞뒤 영업일을 잇는다.
    open_idx = k["observed"].dropna().index
    gap_after = open_idx.to_series().diff().shift(-1) > pd.Timedelta(days=3)
    breaks = open_idx[gap_after.values] + pd.Timedelta(days=1)
    line = k.loc[open_idx].reindex(open_idx.union(breaks))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12.5), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 0.8, 1, 1]})
    fig.patch.set_facecolor(SURFACE)
    xmas = (pd.Timestamp("2010-12-23 12:00"), pd.Timestamp("2011-01-03 12:00"))

    panels = [
        ("① 관측값: 일 순매출", "observed"),
        ("② 요일 효과 (6개 요일 평균 = 0)", "weekday_effect"),
        ("③ 추세 (요일 효과를 뺀 7일 이동평균)", "trend"),
        ("④ 잔차 = 관측값 − 추세 − 요일 효과", "residual"),
    ]
    for ax, (title, col) in zip(axes, panels):
        style(ax)
        ax.axvspan(*xmas, color=CLOSED_SHADE, lw=0, zorder=0)
        ax.set_title(title.replace("−", "-"), loc="left", fontsize=12, color=TEXT_PRIMARY, pad=8)
        ax.set_ylabel("천 파운드", fontsize=9, color=TEXT_SECONDARY)

    axes[0].plot(line.index, line["observed"], color=OBS_COLOR, lw=1.2)
    axes[0].plot(line.index, line["trend"], color=ACCENT, lw=2, alpha=0.9)
    axes[0].text(1.0, 1.02, "회색: 관측값   파란색: 추세(③)", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_ylim(0, None)

    s = k["weekday_effect"].dropna()
    axes[1].bar(s.index, s.values, width=0.8, lw=0,
                color=[POS_COLOR if v >= 0 else NEG_COLOR for v in s.values])
    axes[1].axhline(0, color=BASELINE, lw=0.8)

    axes[2].plot(line.index, line["trend"], color=ACCENT, lw=2)

    r = k["residual"].dropna()
    axes[3].bar(r.index, r.values, width=0.8, lw=0,
                color=[POS_COLOR if v >= 0 else NEG_COLOR for v in r.values])
    axes[3].axhline(0, color=BASELINE, lw=0.8)
    for date in r.abs().nlargest(3).index:
        v = r[date]
        axes[3].annotate(f"{date:%m-%d}  {v:+,.1f}", xy=(date, v),
                         xytext=(5, 3 if v > 0 else -10), textcoords="offset points",
                         fontsize=8.5, color=TEXT_PRIMARY)

    axes[3].xaxis.set_major_locator(mdates.MonthLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3].set_xlim(comp.index.min() - pd.Timedelta(days=2),
                     comp.index.max() + pd.Timedelta(days=2))

    fig.suptitle("일 순매출 분해: 관측값 = 추세 + 요일 효과 + 잔차", x=0.01, ha="left",
                 fontsize=15, color=TEXT_PRIMARY)
    fig.text(0.01, 0.005,
             "영업일 304일 대상. 휴무일(토요일·공휴일)은 비움, 회색 구간은 연말연시 휴무. "
             "②·④의 파란색은 +, 주황색은 -. 세로 격자는 월 경계.",
             fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97), h_pad=1.5)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=SURFACE)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    comp, effect = decompose(daily)

    open_comp = comp[comp["is_closed"] == 0]
    recon = open_comp["trend"] + open_comp["weekday_effect"] + open_comp["residual"]
    assert (recon - open_comp["observed"]).abs().max() < 1e-6
    assert abs(effect.sum()) < 1e-6

    # 요일 효과 비교: 단순 방식(요일 평균 - 6개 요일 평균의 평균)과 분해 방식
    means = open_comp.groupby("dow")["observed"].mean()
    effect_table = pd.DataFrame({
        "weekday": [DOW_NAMES[d] for d in effect.index],
        "effect_decomposition": effect.values,
        "effect_simple": (means - means.mean()).reindex(effect.index).values,
    }, index=effect.index.rename("dow"))

    COMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp.reset_index().to_csv(COMP_PATH, index=False, encoding="utf-8-sig")
    effect_table.reset_index().to_csv(EFFECT_PATH, index=False, encoding="utf-8-sig")
    plot(comp)

    var = open_comp[["observed", "trend", "weekday_effect", "residual"]].var()
    pd.set_option("display.width", 200)
    print(effect_table.to_string())
    print("\n분산:", var.round(0).to_dict())
    print("분산 비율(관측값 대비):", (var / var["observed"]).round(3).to_dict())
    print("잔차 표준편차 1~8월 / 9월~:",
          open_comp.loc[:"2011-08-31", "residual"].std().round(1),
          open_comp.loc["2011-09-01":, "residual"].std().round(1))
    print("잔차 상위:", open_comp["residual"].abs().nlargest(5).round(0).to_dict())
    print("추세 min/max:", open_comp["trend"].idxmin(), round(open_comp["trend"].min()),
          open_comp["trend"].idxmax(), round(open_comp["trend"].max()))


if __name__ == "__main__":
    main()
