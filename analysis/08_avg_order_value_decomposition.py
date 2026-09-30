"""주문당 평균 금액을 추세 + 요일 효과 + 잔차로 분해한다 (05_decomposition.py와 같은 방식).

주문당 평균 금액 = gross_sales / n_invoices (정상 판매 행 기준, 취소 제외)
  - n_invoices가 정상 판매 송장 수이므로 분자도 정상 판매 금액(gross_sales)으로 맞춘다.
  - 일별 값의 단순 이동평균을 쓰므로 송장 수가 적은 날도 같은 비중을 가진다.

입력: data/processed/daily_sales.csv
출력: outputs/tables/aov_decomposition.csv
      outputs/tables/aov_decomposition_weekday_effect.csv
      outputs/tables/aov_residual_check_days.csv
      outputs/figures/avg_order_value_decomposition.png
"""
import importlib.util
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily_sales.csv"
COMP_PATH = ROOT / "outputs" / "tables" / "aov_decomposition.csv"
EFFECT_PATH = ROOT / "outputs" / "tables" / "aov_decomposition_weekday_effect.csv"
CHECK_PATH = ROOT / "outputs" / "tables" / "aov_residual_check_days.csv"
FIG_PATH = ROOT / "outputs" / "figures" / "avg_order_value_decomposition.png"

# 05_decomposition.py의 분해·스타일 함수를 재사용한다 (파일명이 숫자로 시작해 importlib 사용).
_spec = importlib.util.spec_from_file_location("decomp05", ROOT / "analysis" / "05_decomposition.py")
d05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d05)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CHECK_Z = 2.5  # 잔차 로버스트 z 기준


def decompose_aov(daily: pd.DataFrame):
    d = daily.copy()
    # 영업일의 n_invoices는 모두 1 이상이다. 휴무일은 NaN으로 두어 분해에서 제외한다.
    d["aov"] = (d["gross_sales"] / d["n_invoices"]).where(d["is_closed"] == 0)
    d["net_sales"] = d["aov"]  # 05의 decompose가 net_sales 열을 사용하므로 값만 바꿔 전달
    comp, effect = d05.decompose(d)
    return comp, effect


def plot(comp: pd.DataFrame) -> None:
    k = comp[["observed", "trend", "weekday_effect", "residual"]]
    open_idx = k["observed"].dropna().index
    gap_after = open_idx.to_series().diff().shift(-1) > pd.Timedelta(days=3)
    breaks = open_idx[gap_after.values] + pd.Timedelta(days=1)
    line = k.loc[open_idx].reindex(open_idx.union(breaks))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12.5), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 0.8, 1, 1]})
    fig.patch.set_facecolor(d05.SURFACE)
    xmas = (pd.Timestamp("2010-12-23 12:00"), pd.Timestamp("2011-01-03 12:00"))

    panels = [
        ("① 관측값: 일 주문당 평균 금액", "observed"),
        ("② 요일 효과 (6개 요일 평균 = 0)", "weekday_effect"),
        ("③ 추세 (요일 효과를 뺀 7일 이동평균)", "trend"),
        ("④ 잔차 = 관측값 - 추세 - 요일 효과", "residual"),
    ]
    for ax, (title, _) in zip(axes, panels):
        d05.style(ax)
        ax.axvspan(*xmas, color=d05.CLOSED_SHADE, lw=0, zorder=0)
        ax.set_title(title, loc="left", fontsize=12, color=d05.TEXT_PRIMARY, pad=8)
        ax.set_ylabel("파운드/주문", fontsize=9, color=d05.TEXT_SECONDARY)

    axes[0].plot(line.index, line["observed"], color=d05.OBS_COLOR, lw=1.2)
    axes[0].plot(line.index, line["trend"], color=d05.ACCENT, lw=2, alpha=0.9)
    axes[0].text(1.0, 1.02, "회색: 관측값   파란색: 추세(③)", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9, color=d05.TEXT_SECONDARY)
    axes[0].set_ylim(0, None)

    s = k["weekday_effect"].dropna()
    axes[1].bar(s.index, s.values, width=0.8, lw=0,
                color=[d05.POS_COLOR if v >= 0 else d05.NEG_COLOR for v in s.values])
    axes[1].axhline(0, color=d05.BASELINE, lw=0.8)

    axes[2].plot(line.index, line["trend"], color=d05.ACCENT, lw=2)

    r = k["residual"].dropna()
    axes[3].bar(r.index, r.values, width=0.8, lw=0,
                color=[d05.POS_COLOR if v >= 0 else d05.NEG_COLOR for v in r.values])
    axes[3].axhline(0, color=d05.BASELINE, lw=0.8)
    for date in r.abs().nlargest(4).index:
        v = r[date]
        axes[3].annotate(f"{date:%m-%d}  {v:+,.0f}", xy=(date, v),
                         xytext=(5, 3 if v > 0 else -10), textcoords="offset points",
                         fontsize=8.5, color=d05.TEXT_PRIMARY)

    axes[3].xaxis.set_major_locator(mdates.MonthLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3].set_xlim(comp.index.min() - pd.Timedelta(days=2),
                     comp.index.max() + pd.Timedelta(days=2))

    fig.suptitle("일 주문당 평균 금액 분해: 관측값 = 추세 + 요일 효과 + 잔차", x=0.01, ha="left",
                 fontsize=15, color=d05.TEXT_PRIMARY)
    fig.text(0.01, 0.005,
             "주문당 평균 금액 = 정상 판매 금액 / 정상 판매 송장 수. 영업일 304일 대상. "
             "휴무일은 비움, 회색 구간은 연말연시 휴무. ②·④의 파란색은 +, 주황색은 -.",
             fontsize=8.5, color=d05.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97), h_pad=1.5)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, facecolor=d05.SURFACE)


def main() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"]).set_index("date")
    comp, effect = decompose_aov(daily)

    open_comp = comp[comp["is_closed"] == 0]
    recon = open_comp["trend"] + open_comp["weekday_effect"] + open_comp["residual"]
    assert (recon - open_comp["observed"]).abs().max() < 1e-6
    assert abs(effect.sum()) < 1e-6

    means = open_comp.groupby("dow")["observed"].mean()
    effect_table = pd.DataFrame({
        "weekday": [d05.DOW_NAMES[d] for d in effect.index],
        "effect_decomposition": effect.values,
        "effect_simple": (means - means.mean()).reindex(effect.index).values,
    }, index=effect.index.rename("dow"))

    # 잔차 점검일: 중앙값·MAD 기준 로버스트 z
    res = open_comp["residual"]
    mad = (res - res.median()).abs().median() * 1.4826
    z = (res - res.median()) / mad
    check = daily.loc[open_comp.index, ["dow", "gross_sales", "n_invoices", "n_customers",
                                        "quantity", "n_lines", "cancel_amount"]].copy()
    check["aov"] = open_comp["observed"]
    check["trend"] = open_comp["trend"]
    check["residual"] = res
    check["residual_z"] = z
    check["qty_per_invoice"] = check["quantity"] / check["n_invoices"]
    check["lines_per_invoice"] = check["n_lines"] / check["n_invoices"]
    check = check[z.abs() > CHECK_Z].sort_values("residual_z")

    COMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp.reset_index().to_csv(COMP_PATH, index=False, encoding="utf-8-sig")
    effect_table.reset_index().to_csv(EFFECT_PATH, index=False, encoding="utf-8-sig")
    check.reset_index().to_csv(CHECK_PATH, index=False, encoding="utf-8-sig")
    plot(comp)

    var = open_comp[["observed", "trend", "weekday_effect", "residual"]].var()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(effect_table.round(1).to_string())
    print("\n분산 비율(관측값 대비):", (var / var["observed"]).round(3).to_dict())
    print("추세 min/max:", open_comp["trend"].idxmin().date(), round(open_comp["trend"].min(), 1),
          open_comp["trend"].idxmax().date(), round(open_comp["trend"].max(), 1))
    print(f"\n잔차 |z|>{CHECK_Z} 점검일 {len(check)}일")
    print(check[["dow", "aov", "trend", "residual", "residual_z", "n_invoices", "quantity", "qty_per_invoice", "gross_sales"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
