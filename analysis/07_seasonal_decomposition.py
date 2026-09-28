"""
일별 순매출(net_sales)을 관측값·추세·요일 효과(계절성)·잔차로 분해하고,
네 가지를 각각 달마다 분리한 그리드로 그린다 (매출 계산 자체는 statsmodels로 하고,
figure는 월별 4종 x 13개월로 나눠서 출력).
가법 분해(additive)를 쓴다: 매출이 정확히 0인 휴무일이 있어 승법(곱셈) 분해는 나눗셈 문제가 생김.
실행: 프로젝트 루트에서 `python analysis/07_seasonal_decomposition.py`
"""
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_sales.csv", index_col="date", parse_dates=True).asfreq("D")

decomp = seasonal_decompose(daily["net_sales"], model="additive", period=7)
daily["trend"] = decomp.trend
daily["seasonal"] = decomp.seasonal
daily["resid"] = decomp.resid
daily["day_of_month"] = daily.index.day

out_cols = ["net_sales", "trend", "seasonal", "resid"]
daily[out_cols].round(2).to_csv("outputs/tables/seasonal_decomposition.csv", encoding="utf-8-sig")
print("표 저장: outputs/tables/seasonal_decomposition.csv")
print(f"\n잔차 평균: {daily['resid'].mean():.2f}, 잔차 표준편차: {daily['resid'].std():.2f}")
print(f"요일 효과(7일 반복 패턴) 값:\n{decomp.seasonal.iloc[:7].round(2)}")

months = daily["year_month"].dropna().unique().tolist()
n_months = len(months)
n_cols = 4
n_rows = -(-n_months // n_cols)


def monthly_grid(column, title, filename, color, is_partial_col="is_partial_month"):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, m in enumerate(months):
        ax = axes[i]
        sub = daily.loc[daily["year_month"] == m]
        n_days = len(sub)
        tag = f" (부분월, {n_days}일)" if sub[is_partial_col].max() == 1 else f" ({n_days}일)"
        ax.plot(sub["day_of_month"], sub[column], color=color, linewidth=1.6, marker="o", markersize=2)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_title(m + tag, fontsize=10)
        ax.set_xlim(0.5, 31.5)
        ax.grid(alpha=0.3)
    for j in range(len(months), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("일 (day of month)")
    fig.supylabel("순매출 (파운드)")
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{filename}", dpi=150)
    plt.close(fig)
    print(f"그림 저장: outputs/figures/{filename}")


monthly_grid("net_sales", "① 월별 매출 그래프 (관측값)", "09_monthly_observed_grid.png", "#2e86ab")
monthly_grid("trend", "② 월별 추세 그래프 (7일 중심 이동평균 기반)", "10_monthly_trend_grid.png", "#c1440e")
monthly_grid("seasonal", "③ 월별 요일 효과 그래프 (계절성, 7일 주기 반복)", "11_monthly_seasonal_grid.png", "#6a4c93")
monthly_grid("resid", "④ 월별 잔차 그래프 (관측값 - 추세 - 요일 효과)", "12_monthly_resid_grid.png", "#5d6d7e")
