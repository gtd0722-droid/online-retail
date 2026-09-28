"""
월별 합계·일평균 매출을 구하고, 월별 매출·이동평균 그래프를 달마다 분리해서 그린다.
이동평균은 월 경계에 상관없이 전체 기간으로 먼저 계산한 뒤, 달별로 잘라서 보여준다
(그래야 달 첫머리의 이동평균이 이전 달 값을 정상적으로 반영한다).
실행: 프로젝트 루트에서 `python analysis/05_monthly_breakdown.py`
"""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_sales.csv", index_col="date", parse_dates=True)
daily["day_of_month"] = daily.index.day

WINDOWS = [7, 14]
for w in WINDOWS:
    daily[f"ma_{w}d"] = daily["net_sales"].rolling(window=w, min_periods=w).mean()

# ---------------------------------------------------------------
# 1) 월별 합계 · 일평균 (날짜 수로 나눈 값)
# ---------------------------------------------------------------
monthly = daily.groupby("year_month").agg(
    total_net_sales=("net_sales", "sum"),
    mean_net_sales_per_day=("net_sales", "mean"),
    n_days=("net_sales", "size"),
    is_partial_month=("is_partial_month", "max"),
)
monthly = monthly.round(2)
monthly.to_csv("outputs/tables/monthly_summary.csv", encoding="utf-8-sig")
print("표 저장: outputs/tables/monthly_summary.csv")
print("\n월별 합계 · 일평균:")
print(monthly)

months = monthly.index.tolist()
n_months = len(months)
n_cols = 4
n_rows = -(-n_months // n_cols)  # 올림 나눗셈


def month_title(m):
    n_days = int(monthly.loc[m, "n_days"])
    tag = f" (부분월, {n_days}일)" if monthly.loc[m, "is_partial_month"] == 1 else f" ({n_days}일)"
    return m + tag


# ---------------------------------------------------------------
# 2) 월별 매출 그래프 (달마다 분리)
# ---------------------------------------------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
axes = axes.flatten()
for i, m in enumerate(months):
    ax = axes[i]
    sub = daily.loc[daily["year_month"] == m]
    ax.bar(sub["day_of_month"], sub["net_sales"], color="#2e86ab", width=0.8)
    ax.set_title(month_title(m), fontsize=10)
    ax.set_xlim(0.5, 31.5)
    ax.grid(axis="y", alpha=0.3)
for j in range(len(months), len(axes)):
    axes[j].axis("off")
fig.suptitle("월별 일별 순매출", fontsize=14)
fig.supxlabel("일 (day of month)")
fig.supylabel("순매출 (파운드)")
fig.tight_layout()
fig.savefig("outputs/figures/05_monthly_net_sales_grid.png", dpi=150)
plt.close(fig)
print("\n그림 저장: outputs/figures/05_monthly_net_sales_grid.png")

# ---------------------------------------------------------------
# 3) 월별 이동평균 그래프 (달마다 분리, 전체 기간으로 계산한 값을 자름)
# ---------------------------------------------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
axes = axes.flatten()
colors = {7: "#1f6f5c", 14: "#c1440e"}
for i, m in enumerate(months):
    ax = axes[i]
    sub = daily.loc[daily["year_month"] == m]
    for w in WINDOWS:
        ax.plot(sub["day_of_month"], sub[f"ma_{w}d"], color=colors[w], linewidth=1.8, label=f"{w}일 후행 이동평균")
    ax.set_title(month_title(m), fontsize=10)
    ax.set_xlim(0.5, 31.5)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")
for j in range(len(months), len(axes)):
    axes[j].axis("off")
fig.suptitle("월별 순매출 이동평균 (7일·14일 후행, 전체 기간 기준 계산)", fontsize=14)
fig.supxlabel("일 (day of month)")
fig.supylabel("순매출 (파운드)")
fig.tight_layout()
fig.savefig("outputs/figures/06_monthly_moving_average_grid.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/06_monthly_moving_average_grid.png")
