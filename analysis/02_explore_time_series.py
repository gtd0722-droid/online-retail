"""
일별 매출(net_sales) 시계열의 추세·요일 패턴을 확인한다.
실행: 프로젝트 루트에서 `python analysis/02_explore_time_series.py`
"""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_sales.csv", index_col="date", parse_dates=True)

weekday_names = ["월", "화", "수", "목", "금", "토", "일"]

# 1) 일별 순매출 추세 + 7일 이동평균 (휴무일도 0으로 포함)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily.index, daily["net_sales"], color="#9fb8c8", linewidth=1, label="일별 순매출")
ax.plot(daily.index, daily["net_sales"].rolling(7, center=True).mean(), color="#1f6f5c", linewidth=2, label="7일 이동평균")
ax.set_title(f"일별 순매출 추세 ({daily.index.min().date()} ~ {daily.index.max().date()})")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출 (파운드)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/01_daily_net_sales_trend.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/01_daily_net_sales_trend.png")

# 2) 요일별 평균 순매출 (토요일은 휴무로 항상 0)
weekday_avg = daily.groupby("dow")["net_sales"].mean().reindex(range(7))
weekday_avg.index = weekday_names

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(weekday_avg.index, weekday_avg.values, color="#2e86ab")
ax.set_title("요일별 평균 순매출")
ax.set_xlabel("요일")
ax.set_ylabel("평균 순매출 (파운드)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/02_weekday_pattern.png", dpi=150)
plt.close(fig)

weekday_avg.round(2).to_csv("outputs/tables/weekday_avg_net_sales.csv", encoding="utf-8-sig", header=["avg_net_sales"])
print("그림 저장: outputs/figures/02_weekday_pattern.png")
print("표 저장: outputs/tables/weekday_avg_net_sales.csv")

print("\n요일별 평균 순매출:")
print(weekday_avg.round(2))

print(f"\n휴무일(is_closed=1) 수: {int((daily['is_closed'] == 1).sum())}일 / 전체 {len(daily)}일")
