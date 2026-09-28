"""
전체 기간(2010-12-01~2011-12-08)의 요일별 매출 통계를 정리한다.
휴무일을 포함한 전체 평균과, 휴무일을 뺀 영업일 기준 통계를 함께 낸다
(토요일은 항상 휴무라 전체 평균이 0에 가깝게 나오므로, 영업일만의 분포를 따로 봐야 비교가 된다).
실행: 프로젝트 루트에서 `python analysis/06_weekday_stats.py`
"""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_sales.csv", index_col="date", parse_dates=True)
weekday_names = ["월", "화", "수", "목", "금", "토", "일"]

all_stats = daily.groupby("dow")["net_sales"].agg(n_days="size", mean_all="mean")

open_daily = daily.loc[daily["is_closed"] == 0]
open_stats = open_daily.groupby("dow")["net_sales"].agg(
    n_open="size", mean_open="mean", median_open="median", std_open="std", min_open="min", max_open="max"
)

stats = all_stats.join(open_stats).reindex(range(7))
stats["n_closed"] = stats["n_days"] - stats["n_open"]
stats.index = weekday_names
stats.index.name = "요일"
stats = stats[["n_days", "n_open", "n_closed", "mean_all", "mean_open", "median_open", "std_open", "min_open", "max_open"]]
stats = stats.round(2)

stats.to_csv("outputs/tables/weekday_stats.csv", encoding="utf-8-sig")
print("표 저장: outputs/tables/weekday_stats.csv\n")
print(stats)

# 그래프 1: 전체 평균 vs 영업일 평균 막대 비교
fig, ax = plt.subplots(figsize=(9, 5))
x = range(7)
w = 0.35
ax.bar([i - w / 2 for i in x], stats["mean_all"], width=w, color="#9fb8c8", label="전체 평균 (휴무 포함)")
ax.bar([i + w / 2 for i in x], stats["mean_open"], width=w, color="#2e86ab", label="영업일 평균")
ax.set_xticks(list(x))
ax.set_xticklabels(weekday_names)
ax.set_title("요일별 평균 순매출: 전체 vs 영업일만")
ax.set_xlabel("요일")
ax.set_ylabel("순매출 (파운드)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/07_weekday_mean_compare.png", dpi=150)
plt.close(fig)
print("\n그림 저장: outputs/figures/07_weekday_mean_compare.png")

# 그래프 2: 요일별 영업일 매출 분포 (박스플롯) - 평균만으로는 안 보이는 변동성을 확인
fig, ax = plt.subplots(figsize=(9, 5))
box_data = [open_daily.loc[open_daily["dow"] == d, "net_sales"] for d in range(7)]
ax.boxplot(box_data, tick_labels=weekday_names, showmeans=True)
ax.set_title("요일별 영업일 순매출 분포 (박스플롯)")
ax.set_xlabel("요일")
ax.set_ylabel("순매출 (파운드)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/08_weekday_boxplot.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/08_weekday_boxplot.png")
