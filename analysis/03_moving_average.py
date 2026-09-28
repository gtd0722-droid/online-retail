"""
일별 매출(net_sales)에 후행(trailing) 이동평균을 적용해 추세를 확인한다.
후행 이동평균은 해당 날짜와 그 이전 날짜만 사용한다 (미래 값을 참조하지 않음).
실행: 프로젝트 루트에서 `python analysis/03_moving_average.py`
"""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_sales.csv", index_col="date", parse_dates=True)

WINDOWS = [7, 14]  # 일 단위, 후행(trailing)

ma = pd.DataFrame(index=daily.index)
ma["net_sales"] = daily["net_sales"]
for w in WINDOWS:
    # rolling()의 기본 동작이 후행 이동평균: t 시점 값은 [t-w+1, t] 구간의 평균만 사용
    ma[f"ma_{w}d"] = daily["net_sales"].rolling(window=w, min_periods=w).mean()

ma.round(2).to_csv("outputs/tables/daily_net_sales_moving_average.csv", encoding="utf-8-sig")
print("표 저장: outputs/tables/daily_net_sales_moving_average.csv")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily.index, daily["net_sales"], color="#c9d6de", linewidth=1, label="일별 순매출")
colors = {7: "#1f6f5c", 14: "#c1440e"}
for w in WINDOWS:
    ax.plot(ma.index, ma[f"ma_{w}d"], color=colors[w], linewidth=2, label=f"{w}일 후행 이동평균")
ax.set_title(f"일별 순매출과 후행 이동평균 ({daily.index.min().date()} ~ {daily.index.max().date()})")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출 (파운드)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/03_moving_average_7_14d.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/03_moving_average_7_14d.png")

print("\n최근 10일 (원 값 vs 이동평균):")
print(ma.tail(10))
