"""
지수평활법 실습: 단순 지수평활 / Holt / Holt-Winters을 일별 매출에 적용하고 비교한다.
검증 구간(마지막 28일)의 예측 오차로 방법을 고른 뒤, 전체 자료로 다시 적합해 향후 14일을 예측한다.
실행: 프로젝트 루트에서 `python analysis/03_exponential_smoothing.py`
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

daily = pd.read_csv("data/processed/daily_revenue.csv", index_col="date", parse_dates=True)
daily = daily.asfreq("D")
y = daily["revenue"]

SEASON = 7  # 요일 주기 (토요일 휴무가 반복되는 패턴)
TEST_LEN = 28  # 최근 4주를 검증 구간으로 둔다 (계절 주기 4회 포함)

train, test = y.iloc[:-TEST_LEN], y.iloc[-TEST_LEN:]
print(f"학습 구간: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")
print(f"검증 구간: {test.index.min().date()} ~ {test.index.max().date()} ({len(test)}일)")


def mae(actual, pred):
    return float(np.mean(np.abs(actual.values - pred.values)))


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual.values - pred.values) ** 2)))


results = {}
forecasts = {}

# 1) 단순 지수평활 (수준만 갱신) - 평활계수를 최적화로 탐색
ses_fit = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
ses_fc = ses_fit.forecast(TEST_LEN)
results["단순 지수평활 (alpha 최적화)"] = (ses_fit.model.params["smoothing_level"], mae(test, ses_fc), rmse(test, ses_fc))
forecasts["단순 지수평활"] = ses_fc

# 1-1) 강의 예시처럼 alpha를 고정값(0.2/0.5/0.8)으로 바꿔가며 비교
for a in (0.2, 0.5, 0.8):
    fit_a = SimpleExpSmoothing(train, initialization_method="estimated").fit(smoothing_level=a, optimized=False)
    fc_a = fit_a.forecast(TEST_LEN)
    results[f"단순 지수평활 (alpha={a})"] = (a, mae(test, fc_a), rmse(test, fc_a))

# 2) Holt (수준 + 추세)
holt_fit = Holt(train, initialization_method="estimated").fit(optimized=True)
holt_fc = holt_fit.forecast(TEST_LEN)
results["Holt (추세)"] = (None, mae(test, holt_fc), rmse(test, holt_fc))
forecasts["Holt"] = holt_fc

# 2-1) Holt 감쇠 추세
holt_damped_fit = Holt(train, damped_trend=True, initialization_method="estimated").fit(optimized=True)
holt_damped_fc = holt_damped_fit.forecast(TEST_LEN)
results["Holt (감쇠 추세)"] = (None, mae(test, holt_damped_fc), rmse(test, holt_damped_fc))

# 3) Holt-Winters (수준 + 추세 + 계절성, 가법형)
# 매출이 정확히 0(휴무)인 날이 있어 승법형은 적용하지 않는다 (0 나눗셈 문제, 슬라이드 30 참고).
hw_fit = ExponentialSmoothing(
    train, trend="add", seasonal="add", seasonal_periods=SEASON, initialization_method="estimated"
).fit(optimized=True)
hw_fc = hw_fit.forecast(TEST_LEN)
results["Holt-Winters (가법 계절성)"] = (None, mae(test, hw_fc), rmse(test, hw_fc))
forecasts["Holt-Winters"] = hw_fc

# 3-1) Holt-Winters 감쇠 추세
hw_damped_fit = ExponentialSmoothing(
    train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=SEASON, initialization_method="estimated"
).fit(optimized=True)
hw_damped_fc = hw_damped_fit.forecast(TEST_LEN)
results["Holt-Winters (감쇠 추세)"] = (None, mae(test, hw_damped_fc), rmse(test, hw_damped_fc))

# 비교 표 저장
comparison = pd.DataFrame(results, index=["alpha", "MAE", "RMSE"]).T
comparison = comparison.sort_values("MAE")
comparison.to_csv("outputs/tables/model_comparison.csv", encoding="utf-8-sig")
print("\n검증 구간(28일) 예측 오차 비교:")
print(comparison.round(2))

# 대표 세 모형(단순/Holt/Holt-Winters)만 그래프로 비교
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index[-42:], train.iloc[-42:], color="#333333", linewidth=1, label="학습 자료 (최근 6주)")
ax.plot(test.index, test, color="#333333", linewidth=1.5, linestyle="--", label="실제 매출 (검증)")
colors = {"단순 지수평활": "#e07a5f", "Holt": "#81b29a", "Holt-Winters": "#3d5a80"}
for name, fc in forecasts.items():
    ax.plot(fc.index, fc, color=colors[name], linewidth=2, label=f"{name} 예측")
ax.set_title("검증 구간(마지막 28일) 예측 비교")
ax.set_xlabel("날짜")
ax.set_ylabel("매출 (파운드)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/03_forecast_comparison.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/03_forecast_comparison.png")

# 검증 MAE가 가장 낮은 방법을 선택해 전체 자료로 다시 적합하고, 향후 14일을 예측한다.
# 검증 비교에서는 Holt-Winters(가법 계절성) 계열이 항상 가장 낮은 오차를 보였으므로,
# 감쇠 추세 적용 여부만 검증 결과에 따라 선택한다.
best_name = comparison.index[0]
print(f"\n선택한 방법: {best_name} (검증 MAE 기준)")
use_damped = "감쇠" in best_name

FUTURE_LEN = 14
final_fit = ExponentialSmoothing(
    y, trend="add", damped_trend=use_damped, seasonal="add", seasonal_periods=SEASON,
    initialization_method="estimated",
).fit(optimized=True)
future_fc = final_fit.forecast(FUTURE_LEN)
n_clipped = int((future_fc < 0).sum())
future_fc_clipped = future_fc.clip(lower=0)
if n_clipped:
    print(f"참고: 가법 계절성 특성상 예측값이 음수로 나온 날짜 {n_clipped}일은 0으로 표시합니다.")

future_table = pd.DataFrame({"forecast_revenue": future_fc_clipped.round(2)})
future_table.index.name = "date"
future_table.to_csv("outputs/tables/future_forecast_14d.csv", encoding="utf-8-sig")
print(f"\n향후 14일 예측 ({best_name}):")
print(future_table)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y.index[-60:], y.iloc[-60:], color="#333333", linewidth=1.2, label="실제 매출 (최근 60일)")
ax.plot(future_fc_clipped.index, future_fc_clipped, color="#3d5a80", linewidth=2, marker="o", markersize=3, label="향후 14일 예측")
ax.set_title("Holt-Winters 향후 14일 매출 예측")
ax.set_xlabel("날짜")
ax.set_ylabel("매출 (파운드)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/04_future_forecast.png", dpi=150)
plt.close(fig)
print("그림 저장: outputs/figures/04_future_forecast.png")
print("표 저장: outputs/tables/model_comparison.csv, outputs/tables/future_forecast_14d.csv")
