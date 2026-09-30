"""SARIMA로 2011-12 일 순매출 예측 (12_december_forecast.py와 같은 조건).

학습: 2011-01-01 ~ 2011-11-30 (6일 주기 시계열, 토요일 외 휴무일은 전·후 주 평균으로 채움)
차수: 14_sarima_forecast.py에서 학습 구간(1~10월) 내부 검증으로 선택한 차수를 그대로 사용한다.
      (12월 자료를 차수 선택에 쓰지 않는다. 1~11월로 계수만 다시 추정한다.)
예측: 2011-12-01 ~ 2011-12-23 중 토요일이 아닌 20일. 12-24 이후는 휴무 일정을 알 수 없어 예측하지 않는다.

입력: data/processed/daily_sales.csv, outputs/tables/sarima_parameters.csv
출력: outputs/tables/december_forecast_sarima.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"

_spec = importlib.util.spec_from_file_location("s14", ROOT / "analysis" / "14_sarima_forecast.py")
s14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s14)
hw11 = s14.hw11

TRAIN = ("2011-01-01", "2011-11-30")


def main() -> None:
    par = pd.read_csv(TABLES / "sarima_parameters.csv").set_index("name")["value"]
    order = tuple(int(x) for x in str(par["order"]).strip("()").split(","))
    sorder = tuple(int(x) for x in str(par["seasonal_order"]).split(")")[0].strip("(").split(","))
    log = bool(int(float(par["log_transform"])))
    trend = par["trend"]

    daily = pd.read_csv(hw11.DAILY_PATH, parse_dates=["date"]).set_index("date")
    y_slots, filled = hw11.build_series(daily, TRAIN)
    fdates = pd.date_range("2011-12-01", "2011-12-23")
    fdates = fdates[fdates.dayofweek != 5]
    assert y_slots.index[-1] == pd.Timestamp("2011-11-30") and fdates[0].dayofweek == 3
    h = len(fdates)

    fit = s14.fit_sarima(np.log(y_slots) if log else y_slots, order, sorder, trend)
    mean, ci = s14.forecast(fit, h, log)
    out = pd.DataFrame({"sarima_forecast": mean, "lower95": ci[:, 0], "upper95": ci[:, 1]}, index=fdates)
    out.index.name = "date"
    out.reset_index().to_csv(TABLES / "december_forecast_sarima.csv", index=False, encoding="utf-8-sig")

    prev = pd.read_csv(TABLES / "december_forecast.csv", parse_dates=["date"]).set_index("date")
    known = prev.dropna(subset=["actual"])
    e = known["actual"] - out.loc[known.index, "sarima_forecast"]
    print(f"차수 {order}x{sorder}[{s14.S}] 로그={log} 추세={trend}, 채움 {len(filled)}건")
    print(out.round(0).to_string())
    print(f"12-01~12-08 참고 비교(영업일 {len(known)}일): MAE {e.abs().mean():,.0f}  RMSE {np.sqrt((e**2).mean()):,.0f}  "
          f"MAPE {(e.abs()/known['actual']).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
