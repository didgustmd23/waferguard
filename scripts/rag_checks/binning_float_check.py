"""구간 경계 병합(np.unique)이 부동소수점 근사 경계를 합치지 못하는 경우를 합성 배열로 확인한다(제품 코드 미사용).

제품 transform._fit_numeric과 같은 식(분위수: np.quantile(prepared, linspace(0,1,n+1)), 같은 폭: linspace(min,max,n+1),
그 뒤 np.unique(boundaries)[1:-1])을 numpy로 직접 계산한다.
"""
import json
from pathlib import Path
import numpy as np

cases = []
for label, values, strategy, n_bins in (
        ("거의 같은 두 값(1.0, 1.0+2^-52)", np.array([1.0] * 50 + [np.nextafter(1.0, 2.0)] * 50), "uniform", 5),
        ("거의 같은 두 값(1.0, 1.0+2^-52)", np.array([1.0] * 50 + [np.nextafter(1.0, 2.0)] * 50), "quantile", 5),
        ("0.1 단위 세 값", np.array([0.1] * 40 + [0.2] * 30 + [0.3] * 30), "quantile", 10),
        ("완전 상수", np.array([3.0] * 100), "uniform", 5)):
    boundaries = (np.quantile(values, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile"
                  else np.linspace(values.min(), values.max(), n_bins + 1))
    edges = np.unique(boundaries)[1:-1]
    bins = np.searchsorted(edges, values, side="right")
    cases.append({"case": label, "strategy": strategy, "n_bins": n_bins, "distinct_values": int(len(np.unique(values))),
                  "edges": [repr(float(e)) for e in edges], "bins_used": int(len(np.unique(bins))), "max_bins": int(len(edges) + 1)})
Path(__file__).with_suffix('.json').write_text(json.dumps({"numpy": np.__version__, "cases": cases}, ensure_ascii=False, indent=1), encoding='utf-8')
