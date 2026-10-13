# ==========================================
# 같은 검증 fold의 전체·축소 모델 AP로 센서 축소 정책 평가
# - 같은 fold끼리 대응한 AP 평균의 비율을 사용하며 fold별 비율 평균과 구분
# - 원본 센서 수 제한과 AP 하락률 목표·최대 허용을 별도로 확인
# - threshold를 선택하거나 모델을 자동 확정하지 않음
# - 누락 기준 모델·정의 불가 AP·변환 특징은 판정 불가로 기록
# ==========================================
import math

import pandas as pd


def assess_reduction(folds, policy):
    # 입력은 호출자가 같은 실행에서 만든 experiment/fold/AP/센서 기록이다.
    columns = ["baseline", "candidate", "baseline_ap_mean", "candidate_ap_mean",
               "ap_retention_ratio", "ap_loss_ratio", "max_sensor_count_observed",
               "target_ap_loss_ratio", "max_ap_loss_ratio", "ap_status", "sensor_status"]
    
    if policy is None:
        return pd.DataFrame(columns=columns)
    
    records = []

    for baseline, candidate in policy.comparison_pairs:
        reduced = folds[folds.experiment.eq(candidate)]

        if reduced.empty:
            # 실행하지 않은 축소 후보는 결과에 넣지 않는다.
            continue

        reference = folds[folds.experiment.eq(baseline)]
        row = dict.fromkeys(columns)

        row.update(baseline=baseline, candidate=candidate,
                   target_ap_loss_ratio=policy.target_ap_loss_ratio,
                   max_ap_loss_ratio=policy.max_ap_loss_ratio,
                   ap_status="판정 불가", sensor_status="판정 불가")
        
        counts = reduced.sensor_count

        if counts.notna().all() and reduced.is_raw_sensor.all():
            row["max_sensor_count_observed"] = int(counts.max())
            row["sensor_status"] = (
                "충족" if counts.between(1, policy.max_sensor_count).all() else "센서 수 초과"
            )

        # 기준 모델 없이 후보만 실행하거나 fold 범위가 다르면 평균을 비교하지 않는다.
        if (not reference.empty and not reference.fold.duplicated().any()
                and not reduced.fold.duplicated().any()
                and set(reference.fold) == set(reduced.fold)):
            
            paired = reference[["fold", "ap"]].merge(
                reduced[["fold", "ap"]], on="fold", suffixes=("_all", "_reduced"),
                validate="one_to_one",
            )

            # NaN fold를 조용히 제외하면 서로 다른 평가 범위가 되므로 전체를 미판정한다.
            valid = all(
                pd.notna(value) and math.isfinite(value) and 0 <= value <= 1
                for value in paired[["ap_all", "ap_reduced"]].to_numpy().flat
            )

            if valid:
                full_ap, reduced_ap = paired.ap_all.mean(), paired.ap_reduced.mean()
                row.update(baseline_ap_mean=float(full_ap), candidate_ap_mean=float(reduced_ap))
            
                if full_ap > 0:
                    retention = float(reduced_ap / full_ap)
                    loss = 1 - retention
                    row.update(ap_retention_ratio=retention, ap_loss_ratio=loss)
            
                    # 경계값에서는 부동소수점 계산의 작은 오차만 허용한다.
                    if loss <= policy.target_ap_loss_ratio + 1e-12:
                        row["ap_status"] = "목표 달성"
                    elif loss <= policy.max_ap_loss_ratio + 1e-12:
                        row["ap_status"] = "허용 범위·목표 미달"
                    else:
                        row["ap_status"] = "허용 하락률 초과"

        records.append(row)
        
    return pd.DataFrame(records, columns=columns)
