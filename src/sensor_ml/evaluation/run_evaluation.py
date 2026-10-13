# ==========================================
# 변경 내용에 따라 OOF와 시간 검증의 실행 범위 관리
# - 전처리·모델·데이터·split 변경 시 OOF부터 다시 생성
# - threshold 변경 시 같은 실행 폴더의 OOF 비교표를 사용
# - 선행 단계 실패 시 후속 평가를 중단하고 실행 상태 기록
# - OOF 대상 모델은 호출자가 선택하며 Test는 사용하지 않음
# ==========================================

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from src.threshold_policy import policy_candidates, select_policy_threshold
    from src.sensor_ml.experiments.step7_time_validation import compare_time_validation_from_files
    from src.sensor_ml.experiments.step8_threshold_oof import EXPERIMENT_NAME, run_threshold_oof_from_file
except ModuleNotFoundError:
    from modeling_config import DEFAULT_CONFIG_PATH, load_modeling_config
    from threshold_policy import policy_candidates, select_policy_threshold
    from src.sensor_ml.experiments.step7_time_validation import compare_time_validation_from_files
    from src.sensor_ml.experiments.step8_threshold_oof import EXPERIMENT_NAME, run_threshold_oof_from_file


CHANGES = ("preprocessing", "model", "data", "split", "threshold", "policy")


def _validate_oof_reuse(state_path, threshold_path, config, train_path, change, threshold, experiment_name):
    """OOF 재사용 전 성공 상태·학습 조건·모델 일치 여부를 확인한다."""

    # 변경 유형은 호출자가 선언한다. 코드 변경을 자동 감지하는 해시는 사용하지 않는다.
    if change == "threshold" and threshold is None:
        raise ValueError("threshold 변경에는 --threshold 값을 명시하세요.")
    
    if change == "policy" and threshold is not None:
        raise ValueError("정책 변경 시 --threshold를 지정하지 마세요. OOF에서 자동 선택합니다.")
    
    if not state_path.exists() or not threshold_path.exists():
        raise ValueError("성공한 선행 OOF 실행이 없습니다. 전처리·모델·데이터 변경 범위로 먼저 실행하세요.")
    
    previous = json.loads(state_path.read_text(encoding="utf-8"))

    if previous.get("status") != "completed" or not previous.get("oof_ready"):
        raise ValueError("이전 실행이 완료되지 않았습니다. OOF부터 다시 실행하세요.")

    # 정책만 달라진 설정은 재사용하되 학습 조건·입력 경로 변경은 차단한다.
    current = json.loads(json.dumps(asdict(config), default=str))
    original = dict(previous.get("oof_config", previous.get("config", {})))
    current.pop("threshold_policy", None)
    original.pop("threshold_policy", None)

    if current != original or previous.get("train_path") != str(Path(train_path).resolve()):
        raise ValueError("학습 설정 또는 Train 경로가 변경됐습니다. OOF부터 다시 실행하세요.")
    
    if previous.get("model") != experiment_name:
        raise ValueError("OOF 실행과 후보 모델이 다릅니다. 모델 변경으로 OOF부터 다시 실행하세요.")
    
    return previous


# ==========================================
# 실행 범위를 선택해 선행 결과 생성과 후속 검증 연결
# - threshold 생략 시 설정의 Recall·재검사 정책으로 OOF 후보 선택
# - 정책 미충족이면 시간 검증을 실행하지 않고 완료 상태에 미충족 기록
# - threshold만 변경할 때는 명시적인 값과 성공한 선행 실행이 필요
# - 시간 검증 함수가 학습 계약과 threshold 출처를 추가로 검사
# ==========================================
def run_evaluation(config_path, train_path, validation_path, run_dir, *,
                   change, threshold=None, n_jobs=1, experiment_name=EXPERIMENT_NAME):
    if change not in CHANGES:
        raise ValueError(f"지원하지 않는 변경 유형입니다: {change}")
    
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold는 0 이상 1 이하의 유한한 값이어야 합니다.")
    
    root = Path(run_dir)
    state_path = root / "execution.json"
    threshold_path = root / "threshold_compare.csv"
    reuse_oof = change in {"threshold", "policy"}

    config = load_modeling_config(config_path)
    
    previous = _validate_oof_reuse(
        state_path, threshold_path, config, train_path, change, threshold, experiment_name,
    ) if reuse_oof else None

    # 성공·실패 상태를 기록해 실패한 재생성 뒤 남은 과거 로그의 재사용을 차단한다.
    root.mkdir(parents=True, exist_ok=True)

    state = {
        "status": "running", "change": change, "completed_steps": [],
        "oof_ready": False, "model": experiment_name,
        "train_path": str(Path(train_path).resolve()),
        "validation_path": str(Path(validation_path).resolve()),
        "config": asdict(config), "n_jobs": n_jobs,
        "oof_config": previous.get("oof_config", previous.get("config")) if reuse_oof else asdict(config),
        "threshold_rule": "explicit" if threshold is not None else "configured_policy",
        "threshold_policy": asdict(config.threshold_policy),
    }

    def save_state():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")

    save_state()

    try:
        if reuse_oof:
            # 반올림 없는 CSV 값을 사용하며 선택한 값의 출처 검증은 Step 7에 위임한다.
            table = pd.read_csv(threshold_path, float_precision="round_trip")
            state["oof_ready"] = True
        else:
            _, table, _ = run_threshold_oof_from_file(
                config_path, train_path, root / "oof_predictions.csv", threshold_path,
                n_jobs=n_jobs, score_method="single", experiment_name=experiment_name,
            )
            state["completed_steps"].append("oof")
            state["oof_ready"] = True
            save_state()

        if threshold is None:
            # OOF 표에서만 선택한다. Validation 지표를 기준으로 threshold를 고르지 않는다.
            candidates = policy_candidates(table, config.threshold_policy)
            candidates.to_csv(root / "policy_candidates.csv", index=False, encoding="utf-8-sig")
            threshold = select_policy_threshold(table, config.threshold_policy)
            state["policy_status"] = "feasible" if threshold is not None else "infeasible"

            if threshold is None:
                # 과거 검증 CSV는 보존하되 이번 실행에서는 검증하지 않았음을 명시한다.
                state.update(status="completed", threshold=None,
                             time_validation_status="skipped_policy_infeasible")
                save_state()

                return pd.DataFrame([{"policy_status": "infeasible", "threshold": None,
                                      "message": "목표 Recall과 재검사 상한을 만족하는 OOF 후보가 없습니다. 시간 검증을 생략합니다."}])
        else:
            # 명시적 후보 지정은 연구용 수동 실행이며 정책 충족을 뜻하지 않는다.
            state["policy_status"] = "manual_override"

        state["threshold"] = threshold

        save_state()

        result = compare_time_validation_from_files(
            config_path, train_path, validation_path, root / "time_validation.csv",
            experiment_names=(experiment_name,), threshold=threshold,
            threshold_report_path=threshold_path, n_jobs=n_jobs,
        )

        state["completed_steps"].append("time_validation")
        state["time_validation_status"] = "completed"
        state["status"] = "completed"

        save_state()

        return result
    
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        save_state()
        raise


def main():
    # 변경 범위와 입력을 명시해 기능별 실행에도 같은 순서 관리를 적용한다.
    parser = argparse.ArgumentParser(description="변경 내용에 따른 OOF·시간 검증 연속 실행")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--change", choices=CHANGES, required=True)
    parser.add_argument("--experiment", choices=("lightgbm_all", "xgboost",
                                                   "xgboost_scale_pos_weight"),
                        default=EXPERIMENT_NAME,
                        help="평가할 후보 모델입니다. 후보를 바꾸면 --change model로 OOF부터 다시 생성합니다.")
    parser.add_argument("--threshold", type=float,
                        help="OOF 비교표의 수동 후보값. 생략하면 설정의 Recall·재검사 정책으로 선택합니다.")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    result = run_evaluation(args.config, args.train, args.validation, args.run_dir,
                            change=args.change, threshold=args.threshold, n_jobs=args.n_jobs,
                            experiment_name=args.experiment)
    
    print(result.drop(columns="quality_filter_log", errors="ignore").to_string(index=False))
    print(f"실행 기록: {args.run_dir / 'execution.json'}")


if __name__ == "__main__":
    main()
