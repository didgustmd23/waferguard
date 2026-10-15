# WM-811K 전처리 코드 리뷰 순서

에이전트 구현 전까지 생성된 전처리 코드를 리뷰하고, 각 단계의 그래프와 수치를 해석합니다. 생성·진행 순서를 기준으로 정리했으며 자동 실행 순서나 전체 재실행 지시가 아닙니다.

각 단계는 **코드가 하는 일 → 결과·그래프의 의미 → 판단할 수 있는 것과 없는 것** 순서로 확인합니다. 새 기능 구현·실험 실행은 별도 요청 시 진행합니다.

## 1. 핵심 코드

| 순서 | 소스 파일 | 역할 | 리뷰할 핵심 |
| --- | --- | --- | --- |
| 1 | [inspect_pickle.py](../../scripts/wm_sandbox/inspect_pickle.py) | 원본 구조 확인 | 맵·라벨·Lot의 저장 형태와 확인 범위 |
| 2 | [export_canonical.py](../../scripts/wm_sandbox/export_canonical.py) | 숫자 전용 파일로 변환 | 라벨 별칭 처리, 맵 검증, NPZ·manifest 저장 |
| 3 | [validate_canonical.py](../../src/wafer_dl/validate_canonical.py) | 변환 결과 검증 | 파일 정합성, 중복·라벨 충돌, 분할 가능 조건 |
| 4 | [diagnose_split.py](../../src/wafer_dl/diagnose_split.py) | 충돌 제외 영향 진단 | 실제 데이터를 변경하지 않고 제외 전후 비교 |
| 5 | [eda.py](../../src/wafer_dl/eda.py) | EDA 집계·그래프 | 클래스·크기·불량 비율·Lot 분포와 상관관계 |
| 6 | [split.py](../../src/wafer_dl/split.py) | 그룹 기반 분할 | Lot·동일 맵 연결, 충돌 제외, 약 70·15·15 배정 |
| 7 | [transform.py](../../src/wafer_dl/transform.py) | 입력 변환 | 비율 유지 resize, padding, 2채널 구성 |
| 8 | [check_transform.py](../../src/wafer_dl/check_transform.py) | 변환 결과 검사 | 영역 소실·불량 비율 변화·원본과 변환 그림 |
| 9 | [cache.py](../../src/wafer_dl/cache.py) | 변환 결과 저장 | 분할별 NPY 저장, 메모리 버퍼, 색인·완료 표식 |
| 10 | [dataset.py](../../src/wafer_dl/dataset.py) | PyTorch 입력 연결 | 캐시 읽기, Tensor·라벨 구성, 첫 배치 검사 |

원본 구조 확인 코드는 검토만으로 다시 실행할 필요가 없습니다. 외부 pickle을 읽는 작업과 숫자 전용 canonical을 읽는 작업은 구분합니다.

## 2. 그래프·결과 확인 순서

| 코드 단계 | 함께 볼 결과 | 해석할 내용 |
| --- | --- | --- |
| 5. EDA | [analysis.ipynb](../../reports/wm811k/analysis.ipynb)의 EDA 셀, [report.md](../../reports/wm811k/report.md) 1~9절 | 클래스 불균형, 분포 쏠림, 맵 크기·종횡비, 상관관계, 공간 패턴 |
| 6. 분할 | 로컬 `split_summary.csv`, 노트북 9절, 보고서 10~11절 | 충돌 제외 전후 수, 구간별 클래스 수·비율, 그룹 비중첩 |
| 8. 변환 검사 | 노트북 8절, [변환 사례 그림](../../reports/wm811k/transform_v1/examples.png), 보고서 12절 | 원본과 변환의 형태 차이, 전체 소실과 일부 변화의 구분 |
| 9~10. 캐시·Dataset | 노트북 9~10절, 보고서 13절 | 저장 표본·shard 수, 입력 형태·자료형, 검증 범위 |

분할 전 EDA 수치와 충돌 제외 후 집계를 혼용하지 않습니다. Test 분할 완료와 Test 캐시 준비도 서로 다른 단계입니다.

## 3. 보조 코드와 테스트

설치용 PowerShell 스크립트와 `python313._pth`, 환경 의존성 파일은 핵심 코드 리뷰 이후에 확인합니다. 이들은 전처리 알고리즘이 아니라 실행 환경 준비를 돕는 파일입니다.

테스트는 해당 소스와 함께 읽습니다.

| 대상 | 테스트 파일 |
| --- | --- |
| 원본 구조 확인 | `tests/test_wm_sandbox_inspect.py` |
| canonical 변환 | `tests/test_wm_canonical_export.py` |
| canonical 검증 | `tests/test_validate_canonical.py` |
| 제외 영향 진단 | `tests/test_diagnose_split.py` |
| EDA | `tests/test_wafer_eda.py` |
| 그룹 분할 | `tests/test_wafer_split.py` |
| 입력 변환·검사 | `tests/test_wafer_transform.py` |
| 캐시·Dataset | `tests/test_wafer_cache_dataset.py` |

합성 테스트에서 확인한 계약과 실제 데이터에서 관찰한 결과를 구분합니다. 사용 명령과 데이터 전달 방법은 [캐시·Dataset 사용법](CACHE_DATASET.md)을 참고합니다.
