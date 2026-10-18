# 확정 분할 증거와 구간 입력 계약

구현일: 2026-10-16. 기존 확정 분할과 커밋된 내보내기 결과를 연결하는 검증 함수를 구현했다. 이후 [내보내기 worker·부모 검증·승인 CLI·AppContainer 운송·커밋 연결](../guides/PARTITION_EXPORT.md)을 추가했다. 합성 연결 테스트는 통과했으며 실제 Windows 격리 확인과 전처리 변환 실행은 남아 있다.

## 구현 파일

| 파일 | 책임 |
| --- | --- |
| `src/agents/split_evidence.py` | 읽기 전용 승인 DB·커밋·완료 표식 검사, 기존 split_proposal_v1·배정표 검증 |
| `src/agents/partition_input_contract.py` | 분할 증거·내보내기 증거·선택 구간·역할·snapshot 결합과 재검사 |
| `tests/agents/test_partition_input_contract.py` | 합성 산출물과 기존 실제 분할 실행기 코드를 이용한 통합 테스트 |

에이전트 실행 관리 증거를 읽으므로 이 계약은 `src/agents/`에 둔다. 범용 계산 코어가 승인 DB에 의존하도록 바꾸지 않는다. 기존 분할·전처리 알고리즘·모듈 CLI·worker 복사 목록은 변경하지 않았다.

## 1. 확정 분할 증거

신뢰하는 실행 관리자가 보존한 정확한 `ApprovalScope`와 실행 폴더를 `read_confirmed_split()`에 전달한다. 사용자 JSON의 scope나 `partition=train` 선언을 권한 증거로 그대로 사용해서는 안 된다.

다음 조건을 모두 확인한다.

- 동일 주체·실행·단계·원본·설정·계획의 commit_split 작업.
- 승인 consumed·예약 1건·작업 committed. DB는 `mode=ro`로만 열며 ApprovalStore를 생성하거나 승인을 재소비하지 않음.
- 현재 완료 표식과 파일 해시가 기대 ArtifactContext에 일치.
- 파일은 기존 `split.json`·`assignments.csv`·완료 표식만 존재.
- split_proposal_v1의 통과 상태·자동 proposal_id·SplitRules 형식 일치.
- 세 구간의 위치 목록이 비어 있지 않고 정렬·중복 없음·전체 행 위치 완전성 충족.
- 구간 요약의 행 수·비율·클래스 건수와 배정표가 위치 목록에 일치.
- 취소·보안 잠금된 실행에서 자동으로 이어 가지 않음.

위치 목록은 데이터셋 index가 아니라 원본 **행 위치**다. 원본 CSV를 파싱하거나 분할을 다시 생성하지 않는다. Group·Time 경계의 실제 데이터 검증은 기존 분할 확정 시 수행한 검사를 전제로 하며, 여기서 메타데이터만 보고 독립적으로 재증명하지 않는다.

## 2. 구간 내보내기 증거

기존 commit_split은 배정표만 생성한다. Train·Validation CSV가 없으므로 확정 분할만으로 임의 CSV를 구간 입력에 연결하지 않는다. 후속 내보내기 도구가 다음 결과를 승인·커밋해야 한다.

```text
내보내기 attempt/
├─ partition_exports.json
├─ train/
│  ├─ input.bin
│  └─ snapshot.json
├─ validation/
│  ├─ input.bin
│  └─ snapshot.json
└─ AGENT_COMPLETE.json
```

초기 내보내기 작업은 별도 ingest/run_ingest 승인·작업 범위로 연결할 계약이다. 현재 도구가 자동으로 이 작업을 수행하는 것은 아니다. 완료 증거는 기존 작업 예약·승인·표식 형식을 재사용한다.

`partition_exports.json`의 제안 형식은 `partition_exports_v1`이다. 분할 증거 ID·원본 ID·proposal_id·ColumnRoles·두 구간별 행 수·행 위치 목록의 SHA-256·snapshot 크기/해시/상한을 포함한다. 경로는 입력 JSON에서 받지 않고 검증한 attempt 아래의 `train`·`validation`으로 고정한다.

행 위치 해시는 **정확한 위치 목록과 순서의 연결**이지 CSV 내용의 올바른 추출을 수학적으로 증명하는 값이 아니다. 올바른 부분집합 추출은 검토한 내보내기 구현과 테스트에서 보장해야 한다. 임의 CSV와 스스로 작성한 영수증만으로는 승인·커밋 증거 검사를 통과할 수 없다. 단, 승인 DB·표식까지 조작 가능한 공격자에 대한 인증 무결성 보장은 별도 문제다.

## 3. 구간 입력 결합

`bind_partition_input(split, export, partition, roles)`는 다음을 결합한다.

- 다시 확인한 확정 분할 증거와 별도 내보내기 승인·커밋 증거.
- 동일 원본·proposal·분할 증거 ID·기대 ColumnRoles.
- 해당 구간의 행 수와 정확한 행 위치/순서 식별값.
- 실제 snapshot manifest·바이트 해시·크기·복사 상한.

반환된 `PartitionInputContract`는 불변이며 구간·출처·역할·복사본에 결합된 `contract_id`를 가진다. `check()`는 현재 DB·완료 표식·배정·내보내기 기록·복사본을 다시 확인한다. 계약을 만든 뒤 입력이 바뀌면 재사용하지 않는다.

초기 구간은 train·validation뿐이다. Test payload를 내보내거나 읽지 않는다. Test의 위치 배정 메타데이터는 분할 완전성을 확인하는 데 사용한다. 부모는 완료 산출물의 정합성을 위해 Train·Validation 바이트를 해시할 수 있지만 값을 파싱하거나 Validation에서 전처리 기준을 계산하지 않는다. 향후 Train 기준 산출 worker에는 Train 복사본만 전달해야 한다.

## 4. 현재 할 수 있는 것과 남은 것

기존 확정 분할의 읽기 전용 증거 검사와 내보내기 산출물 계약 검증을 제공한다. 초기 계약 테스트는 생산자 fixture를 쓰며, 후속 실행 연결 테스트는 실제 worker·승인 소비·실행기 커밋을 사용하고 OS 운송만 모의한다. 실제 AppContainer 실행 검증과 구별한다.

내보내기 계산·파일 검증·승인·운송·커밋은 [후속 구현](../guides/PARTITION_EXPORT.md)을 따른다. 다음은 합성 네이티브 실행 확인이다. 임의 파일 경로·수동 Train 라벨을 받는 우회 경로는 만들지 않는다. 그 이후 전처리 후보 기준 산출·사용자 적용 승인을 연결한다.

## 테스트

```powershell
python -m unittest tests.agents.test_partition_input_contract -v
python -m unittest discover -s tests/agents -q
```

합성 테스트는 기존 분할 실행기의 제안·승인·커밋 결과 수용, DB 읽기 전용 유지, 역할/행/출처 불일치, 미소비/미커밋, CSV 및 snapshot 변조, 취소 상태, Test 자동 연결 거부를 확인한다. 실제 데이터 분할·모델 학습·변환·AppContainer 실행은 수행하지 않는다.
