# CNN·ResNet Test 전 후보 동결

사용자 결정에 따라 기존 checkpoint 6개의 경로·파일 해시·설정·입력 계보를 기록했다. 재학습·추론·Test 접근은 없다.
이 동결은 검토 가능한 명세이며 운영체제 파일 잠금이나 인증이 아니다. 평가 전에 해시를 다시 대조해야 한다.

| 역할 | 모델 | 조건 | seed | 선택 epoch | Validation Macro-F1 |
| --- | --- | --- | ---: | ---: | ---: |
| primary_candidate | ResNet18 | D4·α0.75·고정 lr0.001 | 42 | 11 | 84.60% |
| primary_candidate | ResNet18 | D4·α0.75·고정 lr0.001 | 43 | 11 | 84.73% |
| primary_candidate | ResNet18 | D4·α0.75·고정 lr0.001 | 44 | 10 | 82.59% |
| comparison_candidate | SmallWaferCNN | D4·α0.75·ReduceLROnPlateau | 42 | 15 | 82.69% |
| comparison_candidate | SmallWaferCNN | D4·α0.75·ReduceLROnPlateau | 43 | 11 | 83.69% |
| comparison_candidate | SmallWaferCNN | D4·α0.75·ReduceLROnPlateau | 44 | 13 | 82.43% |

## 최종 비교 계획

- 같은 Test 구간에서 6개 checkpoint를 각각 평가하고 모델별 3seed 평균·표본표준편차·클래스별 지표를 보고한다.
- 조건을 동결한 후 Test 결과로 설정이나 seed를 고르지 않는다. Test 최고 seed만 대표 성능으로 쓰지 않는다.
- CNN은 스케줄 적용, ResNet은 고정 학습률이므로 최종 튜닝 구성 비교이며 구조만의 효과로 해석하지 않는다.
- Test 캐시 준비·평가 실행은 아직 하지 않았다. 과거 Test와 현재 Test의 표본 겹침은 미확인이다.
- 기존 원본·캐시·모델·SUCCESS는 수정하지 않는다. 상세 계약과 해시는 freeze.json을 따른다.
