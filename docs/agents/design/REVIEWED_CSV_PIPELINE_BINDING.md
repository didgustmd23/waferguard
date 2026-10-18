# CSV 파이프라인 입력 바인딩

`bind_reviewed_csv_pipeline()`은 현재 `tabular_csv_v1`을 위한 첫 입력 계약입니다. CSV snapshot·크기·해시, 데이터 프로파일과 품질 규칙, 정규화된 모듈 계획, 실행 코드 지문을 하나의 바인딩으로 고정합니다.

바인더는 품질 검사를 첫 단계로 요구하고, 이후 단계가 원본 `source.data`를 입력으로 사용하며 품질 단계에 의존하는지 검사합니다. 입력 partition은 `unsplit`만 허용합니다. 따라서 이 경로는 사전에 분할된 Train·Validation·Test 평가를 대신하지 않습니다.

공통 승인은 [`PIPELINE_APPROVAL.md`](../guides/PIPELINE_APPROVAL.md)에 있습니다. 승인 이후에는 기존 `ModulePipelineTool`과 `module_pipeline_worker`를 사용합니다. 자식 프로세스는 고정 허용 목록의 기본 품질·EDA 모듈만 실행합니다. 사용자 작성 모듈을 포함하는 계획은 바인딩 단계에서 거부하며, 향후 별도 검토·worker 격리·결과 계약을 추가해야 합니다.

바인딩은 데이터를 읽거나 변환하지 않습니다. 실행 시 worker는 원본 경로가 아닌 검증된 snapshot 복사본을 읽습니다. 부모 프로세스는 worker 종료 후 요청 결합, 결과 파일 목록, 보고서 구조와 콘텐츠 계약을 재검증합니다.
