"""공통 결과 해설 UI. 화면 표시 자체로 모델을 호출하지 않는다."""

from uuid import uuid4

from .commentary_policy import (default_policy, disclosure_decision, fingerprint,
                                load_policy, model_binding, save_policy)
from .commentary_service import build_packet, failure_code, send_once


STATUS_LABELS = {
    "completed": "해설 완료", "already_attempted": "이미 요청한 해설 · 자동 재전송 안 함",
    "failed_no_retry": "해설 호출 실패 · 전처리 결과는 유지됨",
    "not_sent_or_unconfirmed": "해설 미전송 또는 응답 확인 불가 · 자동 재시도 안 함",
    "new_request_prepared": "새 해설 요청 준비됨 · 아래 자료 전달 확인이 필요함",
}
ERROR_LABELS = {
    "COMMENTARY_ENCODING_REJECTED": "모델 응답의 압축 방식이 허용되지 않습니다.",
    "COMMENTARY_TIMEOUT": "모델 응답 대기 시간이 초과됐습니다.",
    "COMMENTARY_TOKEN_LIMIT": "응답 토큰 상한에 도달해 생성이 중단됐습니다.",
    "OLLAMA_INCOMPLETE_RESPONSE": "모델 응답이 정상 완료되지 않았습니다. 응답 토큰 상한도 확인하세요.",
    "OLLAMA_CONNECTION_FAILED": "로컬 모델 연결 또는 응답 수신에 실패했습니다.",
    "COMMENTARY_CONNECTION_FAILED": "로컬 모델 연결 또는 응답 수신에 실패했습니다.",
    "COMMENTARY_RESPONSE_INVALID": "모델 응답이 요구한 해설 JSON 형식과 일치하지 않습니다.",
    "OLLAMA_RESPONSE_INVALID": "모델 응답의 JSON 형식 또는 인코딩 검증에 실패했습니다.",
    "COMMENTARY_HTTP_REJECTED": "모델 서버가 요청을 정상 수락하지 않았습니다.",
    "COMMENTARY_GRAPH_CONTEXT_INVALID": "그림의 표시 범위와 대응 집계 계약을 확인하지 못했습니다.",
    "OLLAMA_HTTP_FAILED": "모델 서버가 요청을 정상 수락하지 않았습니다.",
}
HTTP_REASONS = {
    "MEMORY_EXHAUSTED": "서버가 메모리 부족을 보고했습니다. GPU VRAM 또는 시스템 메모리 상태를 확인하세요.",
    "CONTEXT_LIMIT": "입력 문맥 한도를 초과했습니다. 그림도 토큰을 사용합니다. 설정에서 문맥 길이를 늘린 뒤 모델 설정 적용·해설 정책 재저장·새 해설 요청을 진행하세요. 자동으로 자료를 자르거나 한도를 변경하지 않습니다.",
    "IMAGES_UNSUPPORTED": "서버가 이 모델의 이미지 입력 미지원을 보고했습니다.",
    "MODEL_OR_ENDPOINT_NOT_FOUND": "모델 이름 또는 API 경로를 찾지 못했습니다.",
    "ACCESS_REJECTED": "서버가 접근을 거부했습니다.",
    "SERVER_BUSY": "서버가 요청 과다 또는 사용량 제한을 반환했습니다.",
    "REDIRECT_REJECTED": "다른 주소로의 이동 요청을 보안 정책에 따라 차단했습니다.",
    "INVALID_REQUEST": "서버가 요청 형식 또는 입력을 거부했습니다. 구체적 사유는 현재 고정 분류로 확인되지 않았습니다.",
    "SERVER_ERROR": "모델 서버 내부 오류입니다. 구체적 사유는 현재 고정 분류로 확인되지 않았습니다.",
    "UNCLASSIFIED": "HTTP 오류의 상세 원인을 안전한 고정 분류로 확인하지 못했습니다.",
}
DECISION_LABELS = {
    "CLASSIFICATION_REQUIRED": "자료 민감도를 먼저 지정하세요.",
    "SENSITIVE_EXCLUDED": "현재 정책에서 민감 자료는 제외합니다.",
    "MODEL_CONFIRMATION_REQUIRED": "현재 모델 설정에 대한 해설 정책을 설정 화면에서 저장하세요.",
}


def render_policy(st):
    st.subheader("LLM 정보 전달 정책")
    try:
        policy = load_policy()
    except (OSError, ValueError, UnicodeError):
        st.error("저장된 해설 정책이 잘못되어 전송을 차단했습니다. 정책을 다시 저장하세요.")
        policy = default_policy()

    with st.form("commentary_policy_form"):
        enabled = st.checkbox("로컬 LLM 결과 자동 해설 기본 허용", value=policy["local_default_allow"],
                              help="현재 모델 설정에 한해 새 작업 완료 후 검증된 결과를 한 번 전달합니다. 기존 작업·새로고침은 자동 호출하지 않습니다.")
        tables = st.checkbox("집계 표 전달", value=policy["tables"],
                             help="허용된 집계 표와 그래프에 대응하는 표시 범위·집계 수치를 전달합니다. 실제 특징명·클래스명·수치가 포함됩니다. 원본 행과 보고서 본문은 제외합니다.")
        graphs = st.checkbox("집계 그래프 전달", value=policy["graphs"],
                             help="허용 목록의 PNG만 전달합니다. 축·제목·수치가 포함됩니다. 원본 맵·예시 이미지는 제외하며 설치 모델이 이미지 입력을 지원해야 합니다.")
        sensitive = st.selectbox("민감 자료 처리", ("항상 제외", "자료별 확인 후 허용"),
                                 index=int(policy["sensitive_action"] == "confirm"),
                                 help="민감도는 사용자가 지정합니다. 미분류 자료는 자동 전달하지 않습니다. 개별 확인은 실제 자료와 모델 설정에 결합됩니다.")
        submitted = st.form_submit_button("해설 전달 정책 저장",
                                          help="로컬 정책 파일에 저장합니다. 저장만으로 모델을 호출하지 않으며 외부 API 기본 허용은 제공하지 않습니다.")

    if submitted:
        try:
            save_policy({"version": 1, "local_default_allow": bool(enabled),
                         "model_binding": model_binding(st.session_state["ui_settings"]["model"]),
                         "sensitive_action": "confirm" if sensitive == "자료별 확인 후 허용" else "exclude",
                         "tables": bool(tables), "graphs": bool(graphs)})
        except (OSError, ValueError, TypeError):
            st.error("해설 정책을 저장하지 못했습니다. 자동 전송하지 않습니다.")
        else:
            st.success("해설 정책을 로컬에 저장했습니다. 모델을 바꾸면 다시 저장해야 합니다.")

    st.caption("외부 API는 실제 전달 자료 확인·건별 승인이 필수입니다. 현재 외부 API 연결은 미지원이며 전송하지 않습니다. LangSmith 추적도 비활성화입니다.")


def scope(controller):
    return fingerprint({"module": getattr(controller, "module_id", "proposal_execution"),
                        "name": controller.name})


def render_classification(st, controller, *, disabled=False):
    st.selectbox("결과 자료 민감도", ("미분류 · 자동 해설 안 함", "일반 자료", "민감 자료"),
                 key="commentary_class_" + scope(controller), disabled=disabled,
                 help="자동 탐지나 무해성 인증이 아닙니다. 실제 특징명·집계·그림에서 추측할 수 있는 정보를 고려해 지정하세요. 실행 시작 후에는 변경하지 않습니다.")


def _classification(st, controller):
    value = st.session_state.get("commentary_class_" + scope(controller))
    return {"일반 자료": "general", "민감 자료": "sensitive"}.get(value, "unknown")


def freeze_classification(st, controller):
    # Streamlit은 다른 화면의 위젯 상태를 정리할 수 있으므로 시작 시 별도 고정한다.
    st.session_state["commentary_frozen_" + scope(controller)] = _classification(st, controller)


def after_completion(st, controller, *, workflow):
    """명시적 실행 명령이 완료된 직후에만 호출한다. 해설 실패는 실행 상태와 분리한다."""
    key = "commentary_result_" + scope(controller)
    try:
        policy = load_policy()
        model = st.session_state["ui_settings"]["model"]
        sensitivity = st.session_state.get("commentary_frozen_" + scope(controller), "unknown")
        if disclosure_decision(policy, "local", sensitivity, model) != "AUTO_ALLOWED":
            return

        if workflow:
            from .workflow_results import load_workflow_results
            result = load_workflow_results(controller)
        else:
            from .result_service import load_results
            result = load_results(controller)

        packet = build_packet(result, policy)
        if load_policy() != policy:
            raise ValueError("COMMENTARY_POLICY_CHANGED")
        with st.spinner("로컬 LLM이 허용된 결과 자료를 해설하고 있습니다."):
            reply = send_once(packet, policy, model, sensitivity, "auto:" + scope(controller))
            # 중복 완료 통지는 원래 해설·요청 시각·진단을 덮어쓰는 근거가 아니다.
            if reply["status"] != "already_attempted" or key not in st.session_state:
                st.session_state[key] = reply
    except Exception as error:
        st.session_state[key] = {"status": "not_sent_or_unconfirmed", "code": failure_code(error)}


def render_commentary(st, controller, result):
    st.subheader("로컬 결과 해설")
    st.caption("해설 화면 코드 버전: pairs-v2")
    key = "commentary_result_" + scope(controller)
    stored = st.session_state.get(key)
    if stored:
        st.text("해설 상태: " + STATUS_LABELS.get(stored["status"], stored["status"]))
        if stored.get("commentary_version"):
            st.caption("이 해설의 요청 버전: " + stored["commentary_version"])
        input_formats = {"sensor_pairs": "센서쌍 목록", "matrix": "상관행렬",
                         "graph_without_numeric_context": "수치 근거 없는 그래프", "tables_only": "표만 전달"}
        if stored.get("input_format") in input_formats:
            st.caption("실제 요청 입력: " + input_formats[stored["input_format"]]
                       + " / 상관 센서쌍 " + str(stored.get("correlation_pair_count", 0)) + "개")
        if stored["status"] == "completed" and (not stored.get("attempted_at") or not stored.get("input_format")):
            st.warning("이 해설에는 요청 시각·입력 형식 기록이 없어 현재 코드로 새로 생성된 응답인지 확인할 수 없습니다.")
        if stored.get("attempted_at"):
            st.caption("표시 중인 요청 시각(UTC): " + stored["attempted_at"])
        if type(stored.get("request_context")) is int:
            st.caption("이 요청에 보낸 문맥 한도: " + str(stored["request_context"]) + " 토큰")
        if type(stored.get("requested_tokens")) is int and type(stored.get("context_tokens")) is int:
            st.warning("서버가 보고한 입력: " + str(stored["requested_tokens"]) + " 토큰 / 허용 한도: "
                       + str(stored["context_tokens"]) + " 토큰")
        if stored.get("code"):
            st.warning(ERROR_LABELS.get(stored["code"], "해설 요청을 완료하지 못했습니다. 로컬 모델 설정을 확인하세요."))
            st.caption("진단 코드: " + stored["code"])
        if type(stored.get("http_status")) is int:
            st.caption("HTTP 상태: " + str(stored["http_status"]))
            st.warning(HTTP_REASONS.get(stored.get("reason"), HTTP_REASONS["UNCLASSIFIED"]))
        for title, field in (("관찰", "observations"), ("해석 한계", "limitations")):
            for text in stored.get("reply", {}).get(field, []):
                # 모델이 만든 링크·HTML·이미지 문법을 실행하지 않는다.
                st.text(title + ": " + text)
        st.caption("LLM의 보조 설명입니다. 결과 검증·품질 판정·실행 승인을 대신하지 않습니다.")

    st.caption("기존 완료 작업은 자동 전송하지 않습니다. 아래에서 실제 전달 자료를 확인한 뒤 건별로 요청할 수 있습니다.")
    label = st.selectbox("이번 해설 자료 민감도", ("미분류", "일반 자료", "민감 자료"),
                         key="commentary_manual_class_" + scope(controller),
                         help="민감 자료 제외 정책을 적용합니다. 사용자 분류이며 자동 민감 정보 탐지가 아닙니다.")
    sensitivity = {"일반 자료": "general", "민감 자료": "sensitive"}.get(label, "unknown")
    try:
        policy = load_policy()
        model = st.session_state["ui_settings"]["model"]
        decision = disclosure_decision(policy, "local", sensitivity, model)
        if decision not in ("AUTO_ALLOWED", "INDIVIDUAL_CONFIRMATION_REQUIRED"):
            st.info(DECISION_LABELS.get(decision, "현재 정책에서 해설 전송을 허용하지 않습니다."))
            return
        packet = build_packet(result, policy)
    except (ValueError, OSError, UnicodeError, TypeError, KeyError):
        st.info("해설에 사용할 허용 자료가 없거나 상한을 초과했습니다. 자동으로 일부만 보내지 않습니다.")
        return

    binding = fingerprint({"packet": packet, "policy": policy, "model": model, "sensitivity": sensitivity})
    st.text("전달 대상: " + model["host"] + ":" + str(model["port"]) + " / " + model["model"])
    st.caption("다음 요청의 적용 문맥 한도: " + str(model["num_ctx"]) + " 토큰")
    with st.expander("실제 전달 자료 미리보기"):
        st.caption("아래 자료만 전달합니다. 원본 행·보고서 본문·로컬 경로·승인 정보는 포함하지 않습니다.")
        for asset in packet["assets"]:
            st.text(asset["name"])
            if "image" in asset:
                import base64
                st.image(base64.b64decode(asset["image"]))
                if "graph_context" in asset:
                    st.json(asset["graph_context"])
            else:
                st.json(asset)
        included = {asset["name"] for asset in packet["assets"]}
        excluded = [item["name"] for item in result["files"]
                    if item["name"].replace("\\", "/").rsplit("/", 1)[-1] not in included]
        if excluded:
            st.text("이번 전달에서 제외한 표시 파일: " + ", ".join(excluded))

    command_key = "commentary_manual_command_" + scope(controller) + binding
    command = st.session_state.setdefault(command_key, uuid4().hex)
    needs_new_request = stored is not None and stored["status"] in {
        "completed", "failed_no_retry", "already_attempted", "not_sent_or_unconfirmed"}
    if needs_new_request:
        st.info("이전 요청은 다시 전송하지 않습니다. 새 해설이 필요하면 새 요청을 준비한 뒤 전달을 다시 확인하세요.")
        acknowledged = st.checkbox("이전 요청이 모델에서 처리됐을 가능성을 이해하고 새 요청을 준비합니다.",
                                    key="commentary_retry_ack_" + command,
                                    help="새 요청은 별도 모델 호출입니다. 이전 기록을 삭제하거나 전처리를 재실행하지 않습니다.")
        if st.button("새 해설 요청 준비", key="commentary_prepare_" + command,
                     disabled=not acknowledged,
                     help="새 요청 ID만 준비합니다. 모델은 호출하지 않으며 다음 전달 확인이 필요합니다."):
            st.session_state[command_key] = uuid4().hex
            st.session_state[key] = {"status": "new_request_prepared"}
            st.rerun()

    with st.form("commentary_send_" + binding + "_" + command):
        confirmed = st.checkbox("표시된 자료를 이 로컬 모델에 전달합니다.",
                                 help="현재 자료·정책·모델 설정 한 건에만 유효합니다. 변경 시 다시 확인해야 합니다.")
        submitted = st.form_submit_button("확인한 자료로 해설 요청",
                                          disabled=needs_new_request,
                                          help="로컬 모델을 한 번 호출합니다. 실패해도 전처리 결과는 유지하며 자동 재시도하지 않습니다.")
    if needs_new_request:
        return
    if submitted and not confirmed:
        st.warning("먼저 ‘표시된 자료를 이 로컬 모델에 전달합니다’를 체크하세요.")
        return
    if submitted and confirmed:
        # 정책 파일이 미리보기 이후 바뀌었다면 승인 범위를 재사용하지 않는다.
        if load_policy() != policy:
            st.error("정책이 바뀌었습니다. 자료를 다시 확인하세요.")
            return
        try:
            with st.spinner("로컬 해설 요청 중입니다."):
                st.session_state[key] = send_once(packet, policy, model, sensitivity,
                                                  command, consent=binding)
        except Exception as error:
            st.session_state[key] = {"status": "not_sent_or_unconfirmed", "code": failure_code(error)}
        st.rerun()
