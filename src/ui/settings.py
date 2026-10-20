"""모델·자원 설정과 결과 해설 전달 정책. 저장 자체로 모델을 호출하지 않는다."""

from copy import deepcopy

from .state import MODES, validate_settings


def render_settings(st):
    """저장 버튼을 누른 경우만 검증하고 이번 세션 설정을 교체한다."""
    st.header("설정")
    st.caption("모델·자원 설정은 이번 세션에 적용합니다. 아래 해설 전달 정책은 별도 저장 버튼으로 로컬 파일에 보존합니다.")
    current = st.session_state["ui_settings"]
    model = current["model"]

    with st.form("settings_form"):
        st.subheader("모델 연결 설정")
        mode = st.selectbox(
            "동작 모드", MODES, index=MODES.index(current["mode"]),
            help="로컬 LLM은 대화로 실행안을 제안합니다. 규칙 기반 화면은 준비 중이며 이 모드에서는 모델을 호출하지 않습니다. 모드 변경은 이전 전송 확인·제안을 무효화합니다.",
        )
        name = st.text_input(
            "Ollama 모델명", value=model["model"],
            help="Ollama에 설치한 정확한 모델 이름과 태그입니다. 예: qwen3.5:9b. 입력만으로 모델을 설치·호출하거나 연결을 확인하지 않습니다.",
        )
        host = st.selectbox(
            "Ollama 주소", ("127.0.0.1", "::1"), index=int(model["host"] == "::1"),
            help="이 PC의 Ollama 서버 주소입니다. 127.0.0.1은 IPv4, ::1은 IPv6 로컬 주소입니다. 외부 서버 주소는 허용하지 않습니다.",
        )
        port = st.number_input(
            "Ollama 포트", min_value=1, max_value=65535, value=model["port"],
            help="Ollama 서버가 받는 포트입니다. 기본 설정은 11434이며 UI 접속 포트인 8501과는 다릅니다.",
        )
        timeout = st.number_input(
            "응답 대기 한도 (초)", min_value=1, max_value=600, value=model["timeout_seconds"],
            help="모델 통신에서 각 소켓 응답을 기다리는 한도입니다. 전체 작업 시간을 강제로 종료하는 상한은 아니며 실패 시 자동 재시도하지 않습니다.",
        )
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0,
                                      value=float(model["temperature"]), step=0.1,
                                      help="응답 생성의 무작위성입니다. 낮을수록 일관된 출력을 유도하지만 0도 정답·완전한 재현성을 보장하지 않습니다. 확률이나 정확도 점수가 아닙니다.")
        context = st.number_input(
            "문맥 길이", min_value=512, max_value=32768, value=model["num_ctx"],
            help="모델이 한 번에 다룰 문맥의 토큰 수입니다. 크게 설정하면 메모리 부담이 늘 수 있습니다. 프로젝트의 입력 길이 제한을 해제하지는 않습니다.",
        )
        predict = st.number_input(
            "응답 토큰 상한", min_value=64, max_value=8192, value=model["num_predict"],
            help="한 번의 응답에서 생성할 최대 토큰 수입니다. 토큰은 글자 수와 다릅니다. 너무 작아 JSON이 잘리면 유효한 실행안으로 처리하지 않습니다.",
        )
        think = st.checkbox(
            "모델 추론 출력 사용", value=model["think"],
            help="지원 모델에 추론 모드 사용을 요청하는 옵션입니다. 응답 시간과 생성량이 늘 수 있으며, 모델의 답이 정확하다는 보증은 아닙니다.",
        )
        st.caption("설치 모델 자동 조회·연결 확인은 아직 연결하지 않았습니다. 모델명은 설정값이며 접속 확인 결과가 아닙니다.")

        st.subheader("작업 기본 자원")
        limits = current["limits"]
        memory = st.number_input(
            "메모리 상한 (MiB)", min_value=1, max_value=32768, value=limits["memory_mib"],
            help="향후 격리 작업에 요청할 전체 프로세스 메모리 상한입니다. 1024 MiB는 1 GiB이며 GPU VRAM·입력 파일 복사 크기와는 별개입니다. 현재는 설정값만 보관합니다.",
        )
        processes = st.number_input(
            "동시 프로세스 상한", min_value=1, max_value=8, value=limits["processes"],
            help="격리 worker와 하위 프로세스 중 동시에 살아 있을 수 있는 프로세스 수입니다. CPU 스레드 수나 작업 전체에서 생성한 누적 프로세스 수가 아닙니다.",
        )
        seconds = st.number_input(
            "작업 시간 상한 (초)", min_value=1, max_value=86400, value=limits["wall_seconds"],
            help="향후 격리 worker 실행에 요청할 실제 경과 시간 상한입니다. 모델의 소켓 응답 대기 한도와 다르며, 실행 연결 후에는 초과 작업의 종료·정리를 확인해야 합니다.",
        )
        submitted = st.form_submit_button(
            "이번 세션에 적용",
            help="폼 전체를 검증한 뒤 이번 브라우저 세션에만 적용합니다. 설정 파일 저장·모델 전송·전처리 실행 승인은 하지 않습니다.",
        )

    if submitted:
        candidate = {
            "model": {"model": name, "host": host, "port": port, "timeout_seconds": timeout,
                      "temperature": temperature, "num_ctx": context, "num_predict": predict, "think": think},
            "limits": {"memory_mib": memory, "processes": processes, "wall_seconds": seconds},
            "mode": mode,
        }

        try:
            checked = validate_settings(candidate)
        except (ValueError, TypeError, KeyError):
            st.error("설정이 허용 범위를 벗어났습니다. 기존 세션 설정은 유지합니다.")
        else:
            # 화면에 남아 있는 이전 모델의 전송 확인·제안은 변경된 설정에서 재사용하지 않는다.
            if checked != current:
                st.session_state["ui_settings"] = deepcopy(checked)
                st.session_state["ui_settings_revision"] += 1
                from .dialogue_view import current_dialogue

                current_dialogue(st)
                st.success("이번 세션에 적용했습니다. 모델 호출·승인 발급·작업 실행은 하지 않았습니다.")
            else:
                st.info("설정 변경이 없습니다. 기존 세션 설정을 유지합니다.")

    st.text("현재 적용 모델: " + st.session_state["ui_settings"]["model"]["model"])
    st.text("현재 적용 문맥 길이: " + str(st.session_state["ui_settings"]["model"]["num_ctx"]) + " 토큰")

    from .commentary_view import render_policy

    render_policy(st)
    st.subheader("추가 기능 상태")
    st.text("외부 API: 미지원\nLangSmith 추적: 비활성화\n대화 파일 저장: 미연결")
