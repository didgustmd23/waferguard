"""AppContainer 호환성 실험 전용 고정 payload. 실제 데이터·설정을 읽지 않는다."""

import ctypes
import json
import os
from pathlib import Path
import socket
import sys

if __package__:
    from .probe_progress import ProbeJournal
else:
    # -I 실행에서도 승인된 검사 도구 복사본의 고정 모듈만 불러온다.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_progress import ProbeJournal


def denied(operation):
    """파일 없음·시간 초과를 차단 성공으로 잘못 해석하지 않는다."""
    try:
        operation()
    except OSError as error:
        blocked = getattr(error, "winerror", None) in {5, 10013} or error.errno in {1, 13}
        outcome = "access_denied" if blocked else "other_error"

        if isinstance(error, TimeoutError):
            outcome = "timeout"
        elif isinstance(error, ConnectionRefusedError):
            outcome = "connection_refused"

        return {
            "denied": blocked, "error_code": getattr(error, "winerror", None) or error.errno,
            "outcome": outcome,
        }

    return {"denied": False, "error_code": 0, "outcome": "connected_or_accessible"}


def main():
    root = Path(sys.argv[1])
    output = root / "output"
    runtime = root / "runtime"
    sys.path.insert(0, str(runtime / "Lib" / "site-packages"))
    handles = []
    journal = ProbeJournal(output / "progress.jsonl")
    journal.checkpoint("BOOT", "START")
    sys.addaudithook(journal.audit)

    # conda의 네이티브 라이브러리도 검사 전용 복사본에서만 찾는다.
    for directory in (runtime, runtime / "DLLs", runtime / "Library" / "bin"):
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))

    checks = {}
    result = {"schema_version": "appcontainer_child_v1", "checks": checks, "real_data_read": False}

    try:
        journal.checkpoint("TOKEN", "START")
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        advapi.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
        advapi.GetTokenInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
        token = ctypes.c_void_p()

        if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise OSError("TOKEN_QUERY_FAILED")

        try:
            flag = ctypes.c_ulong()
            needed = ctypes.c_ulong()
            ok = advapi.GetTokenInformation(token, 29, ctypes.byref(flag), ctypes.sizeof(flag), ctypes.byref(needed))
            checks["appcontainer_token"] = bool(ok and flag.value == 1)

            # TokenCapabilities의 첫 DWORD는 그룹 수다. 조회 실패를 0개로 취급하지 않는다.
            needed = ctypes.c_ulong()
            advapi.GetTokenInformation(token, 30, None, 0, ctypes.byref(needed))
            queried = False
            count = None

            if 4 <= needed.value <= 65536:
                buffer = ctypes.create_string_buffer(needed.value)
                queried = bool(advapi.GetTokenInformation(token, 30, buffer, len(buffer), ctypes.byref(needed)))

                if queried:
                    count = ctypes.c_ulong.from_buffer(buffer).value

            result["token_evidence"] = {"capabilities_queried": queried, "capability_count": count}
            checks["network_capabilities_absent"] = queried and count == 0
        finally:
            kernel.CloseHandle(token)

        journal.checkpoint("TOKEN", "DONE")
        journal.checkpoint("FILE_ACCESS", "START")
        checks["allowed_read"] = (root / "input" / "synthetic.txt").read_text(encoding="utf-8") == "synthetic_only"
        (output / "write_check.txt").write_text("synthetic_only", encoding="utf-8")
        checks["allowed_write"] = True
        checks["protected_read"] = denied(lambda: (root / "protected" / "secret.txt").read_bytes())

        def attempt_write():
            with (root / "input" / "synthetic.txt").open("wb") as stream:
                stream.write(b"unexpected_write")

        checks["input_write"] = denied(attempt_write)
        journal.checkpoint("FILE_ACCESS", "DONE")
        journal.checkpoint("NETWORK", "START")
        config = json.loads((root / "input" / "probe.json").read_text(encoding="utf-8"))

        def tcp(address, port, *, observed=False):
            with socket.socket() as connection:
                connection.settimeout(2)
                connection.connect((address, port))

                if observed:
                    # 연결 뒤 응답만 timeout이어도 실제 연결 성공을 놓치지 않는다.
                    checks["loopback_connected"] = True
                    connection.sendall(config["child_token"].encode("ascii"))
                    connection.recv(2)

        # 부모가 열어 놓은 로컬 서버와 문서용 예약 주소에 합성 연결만 시도한다.
        checks["loopback_connected"] = False
        checks["loopback_network"] = denied(lambda: tcp("127.0.0.1", config["port"], observed=True))
        checks["external_network"] = denied(lambda: tcp("192.0.2.1", 9))
        journal.checkpoint("NETWORK", "DONE")

        for name in ("numpy", "pandas", "sklearn", "matplotlib"):
            stage = "IMPORT_" + name.upper()
            journal.checkpoint(stage, "START")

            try:
                __import__(name)
                checks[f"import_{name}"] = True
                journal.checkpoint(stage, "DONE")
            except Exception as error:
                journal.checkpoint(stage, "FAILED")
                checks[f"import_{name}"] = False
                # 예외 원문·경로는 기록하지 않는다. 누락 모듈명은 제한된 토큰으로만 남긴다.
                diagnostic = {"code": "LIBRARY_IMPORT_FAILED"}

                if isinstance(error, ModuleNotFoundError):
                    import re
                    diagnostic["code"] = "LIBRARY_DEPENDENCY_MISSING"

                    if isinstance(error.name, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", error.name):
                        diagnostic["missing_module"] = error.name

                result.setdefault("import_diagnostics", {})[name] = diagnostic

        if all(checks.get(f"import_{name}") for name in ("numpy", "pandas", "sklearn", "matplotlib")):
            import numpy as np
            import pandas as pd
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            import matplotlib
            matplotlib.use("Agg")
            journal.checkpoint("IMPORT_PYPLOT", "START")
            import matplotlib.pyplot as plt
            journal.checkpoint("IMPORT_PYPLOT", "DONE")

            journal.checkpoint("PIPELINE", "START")
            frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
            transformed = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]).fit_transform(frame)
            checks["numeric_pipeline"] = transformed.shape == (3, 2) and bool(np.isfinite(transformed).all())
            journal.checkpoint("PIPELINE", "DONE")
            journal.checkpoint("CSV", "START")
            frame.to_csv(output / "synthetic.csv", index=False)
            checks["csv_roundtrip"] = pd.read_csv(output / "synthetic.csv").shape == (3, 2)
            journal.checkpoint("CSV", "DONE")
            journal.checkpoint("PLOT_CREATE", "START")
            figure, axes = plt.subplots()
            axes.plot([0, 1], [0, 1])
            journal.checkpoint("PLOT_CREATE", "DONE")
            journal.checkpoint("PLOT_SAVE", "START")
            figure.savefig(output / "synthetic.png")
            plt.close(figure)
            checks["agg_plot"] = (output / "synthetic.png").stat().st_size > 0
            journal.checkpoint("PLOT_SAVE", "DONE")
    except Exception:
        journal.checkpoint(journal.stage, "FAILED")
        result["runtime_error"] = "PROBE_RUNTIME_FAILED"
    finally:
        journal.checkpoint("RESULT", "START")
        (output / "child_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        journal.checkpoint("RESULT", "DONE")


if __name__ == "__main__":
    main()
