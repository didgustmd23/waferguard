"""Windows 로컬 선택 창 전용 프로세스. 파일 내용 읽기·분석·모델 호출은 하지 않는다."""

import json
import sys


KINDS = ("csv_file", "json_file", "npz_file", "data_file", "directory")
# 통합 작업 요청 화면은 등록된 입력 형식을 함께 보여 준다. 선택 뒤 실제 유형은 서버의 어댑터 판별이 정한다.
FILTERS = {"csv_file": (("CSV 파일", "*.csv"),), "json_file": (("JSON 파일", "*.json"),), "npz_file": (("배열 표본 파일", "*.npz"),),
           "data_file": (("CSV·배열 표본 파일", "*.csv *.npz"), ("CSV 파일", "*.csv"), ("배열 표본 파일", "*.npz"))}


def choose_path(kind):
    """Tk를 이 프로세스 안에서만 초기화해 Streamlit 스레드와 분리한다."""
    if kind not in KINDS:
        return {"status": "unavailable"}

    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if kind == "directory":
            path = filedialog.askdirectory(parent=root, title="로컬 데이터 폴더 선택", mustexist=True)
        else:
            filters = FILTERS[kind] + (("모든 파일", "*.*"),)
            path = filedialog.askopenfilename(parent=root, title="로컬 파일 선택", filetypes=filters)

        if not path:
            return {"status": "cancelled"}

        if type(path) is not str or len(path) > 4096 or any(ord(char) < 32 for char in path):
            return {"status": "unavailable"}

        return {"status": "selected", "path": path}
    except Exception:
        # OS·라이브러리 예외 원문에 로컬 경로가 섞일 수 있어 외부로 출력하지 않는다.
        return {"status": "unavailable"}
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def main():
    kind = sys.argv[1] if len(sys.argv) == 2 else ""
    # ASCII JSON으로 출력해 Windows 콘솔 코드 페이지에 의존하지 않는다.
    print(json.dumps(choose_path(kind), ensure_ascii=True))


if __name__ == "__main__":
    main()
