"""결과 보고서의 그림 파일 저장. 검증된 facts의 그래프 값과 검증된 EDA PNG 바이트만 쓴다.

원본·worker·모델을 부르지 않는다. 저장은 보고서 폴더 아래 내용 지문 이름의 새 폴더에만 하며 기존 파일을 덮어쓰지 않는다.
실패는 예외로 완료를 막지 않고 사유를 facts에 남겨 화면과 Markdown이 함께 알린다. 사유는 경로·예외 원문 없는 고정 문구다.
"""

import hashlib
import io
import re

from .artifacts import _is_link


# 한글 라벨을 그릴 수 있는 글꼴. 없으면 글자가 깨진 그림을 저장하지 않고 실패로 알린다.
FONTS = ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
FIGURE_LIMIT = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[a-z0-9_]{1,64}\.png$")
FONT_MISSING = "한글 글꼴을 찾지 못해 그림을 만들지 않았습니다"
DRAW_FAILED = "그림을 만들지 못했습니다"
SOURCE_MISSING = "검증된 그림을 다시 읽지 못했습니다"
TOO_LARGE = "그림 크기가 상한을 넘었습니다"
SAVE_FAILED = "파일 저장에 실패했습니다"
NOT_OVERWRITTEN = "같은 이름의 다른 파일이 있어 덮어쓰지 않았습니다"


def _font():
    from matplotlib import font_manager

    for name in FONTS:
        try:
            return font_manager.FontProperties(fname=font_manager.findfont(font_manager.FontProperties(family=name),
                                                                           fallback_to_default=False))
        except ValueError:
            continue

    raise LookupError("font")


def draw_chart(chart):
    """그래프 사실 한 개를 PNG 바이트로 그린다. 입력 유래 이름은 수식으로 해석하지 않고 파일 메타데이터도 남기지 않는다."""
    import matplotlib

    # 부모 프로세스의 사용자 설정(matplotlibrc)이 입력 유래 이름을 외부 TeX로 넘기지 않게 고정한다.
    with matplotlib.rc_context(rc={"text.usetex": False}):
        return _draw(chart)


def _draw(chart):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    font = _font()
    category, values, rows = chart["category"], chart["values"], chart["rows"]
    labels = [str(row[category]) for row in rows]
    positions = range(len(rows))
    width = 0.8 / len(values)
    figure = Figure(figsize=(8, max(2.5, 0.4 * len(rows) * len(values) + 1.5)) if chart["horizontal"] else (8, 4), dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()

    for index, key in enumerate(values):
        shifted = [position + (index - (len(values) - 1) / 2) * width for position in positions]
        heights = [row[key] for row in rows]
        (axes.barh if chart["horizontal"] else axes.bar)(shifted, heights, width, label=key)

    text = {"fontproperties": font, "parse_math": False}

    if chart["horizontal"]:
        axes.set_yticks(list(positions), labels, **text)
        axes.invert_yaxis()
        axes.set_xlabel(chart["unit"], **text)
    else:
        axes.set_xticks(list(positions), labels, **text)
        axes.set_ylabel(chart["unit"], **text)

    axes.set_title(chart["title"], **text)

    if len(values) > 1:
        axes.legend(prop=font)

    stream = io.BytesIO()
    figure.savefig(stream, format="png", metadata={"Software": None})
    return stream.getvalue()


def _image_name(label, index):
    """파일 이름은 안전한 고정 형식만 쓴다. 입력 유래 글자가 경로·링크에 들어가지 않는다."""
    name = str(label).lower()
    return name if SAFE_NAME.fullmatch(name) else f"eda_{index}.png"


def _collect(facts, images):
    """저장할 (이름표, 파일 이름, 바이트, 실패 사유) 목록. 값이 없는 그래프는 facts의 사유로 이미 안내하므로 뺀다."""
    found = []

    for chart in facts["charts"]:
        if not chart["rows"]:
            continue

        try:
            found.append((chart["title"], chart["file"], draw_chart(chart), None))
        except LookupError:
            found.append((chart["title"], chart["file"], None, FONT_MISSING))
        except Exception:
            found.append((chart["title"], chart["file"], None, DRAW_FAILED))

    verified = {item["label"]: item["data"] for item in images}

    for index, item in enumerate(facts["images"], 1):
        data = verified.get(item["label"])
        found.append((item["label"], _image_name(item["label"], index), data, None if data else SOURCE_MISSING))

    return [(label, name, None, TOO_LARGE) if data and len(data) > FIGURE_LIMIT else (label, name, data, reason)
            for label, name, data, reason in found]


def _same_file(path, data):
    """기존 항목이 링크가 아닌 같은 바이트의 파일인지. 크기가 다르면 읽지 않는다."""
    info = path.lstat()
    return not _is_link(path) and path.is_file() and info.st_size == len(data) and path.read_bytes() == data


def _write(path, data):
    """새 파일에만 쓴다. 같은 바이트의 기존 파일(완료 복구의 재조립)은 그대로 쓴다. 실패 사유를 돌려주며 성공이면 None이다.

    지우는 것은 이번 호출이 만든 부분 파일뿐이다. 기존 항목은 읽지 못해도 건드리지 않는다.
    """
    try:
        if path.exists() or path.is_symlink():
            return None if _same_file(path, data) else NOT_OVERWRITTEN

        stream = path.open("xb")
    except FileExistsError:
        return NOT_OVERWRITTEN
    except OSError:
        return SAVE_FAILED

    try:
        with stream:
            stream.write(data)
    except OSError:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

        return SAVE_FAILED

    return None


def save_figures(facts, images, folder, attempt):
    """facts의 그래프와 검증된 EDA PNG를 저장하고 결과(figures)와 미처리 경고를 더한 facts를 돌려준다."""
    found = _collect(facts, images)
    ready = [(name, data) for _, name, data, reason in found if reason is None]
    digest = hashlib.sha256(b"".join(name.encode() + b"\0" + data for name, data in ready)).hexdigest()[:16]
    directory = f"figures_{attempt}_{digest}"
    target = folder / directory
    items = []

    try:
        # 폴더 이름에 들어가는 값과 재사용할 폴더를 확인한다. 재분석 지점이면 보고서 폴더 밖으로 나갈 수 있어 쓰지 않는다.
        if type(attempt) is not int or _is_link(folder):
            raise OSError("folder")

        if ready:
            target.mkdir(exist_ok=True)

            if _is_link(target):
                raise OSError("link")

        unavailable = None
    except OSError:
        unavailable = SAVE_FAILED

    for label, name, data, reason in found:
        reason = reason or unavailable or _write(target / name, data)
        items.append({"label": label, "file": name, "status": "failed" if reason else "saved", "reason": reason})

    saved = [item for item in items if item["status"] == "saved"]
    failed = [{"구분": "미처리", "내용": f"보고서 그림 저장 실패: {item['label']} ({item['reason']}). 이 그림은 저장 보고서에 없습니다."}
              for item in items if item["status"] == "failed"]
    return dict(facts, figures={"directory": directory if saved else None, "items": items}, warnings=[*facts.get("warnings", []), *failed])
