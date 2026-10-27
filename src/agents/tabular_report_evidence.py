"""표형 결과 보고서의 처리 전 EDA·계획과 결정 절. 검증된 근거(EDA 바이트·진단·확정 계획·후보·적용 기록)만 배치한다.

파일을 읽거나 통계를 새로 계산하지 않는다. EDA 표는 문자열 보존 CSV 파서로 읽어 범주·라벨 표기를 바꾸지 않으며,
숫자 통계 열만 결과 계약에 따라 숫자로 해석한다. 범주형 특징의 실제 범주값은 표시하지 않는다(정책 미결정).
표시 기준(상위 N·희소·상관 절댓값)은 표시용이며 처리·제외·품질 차단에 쓰지 않는다.
"""

from collections import Counter
import heapq

from . import tabular_options
from .eda_result_validation import _rows as _csv_rows
from src.preprocessing.input_profile import MIN_IQR_VALUES
from .strict_json import loads


# 표시용 제안값. 처리·제외·품질 차단·특징 선택 기준이 아니다.
TOP = 20
RARE_COUNT = 5
RARE_RATIO = 0.01
STRONG = 0.8
NAME_LIMIT = 80

MISSING = "미집계"
NOT_COMPUTED = "계산 불가"
BLANK = "(빈 표기)"
EDA_BASIS = ("검증 수준: 분할 EDA 격리 worker 집계. 부모는 파일 구성·헤더·역할·범위·합계를 검사했으며 통계를 다시 계산하지 "
             "않았습니다. 완료 표식 대조는 손상 검사 수준(승인 체인에 기대 해시 없음)입니다.")
DIAGNOSTIC_BASIS = ("검증 수준: 입력 진단 격리 worker 집계. 진단 결과 검증을 통과해 작업 기록에 저장된 값이며 부모가 다시 계산하지 "
                    "않았습니다.")
EDA_FILES = ("전체 표: 기술 상세 reports의 분할 EDA 보고서(eda.md)와 같은 폴더의 missing·numeric_statistics·"
             "categorical_distribution·target_distribution·pearson·spearman·correlation_support CSV")
EDA_SCOPE_NOTE = ("처리 전 EDA는 분할 전 전체 입력 기준입니다. Train·Validation·Test 구간별 처리 전 분포와 같은 구간의 처리 후 분포는 "
                  "산출하지 않았습니다(전후 분포 직접 비교 미산출). Test 구간은 이 보고서에서 새로 읽거나 집계하지 않았습니다.")
DTYPE_LABELS = {"numeric": "수치", "text": "문자열", "boolean": "논리값", "other": "기타"}
SPLIT_LABELS = {"strategy": "분할 방식", "train_ratio": "Train 비율", "validation_ratio": "Validation 비율",
                "test_ratio": "Test 비율", "random_state": "분할 시드", "stratify": "클래스 비율 유지(stratify)",
                "require_all_classes": "모든 구간에 모든 클래스 요구", "timestamp_format": "시간 형식"}


def _name(value):
    text = str(value)
    return text if len(text) <= NAME_LIMIT else text[:NAME_LIMIT - 1] + "…"


def _float(value):
    """EDA 숫자 통계 칸. 빈 칸은 계산 불가(NaN)이며 0으로 바꾸지 않는다."""
    return None if value == "" else float(value)


def _show(value, digits=4):
    return NOT_COMPUTED if value is None else f"{value:.{digits}g}"


def _ratio(count, total):
    return f"{count / total:.1%}" if total else "-"


def _shown_note(shown, total, order):
    omitted = total - shown
    return f"정렬·선정: {order} · 표시 {shown}개 / 전체 {total}개" + (f" / 생략 {omitted}개" if omitted else "")


def _table(title, notes, rows):
    return {"title": title, "notes": [note for note in notes if note], "rows": rows}


def parse_eda(payloads):
    """검증을 마친 EDA 바이트를 표로 해석한다. CSV 칸은 문자열 그대로 두고 summary만 strict JSON으로 읽는다."""
    summary = loads(payloads["summary.json"].decode("utf-8"), len(payloads["summary.json"]))
    tables = {name[:-4]: _csv_rows(payloads[name]) for name in payloads if name.endswith(".csv")}
    return {"summary": summary, **tables}


def _overview(eda, diagnostic, plan):
    summary, columns = diagnostic["summary"], diagnostic["columns"]
    rows_total = summary["row_count"]
    dtypes = Counter(column.get("dtype_group") for column in columns)
    roles = Counter(plan["roles"].get(column["column"], "제외") for column in columns)
    positive = sum(column.get("pos_inf_count") or 0 for column in columns)
    negative = sum(column.get("neg_inf_count") or 0 for column in columns)
    blank = sum(column.get("blank_string_count") or 0 for column in columns)
    infinite = sum(1 for column in columns if column.get("pos_inf_count") or column.get("neg_inf_count"))
    diagnosed = "전체 입력(분할 전) · 모든 컬럼"
    features = len(plan_features(plan))
    eda_summary = eda["summary"]
    rows = [
        {"항목": "행 수", "값": rows_total, "대상·분모": diagnosed, "근거": "입력 진단"},
        {"항목": "컬럼 수", "값": summary["column_count"], "대상·분모": diagnosed, "근거": "입력 진단"},
        {"항목": "자료형 구성", "값": " · ".join(f"{DTYPE_LABELS.get(key, key)} {count}" for key, count in sorted(dtypes.items(), key=lambda item: str(item[0]))),
         "대상·분모": diagnosed, "근거": "입력 진단(값 해석 기준의 자료형 그룹, 역할과 다름)"},
        {"항목": "역할 구성", "값": " · ".join(f"{role} {count}" for role, count in sorted(roles.items())),
         "대상·분모": "모든 컬럼", "근거": "확정 계획"},
        {"항목": "중복 행(모든 컬럼 값이 같은 행)", "값": f"{summary.get('repeated_rows', MISSING)}행",
         "대상·분모": diagnosed + f" · 분모 {rows_total}행", "근거": "입력 진단(중복에 속한 모든 행을 셈, 삭제하지 않음)"},
        {"항목": "특징 값이 같은 행", "값": f"{eda_summary['repeated_feature_rows']}행",
         "대상·분모": f"전체 입력(분할 전) · 특징 역할 컬럼 {features}개", "근거": "분할 EDA(동일 실물 중복의 확정 결과 아님)"},
        {"항목": "비유한 값(+/-무한대)", "값": f"+무한대 {positive} · -무한대 {negative} (컬럼 {infinite}개)",
         "대상·분모": diagnosed + " 중 수치 자료형", "근거": "입력 진단(자동 수정하지 않음)"},
        {"항목": "공백 문자열", "값": blank, "대상·분모": diagnosed + " 중 문자열 자료형", "근거": "입력 진단(자동 수정하지 않음)"},
        {"항목": "target", "값": "있음" if eda_summary["target_present"] else "없음(해당 없음)", "대상·분모": "확정 계획", "근거": "분할 EDA 요약"},
    ]

    if eda_summary["group_count"] is not None:
        rows.append({"항목": "group 수", "값": eda_summary["group_count"], "대상·분모": "전체 입력(분할 전) · group 컬럼",
                     "근거": "분할 EDA 요약"})

    return _table("입력 개요", [DIAGNOSTIC_BASIS, "특징 값이 같은 행·target·group은 분할 EDA 요약입니다. " + EDA_BASIS], rows)


def plan_features(plan):
    return [name for name, role in plan["roles"].items() if role in ("수치형", "범주형")]


def _columns(diagnostic, plan):
    """컬럼별 결측·고유값·비유한 값. 진단(전체 입력·모든 컬럼)만 사용한다."""
    columns, total = diagnostic["columns"], diagnostic["summary"]["row_count"]
    ordered = sorted(enumerate(columns), key=lambda item: (-item[1]["missing_count"], item[0]))
    rows = []

    for _, column in ordered[:TOP]:
        unique = column["observed_unique"]
        rows.append({"컬럼": _name(column["column"]), "역할": plan["roles"].get(column["column"], "제외"),
                     "자료형": DTYPE_LABELS.get(column.get("dtype_group"), column.get("dtype_group") or MISSING),
                     "결측(건)": column["missing_count"], "결측 비율": f"{column['missing_ratio']:.1%}",
                     "관측 고유값": MISSING if unique is None else unique,
                     "상수·전체 결측": "예" if unique is not None and unique <= 1 else "아니오",
                     "공백 문자열": "-" if column.get("blank_string_count") is None else column["blank_string_count"],
                     "+무한대": "-" if column.get("pos_inf_count") is None else column["pos_inf_count"],
                     "-무한대": "-" if column.get("neg_inf_count") is None else column["neg_inf_count"]})

    return _table("컬럼별 결측·고유값·비유한 값", [
        f"대상: 전체 입력(분할 전) · 모든 컬럼 {len(columns)}개 · 분모 {total}행. 특징만의 결측은 분할 EDA missing.csv에 있습니다.",
        _shown_note(len(rows), len(columns), "결측 건수 내림차순, 같으면 입력 순서"),
        "'-'는 해당 자료형이 아니라 세지 않은 항목입니다.", DIAGNOSTIC_BASIS,
        "생략한 컬럼의 진단 행은 이 보고서에 싣지 않았습니다. 특징 역할 컬럼의 전체 결측 표는 " + EDA_FILES[len("전체 표: "):] + " 중 missing에 있습니다."], rows)


def _numeric(eda):
    """수치 특징 기술 통계(분할 EDA). 결측을 제외한 관측값 기준이며 입력 순서 앞쪽을 보인다."""
    table = eda["numeric_statistics"]
    body = table[1:] if table != [[""]] else []
    rows = []

    for row in body[:TOP]:
        values = [_float(value) for value in row[1:]]
        std = values[2]
        rows.append({"특징": _name(row[0]), "관측 수": int(values[0]), "평균": _show(values[1]), "표준편차": _show(std),
                     "최솟값": _show(values[3]), "25%": _show(values[4]), "중앙값": _show(values[5]), "75%": _show(values[6]),
                     "최댓값": _show(values[7]),
                     "비고": "상수(표준편차 0)" if std == 0 else "관측 1개 이하라 표준편차 계산 불가" if std is None else "-"})

    notes = [f"대상: 전체 입력(분할 전) · 수치형 특징 {len(body)}개 · 컬럼별 결측을 제외한 관측값(관측 수가 분모)."]
    notes.append(_shown_note(len(rows), len(body), "입력 순서 앞쪽") if body else "수치형 특징이 없어 해당 없음입니다.")
    return _table("수치 특징 기술 통계", notes + [EDA_BASIS, EDA_FILES if body else ""], rows)


def _outliers(diagnostic, plan):
    """IQR 1.5 이상치 후보(진단). 오류 판정이 아니며 처리 여부는 확정 설정이 정한다."""
    checked = [(index, column) for index, column in enumerate(diagnostic["columns"]) if column.get("iqr_lower") is not None]
    found = sorted((item for item in checked if item[1].get("outlier_candidate_count")),
                   key=lambda item: (-item[1]["outlier_candidate_count"], item[0]))
    numeric = sum(1 for column in diagnostic["columns"] if column.get("dtype_group") == "numeric")
    rows = [{"컬럼": _name(column["column"]), "역할": plan["roles"].get(column["column"], "제외"),
             "유한 관측 수": column.get("finite_count", MISSING), "하한": _show(column["iqr_lower"]), "상한": _show(column["iqr_upper"]),
             "후보 수": column["outlier_candidate_count"],
             "후보 비율": _ratio(column["outlier_candidate_count"], column.get("finite_count") or 0)}
            for _, column in found[:TOP]]
    outlier = tabular_options.value_text("outlier", plan["preparation"].get("outlier"))
    return _table("IQR 이상치 후보 (진단용)", [
        f"대상: 전체 입력(분할 전) · 수치 자료형 컬럼 {numeric}개 중 유한 관측 {MIN_IQR_VALUES}개 이상인 {len(checked)}개(역할 무관) · 분모: 컬럼별 유한 관측 수.",
        _shown_note(len(rows), len(found), "후보 수 내림차순, 같으면 입력 순서") + f" (후보가 없는 컬럼 {len(checked) - len(found)}개는 생략)",
        "통계적 후보이며 오류 판정이 아닙니다. 정상 0/1 값도 포함될 수 있습니다. 처리 여부는 확정 설정 '극단값 자르기'"
        f"({outlier})가 정했고 이 표는 처리·제외에 쓰지 않았습니다.", DIAGNOSTIC_BASIS], rows)


def _categories(eda, plan):
    """범주 특징 요약(분할 EDA). 실제 범주값은 표시하지 않는다. 빈 칸 표기는 결측과 빈 문자열을 구분할 수 없다."""
    total = eda["summary"]["row_count"]
    entries = {}

    # 같은 표기라도 원래 자료형이 다른 범주는 별도 행이다. 표기로 합치지 않고 행 단위로 센다.
    for _, feature, value, count in eda["categorical_distribution"][1:]:
        entries.setdefault(feature, []).append((value, int(count)))

    order = {name: index for index, name in enumerate(name for name, role in plan["roles"].items() if role == "범주형")}
    summary = []

    for feature, items in entries.items():
        observed = [count for value, count in items if value != ""]
        rare = [count for count in observed if count < RARE_COUNT or count / total < RARE_RATIO]
        summary.append((feature, {"특징": _name(feature), "범주 수": len(observed),
                                  "빈 표기(건)": sum(count for value, count in items if value == ""),
                                  "최빈 범주 점유율": _ratio(max(observed), total) if observed else "-",
                                  "희소 범주 수": len(rare), "희소 범주 행 비율": _ratio(sum(rare), total)}))

    summary.sort(key=lambda item: (-item[1]["범주 수"], order.get(item[0], len(order))))
    rows = [row for _, row in summary[:TOP]]
    notes = [f"대상: 전체 입력(분할 전) · 범주형 특징 {len(summary)}개 · 분모 {total}행.",
             _shown_note(len(rows), len(summary), "범주 수 내림차순, 같으면 입력 순서") if summary else "범주형 특징이 없어 해당 없음입니다."]

    if summary:
        notes += [f"희소 범주(표시용 기준): 빈도 {RARE_COUNT} 미만 또는 전체 행의 {RARE_RATIO:.0%} 미만. 처리·제외 기준이 아닙니다.",
                  "실제 범주값은 표시하지 않았습니다(표시 범위 미결정). 빈 표기는 CSV 빈 칸이며 결측과 빈 문자열을 구분하지 못합니다.",
                  "범주 수·점유율은 원래 표기를 바꾸지 않고 센 값입니다('0012'와 '12'는 다른 범주)."]

    return _table("범주 특징 요약", notes + [EDA_BASIS], rows)


def _target_order(eda):
    """target 분포의 표·그래프 공용 선정. 건수 내림차순(같으면 파일 순서) 상위 TOP개와 전체 클래스 수."""
    table = eda["target_distribution"][1:]
    ordered = sorted(enumerate(table), key=lambda item: (-int(item[1][2]), item[0]))
    return [(label, int(count)) for _, (_, label, count) in ordered[:TOP]], len(table)


def _target(eda):
    total = eda["summary"]["row_count"]

    if not eda["summary"]["target_present"]:
        return _table("target 클래스 분포", ["target 역할이 없어 해당 없음입니다(확정 때 '예측 대상 없음')."], [])

    shown, classes = _target_order(eda)
    # 줄인 이름이 서로 같아 보여도 순위로 구분한다. 그래프의 클래스 축도 같은 순위를 쓴다.
    rows = [{"순위": rank, "클래스": _name(label) if label != "" else BLANK, "건수": count, "비율": _ratio(count, total)}
            for rank, (label, count) in enumerate(shown, 1)]
    cut = sum(1 for label, _ in shown if len(label) > NAME_LIMIT)
    return _table("target 클래스 분포", [
        f"대상: 전체 입력(분할 전) · target · 분모 {total}행(결측 포함). 구간별 클래스 수는 분할 결과 표를 보세요.",
        _shown_note(len(rows), classes, "건수 내림차순, 같으면 파일 순서"),
        f"이름이 {NAME_LIMIT}자를 넘는 클래스 {cut}개는 줄여 표시했습니다(순위로 구분)." if cut else "",
        "클래스 이름은 저장된 표기 그대로이며(예: '0012') 로컬 보고서에만 둡니다. 빈 표기는 결측 또는 빈 문자열입니다.", EDA_BASIS], rows)


def _correlations(eda):
    """수치 특징 쌍별 상관. 자기상관·대칭 중복을 빼고, 계산 불가(NaN)를 0으로 바꾸지 않는다."""
    pearson, spearman, support = eda["pearson"], eda["spearman"], eda["correlation_support"]
    names = pearson[0][1:]

    if len(names) < 2:
        return _table("수치 특징 상관 (쌍별)", [
            ("수치형 특징이 없어" if not names else "수치형 특징이 1개라 비교할 쌍이 없어") + " 해당 없음입니다."], [])

    # 전체 쌍 목록을 만들지 않고 상위 TOP개와 개수만 센다. 같은 |r|이면 입력 순서가 앞선 쌍을 먼저 둔다.
    total = len(names) * (len(names) - 1) // 2
    computed = strong = 0
    best, blank = [], []

    for order, (i, j) in enumerate((i, j) for i in range(len(names)) for j in range(i + 1, len(names))):
        r = _float(pearson[i + 1][j + 1])

        if r is None:
            if len(blank) < TOP:
                blank.append((i, j, r))
            continue

        computed += 1
        strong += abs(r) >= STRONG
        item = (abs(r), -order, i, j, r)

        if len(best) < TOP:
            heapq.heappush(best, item)
        elif item > best[0]:
            heapq.heapreplace(best, item)

    # 계산 가능한 쌍을 |Pearson| 내림차순으로, 계산 불가 쌍은 뒤에 입력 순서로 둔다.
    ordered = [(i, j, r) for _, _, i, j, r in sorted(best, reverse=True)] + blank
    rows = [{"특징 A": _name(names[i]), "특징 B": _name(names[j]), "Pearson": _show(r, 3),
             "Spearman": _show(_float(spearman[i + 1][j + 1]), 3), "쌍별 관측 수": int(support[i + 1][j + 1])}
            for i, j, r in ordered[:TOP]]
    return _table("수치 특징 상관 (쌍별)", [
        f"대상: 전체 입력(분할 전) · 수치형 특징 {len(names)}개의 쌍 {total}개(자기상관·대칭 중복 제외) · 분모: 두 특징이 함께 관측된 행.",
        _shown_note(len(rows), total, "|Pearson| 내림차순, 계산 불가는 뒤"),
        f"Pearson 계산 가능 {computed}쌍 · 계산 불가 {total - computed}쌍(관측 3 미만·상수 등, 0으로 바꾸지 않음) · "
        f"|Pearson| {STRONG} 이상 {strong}쌍(표시용 기준, 처리·제외에 쓰지 않음).",
        "쌍별 관측 수가 적은 계수는 확정적인 관계로 해석하지 마세요. 상관은 인과를 뜻하지 않습니다.", EDA_BASIS, EDA_FILES], rows)


def eda_tables(payloads, diagnostic, plan):
    """처리 전 데이터와 기본 EDA 절. 진단(전체 입력·모든 컬럼)과 분할 EDA(전체 입력·특징 역할)를 표마다 분리한다."""
    eda = parse_eda(payloads)
    return [_overview(eda, diagnostic, plan), _columns(diagnostic, plan), _numeric(eda), _outliers(diagnostic, plan),
            _categories(eda, plan), _target(eda), _correlations(eda)]


def _changed(name, candidate, application, expected):
    """설정별 실제 변경 근거. 저장된 후보·적용 집계만 쓰며 없으면 사유를 적는다."""
    impact = (candidate or {}).get("impact")
    removed = (candidate or {}).get("removed") or []

    def measured(key):
        report = (application or {}).get("impact")
        return None if report is None else (report["train"][key], report["validation"][key])

    if name in ("drop_constant", "missing_ratio_threshold"):
        if candidate is None:
            return MISSING
        code = "CONSTANT_OR_ALL_MISSING" if name == "drop_constant" else "HIGH_MISSING_RATIO"
        return f"Train 기준 제거 {sum(1 for item in removed if isinstance(item, dict) and item.get('reason') == code)}개(후보)"

    if name in ("numeric_imputation", "categorical_imputation", "outlier") and (impact is None or "missing" in expected):
        return expected.get("missing", MISSING)

    if name == "numeric_imputation":
        return f"Train 대치 {sum(item['imputed'] for item in impact if item['kind'] == 'numeric')}칸(후보 worker 집계)"

    if name == "categorical_imputation":
        return f"Train 범주 결측 {sum(item['missing'] for item in impact if item['kind'] == 'categorical')}칸 처리(후보 worker 집계)"

    if name == "outlier":
        text = f"Train 값 {expected['clipped_values']}개 clip(후보 worker 집계)"
        values = measured("clipped_values")
        return text + ("" if values is None else f" · 적용 Train {values[0]}·Validation {values[1]}개(적용 worker 집계)")

    if name == "unknown_category":
        values = measured("unknown_categories")
        return MISSING if values is None else f"Validation 미등록 범주 발생 {values[1]}건(특징별 발생 합계, 적용 결과)"

    return "-"


def decision_tables(record, plan, origin, recommendation, candidate, application, expected, split_rules):
    """계획과 결정 절. 결정 출처와 모델 의견(모델 문장)을 계산 사실·실행 기록 대조와 다른 열에 둔다.

    origin(record, path)은 기존 결정 출처 판정(질문 답변 > 사용자 변경 > 모델 권장 > 기본 계획)이며,
    recommendation은 권장안 사용 여부 문장이다. 실행 기록 대조는 후보 계산 설정(binding.rules)과 분할 실행 규칙만 쓴다.
    """
    rules = (candidate or {}).get("binding", {}).get("rules") or {}
    executed = {"split": split_rules or {}, "preparation": rules}
    sources = Counter()
    rows = []

    for section in ("roles", "split", "preparation"):
        for name, value in plan[section].items():
            source, reason = origin(record, section + "." + name)
            sources[source] += 1

            if section == "roles":
                continue

            label = SPLIT_LABELS.get(name, name) if section == "split" else tabular_options.GUIDE.get(name, {}).get("label", name)
            shown = tabular_options.value_text(name, value) if section == "preparation" else str(value)

            if name not in executed[section]:
                check = "실행 기록 없음" if executed[section] else MISSING
            elif executed[section][name] == value:
                check = "실행 값과 같음"
            else:
                check = "다름(실행 값: " + str(executed[section][name]) + ")"

            rows.append({"구분": "분할" if section == "split" else "전처리", "항목": label, "설정 이름": name, "최종 값": shown,
                         "결정 출처": source, "모델 의견": "-" if not reason else str(reason),
                         "실행 기록 대조": check,
                         "실제 변경(근거)": "분할 결과 표 참조" if section == "split" else _changed(name, candidate, application, expected)})

    answers = (record.get("confirmed") or {}).get("answers") or []
    summary = [{"항목": "모델 권장안", "내용": recommendation},
               {"항목": "결정 출처별 항목 수", "내용": " · ".join(f"{key} {count}" for key, count in sorted(sources.items()))
                + " (역할·분할·전처리 전체)"},
               {"항목": "질문 답변", "내용": f"{len(answers)}개" + (" (항목별 값은 참고 사항)" if answers else "")}]
    return [
        _table("권장안 사용과 결정 출처", [
            "결정 출처는 모델 초안·기본 계획과 확정 값을 비교한 결과입니다. 재시도에서 직전 확정 값을 이어 쓴 항목은 '사용자 변경'으로 "
            "보일 수 있습니다. 컬럼별 역할 출처와 제외 사유는 '컬럼별 처리와 제외 사유' 표에 있습니다."], summary),
        _table("최종 실행 설정 (분할·전처리)", [
            "모델 의견은 모델이 쓴 문장이며 계산 사실이 아닙니다. 실행 기록 대조는 분할 실행 규칙과 후보 계산 설정(승인된 실행 값)과의 비교입니다.",
            "실제 변경은 저장된 집계만 적었습니다. 후보 worker 집계는 부모가 형식·범위·결정적 관계만 확인한 값이고 원본 건수를 다시 센 것이 "
            "아닙니다. 대치값·경계·스케일 기준은 '컬럼별 처리와 제외 사유' 표에 있습니다."], rows),
    ]


# 그래프 축 이름 길이. 순위를 앞에 붙여 줄인 이름이 같아도 서로 다른 막대로 남는다.
CHART_NAME_LIMIT = 24
SPLITS = ("train", "validation", "test")


def _chart(title, notes, category, values, unit, rows, horizontal=False):
    """화면 막대 그래프와 Markdown 표·텍스트 막대가 함께 쓰는 사실. rows가 비면 그리지 않고 notes로 사유를 보인다."""
    return {"title": title, "notes": [note for note in notes if note], "category": category, "values": list(values),
            "unit": unit, "horizontal": horizontal, "rows": rows}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def split_chart(split_rules, proposal):
    """요청·확정 분할 비율. 분할 결과 표와 같은 근거(확정 계획의 분할 설정·검증된 분할 제안 summary)만 쓴다."""
    title = "요청·확정 분할 비율 (구간별)"
    partitions = {item["split"]: item for item in (proposal or {}).get("summary", [])}
    rows, absent, actual = [], [], {}

    for name in SPLITS:
        item = partitions.get(name) or {}
        requested, confirmed, count = split_rules.get(name + "_ratio"), item.get("ratio"), item.get("rows")

        if not (_number(requested) and _number(confirmed) and _number(count)):
            absent.append(name)
            continue

        # 확정 비율은 분할 결과 표와 같은 소수 셋째 자리 값이다. 요청 비율은 확정 계획 값 그대로다.
        rows.append({"구간": name, "요청 비율": requested, "확정 비율": round(confirmed, 3), "행 수": count})
        actual[name] = abs(confirmed - requested)

    if absent:
        return _chart(title, ["근거 없음: " + ", ".join(absent) + " 구간의 요청 비율·확정 비율·행 수 중 일부가 검증된 결과에 없어 "
                              "그래프를 그리지 않았습니다(추정하지 않음)."], "구간", ("요청 비율", "확정 비율"), "비율", [])

    # 가장 큰 구간은 반올림 전 실제 차이로 고른다(같으면 구간 순서가 앞선 쪽). 반올림은 표시상 차이가 보이는지만 정한다.
    # 12자리 정리는 표시 반올림이 아니라 2/12와 4/12처럼 실제로 같은 차이가 부동소수 오차로 갈리지 않게 하는 것이다.
    gap = max(rows, key=lambda row: round(actual[row["구간"]], 12))
    if any(round(row["확정 비율"] - row["요청 비율"], 3) for row in rows):
        difference = (f"요청과 확정의 차이가 가장 큰 구간: {gap['구간']} ({gap['요청 비율']} → {gap['확정 비율']:.3f}). "
                      "group·시간 경계나 클래스 비율 유지를 지키면 확정 비율이 요청과 다를 수 있습니다.")
    else:
        # 셋째 자리에서는 차이가 보이지 않아도 실제 비율은 정수 행 배정 때문에 요청과 다를 수 있다. 같다고 쓰지 않는다.
        difference = "행 수를 정수로 배정해 요청 비율과 소폭 차이가 있습니다." if any(actual.values()) else ""
    chart = _chart(title, [
        f"대상: 확정 분할의 행 배정(전처리 전) · 분모 {sum(row['행 수'] for row in rows)}행(세 구간 행 수 합).",
        "요청 비율은 확정 계획의 분할 설정, 확정 비율·행 수는 분할 결과 검증을 통과한 확정 제안의 값입니다(새로 계산하지 않음). "
        "그래프와 '분할 결과' 표는 같은 값입니다.",
        "확정 비율은 소수 셋째 자리로 반올림해 세 구간의 합이 1.000과 다르게 보일 수 있습니다. 정확한 배정은 행 수입니다.",
        difference], "구간", ("요청 비율", "확정 비율"), "비율", rows)
    # 표의 확정 비율은 분할 결과 표와 같은 셋째 자리 형식으로 보인다(그래프 값은 숫자 그대로).
    chart["digits"] = {"확정 비율": 3}
    return chart


def _short(text):
    return text if len(text) <= CHART_NAME_LIMIT else text[:CHART_NAME_LIMIT - 1] + "…"


def target_chart(payloads):
    """target 클래스 분포. 처리 전 EDA 절의 target 표와 같은 검증된 EDA 바이트·같은 선정(_target_order)을 쓴다."""
    title = "target 클래스 분포 (전체 입력, 분할 전)"
    eda = parse_eda({name: payloads[name] for name in ("summary.json", "target_distribution.csv")})
    total = eda["summary"]["row_count"]

    if not eda["summary"]["target_present"]:
        return _chart(title, ["해당 없음: target 역할이 없어 클래스 분포 그래프가 없습니다(확정 때 '예측 대상 없음')."],
                      "클래스", ("건수",), "건수(행)", [])

    shown, classes = _target_order(eda)
    rows = [{"클래스": f"{rank:02d}. " + (_short(label) if label != "" else BLANK), "건수": count, "비율": _ratio(count, total)}
            for rank, (label, count) in enumerate(shown, 1)]
    omitted = total - sum(count for _, count in shown)
    return _chart(title, [
        f"대상: 전체 입력(분할 전·전처리 전) · target · 분모 {total}행(결측 포함). 구간별·처리 후 분포가 아닙니다.",
        _shown_note(len(rows), classes, "건수 내림차순, 같으면 파일 순서")
        + (f" · 생략한 클래스의 행 {omitted}개" if classes > len(rows) else ""),
        f"클래스 이름은 {CHART_NAME_LIMIT}자까지 줄이고 앞의 번호(처리 전 EDA 절 target 표의 순위)로 구분합니다. 그 표는 "
        f"{NAME_LIMIT}자까지 보이며, 줄이지 않은 이름은 기술 상세 분할 EDA 폴더의 target_distribution.csv에 있습니다.",
        "클래스 1개뿐이라 클래스 간 불균형은 비교할 수 없습니다." if classes == 1 else "",
        EDA_BASIS], "클래스", ("건수",), "건수(행)", rows, horizontal=True)
