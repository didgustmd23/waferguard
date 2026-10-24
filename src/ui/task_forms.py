"""작업 유형별 계획 편집 부품 등록부와 항목 명세로 그리는 일반 양식. 공통 작업 화면이 이 등록부만 사용한다."""

from . import array_task_form, tabular_task_form


# v2 표형 작업은 진단 표·계획 구조가 CSV v1과 같아 같은 부품을 쓴다.
FORMS = {"tabular_csv": tabular_task_form.render, "tabular_v2": tabular_task_form.render, "array_npz": array_task_form.render}


def generic_form(st, record, plan, fields, key):
    """항목 명세(enum·boolean·integer·number)만으로 계획 양식을 그린다. 전용 부품이 없는 유형에 쓴다."""
    updated = {section: dict(values) for section, values in plan.items()}

    for number, field in enumerate(fields, start=1):
        section, name = field["path"].split(".", 1)
        value = updated[section][name]
        widget_key = key + "_" + str(number)
        # 위젯 이름은 마크다운으로 해석되므로 고정 번호만 쓰고, 항목 경로(입력 이름 포함 가능)는 일반 텍스트로 보인다.
        label = f"항목 {number}"
        st.text(label + ": " + field["path"])

        if field["type"] == "enum":
            options = [option for option in field["options"] if option not in field.get("unsupported", [])]
            updated[section][name] = st.selectbox(label, options, key=widget_key,
                                                  index=options.index(value) if value in options else None)
        elif field["type"] == "boolean":
            updated[section][name] = st.checkbox(label, value=bool(value), key=widget_key)
        elif field["type"] == "integer":
            updated[section][name] = int(st.number_input(label, int(field.get("minimum", 0)),
                                                         int(field.get("maximum", 2 ** 31)), int(value), key=widget_key))
        elif field["type"] == "number":
            updated[section][name] = float(st.number_input(label, float(field.get("minimum", 0.0)),
                                                           float(field.get("maximum", 1e12)), float(value), key=widget_key))

    return updated


def form_for(kind):
    return FORMS.get(kind, generic_form)
