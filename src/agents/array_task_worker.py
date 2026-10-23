"""배열 표본 묶음(.npz) 통합 작업의 격리 worker. 고정 단계 3개(진단·Train 기준 후보·적용)만 실행한다.

부모가 승인 소비 후 복사한 작업 입력만 읽는다. 원본 폴더·승인 DB·다른 작업은 보지 않는다. 계산은
src.preprocessing.array_dataset이 하고, 이 모듈은 고정 경로·설정·snapshot 대조와 결과 파일 쓰기만 맡는다.
"""

from pathlib import Path
import sys


STAGES = {"audit": "run_audit", "derive_preparation": "derive_preparation", "apply_preparation": "apply_preparation"}
OUTPUTS = {"audit": "profile", "derive_preparation": "candidate", "apply_preparation": "prepared"}
POLICY_VERSION = "array_task_v1"
MAX_RECORD_BYTES = 64 * 1024 * 1024


def _diagnostic(root, stage, code, detail=None):
    """고정 단계·분류만 scratch에 기록한다. 세부 코드는 계산 코어의 고정 코드만 덧붙인다."""
    import json

    from src.agents.snapshot_eda_worker import _diagnostic as base

    base(root, stage, code)

    if detail is not None:
        path = root / "scratch" / "task" / "diagnostic.json"

        try:
            item = json.loads(path.read_text(encoding="utf-8"))

            if type(item) is dict and set(item) == {"stage", "code"}:
                path.write_text(json.dumps(dict(item, detail=detail)), encoding="utf-8")
        except (OSError, ValueError):
            # 세부 코드 기록 실패가 기존 분류를 바꾸지 않는다.
            pass


def _write_json(path, value):
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")

    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("결과 기록 크기 상한을 초과했습니다.")

    with path.open("xb") as stream:
        stream.write(payload)


def _read_json(path):
    from src.agents import strict_json

    with path.open("rb") as stream:
        data = stream.read(MAX_RECORD_BYTES + 1)

    return strict_json.loads(data.decode("utf-8"), MAX_RECORD_BYTES)


def _split_ids(path, headers, mapping, count):
    """표본 메타데이터(라벨·그룹·ID)를 표본 위치 순서의 파이썬 값 목록으로 읽는다."""
    from src.preprocessing.array_dataset import ArrayInputError, load_array, meta_missing, meta_text

    def values(role):
        key = mapping[role]
        return None if key is None else load_array(path, key, headers[key])

    labels, groups, ids = values("label"), values("group"), values("sample_id")

    if groups is not None and meta_missing(groups).any():
        raise ArrayInputError("GROUP_MISSING", "그룹 값이 비어 있는 표본이 있습니다.")

    if ids is not None and (meta_missing(ids).any() or len(set(ids.tolist())) != count):
        raise ArrayInputError("SAMPLE_ID_INVALID", "표본 ID가 비어 있거나 중복됩니다.")

    return (labels, None if groups is None else [meta_text(value) for value in groups],
            [meta_text(value) for value in ids] if ids is not None else [str(index) for index in range(count)])


def _prepare(path, packet, output):
    """분할(메타데이터만) → Train 채널 기준 → Train 예상 영향. 변환 파일은 만들지 않는다."""
    import numpy as np

    from src.preprocessing.array_dataset import (ArrayInputError, channel_first, check_mapping, derive_state, apply_state,
                                                 load_array, meta_missing, meta_text, sample_hashes, split_frame)
    from src.preprocessing.split import SPLITS, SplitRules, propose_split

    headers, mapping, options = packet["headers"], packet["mapping"], packet["preparation"]
    count, _ = check_mapping(headers, mapping)
    samples = load_array(path, mapping["samples"], headers[mapping["samples"]])
    labels, groups, ids = _split_ids(path, headers, mapping, count)
    missing = meta_missing(labels) if labels is not None else np.zeros(count, dtype=bool)

    if missing.any() and options["missing_labels"] is None:
        raise ArrayInputError("LABEL_MISSING_UNDECIDED", "라벨 결측 표본의 처리 방법이 정해지지 않았습니다.")

    excluded = [int(index) for index in np.flatnonzero(missing)] if options["missing_labels"] == "exclude" else []
    dropped = set(excluded)
    kept = [index for index in range(count) if index not in dropped]
    label_text = None if labels is None else [None if missing[index] else meta_text(labels[index]) for index in range(count)]
    use_label = labels is not None and not missing[kept].any()
    hashes = sample_hashes(samples)
    # 같은 표본 묶기를 고르면 그룹 단위로 나눈다(사용자가 확정한 선택). random이면 그룹 역할을 분할에 넘기지 않는다.
    grouped = packet["split"]["strategy"] == "group" or options["duplicate_grouping"]
    data = split_frame(kept, label_text, groups if grouped else None, ids, hashes, use_label=use_label,
                       duplicate_grouping=options["duplicate_grouping"])
    rules = SplitRules(**dict(packet["split"], strategy="group" if grouped else "random"), require_all_classes=False)
    from src.agents.input_snapshot import Snapshot

    proposal = propose_split(data, rules, source_id=Snapshot(**packet["snapshot"]).source_id, config_id=packet["plan_digest"])
    partitions = {name: [kept[index] for index in proposal["partitions"][name]] for name in SPLITS}
    owner = {position: name for name in SPLITS for position in partitions[name]}
    first = {}
    cross = 0

    # 구간을 넘는 완전 동일 표본 수. 묶기를 고르지 않았으면 누수 경고 근거로 남긴다.
    for position in kept:
        seen = first.setdefault(hashes[position], owner[position])
        cross += seen != owner[position]

    group_overlap = 0

    if groups is not None:
        members = {name: {groups[position] for position in partitions[name]} for name in SPLITS}
        group_overlap = sum(len(members[a] & members[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))

    view = channel_first(samples, mapping["channel_axis"])
    train = view[partitions["train"]]

    if not len(partitions["train"]) or not len(partitions["validation"]):
        raise ArrayInputError("SPLIT_EMPTY", "Train 또는 Validation 구간이 비었습니다.")

    state = derive_state(train, options)
    _, impact = apply_state(train, state, options)
    summary = [{"split": name, "samples": len(partitions[name]),
                "classes": ([{"value": value, "count": int(sum(1 for position in partitions[name] if label_text[position] == value))}
                             for value in sorted({label_text[position] for position in partitions[name] if label_text[position] is not None})]
                            if label_text is not None else []),
                "groups": len({groups[position] for position in partitions[name]}) if groups is not None else None}
               for name in SPLITS]
    folder = output / "candidate"
    folder.mkdir()
    _write_json(folder / "assignment.json", {"version": "array_assignment_v1", "samples": count, "partitions": partitions,
                                              "excluded": excluded, "summary": summary, "cross_split_duplicates": cross,
                                              "group_overlap": group_overlap, "warnings": proposal["warnings"]})
    _write_json(folder / "state.json", state)
    _write_json(folder / "candidate.json", {"version": "array_candidate_v1", "binding": packet, "impact": impact,
                                             "transformed": False, "model_trained": False})


def _apply(path, packet, inputs, output):
    """검증된 분할·기준으로 Train·Validation만 변환한다. Test는 읽지도 쓰지도 않는다."""
    import numpy as np

    from src.preprocessing.array_dataset import (apply_state, channel_first, check_mapping, impact_totals, load_array,
                                                 meta_missing, restore_layout)

    headers, mapping, options = packet["headers"], packet["mapping"], packet["preparation"]
    check_mapping(headers, mapping)
    assignment, state = _read_json(inputs / "assignment.json"), _read_json(inputs / "state.json")
    samples = channel_first(load_array(path, mapping["samples"], headers[mapping["samples"]]), mapping["channel_axis"])
    labels, groups, ids = _split_ids(path, headers, mapping, headers[mapping["samples"]]["shape"][0])
    missing = None if labels is None else meta_missing(labels)
    folder = output / "prepared"
    folder.mkdir()
    impact = {}

    for name in ("train", "validation"):
        positions = assignment["partitions"][name]
        transformed, items = apply_state(samples[positions], state, options)

        with (folder / f"{name}_samples.npy").open("xb") as stream:
            np.save(stream, np.ascontiguousarray(restore_layout(transformed, mapping["channel_axis"])), allow_pickle=False)

        _write_json(folder / f"{name}_meta.json", {
            "positions": positions, "sample_id": [ids[position] for position in positions],
            "label": None if labels is None else [None if missing[position] else labels[position].item() for position in positions],
            "group": None if groups is None else [groups[position] for position in positions]})
        impact[name] = impact_totals(items)

    _write_json(folder / "preprocessor.json", state)
    _write_json(folder / "application.json", {"version": "array_application_v1", "binding": packet, "impact": impact,
                                               "transformed": True, "refit": False, "test_used": False, "model_trained": False})


def _profile(path, packet, output):
    """키별 진단 요약과 표본 미리보기(기본 매핑의 표본 배열, 앞에서부터 최대 12개)."""
    from src.preprocessing.array_dataset import diagnose, load_array, previews

    summary = diagnose(path, packet["max_total_bytes"])
    folder = output / "profile"
    folder.mkdir()
    shown = []
    key = summary["base_mapping"]["samples"]

    if key is not None:
        (folder / "previews").mkdir()

        for position, data in previews(load_array(path, key, summary["headers"][key])):
            name = f"preview_{position}.png"
            (folder / "previews" / name).write_bytes(data)
            shown.append({"file": name, "position": position, "key": key})

    _write_json(folder / "summary.json", dict(summary, previews=shown))


def execute_request(request, root):
    """요청 단계·고정 경로·설정 해시·snapshot을 확인한 뒤 단계 계산을 한다."""
    import zipfile

    from src.agents.input_snapshot import Snapshot
    from src.agents.snapshot_eda_worker import _verify_staged_snapshot
    from src.agents.worker_contract import WorkerResult, _decode
    from src.preprocessing.array_dataset import ArrayInputError
    from src.preprocessing.split import _digest

    root = Path(root).resolve()
    source = root / "input/snapshot/input.bin"
    settings = root / "input/array_task.json"
    output = root / "output/result"
    candidate = root / "input/candidate"
    stage = "REQUEST_PATHS"

    try:
        files = (str(source), str(source.parent / "snapshot.json"), str(settings))

        if request.stage == "apply_preparation":
            files += tuple(str(candidate / name) for name in ("assignment.json", "state.json", "candidate.json"))

        if (request.task_id != root.name or STAGES.get(request.stage) != request.action_id or request.policy_version != POLICY_VERSION
                or request.dependency_directories or request.input_files != files or request.output_directory != str(output)
                or request.scratch_directory != str(root / "scratch/task")):
            raise ValueError("배열 작업의 고정 요청 경로가 다릅니다.")

        stage = "SETTINGS"

        with settings.open("rb") as stream:
            packet = _decode(stream.read(65537).decode("utf-8"))

        if (type(packet) is not dict or packet.get("stage") != request.stage or packet.get("version") != POLICY_VERSION
                or _digest(packet) != request.config_id or packet.get("limits") != {name: getattr(request.limits, name) for name in ("wall_seconds", "memory_mib", "processes")}):
            raise ValueError("배열 작업 설정이 승인 계약과 다릅니다.")

        expected = Snapshot(**packet["snapshot"])

        if request.source_id != expected.source_id:
            raise ValueError("배열 작업 출처가 다릅니다.")

        stage = "SNAPSHOT"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("작업 입력 복사본이 변경되었습니다.")

        stage = "INPUT_VALIDATION"

        if request.stage == "audit":
            _profile(source, packet, output)
        elif request.stage == "derive_preparation":
            _prepare(source, packet, output)
        else:
            _apply(source, packet, candidate, output)

        stage = "SNAPSHOT_RECHECK"

        if not _verify_staged_snapshot(root, expected):
            raise ValueError("계산 중 입력 복사본이 변경되었습니다.")
    except ArrayInputError as error:
        _diagnostic(root, stage, "INPUT_INVALID", error.code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, IndexError, zipfile.BadZipFile) as error:
        # 손상된 zip은 입력 문제다. 파일 접근·입출력 실패와 구분한다.
        code = "ACCESS_DENIED" if isinstance(error, PermissionError) else "IO_FAILED" if isinstance(error, OSError) else "INPUT_INVALID"
        _diagnostic(root, stage, code)
        return WorkerResult(request.task_id, request.request_id, "blocked_input", "WORKER_INPUT_BLOCKED")
    except Exception:
        _diagnostic(root, stage, "RUNTIME_FAILED")
        return WorkerResult(request.task_id, request.request_id, "failed_runtime", "WORKER_FAILED")

    _diagnostic(root, "COMPLETE", "PASSED")
    return WorkerResult(request.task_id, request.request_id, "succeeded", "WORKER_EXITED_OK", (OUTPUTS[request.stage],))


def main():
    """고정 복사 런타임을 초기화한다. 요청 식별값이 다르면 실행하지 않는다."""
    import os

    root = Path(sys.argv[1]).resolve()
    runtime = root / "runtime"
    sys.path[:0] = [str(root / "tools"), str(runtime / "Lib/site-packages")]
    handles = [os.add_dll_directory(str(path)) for path in (runtime, runtime / "DLLs", runtime / "Library/bin") if path.is_dir()]

    from src.agents.worker_contract import WorkerRequest

    try:
        with (root / "input/request.json").open("rb") as stream:
            request = WorkerRequest.from_json(stream.read(65537).decode("utf-8"))

        if request.request_id != sys.argv[2]:
            return 2

        result = execute_request(request, root)
        (root / "output/result/worker_result.json").write_text(result.to_json(), encoding="utf-8")
        return 0 if result.status == "succeeded" else 2
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
