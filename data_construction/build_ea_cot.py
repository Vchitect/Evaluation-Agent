#!/usr/bin/env python3
"""Build EA-CoT Alpaca records from curated Evaluation Agent trajectories.

This utility intentionally performs format conversion only. Release curation and
evidence-faithfulness filtering must happen before its input directory is passed
to the builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCORE_TABLES = {
    "t2v": REPO_ROOT / "eval_agent" / "vbench_dimension_scores.tsv",
    "t2i": REPO_ROOT / "eval_agent" / "t2i_comp_dimension_scores.tsv",
}

TRAINING_PROMPT_KEYS = {
    "t2v": "eval-agent-vbench-training-sys_v1",
    "t2i": "eval-agent-t2i-training-sys",
}
FORMAT_PROMPT_KEY = "eval-agent-format-sys"

SUBASPECT_TEMPLATES = (
    "I will evaluate the model's sub-aspect: {subaspect}.",
    "I will focus on the {subaspect} sub-aspect of the model.",
    "Let me assess the {subaspect} sub-aspect of this model.",
    "I need to examine the model's {subaspect} sub-aspect.",
    "Now I will analyze the {subaspect} sub-aspect dimension.",
    "I'll investigate the {subaspect} sub-aspect quality of the model.",
    "Time to evaluate the {subaspect} sub-aspect performance.",
    "I should check the model's {subaspect} sub-aspect capabilities.",
)


class TrajectoryError(ValueError):
    """Raised when one source trajectory cannot be converted safely."""


@dataclass
class ConversionResult:
    records: List[Dict[str, Any]]
    chat_messages_stripped: bool
    tool_records: int
    summary_records: int


@dataclass
class BuildReport:
    modality: str
    input_dir: str
    output: str
    score_table: str
    seed: int
    source_content_deduplication: bool
    discovered_files: int = 0
    processed_files: int = 0
    records: int = 0
    tool_records: int = 0
    summary_records: int = 0
    chat_message_lists_stripped: int = 0
    duplicate_files_skipped: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)


def resolve_from_repo(raw_path: str) -> Path:
    """Resolve a CLI path; relative paths are stable against the repository root."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalized_dimension(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().casefold())


def load_score_table(path: Path) -> Dict[str, Dict[str, str]]:
    required_columns = (
        "Dimension",
        "Very High",
        "High",
        "Moderate",
        "Low",
        "Very Low",
    )

    if not path.is_file():
        raise FileNotFoundError(f"Score table does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [
            column
            for column in required_columns
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"Score table {path} is missing columns: {', '.join(missing)}"
            )

        result: Dict[str, Dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            dimension = (row.get("Dimension") or "").strip()
            if not dimension:
                raise ValueError(f"Empty Dimension in {path}:{line_number}")
            key = normalized_dimension(dimension)
            if key in result:
                raise ValueError(f"Duplicate Dimension {dimension!r} in {path}")
            result[key] = {
                column: (row.get(column) or "").strip() for column in required_columns
            }

    if not result:
        raise ValueError(f"Score table is empty: {path}")
    return result


def load_system_prompt(modality: str) -> str:
    """Load release prompts from eval_agent/system_prompts.py."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        prompt_module = importlib.import_module("eval_agent.system_prompts")
        prompts = prompt_module.sys_prompts
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Could not import sys_prompts from eval_agent/system_prompts.py"
        ) from exc

    training_key = TRAINING_PROMPT_KEYS[modality]
    missing = [key for key in (training_key, FORMAT_PROMPT_KEY) if key not in prompts]
    if missing:
        raise KeyError("Missing required system prompt key(s): " + ", ".join(missing))

    # Preserve the concatenation used to construct the released training data.
    return prompts[training_key] + prompts[FORMAT_PROMPT_KEY]


def first_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def require_text(value: Any, label: str, source: Path, step_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryError(
            f"step {step_index}: {label} must be a non-empty string in {source}"
        )
    return value.strip()


def render_narrative(value: Any, label: str, source: Path, step_index: int) -> str:
    """Render legacy narrative fields without discarding structured content."""
    if isinstance(value, str):
        if value.strip():
            return value.strip()
    elif isinstance(value, (dict, list)):
        if value:
            # The original release builder interpolated these values in an
            # f-string. Preserve that representation for legacy trajectories
            # whose Analysis or Summary was emitted as structured JSON.
            return str(value)

    raise TrajectoryError(
        f"step {step_index}: {label} must be non-empty text, object, or list in {source}"
    )


def is_chat_messages_list(value: Any) -> bool:
    """Return true only for a non-empty OpenAI-style messages list."""
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and "content" in message
        for message in value
    )


def format_score(value: Any, label: str, source: Path, step_index: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(
            f"step {step_index}: {label} is not numeric in {source}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise TrajectoryError(
            f"step {step_index}: {label} is not finite in {source}: {value!r}"
        )
    return f"{number:.4f}"


def score_reference(tool: str, table: Mapping[str, Mapping[str, str]]) -> str:
    row = table.get(normalized_dimension(tool))
    if row is None:
        available = ", ".join(sorted(item["Dimension"] for item in table.values()))
        raise TrajectoryError(
            f"tool {tool!r} has no score-table row; available dimensions: {available}"
        )
    return (
        f"{row['Dimension']}: "
        f"Very High -> {row['Very High']}, "
        f"High -> {row['High']}, "
        f"Moderate -> {row['Moderate']}, "
        f"Low -> {row['Low']}, "
        f"Very Low -> {row['Very Low']}"
    )


def format_observation(
    step: Mapping[str, Any],
    tool: str,
    modality: str,
    table: Mapping[str, Mapping[str, str]],
    source: Path,
    step_index: int,
) -> str:
    eval_results = step.get("eval_results")
    if not isinstance(eval_results, dict):
        raise TrajectoryError(
            f"step {step_index}: eval_results must be an object in {source}"
        )

    score_payload = eval_results.get("score")
    if not isinstance(score_payload, (list, tuple)) or len(score_payload) < 2:
        raise TrajectoryError(
            f"step {step_index}: eval_results.score must be [overall, details] in {source}"
        )

    details = score_payload[1]
    if not isinstance(details, list):
        raise TrajectoryError(
            f"step {step_index}: per-prompt score details must be a list in {source}"
        )

    score_key = "video_results" if modality == "t2v" else "image_results"
    lines = [
        f"Scoring Reference Table of '{tool}': {score_reference(tool, table)}",
        "",
        "Results:",
        f"- Overall score: {format_score(score_payload[0], 'overall score', source, step_index)}",
        "- Per-prompt scores:",
    ]

    for detail_index, detail in enumerate(details):
        if not isinstance(detail, dict):
            raise TrajectoryError(
                f"step {step_index}: detail {detail_index} must be an object in {source}"
            )
        prompt = require_text(
            detail.get("prompt"),
            f"detail {detail_index} prompt",
            source,
            step_index,
        )
        if score_key not in detail:
            raise TrajectoryError(
                f"step {step_index}: detail {detail_index} has no {score_key!r} in {source}"
            )
        score = format_score(
            detail[score_key],
            f"detail {detail_index} {score_key}",
            source,
            step_index,
        )
        lines.append(f'  • "{prompt}": {score}')

    return "<information>" + "\n".join(lines) + "\n</information>"


def convert_trajectory(
    payload: Any,
    source: Path,
    modality: str,
    table: Mapping[str, Mapping[str, str]],
    system_prompt: str,
    rng: random.Random,
) -> ConversionResult:
    if not isinstance(payload, list) or len(payload) < 2:
        raise TrajectoryError(
            f"trajectory must be a JSON list with at least two items: {source}"
        )

    items = list(payload)
    stripped_messages = False
    if items and is_chat_messages_list(items[-1]):
        items.pop()
        stripped_messages = True

    if len(items) < 2:
        raise TrajectoryError(f"trajectory has no decision or summary steps: {source}")

    user_query = require_text(items[0], "user query", source, 0)
    steps = items[1:]
    history: List[List[str]] = []
    current_instruction = user_query
    records: List[Dict[str, Any]] = []
    tool_records = 0
    summary_records = 0
    saw_summary = False

    for offset, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            raise TrajectoryError(
                f"step {offset}: expected an object, got {type(raw_step).__name__} in {source}"
            )
        if saw_summary:
            raise TrajectoryError(
                f"step {offset}: data appears after the final summary in {source}"
            )

        tool_value = first_value(raw_step, ("Tool", "tool"))
        summary_value = first_value(raw_step, ("Summary", "summary"))

        if tool_value is not None:
            tool = require_text(tool_value, "tool", source, offset)
            thought = require_text(
                first_value(raw_step, ("Thought", "thought")),
                "thought",
                source,
                offset,
            )
            subaspect = require_text(
                first_value(raw_step, ("Sub-aspect", "sub_aspect", "subaspect")),
                "sub-aspect",
                source,
                offset,
            )
            subaspect_sentence = rng.choice(SUBASPECT_TEMPLATES).format(
                subaspect=subaspect
            )
            output = f"<think>{thought} {subaspect_sentence}</think><tool>{tool}</tool>"
            observation = format_observation(
                raw_step,
                tool,
                modality,
                table,
                source,
                offset,
            )
            tool_records += 1

        elif summary_value is not None:
            thought = require_text(
                first_value(raw_step, ("Thought", "thought")),
                "summary thought",
                source,
                offset,
            )
            summary = render_narrative(summary_value, "summary", source, offset)
            analysis_value = first_value(raw_step, ("Analysis", "analysis"))
            if analysis_value is None:
                summary_body = f"Summary: {summary}"
            else:
                analysis = render_narrative(analysis_value, "analysis", source, offset)
                summary_body = f"Analysis: {analysis}\nSummary: {summary}"
            output = f"<think>{thought}</think><summary>{summary_body}</summary>"
            observation = None
            summary_records += 1
            saw_summary = True

        else:
            raise TrajectoryError(
                f"step {offset}: expected either Tool/tool or Summary/summary in {source}"
            )

        records.append(
            {
                "instruction": current_instruction,
                "input": "",
                "output": output,
                "system": system_prompt,
                "history": [pair[:] for pair in history],
            }
        )

        if observation is not None:
            history.append([current_instruction, output])
            current_instruction = observation

    if not saw_summary:
        raise TrajectoryError(f"trajectory has no final Summary/summary step: {source}")
    if summary_records != 1:
        raise TrajectoryError(
            f"trajectory must contain exactly one summary step: {source}"
        )

    return ConversionResult(
        records=records,
        chat_messages_stripped=stripped_messages,
        tool_records=tool_records,
        summary_records=summary_records,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def relative_display(path: Path, input_dir: Path) -> str:
    try:
        return path.relative_to(input_dir).as_posix()
    except ValueError:
        return str(path)


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_sources(input_dir: Path, excluded: Sequence[Path]) -> List[Path]:
    excluded_resolved = {path.resolve() for path in excluded}
    return sorted(
        (
            path
            for path in input_dir.rglob("*.json")
            if path.is_file() and path.resolve() not in excluded_resolved
        ),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )


def build_dataset(
    modality: str,
    input_dir: Path,
    output: Path,
    score_table_path: Path,
    seed: int,
    deduplicate_source_content: bool,
    report_path: Optional[Path],
) -> BuildReport:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    table = load_score_table(score_table_path)
    system_prompt = load_system_prompt(modality)
    excluded = [output]
    if report_path is not None:
        excluded.append(report_path)
    sources = discover_sources(input_dir, excluded)
    if not sources:
        raise FileNotFoundError(f"No JSON files found recursively under: {input_dir}")

    report = BuildReport(
        modality=modality,
        input_dir=str(input_dir),
        output=str(output),
        score_table=str(score_table_path),
        seed=seed,
        source_content_deduplication=deduplicate_source_content,
        discovered_files=len(sources),
    )
    rng = random.Random(seed)
    all_records: List[Dict[str, Any]] = []
    seen_digests: Dict[str, Path] = {}

    for source in sources:
        display = relative_display(source, input_dir)

        if deduplicate_source_content:
            try:
                digest = source_digest(source)
            except OSError as exc:
                message = f"could not hash source: {exc}"
                report.errors.append({"file": display, "error": message})
                print(f"[error] {display}: {message}", file=sys.stderr)
                continue
            if digest in seen_digests:
                first = relative_display(seen_digests[digest], input_dir)
                reason = f"exact source-content duplicate of {first}"
                report.duplicate_files_skipped += 1
                report.skipped.append({"file": display, "reason": reason})
                print(f"[skip] {display}: {reason}", file=sys.stderr)
                continue
            seen_digests[digest] = source

        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            converted = convert_trajectory(
                payload=payload,
                source=Path(display),
                modality=modality,
                table=table,
                system_prompt=system_prompt,
                rng=rng,
            )
        except (OSError, json.JSONDecodeError, TrajectoryError) as exc:
            report.errors.append({"file": display, "error": str(exc)})
            print(f"[error] {display}: {exc}", file=sys.stderr)
            continue

        all_records.extend(converted.records)
        report.processed_files += 1
        report.tool_records += converted.tool_records
        report.summary_records += converted.summary_records
        if converted.chat_messages_stripped:
            report.chat_message_lists_stripped += 1

    if not all_records:
        raise RuntimeError(
            "No training records were produced; inspect the reported source errors"
        )

    report.records = len(all_records)
    atomic_write_json(output, all_records)
    if report_path is not None:
        atomic_write_json(report_path, asdict(report))
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert curated Evaluation Agent trajectories into EA-CoT Alpaca records."
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=("t2v", "t2i"),
        help="Use VBench video scores (t2v) or T2I-CompBench image scores (t2i).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory searched recursively for curated trajectory JSON files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Alpaca JSON array. Relative paths resolve from the repository root.",
    )
    parser.add_argument(
        "--score-table",
        help=(
            "TSV score reference. Defaults to eval_agent/vbench_dimension_scores.tsv "
            "for t2v or eval_agent/t2i_comp_dimension_scores.tsv for t2i."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic sub-aspect wording (default: 0).",
    )
    parser.add_argument(
        "--deduplicate-source-content",
        action="store_true",
        help=(
            "Skip byte-for-byte duplicate source JSON files by SHA-256. Disabled by "
            "default so the builder never silently changes a released dataset count."
        ),
    )
    parser.add_argument(
        "--report",
        help="Optional path for a machine-readable build report JSON.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    input_dir = resolve_from_repo(args.input_dir)
    output = resolve_from_repo(args.output)
    score_table_path = (
        resolve_from_repo(args.score_table)
        if args.score_table
        else DEFAULT_SCORE_TABLES[args.modality].resolve()
    )
    report_path = resolve_from_repo(args.report) if args.report else None

    try:
        report = build_dataset(
            modality=args.modality,
            input_dir=input_dir,
            output=output,
            score_table_path=score_table_path,
            seed=args.seed,
            deduplicate_source_content=args.deduplicate_source_content,
            report_path=report_path,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        KeyError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {report.records} records to {output}")
    print(
        "Processed "
        f"{report.processed_files}/{report.discovered_files} source files; "
        f"tool records={report.tool_records}, summary records={report.summary_records}"
    )
    print(
        f"Stripped {report.chat_message_lists_stripped} final chat messages lists; "
        f"skipped {report.duplicate_files_skipped} exact source duplicates; "
        f"errors={len(report.errors)}"
    )
    if report_path is not None:
        print(f"Wrote build report to {report_path}")
    if report.errors:
        print(
            "Build completed with source errors. Review stderr or --report before release use.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
