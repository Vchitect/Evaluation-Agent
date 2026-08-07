"""Parsing and serialization helpers for the local Open-EA runtime."""

import json
import re
from pathlib import Path

VBENCH_TOOLS = (
    "subject consistency",
    "background consistency",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "appearance style",
    "temporal style",
    "overall consistency",
    "multiple objects",
    "object class",
    "dynamic degree",
    "human action",
    "color",
    "spatial relationship",
    "scene",
)


def extract_tag(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def parse_plan(text):
    """Parse EA-3B's tagged action or terminal summary."""
    if not isinstance(text, str):
        raise ValueError("Planner returned an empty or non-string response.")

    thought = extract_tag(text, "think")
    if not thought:
        raise ValueError(f"Planner response has no non-empty <think> tag: {text[:300]}")

    summary = extract_tag(text, "summary")
    if summary is not None:
        if not summary:
            raise ValueError(
                f"Planner response has an empty <summary> tag: {text[:300]}"
            )
        return {"thought": thought, "summary": summary}

    tool = extract_tag(text, "tool")
    if not tool:
        raise ValueError(
            f"Planner response contains neither <tool> nor <summary>: {text[:300]}"
        )
    return {"thought": thought, "tool": tool}


def normalize_vbench_tool(tool_name):
    """Map a generated tool label to the implementation's snake-case name."""
    normalized = tool_name.casefold().replace("_", " ").strip()
    for candidate in VBENCH_TOOLS:
        if candidate in normalized:
            return candidate.replace(" ", "_")
    return None


def parse_json_object(text):
    """Parse a JSON object, including the common fenced-output form."""
    if not isinstance(text, str):
        raise ValueError("Prompt model returned an empty or non-string response.")

    candidates = [text.strip()]
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        candidates.append(text[left : right + 1])

    errors = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object.")
            return value
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError(f"Could not parse prompt-selection JSON: {'; '.join(errors)}")


def save_json(content, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        content, ensure_ascii=False, indent=2, default=_json_default
    )
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(serialized + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value):
    """Convert common metric-library scalar/array values to JSON-native types."""
    if isinstance(value, Path):
        return str(value)

    # NumPy scalars expose ``item`` and arrays expose ``tolist``. Keeping the
    # conversion duck-typed avoids making NumPy a dependency of this helper.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (RuntimeError, TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
