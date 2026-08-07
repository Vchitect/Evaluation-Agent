"""Run VBench evaluation with the local EA-3B planning backbone."""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from base_agent import BaseAgentSFT
from open_ea_utils import (
    normalize_vbench_tool,
    parse_json_object,
    parse_plan,
    save_json,
)
from system_prompts import sys_prompts
from tools import ToolCalling

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a T2V model with Open-EA and VBench tools."
    )
    parser.add_argument(
        "--user_query", required=True, help="Capability question to evaluate."
    )
    parser.add_argument(
        "--model",
        default="latte1",
        help="Visual generator registered in tools.GenModel.",
    )
    parser.add_argument(
        "--planner-base-url",
        default=os.environ.get("OPEN_EA_PLANNER_URL", "http://127.0.0.1:12333/v1"),
        help="OpenAI-compatible base URL serving EA-3B.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("OPEN_EA_PLANNER_MODEL", "eval-agent"),
        help="Served model name for EA-3B.",
    )
    parser.add_argument(
        "--prompt-base-url",
        default=os.environ.get("OPEN_EA_PROMPT_URL", "http://127.0.0.1:12334/v1"),
        help="OpenAI-compatible base URL serving the prompt-selection model.",
    )
    parser.add_argument(
        "--prompt-model",
        default=os.environ.get("OPEN_EA_PROMPT_MODEL", "prompt-agent"),
        help="Served model name for prompt selection.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10, help="Safety limit for tool rounds."
    )
    parser.add_argument(
        "--max-agent-attempts",
        type=int,
        default=3,
        help="Maximum attempts for malformed planner or prompt-selection output.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "eval_vbench_results",
        help="Directory for generated samples and the saved trajectory.",
    )
    parser.add_argument(
        "--score-table",
        type=Path,
        default=SCRIPT_DIR / "vbench_dimension_scores.tsv",
        help="VBench score-to-tier reference table.",
    )
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    if args.max_agent_attempts < 1:
        parser.error("--max-agent-attempts must be at least 1")
    return args


class OpenEvalAgent:
    def __init__(
        self,
        sample_model,
        planner_base_url,
        planner_model,
        prompt_base_url,
        prompt_model,
        score_table,
        output_dir,
        max_rounds=10,
        max_agent_attempts=3,
        temperature=0.5,
    ):
        self.tools = ToolCalling(sample_model=sample_model, save_mode="video")
        self.sample_model = sample_model
        self.score_table = Path(score_table)
        self.output_dir = Path(output_dir)
        self.max_rounds = max_rounds
        self.max_agent_attempts = max_agent_attempts
        self.planner = BaseAgentSFT(
            system_prompt=(
                sys_prompts["eval-agent-vbench-training-sys_v1"]
                + sys_prompts["eval-agent-format-sys"]
            ),
            use_history=True,
            temp=temperature,
            base_url=planner_base_url,
            model_name=planner_model,
        )
        self.prompt_agent = BaseAgentSFT(
            system_prompt=sys_prompts["vbench-prompt-sys-open"],
            use_history=False,
            temp=temperature,
            base_url=prompt_base_url,
            model_name=prompt_model,
        )

    def _prepare_output(self, query):
        clean_query = re.sub(r"[^A-Za-z0-9._-]+", "_", query).strip("_")[:80] or "query"
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        run_name = f"{datetime.now():%Y%m%d-%H%M%S-%f}-{clean_query}-{digest}"
        save_path = self.output_dir / self.sample_model / run_name
        video_folder = save_path / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        return video_folder, save_path / "eval_results.json"

    @staticmethod
    def _format_reference_table(dataframe, tool_name):
        normalized = (
            dataframe["Dimension"].str.casefold().str.replace(" ", "_", regex=False)
        )
        row = dataframe.loc[normalized == tool_name]
        if row.empty:
            raise ValueError(f"No score reference found for tool: {tool_name}")
        row = row.iloc[0]
        reference = (
            f"{row['Dimension']}: Very High -> {row['Very High']}, "
            f"High -> {row['High']}, Moderate -> {row['Moderate']}, "
            f"Low -> {row['Low']}, Very Low -> {row['Very Low']}"
        )
        return str(row["Dimension"]), reference

    @staticmethod
    def _format_eval_results(tool_name, eval_results, reference_table):
        average_score, per_prompt_results = eval_results["score"]
        lines = [
            f"Scoring Reference Table of '{tool_name}': {reference_table}",
            "",
            "Results:",
            f"- Overall score: {float(average_score):.4f}",
            "- Per-prompt scores:",
        ]
        for result in per_prompt_results:
            lines.append(
                f'  • "{result["prompt"]}": {float(result["video_results"]):.4f}'
            )
        return "<information>" + "\n".join(lines) + "\n</information>"

    @staticmethod
    def _select_prompt_records(response, prompt_list):
        by_prompt = {item["prompt"]: item for item in prompt_list}
        selected = []
        selected_prompts = set()
        for value in response.values():
            if not isinstance(value, dict):
                continue
            generated_prompt = value.get("prompt") or value.get("Prompt")
            if not isinstance(generated_prompt, str) or not generated_prompt.strip():
                continue
            canonical = by_prompt.get(generated_prompt)
            if canonical is None:
                import difflib

                matches = difflib.get_close_matches(
                    generated_prompt, by_prompt, n=1, cutoff=0.6
                )
                if not matches:
                    continue
                canonical = by_prompt[matches[0]]
            if canonical["prompt"] in selected_prompts:
                continue
            selected.append(canonical)
            selected_prompts.add(canonical["prompt"])
        if not 3 <= len(selected) <= 9:
            raise ValueError(
                "Prompt model must select 3–9 distinct prompts; "
                f"received {len(selected)}."
            )
        return selected

    def _request_plan(self, planner_input):
        responses = []
        error = None
        history_length = len(self.planner.messages)
        try:
            for attempt in range(self.max_agent_attempts):
                # Retry instructions are only temporary context for this request. Keep
                # the accepted trajectory aligned with the released training format.
                del self.planner.messages[history_length:]
                if attempt == 0:
                    request = planner_input
                else:
                    request = (
                        planner_input
                        + "\n\nYour previous response for this input was invalid: "
                        f"{error}. Return a corrected response using exactly one "
                        "<think>...</think><tool>...</tool> action or one "
                        "<think>...</think><summary>...</summary> conclusion."
                    )

                raw_plan = self.planner(request)
                responses.append(raw_plan)
                try:
                    plan = parse_plan(raw_plan)
                    if "tool" in plan:
                        tool_name = normalize_vbench_tool(plan["tool"])
                        if tool_name is None:
                            raise ValueError(
                                f"unsupported VBench tool selected: {plan['tool']}"
                            )
                        plan["tool"] = tool_name
                    self.planner.messages[history_length:] = [
                        {"role": "user", "content": planner_input},
                        {"role": "assistant", "content": raw_plan},
                    ]
                    return plan, responses
                except ValueError as exc:
                    error = exc

            raise ValueError(
                "Planner failed to return a valid action after "
                f"{self.max_agent_attempts} attempts."
            ) from error
        except Exception:
            del self.planner.messages[history_length:]
            raise

    def _request_prompt_selection(self, prompt_request, prompt_list):
        responses = []
        error = None
        for attempt in range(self.max_agent_attempts):
            if attempt == 0:
                request = prompt_request
            else:
                request = (
                    prompt_request
                    + "\n\nThe previous response was invalid: "
                    + str(error)
                    + " Return valid JSON selecting 3–9 distinct prompts exactly from "
                    "the supplied list."
                )

            raw_response = self.prompt_agent(request)
            responses.append(raw_response)
            try:
                parsed = parse_json_object(raw_response)
                selected = self._select_prompt_records(parsed, prompt_list)
                return selected, responses
            except ValueError as exc:
                error = exc

        raise ValueError(
            "Prompt selector failed to return 3–9 valid prompts after "
            f"{self.max_agent_attempts} attempts."
        ) from error

    @staticmethod
    def _load_prompt_list(tool_name):
        with (SCRIPT_DIR / "eval_tools" / "vbench" / "VBench_full_info.json").open(
            "r", encoding="utf-8"
        ) as handle:
            source = json.load(handle)

        prompts = []
        for item in source:
            if tool_name not in item["dimension"]:
                continue
            record = {"prompt": item["prompt_en"]}
            auxiliary = item.get("auxiliary_info", {}).get(tool_name)
            if auxiliary:
                record["auxiliary_info"] = next(iter(auxiliary.values()))
            prompts.append(record)
        if not prompts:
            raise ValueError(f"No VBench prompts found for tool: {tool_name}")
        return prompts

    def _sample_and_eval(self, selected, video_folder, tool_name):
        prompts = [item["prompt"] for item in selected]
        video_pairs = self.tools.sample(prompts, str(video_folder))
        auxiliary = {item["prompt"]: item.get("auxiliary_info") for item in selected}
        for pair in video_pairs:
            if auxiliary.get(pair["prompt"]) is not None:
                pair["auxiliary_info"] = auxiliary[pair["prompt"]]
        return self.tools.eval(tool_name, video_pairs)

    def run(self, query):
        video_folder, result_file = self._prepare_output(query)
        score_dataframe = pd.read_csv(self.score_table, sep="\t")
        transcript = [{"user_query": query}]
        planner_input = query
        started_at = time.time()

        tool_rounds = 0
        while True:
            round_index = tool_rounds + 1
            plan, planner_responses = self._request_plan(planner_input)
            plan["raw_response"] = planner_responses[-1]
            if len(planner_responses) > 1:
                plan["planner_responses"] = planner_responses
            plan["round"] = round_index

            if "summary" in plan:
                plan["elapsed_seconds"] = time.time() - started_at
                transcript.append(plan)
                save_json(transcript, result_file)
                print(plan["summary"])
                print(f"Results saved to {result_file}")
                return transcript

            if tool_rounds >= self.max_rounds:
                plan["not_executed"] = "max_rounds_reached"
                transcript.append(plan)
                transcript.append(
                    {
                        "stopped": "max_rounds_reached",
                        "max_rounds": self.max_rounds,
                        "elapsed_seconds": time.time() - started_at,
                    }
                )
                save_json(transcript, result_file)
                print(
                    f"Stopped after {self.max_rounds} tool rounds without a "
                    "<summary> response."
                )
                print(f"Partial results saved to {result_file}")
                return transcript

            tool_name = plan["tool"]

            display_tool_name, reference_table = self._format_reference_table(
                score_dataframe, tool_name
            )
            prompt_list = self._load_prompt_list(tool_name)
            prompt_request = (
                "## Context\n"
                + json.dumps(
                    {"thought": plan["thought"], "tool": tool_name}, ensure_ascii=False
                )
                + "\n\n## Prompt List\n"
                + json.dumps(prompt_list, ensure_ascii=False)
            )
            selected, prompt_responses = self._request_prompt_selection(
                prompt_request, prompt_list
            )
            eval_results = self._sample_and_eval(selected, video_folder, tool_name)
            observation = self._format_eval_results(
                display_tool_name, eval_results, reference_table
            )

            plan["prompt_request"] = prompt_request
            plan["raw_prompt_response"] = prompt_responses[-1]
            if len(prompt_responses) > 1:
                plan["prompt_responses"] = prompt_responses
            plan["selected_prompts"] = selected
            plan["eval_results"] = eval_results
            plan["observation"] = observation
            transcript.append(plan)
            save_json(transcript, result_file)
            planner_input = observation
            tool_rounds += 1


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    score_table = args.score_table.resolve()

    # Existing model and metric adapters resolve their assets relative to eval_agent/.
    os.chdir(SCRIPT_DIR)
    agent = OpenEvalAgent(
        sample_model=args.model,
        planner_base_url=args.planner_base_url,
        planner_model=args.planner_model,
        prompt_base_url=args.prompt_base_url,
        prompt_model=args.prompt_model,
        score_table=score_table,
        output_dir=output_dir,
        max_rounds=args.max_rounds,
        max_agent_attempts=args.max_agent_attempts,
        temperature=args.temperature,
    )
    agent.run(args.user_query)


if __name__ == "__main__":
    main()
