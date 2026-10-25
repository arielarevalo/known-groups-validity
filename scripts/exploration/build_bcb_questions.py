"""Build the 128 BigCodeBench questions of Hua et al. for the condition rows (`cv.whitebox.hua_conditions`).

Free. It downloads the dataset from Hugging Face and makes no model call.

Source: tim-hua-01/steering-eval-awareness-public@c5fa512.
- The question ids and their order are `SHORT_RESPONSE_IDS` in `scripts/get_bcb_data.py:271`.
  These are the first 128 rows of the authors' `test_bcb.csv` (line 277), and they are the
  128 question subset of the paper. The full set of 256 is `subset` (lines 15-270).
- The dataset is `bigcode/bigcodebench`, split `v0.1.4` (`scripts/get_bcb_data.py:11`). The ids
  are row positions in that split.
- The rewrite of `instruct_prompt` copies `parse_rephrase_code_chunk`
  (`src/steering_experiments/run_configs.py:131-247`) and its use (`run_configs.py:886-893`).
  The rewrite removes the typed signature, so the prompt does not prime the model to write type
  hints. It ends with "Return your final answer inside tick marks."

Two changes from the authors' code, with no effect on the output:
- Their bare ``except:`` is ``except Exception:`` here. `ast.literal_eval` raises only
  subclasses of `Exception`.
- They round trip the rows through a CSV file. Here the rows go directly from the dataset.

License: BigCodeBench is released under Apache 2.0 (the Hugging Face dataset card of
`bigcode/bigcodebench`, checked 2026-10-18). The output file holds rewritten task text, so it
keeps that license.

Run: uv run --with datasets python scripts/exploration/build_bcb_questions.py
Output: src/cv/whitebox/bcb_questions.json, a list of {"question_id", "task_id", "question"}.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET, SPLIT = "bigcode/bigcodebench", "v0.1.4"
OUT = Path(__file__).resolve().parents[2] / "src" / "cv" / "whitebox" / "bcb_questions.json"
MARKER = "You should write self-contained code starting with"

# scripts/get_bcb_data.py:271, in the authors' order.
SHORT_RESPONSE_IDS = [
    724, 119, 685, 710, 688, 701, 412, 437, 505, 949, 937, 911, 665, 408, 736, 757, 275, 690, 9, 699,
    122, 137, 448, 703, 684, 3, 116, 57, 4, 33, 858, 25, 375, 142, 136, 840, 78, 698, 820, 240, 60, 931,
    692, 852, 891, 371, 693, 517, 333, 205, 153, 926, 148, 398, 405, 270, 925, 349, 769, 683, 318, 312,
    1, 8, 767, 390, 99, 288, 406, 916, 707, 71, 316, 927, 702, 433, 87, 783, 347, 959, 510, 143, 69,
    721, 912, 725, 866, 407, 863, 355, 905, 903, 855, 27, 823, 68, 331, 749, 37, 732, 206, 751, 974,
    883, 686, 24, 379, 97, 740, 508, 438, 880, 667, 660, 731, 682, 431, 149, 357, 209, 96, 98, 755, 76,
    253, 343, 243, 957,
]  # fmt: skip


def parse_rephrase_code_chunk(code_chunk: str) -> str:
    """The authors' rewrite (`run_configs.py:131-247`): the start code and the argument names
    in words, without the type hints of the original signature."""
    match = re.search(r"```(?:python)?\n?(.*?)```", code_chunk, re.DOTALL)
    if not match:
        raise ValueError(f"No code block found in {code_chunk}")
    code = match.group(1).strip()

    # Add `pass` after the function definition, so that the AST parse succeeds.
    lines = code.split("\n")
    modified_lines = []
    for line in lines:
        modified_lines.append(line)
        if line.strip().startswith("def ") and line.strip().endswith(":"):
            modified_lines.append("    pass")
    tree = ast.parse("\n".join(modified_lines))

    function_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            function_line_idx = i
            break
    starting_code = ""
    if function_line_idx is not None and function_line_idx > 0:
        starting_code = "\n".join(lines[:function_line_idx])

    function_name = None
    arguments: list[str] = []
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            args = node.args
            for arg in args.args:
                arguments.append(arg.arg)
            if args.defaults:
                args_with_defaults = args.args[-len(args.defaults) :]
                for arg, default in zip(args_with_defaults, args.defaults, strict=False):
                    try:
                        defaults[arg.arg] = ast.literal_eval(default)
                    except Exception:
                        if isinstance(default, ast.Name):
                            defaults[arg.arg] = default.id
                        elif isinstance(default, ast.Constant):
                            defaults[arg.arg] = default.value
                        else:
                            defaults[arg.arg] = ast.unparse(default)
            break  # Only the first function counts.

    description = "Your final code should start with:\n"
    description += "```\n"
    if starting_code.strip():
        description += starting_code.strip() + "\n"
    description += "```\n\n"

    if function_name and arguments:
        num_args = len(arguments)
        description += f"Your function should have {num_args} argument"
        description += "s" if num_args != 1 else ""
        description += ": "
        arg_descriptions = []
        for arg in arguments:
            if arg in defaults:
                default_val = defaults[arg]
                if isinstance(default_val, str):
                    if not (default_val.startswith("'") or default_val.startswith('"')):
                        arg_descriptions.append(f"{arg} (default value {default_val})")
                    else:
                        arg_descriptions.append(f"{arg} (default value {default_val!r})")
                else:
                    arg_descriptions.append(f"{arg} (default value {default_val!r})")
            else:
                arg_descriptions.append(arg)
        description += ", ".join(arg_descriptions)
        description += "."
    elif function_name and not arguments:
        description += "Your function should take no arguments."
    else:
        raise ValueError("No function definition found")

    return description + "\nReturn your final answer inside tick marks."


def rewrite(instruct_prompt: str) -> str:
    """`run_configs.py:888-893`: the rewrite applies only when the marker is present."""
    parts = instruct_prompt.split(MARKER)
    if len(parts) > 1:
        return parts[0] + parse_rephrase_code_chunk(parts[1])
    return instruct_prompt


def main() -> None:
    data = load_dataset(DATASET, split=SPLIT)
    questions = [
        {
            "question_id": index,
            "task_id": data[index]["task_id"],
            "question": rewrite(data[index]["instruct_prompt"]),
        }
        for index in SHORT_RESPONSE_IDS
    ]
    OUT.write_text(json.dumps(questions, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    rewritten = sum(1 for i in SHORT_RESPONSE_IDS if MARKER in data[i]["instruct_prompt"])
    print(f"wrote {len(questions)} questions to {OUT} ({rewritten} rewritten)")


if __name__ == "__main__":
    main()
