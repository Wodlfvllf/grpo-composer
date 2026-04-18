import re

from datasets import load_dataset
from pathlib import Path
from scripts.modal_app import DATA_ROOT

def _extract_gsm8k_solution(answer: str) -> str:
    match = re.search(r"####\s*(-?[0-9\.,]+)", answer)
    if not match:
        raise ValueError(f"Cannot extract GSM8K solution from answer: {answer[:120]}...")
    return match.group(1).replace(",", "")

def _prepare_gsm8k_dataset(output_dir: Path) -> tuple[str, str]:

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_gsm8k.parquet"
    val_path = output_dir / "test_gsm8k.parquet"

    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    dataset = load_dataset("openai/gsm8k", "main")
    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question = example["question"] + " " + instruction
            solution = _extract_gsm8k_solution(example["answer"])
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }

        return process_fn

    train_dataset = dataset["train"].map(make_map_fn("train"), with_indices=True)
    val_dataset = dataset["test"].map(make_map_fn("test"), with_indices=True)
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))
    return str(train_path), str(val_path)

def _extract_boxed_answer(solution: str) -> str:
    """Extract content inside the last \\boxed{...} from a MATH solution."""
    idx = solution.rfind("\\boxed{")
    if idx < 0:
        raise ValueError(f"No \\boxed{{}} found in: {solution[:120]}...")
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[idx + len("\\boxed{"):i - 1]

def _prepare_light_eval_maths_dataset(output_dir : Path) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_lighteval.parquet"
    val_path = output_dir / "test_lighteval.parquet"

    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    dataset = load_dataset("lighteval/MATH-Hard")
    instruction = "Put your final answer in \\boxed{}."

    def make_map_fn(split : str):
        def process_fn(example, idx):
            question = example['problem'] + " " + instruction
            solution = _extract_boxed_answer(example['solution'])

            return {
                "data_source" : "lighteval/MATH",
                "prompt" : [{"role" : "user", "content" : question}],
                "ability" : "math",
                "reward_model" : {
                    "style" : "rule",
                    "ground_truth" : solution
                },
                "extra_info" : {
                    "split" : split,
                    "index" : idx
                },
            }

        return process_fn
    
    train_dataset = dataset["train"].map(make_map_fn("train"), with_indices=True)
    val_dataset = dataset["test"].map(make_map_fn("test"), with_indices=True)
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))

    return str(train_path), str(val_path)


# ──────────────────────────────────────────────────────────────────
# Additional math datasets — single rule-based reward, GSM8K-shaped.
# ──────────────────────────────────────────────────────────────────

def _prepare_dapo_math_17k_dataset(output_dir: Path) -> tuple[str, str]:
    """DAPO paper's curated competition-math set (~17K problems).

    HF: BytedTsinghua-SIA/DAPO-Math-17k. Only a train split; we carve out a
    held-out val split deterministically.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_dapo_math.parquet"
    val_path = output_dir / "test_dapo_math.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    instruction = "Put your final answer in \\boxed{}."

    def make_map_fn(split: str):
        def process_fn(example, idx):
            prompt_text = example.get("prompt") or example.get("problem") or example.get("question")
            answer = (
                example.get("answer")
                or example.get("ground_truth")
                or example.get("solution")
                or ""
            )
            if isinstance(answer, str) and "\\boxed{" in answer:
                try:
                    answer = _extract_boxed_answer(answer)
                except ValueError:
                    pass
            return {
                "data_source": "BytedTsinghua-SIA/DAPO-Math-17k",
                "prompt": [{"role": "user", "content": f"{prompt_text} {instruction}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
        return process_fn

    split = ds.train_test_split(test_size=0.05, seed=42)
    split["train"].map(make_map_fn("train"), with_indices=True).to_parquet(str(train_path))
    split["test"].map(make_map_fn("test"), with_indices=True).to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_aime_dataset(output_dir: Path) -> tuple[str, str]:
    """AIME 2024 — 30 problems, eval-only really; we use it as both."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_aime.parquet"
    val_path = output_dir / "test_aime.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    instruction = "Put your final integer answer in \\boxed{}."

    def process_fn(example, idx):
        return {
            "data_source": "Maxwell-Jia/AIME_2024",
            "prompt": [{"role": "user", "content": f"{example['Problem']} {instruction}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(example["Answer"])},
            "extra_info": {"split": "train", "index": idx},
        }

    mapped = ds.map(process_fn, with_indices=True)
    # Tiny dataset — full set in train, full set in val (eval-only typical use)
    mapped.to_parquet(str(train_path))
    mapped.to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_amc_dataset(output_dir: Path) -> tuple[str, str]:
    """AMC validation set (AI-MO/aimo-validation-amc)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_amc.parquet"
    val_path = output_dir / "test_amc.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    instruction = "Put your final answer in \\boxed{}."

    def process_fn(example, idx):
        return {
            "data_source": "AI-MO/aimo-validation-amc",
            "prompt": [{"role": "user", "content": f"{example['problem']} {instruction}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
            "extra_info": {"split": "train", "index": idx},
        }

    mapped = ds.map(process_fn, with_indices=True)
    split = mapped.train_test_split(test_size=0.2, seed=42)
    split["train"].to_parquet(str(train_path))
    split["test"].to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_gpqa_dataset(output_dir: Path) -> tuple[str, str]:
    """GPQA (graduate-level science MCQ). Uses gpqa_main subset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_gpqa.parquet"
    val_path = output_dir / "test_gpqa.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    instruction = (
        "Choose the correct option (A, B, C, or D). "
        "Put your final answer letter in \\boxed{}."
    )

    def process_fn(example, idx):
        # Build the MCQ prompt
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        # Deterministic shuffle by idx so the correct answer isn't always A
        order = [(idx + i) % 4 for i in range(4)]
        seen = set()
        order = [o for o in order if not (o in seen or seen.add(o))]
        labels = ["A", "B", "C", "D"]
        rendered = "\n".join(f"{labels[i]}. {choices[order[i]]}" for i in range(4))
        correct_label = labels[order.index(0)]
        prompt = f"Question: {example['Question']}\n\nOptions:\n{rendered}\n\n{instruction}"
        return {
            "data_source": "Idavidrein/gpqa",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "science_mcq",
            "reward_model": {"style": "rule", "ground_truth": correct_label},
            "extra_info": {"split": "train", "index": idx},
        }

    mapped = ds.map(process_fn, with_indices=True)
    split = mapped.train_test_split(test_size=0.1, seed=42)
    split["train"].to_parquet(str(train_path))
    split["test"].to_parquet(str(val_path))
    return str(train_path), str(val_path)


# ──────────────────────────────────────────────────────────────────
# Code datasets — ground_truth is JSON-encoded test cases. Requires a
# code-execution reward manager (verl ships `prime` style); otherwise the
# rule-based naive manager will score them all 0. The YAML overlays in
# configs/data/{humaneval,mbpp,...}.yaml set reward_manager appropriately
# and warn if the right manager is not installed.
# ──────────────────────────────────────────────────────────────────

def _prepare_humaneval_dataset(output_dir: Path) -> tuple[str, str]:
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_humaneval.parquet"
    val_path = output_dir / "test_humaneval.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset("openai_humaneval", split="test")  # only `test` exists
    instruction = "Complete the function. Output only the function body."

    def process_fn(example, idx):
        gt = json.dumps({
            "test": example["test"],
            "entry_point": example["entry_point"],
            "canonical_solution": example["canonical_solution"],
        })
        return {
            "data_source": "openai_humaneval",
            "prompt": [{"role": "user", "content": f"{example['prompt']}\n\n{instruction}"}],
            "ability": "code",
            "reward_model": {"style": "code_test", "ground_truth": gt},
            "extra_info": {"split": "test", "index": idx, "task_id": example["task_id"]},
        }

    mapped = ds.map(process_fn, with_indices=True)
    split = mapped.train_test_split(test_size=0.2, seed=42)
    split["train"].to_parquet(str(train_path))
    split["test"].to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_mbpp_dataset(output_dir: Path) -> tuple[str, str]:
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_mbpp.parquet"
    val_path = output_dir / "test_mbpp.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    train = load_dataset("mbpp", "sanitized", split="train")
    val = load_dataset("mbpp", "sanitized", split="test")
    instruction = "Write a Python function that satisfies the description."

    def process_fn(split_name):
        def fn(example, idx):
            gt = json.dumps({
                "test_list": example["test_list"],
                "code": example["code"],
            })
            return {
                "data_source": "mbpp",
                "prompt": [{"role": "user", "content": f"{example['prompt']}\n\n{instruction}"}],
                "ability": "code",
                "reward_model": {"style": "code_test", "ground_truth": gt},
                "extra_info": {"split": split_name, "index": idx, "task_id": example["task_id"]},
            }
        return fn

    train.map(process_fn("train"), with_indices=True).to_parquet(str(train_path))
    val.map(process_fn("test"), with_indices=True).to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_code_contests_dataset(output_dir: Path) -> tuple[str, str]:
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_code_contests.parquet"
    val_path = output_dir / "test_code_contests.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    train = load_dataset("deepmind/code_contests", split="train", streaming=False)
    val = load_dataset("deepmind/code_contests", split="valid", streaming=False)
    instruction = "Write a complete Python solution that reads from stdin and writes to stdout."

    def process_fn(split_name):
        def fn(example, idx):
            gt = json.dumps({
                "public_tests": example.get("public_tests", {}),
                "private_tests": example.get("private_tests", {}),
                "generated_tests": example.get("generated_tests", {}),
            })
            return {
                "data_source": "deepmind/code_contests",
                "prompt": [{"role": "user", "content": f"{example['description']}\n\n{instruction}"}],
                "ability": "code",
                "reward_model": {"style": "code_io", "ground_truth": gt},
                "extra_info": {"split": split_name, "index": idx, "name": example.get("name", "")},
            }
        return fn

    train.map(process_fn("train"), with_indices=True).to_parquet(str(train_path))
    val.map(process_fn("valid"), with_indices=True).to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_livecodebench_dataset(output_dir: Path) -> tuple[str, str]:
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_lcb.parquet"
    val_path = output_dir / "test_lcb.parquet"
    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag="release_v1",
        split="test",
        trust_remote_code=True,
    )
    instruction = "Write a complete Python solution."

    def process_fn(example, idx):
        gt = json.dumps({
            "public_test_cases": example.get("public_test_cases", ""),
            "private_test_cases": example.get("private_test_cases", ""),
            "starter_code": example.get("starter_code", ""),
        })
        return {
            "data_source": "livecodebench/code_generation_lite",
            "prompt": [{"role": "user", "content": f"{example['question_content']}\n\n{instruction}"}],
            "ability": "code",
            "reward_model": {"style": "code_io", "ground_truth": gt},
            "extra_info": {"split": "test", "index": idx, "question_id": example.get("question_id", "")},
        }

    mapped = ds.map(process_fn, with_indices=True)
    split = mapped.train_test_split(test_size=0.2, seed=42)
    split["train"].to_parquet(str(train_path))
    split["test"].to_parquet(str(val_path))
    return str(train_path), str(val_path)


_PRESET_REGISTRY = {
    "gsm8k":           ("gsm8k",          _prepare_gsm8k_dataset),
    "math":            ("lighteval_math", _prepare_light_eval_maths_dataset),
    "math-hard":       ("lighteval_math", _prepare_light_eval_maths_dataset),
    "lighteval":       ("lighteval_math", _prepare_light_eval_maths_dataset),
    "dapo_math_17k":   ("dapo_math_17k",  _prepare_dapo_math_17k_dataset),
    "dapo":            ("dapo_math_17k",  _prepare_dapo_math_17k_dataset),
    "aime":            ("aime",           _prepare_aime_dataset),
    "amc":             ("amc",            _prepare_amc_dataset),
    "gpqa":            ("gpqa",           _prepare_gpqa_dataset),
    "humaneval":       ("humaneval",      _prepare_humaneval_dataset),
    "mbpp":            ("mbpp",           _prepare_mbpp_dataset),
    "code_contests":   ("code_contests",  _prepare_code_contests_dataset),
    "livecodebench":   ("lcb",            _prepare_livecodebench_dataset),
    "lcb":             ("lcb",            _prepare_livecodebench_dataset),
}


def _prepare_dataset(train_files: str, val_files: str, dataset_preset: str) -> tuple[str, str]:
    if train_files and val_files:
        return train_files, val_files
    if train_files or val_files:
        raise ValueError("Provide both train_files and val_files, or provide neither and use dataset_preset.")

    preset = dataset_preset.strip().lower()
    if preset not in _PRESET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset_preset='{dataset_preset}'. "
            f"Supported: {sorted(_PRESET_REGISTRY)}"
        )
    subdir, fn = _PRESET_REGISTRY[preset]
    return fn(DATA_ROOT / subdir)