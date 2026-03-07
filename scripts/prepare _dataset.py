import re

from datasets import load_dataset
from pathlib import Path

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


