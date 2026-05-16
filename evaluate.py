import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from config import MAX_FEW_SHOT, FEW_SHOT_COUNTS
from api import classify, classify_chat
from goqa_data import format_example
from icm import SYSTEM_PROMPT


def _accuracy(preds: List[str], examples: List[Dict]) -> float:
    if not examples:
        return 0.0
    return sum(1 for p, ex in zip(preds, examples)
               if p == ex["label"]) / len(examples)


def eval_zero_shot_base(test: List[Dict], desc_prefix: str = "") -> float:
    preds = []
    for ex in tqdm(test, desc=f"{desc_prefix}Zero-shot (Base)"):
        try:
            prompt = f"{SYSTEM_PROMPT}\n\n{format_example(ex)}"
            preds.append(classify(prompt))
        except Exception as e:
            print(f"\n  Eval error, defaulting to True: {e}")
            preds.append("True")
    acc = _accuracy(preds, test)
    print(f"  {desc_prefix}Zero-shot (Base): {acc:.4f}")
    return acc


def eval_zero_shot_chat(test: List[Dict], desc_prefix: str = "") -> float:
    preds = []
    for ex in tqdm(test, desc=f"{desc_prefix}Zero-shot (Chat)"):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_example(ex)},
            ]
            preds.append(classify_chat(messages))
        except Exception as e:
            print(f"\n  Eval error, defaulting to True: {e}")
            preds.append("True")
    acc = _accuracy(preds, test)
    print(f"  {desc_prefix}Zero-shot (Chat): {acc:.4f}")
    return acc


def eval_few_shot(
    test: List[Dict],
    demo_examples: List[Dict],
    demo_labels: List[str],
    max_shots: int = MAX_FEW_SHOT,
    label: str = "Few-shot",
    desc_prefix: str = "",
) -> float:
    demos_ex = demo_examples[:max_shots]
    demos_lbl = demo_labels[:max_shots]

    demos_text = "\n\n".join(
        format_example(ex, lbl)
        for ex, lbl in zip(demos_ex, demos_lbl)
    )

    preds = []
    for ex in tqdm(test, desc=f"{desc_prefix}{label}"):
        try:
            prompt = f"{SYSTEM_PROMPT}\n\n{demos_text}\n\n{format_example(ex)}"
            preds.append(classify(prompt))
        except Exception as e:
            print(f"\n  Eval error, defaulting to True: {e}")
            preds.append("True")

    acc = _accuracy(preds, test)
    print(f"  {desc_prefix}{label}: {acc:.4f}")
    return acc


def run_all_conditions(
    test: List[Dict],
    train: List[Dict],
    icm_examples: List[Dict],
    icm_labels: List[str],
    country: str = "",
    skip_chat: bool = False,
) -> Dict[str, float]:
    prefix = f"[{country}] " if country else ""
    results = {}
    print(f"\n{'═' * 60}")
    print(f"  Evaluating: {country or 'aggregate'}")
    print(f"{'═' * 60}")

    results["Zero-shot"] = eval_zero_shot_base(test, prefix)

    if skip_chat:
        print(f"  {prefix}Zero-shot (Chat): SKIPPED (run --chat_only after switching to instruct model)")
        results["Zero-shot (Chat)"] = 0.0
    else:
        results["Zero-shot (Chat)"] = eval_zero_shot_chat(test, prefix)

    golden_labels = [ex["label"] for ex in train]
    results["Prompt-Golden"] = eval_few_shot(
        test, train, golden_labels,
        label="Prompt-Golden", desc_prefix=prefix,
    )

    results["Prompt-ICM (Ours)"] = eval_few_shot(
        test, icm_examples, icm_labels,
        label="Prompt-ICM (Ours)", desc_prefix=prefix,
    )

    return results


def eval_accuracy_vs_num_examples(
    test: List[Dict],
    train: List[Dict],
    icm_examples: List[Dict],
    icm_labels: List[str],
    shot_counts: List[int] = None,
    country: str = "",
    seed: int = 42,
) -> Dict[str, Dict[int, float]]:
    if shot_counts is None:
        shot_counts = FEW_SHOT_COUNTS

    rng = random.Random(seed)
    prefix = f"[{country}] " if country else ""

    random_labels = [rng.choice(["True", "False"]) for _ in train]

    gold_labels = [ex["label"] for ex in train]

    results = {"ICM": {}, "Random": {}, "Gold": {}}

    for n_shots in shot_counts:
        print(f"\n  {prefix}Evaluating with {n_shots} shots...")
        results["ICM"][n_shots] = eval_few_shot(
            test, icm_examples, icm_labels,
            max_shots=n_shots,
            label=f"ICM ({n_shots} shots)",
            desc_prefix=prefix,
        )

        results["Random"][n_shots] = eval_few_shot(
            test, train, random_labels,
            max_shots=n_shots,
            label=f"Random ({n_shots} shots)",
            desc_prefix=prefix,
        )

        results["Gold"][n_shots] = eval_few_shot(
            test, train, gold_labels,
            max_shots=n_shots,
            label=f"Gold ({n_shots} shots)",
            desc_prefix=prefix,
        )

    return results
