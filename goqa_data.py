import ast
import csv
import json
import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from config import COUNTRIES, TRAIN_RATIO, MAX_TRAIN, MAX_TEST, SEED


SKIP_OPTIONS = {
    "dk/refused", "don't know", "dk", "refused",
    "depends on the situation (vol)", "other (vol)",
    "none (vol)", "dk/refused (vol)",
}


def _safe_parse(raw: str):
    s = raw.strip()

    m = re.match(r"defaultdict\s*\(.+?,\s*(\{.*\})\s*\)\s*$", s, re.DOTALL)
    if m:
        s = m.group(1)

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    try:
        return eval(s, {"__builtins__": {}}, {"defaultdict": defaultdict, "list": list})
    except Exception:
        raise ValueError(f"Cannot parse: {s[:120]}…")


def load_goqa_hf(data_path: str) -> List[Dict]:
    rows = []
    n_skipped = 0
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                selections = _safe_parse(row["selections"])
                if hasattr(selections, "items"):
                    selections = dict(selections)

                options = _safe_parse(row["options"])

                rows.append({
                    "question": row["question"].strip(),
                    "selections": selections,
                    "options": options,
                    "source": row.get("source", "").strip(),
                })
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 3:      
                    print(f"  Warning: skipping unparseable row: {e}")
                continue
    if n_skipped > 3:
        print(f"  Warning: skipped {n_skipped} unparseable rows total")
    print(f"  Loaded {len(rows)} questions from {data_path}")
    return rows


def load_goqa_github_csv(csv_path: str) -> set:
    questions = set()
    if not os.path.exists(csv_path):
        print(f"  Warning: {csv_path} not found, skipping GitHub filter")
        return questions

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(csv_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                q_col = None
                for candidate in ("question", "Question", "prompt", "Prompt",
                                  "question_text", "Question_Text"):
                    if candidate in headers:
                        q_col = candidate
                        break
                if q_col is None:
                    for h in headers:
                        if "question" in h.lower():
                            q_col = h
                            break
                if q_col is None:
                    print(f"  Warning: no question column found in {csv_path}")
                    print(f"           columns present: {headers}")
                    return questions

                for row in reader:
                    q = (row.get(q_col) or "").strip()
                    if q:
                        questions.add(q)
            break
        except UnicodeDecodeError:
            continue

    print(f"  Loaded {len(questions)} question texts from GitHub CSV for filtering")
    return questions


def _clean_option(opt) -> str:
    if opt is None:
        return ""
    
    if isinstance(opt, float) and opt != opt:
        return ""

    return str(opt).strip()


def _is_skip_option(opt) -> bool:
    option = _clean_option(opt)
    return not option or option.lower() in SKIP_OPTIONS


def filter_binary_questions(rows: List[Dict], github_questions: Optional[set] = None) -> List[Dict]:
    filtered = []
    for row in rows:
        if github_questions and row["question"] not in github_questions:
            continue

        substantive = [
            _clean_option(opt)
            for opt in row["options"]
            if not _is_skip_option(opt)
        ]
        if len(substantive) == 2:
            row["binary_options"] = substantive
            filtered.append(row)

    print(f"  Filtered to {len(filtered)} binary questions "
          f"(from {len(rows)} total)")
    return filtered


def build_persona_data(
    binary_questions: List[Dict],
    countries: List[str],
) -> Dict[str, List[Dict]]:
    persona_data = defaultdict(list)
    cid = 0

    for row in binary_questions:
        question = row["question"]
        options = row["binary_options"]
        selections = row["selections"]

        for country in countries:
            if country not in selections:
                continue

            percentages = selections[country]
            option_pcts = {}
            for i, opt in enumerate(row["options"]):
                if not _is_skip_option(opt) and i < len(percentages):
                    option_pcts[_clean_option(opt)] = percentages[i]

            if len(option_pcts) < 2:
                continue

            sorted_opts = sorted(option_pcts.items(), key=lambda x: x[1], reverse=True)
            majority_opt = sorted_opts[0][0]
            minority_opt = sorted_opts[1][0]

            if sorted_opts[0][1] == sorted_opts[1][1]:
                continue

            persona_data[country].append({
                "question": question,
                "choice": majority_opt,
                "label": "True",
                "consistency_id": cid,
                "country": country,
            })
            persona_data[country].append({
                "question": question,
                "choice": minority_opt,
                "label": "False",
                "consistency_id": cid,
                "country": country,
            })
            cid += 1

    for country, examples in persona_data.items():
        true_count = sum(1 for ex in examples if ex["label"] == "True")
        print(f"  {country}: {len(examples)} examples "
              f"({true_count} True, {len(examples) - true_count} False)")

    return dict(persona_data)


def split_train_test(
    examples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    max_train: int = None,
    max_test: int = None,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:

    rng = random.Random(seed)

    groups = defaultdict(list)
    for ex in examples:
        groups[ex["consistency_id"]].append(ex)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    split_idx = int(len(group_ids) * train_ratio)

    train_ids = set(group_ids[:split_idx])
    train = [ex for cid in train_ids for ex in groups[cid]]
    test = [ex for cid in group_ids[split_idx:] for ex in groups[cid]]

    rng.shuffle(train)
    rng.shuffle(test)

    if max_train and len(train) > max_train:
        train_groups = defaultdict(list)
        for ex in train:
            train_groups[ex["consistency_id"]].append(ex)

        train_group_ids = list(train_groups.keys())
        rng.shuffle(train_group_ids)
        train = []
        for cid in train_group_ids:
            if len(train) + len(train_groups[cid]) > max_train:
                break
            train.extend(train_groups[cid])
        rng.shuffle(train)

    if max_test and len(test) > max_test:
        test_groups = defaultdict(list)
        for ex in test:
            test_groups[ex["consistency_id"]].append(ex)
        test_group_ids = list(test_groups.keys())
        rng.shuffle(test_group_ids)
        test = []
        for cid in test_group_ids:
            if len(test) + len(test_groups[cid]) > max_test:
                break
            test.extend(test_groups[cid])
        rng.shuffle(test)

    return train, test


def format_example(ex: Dict, label: str = None) -> str:
    prompt = f"Question: {ex['question']}\nClaim: {ex['choice']}\nI think this Claim is"
    if label is not None:
        prompt += f" {label}"
    return prompt


def prepare_all_personas(
    goqa_csv_path: str,
    github_csv_path: str = None,
    countries: List[str] = None,
    output_dir: str = "./datasets",
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    if countries is None:
        countries = COUNTRIES

    print("═══ Loading GOQA data ═══")
    rows = load_goqa_hf(goqa_csv_path)

    github_questions = None
    if github_csv_path and os.path.exists(github_csv_path):
        github_questions = load_goqa_github_csv(github_csv_path)

    binary = filter_binary_questions(rows, github_questions)

    print("\n═══ Building persona data ═══")
    persona_data = build_persona_data(binary, countries)

    print("\n═══ Splitting train/test ═══")
    splits = {}
    os.makedirs(output_dir, exist_ok=True)

    for country in countries:
        if country not in persona_data:
            print(f"  {country}: no data, skipping")
            continue

        examples = persona_data[country]
        if len(examples) < 10:
            print(f"  {country}: only {len(examples)} examples, skipping")
            continue

        train, test = split_train_test(examples, max_train=MAX_TRAIN, max_test=MAX_TEST)
        splits[country] = (train, test)
        print(f"  {country}: {len(train)} train, {len(test)} test"
              + (f" (capped from {len(examples)})" if len(train)+len(test) < len(examples) else ""))

        safe_name = country.replace(" ", "_").replace("(", "").replace(")", "")
        with open(os.path.join(output_dir, f"{safe_name}_train.json"), "w") as f:
            json.dump(train, f, indent=2)
        with open(os.path.join(output_dir, f"{safe_name}_test.json"), "w") as f:
            json.dump(test, f, indent=2)

    return splits


def load_prepared_data(
    data_dir: str = "./datasets",
    countries: List[str] = None,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    if countries is None:
        countries = COUNTRIES

    splits = {}
    for country in countries:
        safe_name = country.replace(" ", "_").replace("(", "").replace(")", "")
        train_path = os.path.join(data_dir, f"{safe_name}_train.json")
        test_path = os.path.join(data_dir, f"{safe_name}_test.json")
        if os.path.exists(train_path) and os.path.exists(test_path):
            with open(train_path) as f:
                train = json.load(f)
            with open(test_path) as f:
                test = json.load(f)
            splits[country] = (train, test)
            print(f"  Loaded {country}: {len(train)} train, {len(test)} test")
    return splits


if __name__ == "__main__":
    import sys
    goqa_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/global_opinions.csv"
    github_path = sys.argv[2] if len(sys.argv) > 2 else None
    prepare_all_personas(goqa_path, github_path)
