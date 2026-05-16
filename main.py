import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

from config import SEED, COUNTRIES, NUM_ICM_ITERATIONS, FEW_SHOT_COUNTS
from goqa_data import prepare_all_personas, load_prepared_data
from icm import ICM
from evaluate import run_all_conditions, eval_accuracy_vs_num_examples, eval_zero_shot_chat
from plot import (
    plot_figure1, plot_figure1_per_persona,
    plot_figure2_per_persona,
)


def run_icm_for_persona(
    train: List[Dict],
    country: str,
    num_iters: int = NUM_ICM_ITERATIONS,
    output_dir: str = "./outputs",
) -> Tuple[List[Dict], List[str], float]:
    print(f"\n{'─' * 50}")
    print(f"  Running ICM for: {country}")
    print(f"  Training examples: {len(train)}")
    print(f"{'─' * 50}")

    labeler = ICM(train, num_iters=num_iters, persona_name=country)
    icm_label_map = labeler.run()
    labeler.flip_if_inverted() 

    icm_acc = labeler.accuracy_vs_golden()
    print(f"  ICM label accuracy vs gold: {icm_acc:.4f}")

    icm_examples, icm_labels = labeler.get_labeled()
    print(f"  ICM labeled {len(icm_examples)} / {len(train)} examples")

    safe_name = country.replace(" ", "_").replace("(", "").replace(")", "")
    try:
        with open(os.path.join(output_dir, f"icm_labels_{safe_name}.jsonl"), "w") as f:
            for ex, lbl in zip(icm_examples, icm_labels):
                f.write(json.dumps({
                    "question": ex["question"],
                    "claim": ex["choice"],
                    "icm_label": lbl,
                    "golden_label": ex["label"],
                    "country": country,
                }) + "\n")
    except Exception as e:
        print(f"  Warning: failed to save ICM labels: {e}")

    return icm_examples, icm_labels, icm_acc


def _run_chat_only(args, countries):

    results_path = os.path.join(args.output_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run the base-model pass first.")
        return

    with open(results_path) as f:
        saved = json.load(f)

    splits = load_prepared_data(args.data_dir, countries)
    if not splits:
        print("ERROR: No prepared data found.")
        return

    print("═══ Running zero-shot chat (instruct model) ═══")
    for country in countries:
        if country not in splits:
            continue
        _, test = splits[country]
        prefix = f"[{country}] "
        acc = eval_zero_shot_chat(test, desc_prefix=prefix)
        saved["per_persona"][country]["test_accuracies"]["Zero-shot (Chat)"] = acc
        print(f"  {country}: Zero-shot (Chat) = {acc*100:.2f}%")

    conditions = ["Zero-shot", "Zero-shot (Chat)", "Prompt-Golden", "Prompt-ICM (Ours)"]
    for cond in conditions:
        vals = [saved["per_persona"][p]["test_accuracies"].get(cond, 0)
                for p in saved["per_persona"]]
        saved["aggregate"][cond] = sum(vals) / len(vals) if vals else 0.0

    with open(results_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\n  Merged results saved to {results_path}")

    try:
        plot_figure1(saved["aggregate"],
                     os.path.join(args.output_dir, "figure1_aggregate.png"))
        per = {c: d["test_accuracies"] for c, d in saved["per_persona"].items()}
        if len(per) > 1:
            plot_figure1_per_persona(per, os.path.join(args.output_dir,
                                                       "figure1_per_persona.png"))
    except Exception as e:
        print(f"  Figure generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ICM on GlobalOpinionQA for pluralistic alignment"
    )
    parser.add_argument("--goqa_csv", type=str,
                        default="./datasets/global_opinions.csv",
                        help="Path to HuggingFace GOQA CSV")
    parser.add_argument("--github_csv", type=str, default=None,
                        help="Path to ariba-k global_opinion_data.csv (optional filter)")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Directory for prepared train/test splits")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory for results and figures")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Load previously prepared data from data_dir")
    parser.add_argument("--skip_figure2", action="store_true",
                        help="Skip accuracy-vs-shots evaluation (saves API calls)")
    parser.add_argument("--skip_chat", action="store_true",
                        help="Skip zero-shot chat condition (run on instruct model later)")
    parser.add_argument("--chat_only", action="store_true",
                        help="Run ONLY zero-shot chat condition, merging into existing results.json")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override NUM_ICM_ITERATIONS")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Override country list")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    countries = args.countries or COUNTRIES
    num_iters = args.iters or NUM_ICM_ITERATIONS

    if args.chat_only:
        _run_chat_only(args, countries)
        return

    if args.skip_data_prep:
        print("═══ Loading prepared data ═══")
        splits = load_prepared_data(args.data_dir, countries)
    else:
        print("═══ Preparing GOQA data ═══")
        splits = prepare_all_personas(
            goqa_csv_path=args.goqa_csv,
            github_csv_path=args.github_csv,
            countries=countries,
            output_dir=args.data_dir,
        )

    if not splits:
        print("ERROR: No persona data available. Check your data paths.")
        return

    per_persona_results: Dict[str, Dict[str, float]] = {}
    per_persona_icm_acc: Dict[str, float] = {}
    per_persona_shots: Dict[str, Dict[str, Dict[int, float]]] = {}

    all_icm_data: Dict[str, Tuple[List[Dict], List[str]]] = {}

    for country in countries:
        if country not in splits:
            print(f"\n  Skipping {country}: no data")
            continue

        train, test = splits[country]
        print(f"\n{'═' * 60}")
        print(f"  PERSONA: {country}")
        print(f"  Train: {len(train)}, Test: {len(test)}")
        print(f"{'═' * 60}")

        icm_examples, icm_labels, icm_acc = run_icm_for_persona(
            train, country, num_iters=num_iters, output_dir=args.output_dir
        )
        per_persona_icm_acc[country] = icm_acc
        all_icm_data[country] = (icm_examples, icm_labels)

        if len(icm_examples) < 5:
            print(f"  WARNING: Too few ICM labels for {country}, "
                  f"results will be unreliable.")

        results = run_all_conditions(
            test, train, icm_examples, icm_labels, country=country,
            skip_chat=args.skip_chat,
        )
        per_persona_results[country] = results

        if not args.skip_figure2:
            max_available = min(len(icm_examples), len(train))
            valid_shots = [s for s in FEW_SHOT_COUNTS if s <= max_available]
            if valid_shots:
                shots_results = eval_accuracy_vs_num_examples(
                    test, train, icm_examples, icm_labels,
                    shot_counts=valid_shots, country=country,
                )
                per_persona_shots[country] = shots_results

    print("\n" + "═" * 60)
    print("  AGGREGATE RESULTS")
    print("═" * 60)

    conditions = ["Zero-shot", "Zero-shot (Chat)",
                   "Prompt-Golden", "Prompt-ICM (Ours)"]
    aggregate_results = {}
    for cond in conditions:
        values = [per_persona_results[p][cond]
                  for p in per_persona_results if cond in per_persona_results[p]]
        if values:
            aggregate_results[cond] = sum(values) / len(values)
        else:
            aggregate_results[cond] = 0.0

    print("\n  Per-persona results:")
    for country, results in per_persona_results.items():
        print(f"    {country}:")
        for cond, acc in results.items():
            print(f"      {cond}: {acc * 100:.2f}%")
        print(f"      ICM label accuracy: {per_persona_icm_acc.get(country, 0) * 100:.2f}%")

    print("\n  Aggregate results:")
    for cond, acc in aggregate_results.items():
        print(f"    {cond}: {acc * 100:.2f}%")

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "countries": list(per_persona_results.keys()),
        "per_persona": {
            country: {
                "test_accuracies": results,
                "icm_label_accuracy": per_persona_icm_acc.get(country, 0),
            }
            for country, results in per_persona_results.items()
        },
        "aggregate": aggregate_results,
    }

    if per_persona_shots:
        full_results["accuracy_vs_shots"] = {
            country: {
                method: {str(k): v for k, v in shots.items()}
                for method, shots in methods.items()
            }
            for country, methods in per_persona_shots.items()
        }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\n═══ Generating figures ═══")

    try:
        plot_figure1(
            aggregate_results,
            output_path=os.path.join(args.output_dir, "figure1_aggregate.png"),
        )
    except Exception as e:
        print(f"  Failed to generate Figure 1 (aggregate): {e}")

    if len(per_persona_results) > 1:
        try:
            plot_figure1_per_persona(
                per_persona_results,
                output_path=os.path.join(args.output_dir,
                                         "figure1_per_persona.png"),
            )
        except Exception as e:
            print(f"  Failed to generate Figure 1 (per-persona): {e}")

    if per_persona_shots:
        try:
            plot_figure2_per_persona(
                per_persona_shots,
                output_path_prefix=os.path.join(args.output_dir, "figure2"),
            )
        except Exception as e:
            print(f"  Failed to generate Figure 2: {e}")

    print("\n═══ Done! ═══")


if __name__ == "__main__":
    main()