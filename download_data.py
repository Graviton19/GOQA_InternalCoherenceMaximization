import argparse
import os
import subprocess
import sys
from datasets import load_dataset

def download_hf_dataset(output_dir: str):
    csv_path = os.path.join(output_dir, "global_opinions.csv")
    if os.path.exists(csv_path):
        print(f"  Already exists: {csv_path}")
        return csv_path

    print("  Downloading from HuggingFace...")
    try:
        ds = load_dataset("Anthropic/llm_global_opinions", split="train")
        ds.to_csv(csv_path)
        print(f"  Saved to: {csv_path}")
        return csv_path
    except ImportError:
        print("  'datasets' library not found, trying direct download...")
    except Exception as e:
        print(f"  datasets library failed: {e}, trying direct download...")

    url = ("https://huggingface.co/datasets/Anthropic/llm_global_opinions"
           "/resolve/main/global_opinions.csv")
    try:
        subprocess.run(
            ["curl", "-L", "-o", csv_path, url],
            check=True, capture_output=True,
        )
        print(f"  Downloaded to: {csv_path}")
        return csv_path
    except Exception:
        pass

    try:
        subprocess.run(
            ["wget", "-O", csv_path, url],
            check=True, capture_output=True,
        )
        print(f"  Downloaded to: {csv_path}")
        return csv_path
    except Exception:
        pass

    print("  ERROR: Could not download GOQA data.")
    print("  Please manually download from:")
    print(f"    {url}")
    print(f"  And save to: {csv_path}")
    return None


def download_github_csv(output_dir: str):
    csv_path = os.path.join(output_dir, "global_opinion_data.csv")
    if os.path.exists(csv_path):
        print(f"  Already exists: {csv_path}")
        return csv_path

    url = ("https://raw.githubusercontent.com/ariba-k/"
           "llm-cultural-alignment-evaluation/main/"
           "steerability/data/global_opinion_data.csv")
    print("  Downloading from GitHub...")
    try:
        subprocess.run(
            ["curl", "-L", "-o", csv_path, url],
            check=True, capture_output=True,
        )
        print(f"  Downloaded to: {csv_path}")
        return csv_path
    except Exception:
        pass

    try:
        subprocess.run(
            ["wget", "-O", csv_path, url],
            check=True, capture_output=True,
        )
        print(f"  Downloaded to: {csv_path}")
        return csv_path
    except Exception:
        pass

    print("  WARNING: Could not download GitHub CSV (optional).")
    print("  You can continue without it.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download GOQA dataset files"
    )
    parser.add_argument("--output_dir", type=str, default="./datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("═══ Downloading GOQA Data ═══")
    hf_path = download_hf_dataset(args.output_dir)
    github_path = download_github_csv(args.output_dir)

    print("\n═══ Summary ═══")
    if hf_path:
        print(f"  GOQA CSV:      {hf_path}")
    else:
        print("  GOQA CSV:      NOT DOWNLOADED (required)")
    if github_path:
        print(f"  GitHub CSV:    {github_path}")
    else:
        print("  GitHub CSV:    not downloaded (optional)")

    print("\nNext step:")
    if hf_path and github_path:
        print(f"  python main.py --goqa_csv {hf_path} --github_csv {github_path}")
    elif hf_path:
        print(f"  python main.py --goqa_csv {hf_path}")
    else:
        print("  Download the data manually first.")


if __name__ == "__main__":
    main()
