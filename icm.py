import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import (
    ALPHA, T0, T_MIN, BETA, K_INIT, NUM_ICM_ITERATIONS, CONTEXT_SIZE,
)
from api import get_label_probs
from goqa_data import format_example


SYSTEM_PROMPT = (
    "You are evaluating claims as True or False. "
    "Respond with exactly one word: True or False."
)

OPPOSITE = {"True": "False", "False": "True"}


def _build_prompt(target_ex: Dict, context_examples: List[Dict],
                  context_labels: List[str], max_context: int) -> str:
    if len(context_examples) > max_context:
        idx = random.sample(range(len(context_examples)), max_context)
        context_examples = [context_examples[i] for i in idx]
        context_labels = [context_labels[i] for i in idx]

    demos = "\n\n".join(
        format_example(ex, lbl)
        for ex, lbl in zip(context_examples, context_labels)
    )
    target = format_example(target_ex)

    if demos:
        return f"{SYSTEM_PROMPT}\n\n{demos}\n\n{target}"
    else:
        return f"{SYSTEM_PROMPT}\n\n{target}"


class ICM:
    def __init__(self, examples: List[Dict],
                 alpha: float = ALPHA, t0: float = T0, t_min: float = T_MIN,
                 beta: float = BETA, k_init: int = K_INIT,
                 max_context: int = CONTEXT_SIZE,
                 num_iters: int = NUM_ICM_ITERATIONS,
                 persona_name: str = ""):
        self.examples = examples
        self.N = len(examples)
        self.alpha = alpha
        self.t0 = t0
        self.t_min = t_min
        self.beta = beta
        self.k_init = k_init
        self.max_context = max_context
        self.num_iters = num_iters
        self.persona_name = persona_name

        self.labels: Dict[int, str] = {}
        self.scores: Dict[int, float] = {}

        self.partner: Dict[int, int] = {}
        cid_groups: Dict[int, List[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            cid = ex.get("consistency_id")
            if cid is not None:
                cid_groups[cid].append(i)
        for cid, indices in cid_groups.items():
            if len(indices) == 2:
                self.partner[indices[0]] = indices[1]
                self.partner[indices[1]] = indices[0]

        n_paired = sum(1 for i in range(self.N) if i in self.partner)
        tag = f"[{self.persona_name}]" if self.persona_name else ""
        print(f"  {tag} Equivalence pairs: {n_paired // 2} "
              f"({n_paired} examples paired, "
              f"{self.N - n_paired} unpaired)")

    def _context(self, exclude: int) -> Tuple[List[Dict], List[str]]:
        exclude_set = {exclude}
        if exclude in self.partner:
            exclude_set.add(self.partner[exclude])

        idxs = [i for i in sorted(self.labels) if i not in exclude_set]
        return ([self.examples[i] for i in idxs],
                [self.labels[i] for i in idxs])

    def _score_example(self, idx: int) -> float:
        ctx_ex, ctx_lbl = self._context(exclude=idx)
        if not ctx_ex:
            return 0.0
        prompt = _build_prompt(self.examples[idx], ctx_ex, ctx_lbl,
                               self.max_context)
        probs = get_label_probs(prompt)
        return probs[self.labels[idx]]

    def _best_label(self, idx: int) -> Tuple[str, dict]:
        ctx_ex, ctx_lbl = self._context(exclude=idx)
        prompt = _build_prompt(self.examples[idx], ctx_ex, ctx_lbl,
                               self.max_context)
        probs = get_label_probs(prompt)
        best = "True" if probs["True"] >= probs["False"] else "False"
        return best, probs

    def _set_label(self, idx: int, label: str, score: float):
        self.labels[idx] = label
        self.scores[idx] = score

        p = self.partner.get(idx)
        if p is not None:
            self.labels[p] = OPPOSITE[label]
            self.scores[p] = score - 0.5

    def _initialize(self):
        tag = f"[{self.persona_name}]" if self.persona_name else ""
        print(f"  {tag} Initializing with K={self.k_init} random labels...")

        init_idx = []
        candidates = list(range(self.N))
        random.shuffle(candidates)
        labeled_set = set()
        for i in candidates:
            if len(init_idx) >= self.k_init:
                break
            if i not in labeled_set:
                init_idx.append(i)
                labeled_set.add(i)
                p = self.partner.get(i)
                if p is not None:
                    labeled_set.add(p)

        for i in init_idx:
            lbl = random.choice(["True", "False"])
            self._set_label(i, lbl, -5.0)

        for i in list(self.labels.keys()):
            try:
                self.scores[i] = self._score_example(i)
            except Exception as e:
                print(f"    API error for idx {i}, using score -100: {e}")
                self.scores[i] = -100.0

    def run(self) -> Dict[int, str]:
        tag = f"[{self.persona_name}]" if self.persona_name else ""
        print(f"  {tag} Starting ICM: N={self.N}, iters={self.num_iters}, "
              f"K={self.k_init}, alpha={self.alpha}")

        try:
            self._initialize()
        except Exception as e:
            print(f"  {tag} FATAL init error: {e}")
            for i in self.labels:
                self.scores[i] = -100.0

        accepted = rejected = skipped = 0

        for n in tqdm(range(1, self.num_iters + 1),
                      desc=f"ICM {self.persona_name}"):
            T = max(self.t_min, self.t0 / (1 + self.beta * math.log(n)))

            # Prefer unlabeled examples 80% of the time
            unlabeled = [i for i in range(self.N) if i not in self.labels]
            if unlabeled and random.random() < 0.8:
                idx = random.choice(unlabeled)
            else:
                idx = random.choice(range(self.N))

            try:
                new_label, probs = self._best_label(idx)
                new_score = probs[new_label]
            except Exception as e:
                print(f"\n  {tag} [iter {n}] API error, skipping: {e}")
                skipped += 1
                continue

            is_new = idx not in self.labels
            if is_new:
                self._set_label(idx, new_label, new_score)
                accepted += 1
            else:
                old_score = self.scores.get(idx, -100.0)
                delta = self.alpha * (new_score - old_score)

                if delta > 0 or random.random() < math.exp(delta / T):
                    self._set_label(idx, new_label, new_score)
                    accepted += 1
                else:
                    rejected += 1

            if n % 50 == 0:
                total_score = self.alpha * sum(self.scores.values())
                true_count = sum(1 for v in self.labels.values()
                                 if v == "True")
                print(f"    {tag} [iter {n}] labeled={len(self.labels)}/{self.N} "
                      f"score={total_score:.2f} T={T:.3f} "
                      f"acc/rej/skip={accepted}/{rejected}/{skipped} "
                      f"T_frac={true_count / max(1, len(self.labels)):.2f}")

        print(f"  {tag} Done: labeled {len(self.labels)}/{self.N}, "
              f"accepted={accepted} rejected={rejected} skipped={skipped}")
        return dict(self.labels)

    def accuracy_vs_golden(self) -> float:
        if not self.labels:
            return 0.0
        correct = sum(1 for i, lbl in self.labels.items()
                      if lbl == self.examples[i]["label"])
        return correct / len(self.labels)

    def flip_if_inverted(self) -> bool:
        acc = self.accuracy_vs_golden()
        if acc < 0.5:
            print(f"  [{self.persona_name}] ICM accuracy {acc:.4f} < 0.5 "
                  f"— flipping all labels (inverted solution detected)")
            self.labels = {i: OPPOSITE[lbl] for i, lbl in self.labels.items()}
            new_acc = self.accuracy_vs_golden()
            print(f"  [{self.persona_name}] Accuracy after flip: {new_acc:.4f}")
            return True
        return False

    def get_labeled(self) -> Tuple[List[Dict], List[str]]:
        idxs = sorted(self.labels.keys())
        return [self.examples[i] for i in idxs], [self.labels[i] for i in idxs]