import time
import openai
from config import (
    API_KEY, API_BASE_URL,
    CHAT_API_KEY, CHAT_BASE_URL,
    BASE_MODEL, CHAT_MODEL, LOGPROBS_TOP,
)

base_client = openai.OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

chat_client = openai.OpenAI(
    base_url=CHAT_BASE_URL,
    api_key=CHAT_API_KEY,
)

_DEBUG = True
_debug_calls = 0
_DEBUG_MAX = 3


def _complete(prompt: str, max_tokens: int = 1, temperature: float = 0.0,
              max_retries: int = 5) -> openai.types.Completion:
    for attempt in range(max_retries):
        try:
            return base_client.completions.create(
                model=BASE_MODEL,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=LOGPROBS_TOP,
            )
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except openai.APIStatusError as e:
            print(f"  API error {e.status_code}: {str(e)[:200]}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except openai.APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded")


def _chat(model: str, messages: list, max_tokens: int = 1,
          temperature: float = 0.0, max_retries: int = 5) -> openai.types.chat.ChatCompletion:
    for attempt in range(max_retries):
        try:
            return chat_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=LOGPROBS_TOP,
            )
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except openai.APIStatusError as e:
            print(f"  API error {e.status_code}: {str(e)[:200]}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except openai.APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded")


def _debug_log_completion(response: openai.types.Completion) -> None:
    global _debug_calls
    if not _DEBUG or _debug_calls >= _DEBUG_MAX:
        return
    _debug_calls += 1
    choice = response.choices[0]
    print(f"\n--- DEBUG call #{_debug_calls} ---")
    print(f"  text: '{(choice.text or '')[:80]}'")
    lp = choice.logprobs
    if lp and lp.top_logprobs:
        print(f"  top logprobs at pos 0: {dict(list(lp.top_logprobs[0].items())[:5])}")
    else:
        print(f"  logprobs: {lp}")
    print("--- END DEBUG ---\n")


def _debug_log_chat(response: openai.types.chat.ChatCompletion) -> None:
    global _debug_calls
    if not _DEBUG or _debug_calls >= _DEBUG_MAX:
        return
    _debug_calls += 1
    choice = response.choices[0]
    print(f"\n--- DEBUG call #{_debug_calls} ---")
    print(f"  text: {(choice.message.content or '')[:80]}")
    lp = choice.logprobs
    if lp and lp.content:
        tok_info = lp.content[0]
        print(f"  token: '{tok_info.token}'  logprob: {tok_info.logprob:.4f}")
        for t in (tok_info.top_logprobs or [])[:5]:
            print(f"    '{t.token}': {t.logprob:.4f}")
    else:
        print(f"  logprobs: {lp}")
    print("--- END DEBUG ---\n")


def get_label_probs(prompt: str, model: str = None) -> dict:
    response = _complete(prompt, max_tokens=1)
    _debug_log_completion(response)

    choice = response.choices[0]
    probs = {"True": -100.0, "False": -100.0}

    lp = choice.logprobs
    if lp and lp.top_logprobs:
        for token, logprob in lp.top_logprobs[0].items():
            tok = token.strip().lower()
            if tok == "true":
                probs["True"] = max(probs["True"], logprob)
            elif tok == "false":
                probs["False"] = max(probs["False"], logprob)

    if probs["True"] == -100.0 and probs["False"] == -100.0:
        text = (choice.text or "").strip().lower()
        if "true" in text:
            probs["True"] = -0.5
            probs["False"] = -5.0
        elif "false" in text:
            probs["True"] = -5.0
            probs["False"] = -0.5

    return probs


def get_logprob_for_label(prompt: str, label: str, model: str = None) -> float:
    return get_label_probs(prompt)[label]


def classify(prompt: str, model: str = None) -> str:
    probs = get_label_probs(prompt)
    return "True" if probs["True"] >= probs["False"] else "False"


def classify_chat(messages: list) -> str:
    response = _chat(CHAT_MODEL, messages)
    _debug_log_chat(response)

    choice = response.choices[0]
    top_lps = []
    lp = choice.logprobs
    if lp and lp.content:
        top_lps = lp.content[0].top_logprobs or []

    if top_lps:
        true_lp, false_lp = -100.0, -100.0
        for entry in top_lps:
            tok = entry.token.strip().lower()
            if tok == "true":
                true_lp = max(true_lp, entry.logprob)
            elif tok == "false":
                false_lp = max(false_lp, entry.logprob)
        if true_lp > -100.0 or false_lp > -100.0:
            return "True" if true_lp >= false_lp else "False"

    text = (choice.message.content or "").strip().lower()
    if "true" in text[:15]:
        return "True"
    if "false" in text[:15]:
        return "False"
    return "True"
