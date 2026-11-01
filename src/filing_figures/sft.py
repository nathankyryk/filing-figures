"""Training examples for fine-tuning: the exact prompt the model gets at inference, then the
answer-key JSON, ending with the chat template's end-of-turn token.

The loss covers the answer only, and not the value of a blank field. A blank in the key means
"not scored" (several share classes, a tag the statements don't show, a figure the statement
finder missed), not "absent from the filing". Teaching the model to answer null there would
teach it to give up on figures that are in the text. The keys and the JSON layout around a blank
field are still trained, so the answer keeps its shape.
"""

from __future__ import annotations

import json

from filing_figures import prompts
from filing_figures.labels import FIELD_NAMES

PARTS = ("value", "scale", "period_end", "fiscal_year")
_PROBE = "FILING_FIGURES_PROBE"


def answer_text(rec: dict) -> tuple[str, list[tuple[int, int]]]:
    """The gold answer exactly as `json.dumps` writes it, and the character spans of the blank
    fields' values (the parts the loss skips)."""
    text, hidden = "{", []
    for i, name in enumerate(FIELD_NAMES):
        gold = rec["gold"][name]
        part = {k: (gold[k] if gold else None) for k in PARTS}
        text += ("" if i == 0 else ", ") + json.dumps(name) + ": "
        body = json.dumps(part)
        if gold is None:
            hidden.append((len(text), len(text) + len(body)))
        text += body
    return text + "}", hidden


def chat_prompt(tokenizer, rec: dict) -> str:
    """The prompt as vLLM renders it for a chat request with thinking off."""
    return tokenizer.apply_chat_template(
        prompts.messages(rec["input"]), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def end_of_turn(tokenizer) -> str:
    """What the chat template writes after an assistant message ("<|im_end|>" for Qwen,
    "<turn|>" for Gemma 4), without the newline that follows it."""
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": _PROBE}]
    full = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    tail = full[full.rindex(_PROBE) + len(_PROBE) :].rstrip("\n")
    return tail or tokenizer.eos_token


def encode(tokenizer, rec: dict, eot: str) -> dict:
    """input_ids, attention_mask and labels (-100 = no loss) for one filing.

    Prompt and answer are tokenized separately, as at inference: vLLM tokenizes the prompt and
    the model generates from there. The template already holds any BOS token, so none is added.
    """
    prompt = chat_prompt(tokenizer, rec)
    answer, hidden = answer_text(rec)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    enc = tokenizer(answer + eot, add_special_tokens=False, return_offsets_mapping=True)
    labels = [-100] * len(p_ids)
    for tid, (start, end) in zip(enc["input_ids"], enc["offset_mapping"], strict=True):
        skip = any(start < h_end and end > h_start for h_start, h_end in hidden)
        labels.append(-100 if skip else tid)
    ids = p_ids + list(enc["input_ids"])
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def calibration_text(tokenizer, rec: dict, eot: str) -> str:
    """Prompt and gold answer as one string: what the model sees while it writes the answer."""
    return chat_prompt(tokenizer, rec) + answer_text(rec)[0] + eot
