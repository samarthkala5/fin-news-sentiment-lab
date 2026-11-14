"""Small instruct LLM for 3-class sentiment: zero-shot prompting and LoRA fine-tuning.

The LLM reads a headline and answers with one word; parse_label() maps the answer to a label.
LoRA freezes every original weight W and trains only a small low-rank update (alpha/r) * B @ A
next to chosen linear layers. QLoRA also stores the frozen weights in 4 bits, so a 1.5B model
fits on a free Colab T4. TRL's SFTTrainer trains on prompt-completion pairs, so the loss counts
only the answer tokens. Unsloth (GPU only) makes this faster; the plain Hugging Face path runs
the same steps on CPU for smoke tests.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from fnsl import config

if TYPE_CHECKING:  # only for type hints; datasets is imported lazily at runtime
    from datasets import Dataset

# Bump PROMPT_VERSION whenever SYSTEM_PROMPT or build_messages() changes: an adapter trained
# with one prompt must not be evaluated with another (see build_messages).
PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (
    "Classify the sentiment of this financial news headline as Bearish, Bullish or Neutral. "
    "Answer with one word."
)

# Verified on the Hub (Nov 2026): not gated, Apache-2.0, an instruct (non-"thinking") model.
# "unsloth-bnb-4bit" = Unsloth's dynamic 4-bit quant: weights are already stored in 4 bits.
DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"
# 135M-parameter chat model (Apache-2.0) used for --smoke runs and tests on CPU.
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# All attention and MLP projections, as the Unsloth LoRA guide recommends.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Words parse_label() accepts. "positive"/"negative" are documented synonyms because
# instruct models often answer in those terms when they are not fine-tuned.
LABEL_WORDS = {
    "bearish": config.LABEL2ID["Bearish"],
    "bullish": config.LABEL2ID["Bullish"],
    "neutral": config.LABEL2ID["Neutral"],
    "positive": config.LABEL2ID["Bullish"],
    "negative": config.LABEL2ID["Bearish"],
}


def build_messages(text: str, label: int | None = None) -> list[dict[str, str]]:
    """The chat messages for one headline; with `label`, also the assistant's answer.

    This ONE function builds the prompt for training, zero-shot and evaluation. A fine-tuned
    model learns to answer right after this exact sequence of tokens. If inference used a
    different instruction, a different role layout or even different whitespace, the model
    would see inputs unlike anything it was trained on, and its answers (and their format)
    could change unpredictably, so measured accuracy would no longer describe the trained
    model. The tokenizer's chat template turns these messages into the model's own format.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Headline: {text}"},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": config.ID2LABEL[int(label)]})
    return messages


def parse_label(output: str) -> int:
    """Map a model's free-text answer to a label id, or config.UNPARSEABLE.

    Case-insensitive whole-word match of Bearish/Bullish/Neutral, plus the synonyms
    positive -> Bullish and negative -> Bearish. Markdown, punctuation and whitespace are
    ignored because only runs of letters are compared. If no label word appears, or words
    for MORE THAN ONE distinct label appear ("bullish or bearish"), the output is
    unparseable: an ambiguous answer counts as wrong and is never guessed. Negation is not
    understood ("not bullish" -> Bullish); with a 5-token answer this is rare.
    """
    words = re.findall(r"[a-z]+", str(output).lower())
    found = {LABEL_WORDS[word] for word in words if word in LABEL_WORDS}
    if len(found) == 1:
        return found.pop()
    return config.UNPARSEABLE


def to_sft_dataset(df: pd.DataFrame) -> Dataset:
    """Rows (text, label) -> a datasets.Dataset in TRL's conversational prompt-completion format.

    Each example is {"prompt": [system, user messages], "completion": [assistant message]}.
    SFTTrainer renders prompt + completion with the chat template and, because the dataset
    has separate prompt and completion columns, computes the loss on the completion tokens
    only (the label word and the end-of-turn token).
    """
    from datasets import Dataset  # lazy: keeps `import fnsl.llm_lora` cheap

    rows = []
    for text, label in zip(df["text"], df["label"], strict=True):
        messages = build_messages(text, int(label))
        rows.append({"prompt": messages[:-1], "completion": messages[-1:]})
    return Dataset.from_list(rows)


def sample_rows(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """A fixed random subset of n rows (sorted by id); n <= 0 or n >= len(df) keeps all rows."""
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).sort_values("id").reset_index(drop=True)


def gpu_dtype() -> Any:
    """The 16-bit float type for GPU runs: bfloat16 on Ampere or newer GPUs, else float16 (T4).

    Model weights and mixed-precision training must use the same type (Unsloth refuses a
    mismatch), so load_model() and train_lora() both ask here. torch's default bf16 check also
    counts slow emulation and would say yes on a T4, so only native support counts.
    """
    import torch

    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)  # False without CUDA
    return torch.bfloat16 if native_bf16 else torch.float16


def load_model(
    backend: str,
    model_name: str,
    max_seq_length: int = 128,
    load_in_4bit: bool = True,
    adapter_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load (model, tokenizer); with `adapter_dir`, attach a saved LoRA adapter to its base model.

    backend="unsloth": unsloth.FastLanguageModel.from_pretrained (needs an NVIDIA GPU). Given
    an adapter folder, Unsloth reads adapter_config.json, loads the base model named there
    and attaches the adapter. Unsloth cuts every input to max_seq_length tokens, so inference
    must load with a limit that fits the longest prompt. backend="hf": transformers + peft, on
    CPU (smoke runs, tests) or GPU (fallback if Unsloth cannot be installed); max_seq_length
    is not used. 4-bit loading needs a CUDA GPU.
    """
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir) if adapter_dir is not None else model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Unsloth picks float16 on a T4, bfloat16 on newer GPUs
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer
    if backend != "hf":
        raise ValueError(f"backend must be 'unsloth' or 'hf', got {backend!r}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # transformers 5 renamed torch_dtype -> dtype and defaults to "auto" (the checkpoint's
    # dtype, often bfloat16), so set it explicitly: float32 on CPU; on a GPU the 16-bit type
    # that train_lora() also uses for mixed precision.
    dtype = gpu_dtype() if device == "cuda" else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype}
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
    elif load_in_4bit:
        print("[llm] 4-bit loading needs a CUDA GPU (bitsandbytes); using float32 on CPU.")

    # The adapter folder holds the tokenizer saved after training (same chat template).
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **kwargs)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model, tokenizer


def add_lora(
    model: Any,
    backend: str,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    seed: int = config.SEED,
) -> Any:
    """Freeze the model and attach trainable LoRA adapters to `target_modules`.

    r is the rank of the update B @ A (more rank = more capacity and memory); alpha scales it
    (the update is multiplied by alpha / r). B starts at zero, so training starts exactly
    from the base model.
    """
    target_modules = list(target_modules or DEFAULT_TARGET_MODULES)
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's memory-saving checkpointing
            random_state=seed,
        )
    if backend != "hf":
        raise ValueError(f"backend must be 'unsloth' or 'hf', got {backend!r}")

    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # the random init of the A matrices
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def generate_labels(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: int = 5,
    backend: str = "hf",
) -> tuple[list[str], list[int]]:
    """Ask the model about each headline; return (raw text answers, parsed label ids).

    Greedy decoding (always the most likely next token) makes the output deterministic.
    Prompts in a batch have different lengths, so they are padded on the LEFT: the model
    continues from the last position, which must be the real end of every prompt. Only the
    newly generated tokens are decoded.
    """
    import torch

    if backend == "unsloth":
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)  # Unsloth's faster generation mode
    else:
        model.eval()
    tokenizer.padding_side = "left"

    texts = list(texts)
    raw_outputs: list[str] = []
    for start in range(0, len(texts), batch_size):
        conversations = [build_messages(text) for text in texts[start : start + batch_size]]
        # Same call TRL uses when it tokenizes training prompts, so the tokens match exactly.
        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,  # end with the assistant header: "your answer here"
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        # Unsloth silently cuts inputs longer than the max_seq_length it was loaded with (plain
        # transformers models have no such attribute). With left padding that drops the END of
        # the prompts, so stop here instead of answering truncated prompts.
        limit = getattr(model, "max_seq_length", None)
        n_tokens = inputs["input_ids"].shape[1] + max_new_tokens
        if limit is not None and n_tokens > limit:
            raise ValueError(
                f"Prompt + answer need {n_tokens} tokens but the model was loaded with "
                f"max_seq_length={limit}: load it with a larger max_seq_length."
            )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # The model's own generation_config (e.g. Qwen2.5: repetition_penalty=1.1,
                # temperature=0.7) still applies unless overridden here. A repetition penalty
                # would push the model AWAY from the label words, which appear in the prompt.
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        raw_outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return raw_outputs, [parse_label(output) for output in raw_outputs]


def summarize_outputs(
    raw_outputs: Sequence[str], label_ids: Sequence[int], n_examples: int = 10
) -> dict[str, Any]:
    """What the model actually wrote: distinct unparseable answers and the most common answers."""
    unparseable = [
        output
        for output, label in zip(raw_outputs, label_ids, strict=True)
        if label == config.UNPARSEABLE
    ]
    return {
        # dict.fromkeys keeps the first occurrence of each distinct answer, in order
        "unparseable_examples": list(dict.fromkeys(unparseable))[:n_examples],
        "most_common_outputs": Counter(raw_outputs).most_common(n_examples),
    }


def count_too_long(tokenizer: Any, df: pd.DataFrame, max_seq_length: int) -> int:
    """How many rows (prompt + answer) are longer than max_seq_length tokens.

    The trainer cuts such rows at the end, which would remove the answer we train on.
    """
    conversations = [
        build_messages(t, int(y)) for t, y in zip(df["text"], df["label"], strict=True)
    ]
    ids = tokenizer.apply_chat_template(conversations, tokenize=True, return_dict=True)["input_ids"]
    return sum(len(row) > max_seq_length for row in ids)


def sft_trainer_class(use_cuda: bool) -> type:
    """TRL's SFTTrainer on a GPU; on CPU, a subclass that uses the plain Hugging Face loss.

    TRL 1.15 computes the SFT loss with a Triton kernel (its "fused LM head") that only runs
    on a GPU. On CPU, compute_loss falls back to transformers.Trainer's: the model's own
    cross-entropy on the same labels (TRL has already set prompt tokens to -100), so it is
    the same answer-only loss, just slower. TRL's extra metrics (token accuracy) are skipped.
    """
    from transformers import Trainer
    from trl import SFTTrainer

    if use_cuda:
        return SFTTrainer

    class CpuSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return Trainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

    return CpuSFTTrainer


def train_lora(
    model: Any,
    tokenizer: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    backend: str = "hf",
    max_steps: int = 300,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    grad_accum: int = 2,
    warmup_steps: int = 30,
    eval_steps: int = 50,
    logging_steps: int = 10,
    max_seq_length: int = 128,
    seed: int = config.SEED,
) -> list[dict[str, Any]]:
    """Supervised fine-tuning of the LoRA adapters with TRL's SFTTrainer.

    Each optimizer step sees batch_size * grad_accum examples. The learning rate rises
    linearly for warmup_steps, then decays linearly to 0 at max_steps. Every eval_steps
    (and once before training) the loss on `val_df` is measured; the test split is never
    used here. Returns the trainer's log history (train "loss" and "eval_loss" entries).
    """
    import torch
    from trl import SFTConfig

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and gpu_dtype() == torch.bfloat16  # float16 on a T4 (no native bfloat16)
    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        eval_on_start=True,  # val loss of the untouched model, so the drop is visible
        save_strategy="no",  # the adapter is saved explicitly after training
        max_length=max_seq_length,  # TRL 1.x name (older docs say max_seq_length)
        completion_only_loss=True,  # loss on the answer tokens only
        # Unsloth manages its own checkpointing; prompts are short, so the HF path skips it.
        gradient_checkpointing=backend == "unsloth",
        fp16=use_cuda and not bf16,
        bf16=bf16,  # TRL would default this to True, which fails on a T4 and slows a CPU
        dataloader_pin_memory=use_cuda,  # pinned memory only helps CPU -> GPU copies
        seed=seed,
        report_to="none",
    )
    trainer = sft_trainer_class(use_cuda)(
        model=model,
        args=args,
        train_dataset=to_sft_dataset(train_df),
        eval_dataset=to_sft_dataset(val_df),
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer.state.log_history


def split_log_history(log_history: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Pick the train-loss and val-loss points out of a Trainer log history."""
    train = [{"step": int(e["step"]), "loss": float(e["loss"])} for e in log_history if "loss" in e]
    val = [
        {"step": int(e["step"]), "eval_loss": float(e["eval_loss"])}
        for e in log_history
        if "eval_loss" in e
    ]
    return train, val


def trainable_parameters(model: Any) -> tuple[int, int]:
    """(trainable, total) parameter counts of a PEFT model.

    peft's own counter also counts 4-bit weights correctly (two values per stored byte).
    """
    trainable, total = model.get_nb_trainable_parameters()
    return int(trainable), int(total)


TRAIN_INFO_FILE = "train_info.json"


def read_adapter_info(adapter_dir: str | Path) -> dict[str, Any]:
    """LoRA settings and training record stored in a saved adapter folder.

    adapter_config.json is written by peft (rank, alpha, dropout, target modules, base model);
    train_info.json is written by scripts/train_unsloth.py --mode train. Raises if either is
    missing, or if the adapter was trained with a different prompt version than this code builds.
    """
    adapter_dir = Path(adapter_dir)
    for name in ("adapter_config.json", TRAIN_INFO_FILE):
        if not (adapter_dir / name).exists():
            raise FileNotFoundError(
                f"{adapter_dir / name} not found: train an adapter first (--mode train)"
            )
    peft_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    train_info = json.loads((adapter_dir / TRAIN_INFO_FILE).read_text(encoding="utf-8"))

    trained_prompt = train_info.get("prompt_version")
    if trained_prompt != PROMPT_VERSION:
        raise ValueError(
            f"The adapter in {adapter_dir} was trained with prompt {trained_prompt}, but this code "
            f"builds prompt {PROMPT_VERSION}. Evaluate with the matching prompt or retrain."
        )
    return {
        "base_model": peft_config.get("base_model_name_or_path"),
        "lora": {
            "r": peft_config.get("r"),
            "alpha": peft_config.get("lora_alpha"),
            "dropout": peft_config.get("lora_dropout"),
            # peft may store target_modules as a set-like list in any order
            "target_modules": sorted(peft_config.get("target_modules") or []),
        },
        "train_info": train_info,
    }
