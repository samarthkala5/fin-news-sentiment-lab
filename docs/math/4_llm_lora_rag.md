# Mathematics, part 4: Small LLM with LoRA, and RAG (section 8.7–8.8)

Part of the [project guide](../PROJECT_GUIDE.md#8-the-mathematics-part-by-part) for fin-news-sentiment-lab, Days 8-12 of `PROJECT_PLAN.md`. Checked against the code at commit `98a9d42`; the one later code change (`aaf142e`) makes the Unsloth resume claim require an adapter trained with `--backend unsloth`.

> **Numbers rule.** A *result* is copied from the results file named next to it. Every other number is a setting read from the code, a check you can rerun (named as such), or arithmetic shown in full. Models that have only had a `--smoke` run have no results, and none are stated.

Each concept follows the same pattern: the intuition, the mathematics (every symbol defined), how this code implements it, and why it matters or where it can go wrong. Formulas use GitHub's math rendering.

## Contents

- [8.7 Small LLM: zero-shot prompting and LoRA fine-tuning (Unsloth)](#87-small-llm-zero-shot-prompting-and-lora-fine-tuning-unsloth)
  - [8.7.1 Causal language modelling and the next-token cross-entropy](#871-causal-language-modelling-and-the-next-token-cross-entropy)
  - [8.7.2 Chat template (ChatML) and the exact prompt](#872-chat-template-chatml-and-the-exact-prompt)
  - [8.7.3 Zero-shot evaluation and greedy decoding](#873-zero-shot-evaluation-and-greedy-decoding)
  - [8.7.4 The label parser and UNPARSEABLE](#874-the-label-parser-and-unparseable)
  - [8.7.5 4-bit quantisation (NF4, bitsandbytes) and the QLoRA idea](#875-4-bit-quantisation-nf4-bitsandbytes-and-the-qlora-idea)
  - [8.7.6 LoRA: what is trained and what is frozen](#876-lora-what-is-trained-and-what-is-frozen)
  - [8.7.7 Supervised fine-tuning with a completion-only loss (TRL)](#877-supervised-fine-tuning-with-a-completion-only-loss-trl)
  - [8.7.8 Optimiser, learning-rate schedule, gradient accumulation and validation](#878-optimiser-learning-rate-schedule-gradient-accumulation-and-validation)
  - [8.7.9 CPU smoke path: replacing TRL's Triton loss](#879-cpu-smoke-path-replacing-trls-triton-loss)
  - [8.7.10 Unsloth input truncation and inference_max_seq_length](#8710-unsloth-input-truncation-and-inference_max_seq_length)
  - [8.7.11 Outputs](#8711-outputs)
  - [8.7.12 Interview answers (plan "Explain" questions, Days 8-10)](#8712-interview-answers-plan-explain-questions-days-8-10)
- [8.8 Retrieval-augmented generation (RAG) and hit@k](#88-retrieval-augmented-generation-rag-and-hitk)
  - [8.8.1 The corpus and its licences](#881-the-corpus-and-its-licences)
  - [8.8.2 Loading: header lines become metadata](#882-loading-header-lines-become-metadata)
  - [8.8.3 Chunking with RecursiveCharacterTextSplitter](#883-chunking-with-recursivecharactertextsplitter)
  - [8.8.4 Embeddings: MiniLM sentence vectors of unit length](#884-embeddings-minilm-sentence-vectors-of-unit-length)
  - [8.8.5 Vector search: FAISS IndexFlatL2 and the cosine identity](#885-vector-search-faiss-indexflatl2-and-the-cosine-identity)
  - [8.8.6 Top-k retrieval and the answer step](#886-top-k-retrieval-and-the-answer-step)
  - [8.8.7 Saving and loading the index (the pickle risk)](#887-saving-and-loading-the-index-the-pickle-risk)
  - [8.8.8 The evaluation questions and their validation gates](#888-the-evaluation-questions-and-their-validation-gates)
  - [8.8.9 hit@k](#889-hitk)
  - [8.8.10 Comparing two chunk sizes, and why the winner's score is optimistic](#8810-comparing-two-chunk-sizes-and-why-the-winners-score-is-optimistic)
  - [8.8.11 What hit@k does not measure, and why retrieval can fail](#8811-what-hitk-does-not-measure-and-why-retrieval-can-fail)
  - [8.8.12 Outputs](#8812-outputs)
  - [8.8.13 Interview answers (plan "Explain" questions, Days 11-12)](#8813-interview-answers-plan-explain-questions-days-11-12)
- [Files this part reads and writes](#files-this-part-reads-and-writes)

---

## 8.7 Small LLM: zero-shot prompting and LoRA fine-tuning (Unsloth)

> **Status at HEAD 98a9d42.** The code is in `src/fnsl/llm_lora.py` and `scripts/train_unsloth.py`. Tests: `tests/test_llm_parser.py` (CPU, runs in CI, passes at HEAD) and `tests/test_llm_smoke.py` (marked heavy: deselected in CI, passes locally with `pytest -m heavy`). Only the CPU smoke path has run: `python scripts/train_unsloth.py --smoke --mode all`. It uses the plain Hugging Face backend, `HuggingFaceTB/SmolLM2-135M-Instruct` in float32, 5 optimiser steps (`scripts/train_unsloth.py` → `SMOKE_OVERRIDES`) and the 64 shared smoke test rows (`src/fnsl/config.py` → `SMOKE_SIZES`), and writes to the gitignored `results/smoke/` and `checkpoints/smoke/`. The Unsloth backend and 4-bit loading have **never run**: there is no GPU here, and Unsloth and bitsandbytes are not installed locally. `results/metrics/` has no `llm_zeroshot.json`, `llm_lora_train.json` or `llm_lora.json`, so this section states no LLM result.
>
> **What you need to do.** On a Colab or Kaggle GPU, run section 7 of `notebooks/colab_runner.ipynb`: `--mode zero-shot` (Day 8), `--mode train` (Day 9) and `--mode eval` (Day 10). Run train and eval in the same session, because the adapter is saved in `checkpoints/`, which Colab deletes when the session ends (or set `INCLUDE_LORA_ADAPTER = True` in step 9 to download it). Optionally run `--mode zero-shot --n-test 0` so that McNemar's test can pair zero-shot with the other models; it overwrites the 50-row zero-shot files. Day 9 is done when the adapter is saved and the validation loss falls: compare `val_loss_start` and `val_loss_end` in `llm_lora_train.json`. Download `results.zip`, unpack it in the repo, run `python scripts/compare_models.py` and `python scripts/make_results_table.py --update-readme --count-tests`, then `pytest -m "not heavy"` and `ruff check .`, and commit. `RESUME_NUMBERS.md` allows the Unsloth claim only when `params.train_backend` in `results/metrics/llm_lora.json` is `unsloth` (see the pitfall in 8.7.6). If the Unsloth install or model load fails twice, the plan (section 10) says: stop at Day 8, skip Days 9-10 and do not claim Unsloth. Fill the LoRA entry in `reading_log.md` (Day 10).

How the three stages fit together (all in `scripts/train_unsloth.py`):

```text
zero-shot  load base model (4-bit) -> build_messages -> chat template -> greedy generate (<= 5 new tokens)
           -> parse_label -> metrics.save_results                      (run_zero_shot)
train      load base model (4-bit) -> add_lora -> TRL SFTTrainer on the train split
           (answer-only loss, val loss at step 0 and every 50 steps) -> save adapter + llm_lora_train.json   (run_train)
eval       read_adapter_info (prompt-version check) -> load base model + adapter
           -> same prompt, decoding and parser on all test rows -> metrics.save_results       (run_eval)
```

Library versions checked for this section (installed in `.venv`): transformers 5.17.0, TRL 1.15.0, PEFT 0.21.3, torch 2.14.1 (CPU build). `requirements-gpu.txt` keeps transformers at or below 5.17.0 and TRL at or below 1.15.0 because Unsloth 2026.11 accepts only those versions. Colab installs whatever versions fall inside those ranges (step 3 of the notebook prints them), so "checked" below means checked in the local versions.

### 8.7.1 Causal language modelling and the next-token cross-entropy

**Intuition.** A causal language model reads tokens from left to right. At every position it outputs a probability for each possible next token. Training raises the probability of the token that really comes next.

**Maths.** A text is a sequence of token ids $`x_1, \dots, x_T`$ from a vocabulary of $`V`$ tokens. After reading $`x_1, \dots, x_t`$, the model with parameters $`\theta`$ outputs a logit vector $`z_t \in \mathbb{R}^V`$ (one score per vocabulary token). The softmax turns the logits into probabilities:

```math
p_\theta(x_{t+1} = v \mid x_{\le t}) = \frac{\exp(z_{t,v})}{\sum_{u=1}^{V} \exp(z_{t,u})}
```

"Causal" means that position $`t`$ may only look at positions $`\le t`$ (masked self-attention), so one forward pass scores every position at once. The loss is the average negative log-probability of the true next tokens. A loss mask $`\omega_t \in \{0, 1\}`$ says which positions count:

```math
\mathcal{L}(\theta) = -\frac{1}{\sum_{t=2}^{T} \omega_t} \sum_{t=2}^{T} \omega_t \, \log p_\theta(x_t \mid x_{<t})
```

The sums start at $`t = 2`$ because the first token has nothing before it. In pre-training $`\omega_t = 1`$ everywhere. In this project's fine-tuning $`\omega_t = 1`$ only on the answer tokens (8.7.7). Over a batch, both sums also run over every sequence in the batch. The logarithm is natural. Toy example: if the model assigns probability 0.6 to the correct answer token, that token adds $`-\ln 0.6 \approx 0.51`$ to the sum; at probability 0.95 it adds about 0.05.

**In this code.** The loss is not written by hand. With the `hf` backend on a GPU, TRL's `SFTTrainer` computes it: per-token loss $`-\log p`$, summed, divided by the number of counted tokens. TRL's kernel sends only the counted positions through the output layer, so it never builds the full table of logits (checked in TRL 1.15, `SFTTrainer.compute_loss`). With `--backend unsloth`, Unsloth patches transformers, TRL and PEFT when it is imported (comment in `scripts/train_unsloth.py` → `main`), so which kernel computes the loss there could not be checked; the objective is the same. On the CPU smoke path, `transformers`' `ForCausalLMLoss` computes it: it shifts the labels by one position and calls `cross_entropy(..., ignore_index=-100)` (see 8.7.9). A label of −100 means $`\omega_t = 0`$. Checked with the cached tokenizer of the default model: the three answers start with three different tokens. `Neutral` is one token, `Bearish` is `Bear` + `ish`, and `Bullish` is `B` + `ull` + `ish`. So the loss on the first answer token acts like a three-way classification loss, except that the softmax runs over the whole vocabulary, not over three words.

**Why it matters here.** Fine-tuning uses the same next-token loss as pre-training; only the mask changes. Because the softmax covers every token, the model can put probability on words that are not labels ("The", "Sentiment", "**"). That is why a parser is needed (8.7.4).

### 8.7.2 Chat template (ChatML) and the exact prompt

**Intuition.** Instruct models were tuned on conversations written in a fixed text format with special role markers. A prompt in that format gets sensible answers. A fine-tuned model learns to answer right after one exact sequence of tokens, so training and inference must produce exactly that sequence.

**Maths.** No new maths. Write $`\tau`$ for the chat template: a fixed function from a list of messages to a token sequence. Fine-tuning fits $`p_\theta(\text{answer} \mid \tau(\text{messages}))`$. If inference used a different function $`\tau'`$, the model would be scored on inputs it never saw in training, and the measured accuracy would not describe the trained model.

**In this code.** `src/fnsl/llm_lora.py` → `build_messages(text, label=None)` is the only place a prompt is built. It returns a system message (`SYSTEM_PROMPT`), a user message `"Headline: <text>"` and, when `label` is given, an assistant message that holds the label word (`Bearish`, `Bullish` or `Neutral`). Qwen2.5 uses the ChatML template. Rendered with the default model's own tokenizer (`unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit`, cached locally) and `add_generation_prompt=True`, the inference prompt for the headline used in `tests/test_llm_parser.py` is exactly:

```text
<|im_start|>system
Classify the sentiment of this financial news headline as Bearish, Bullish or Neutral. Answer with one word.<|im_end|>
<|im_start|>user
Headline: $AAPL beats estimates, raises guidance<|im_end|>
<|im_start|>assistant
```

The training sequence is the same text followed by `Bullish<|im_end|>` and a newline. The last line of the prompt, `<|im_start|>assistant` plus a newline, is the "generation prompt": it tells the model that its turn starts here.

- Training: `to_sft_dataset` stores each row as `{"prompt": [system, user], "completion": [assistant]}`. TRL renders the prompt with `add_generation_prompt=True`, renders prompt + completion, and starts the completion where the two token lists stop agreeing (checked in TRL 1.15). For Qwen2.5 the prompt tokens are an exact prefix of the full sequence (checked with the tokenizer), so nothing is lost at the boundary.
- Inference: `generate_labels` calls `tokenizer.apply_chat_template(conversations, add_generation_prompt=True, padding=True, ...)` on `build_messages(text)`: the same template call.
- `PROMPT_VERSION = "v1"` is written into `llm_lora_train.json` and into `train_info.json` next to the adapter. `read_adapter_info` refuses to evaluate an adapter whose stored version differs. The code comment says to bump the version whenever `SYSTEM_PROMPT` or `build_messages` changes.
- `run_train` saves the tokenizer (with its chat template) next to the adapter, and the `hf` backend loads the tokenizer from that folder at evaluation.
- Tests: `tests/test_llm_parser.py` → `test_inference_prompt_is_the_training_prompt_without_the_answer` (CPU) checks that the training messages without the answer equal the inference messages. `tests/test_llm_smoke.py` → `test_trl_trains_on_exactly_the_prompt_tokens_used_at_inference` (heavy, run with the smoke model) checks, token by token, that the tokens TRL puts before the answer equal the output of the same `apply_chat_template` call that `generate_labels` makes (without the padding).

**Pitfall.** Small changes break the match without any error. Drop the system message and Qwen2.5's template inserts its own default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."). Forget `add_generation_prompt=True` and the prompt ends after the user turn, so the model is never told that it is its turn. Neither raises an error; the accuracy just changes.

### 8.7.3 Zero-shot evaluation and greedy decoding

**Intuition.** Zero-shot means asking the untouched instruct model, with no training. It is the baseline that LoRA has to beat. It uses the same prompt, decoding and parser as the LoRA evaluation, so apart from the fine-tuning the only difference is how many test rows are scored (50 by default for zero-shot, all for the LoRA evaluation). Greedy decoding always takes the single most likely next token, so the same prompt always gives the same answer.

**Maths.** Greedy decoding picks, at each step,

```math
\hat{x}_{t+1} = \arg\max_{v \in \{1, \dots, V\}} p_\theta(v \mid x_{\le t}) = \arg\max_{v} z_{t,v}
```

The softmax keeps the order of the logits, so the arg max of the probabilities equals the arg max of the logits. Decoding repeats until the end-of-turn token or `max_new_tokens`. Sampling would instead draw token $`v`$ with probability proportional to $`\exp(z_{t,v} / T_{\text{temp}})`$, where $`T_{\text{temp}}`$ is the temperature, often after keeping only the few most likely tokens (top-k) or the smallest set of tokens whose probabilities add up to a threshold (top-p). A repetition penalty $`\rho`$ changes the logit of every token $`v`$ that already appears in the context (prompt or generated text) before the arg max:

```math
z'_{t,v} = \begin{cases} z_{t,v} \cdot \rho & \text{if } z_{t,v} < 0 \\ z_{t,v} / \rho & \text{otherwise} \end{cases}
```

With $`\rho > 1`$ repeated tokens become less likely; $`\rho = 1`$ changes nothing.

**In this code.** `scripts/train_unsloth.py` → `run_zero_shot` picks the test rows with `llm_lora.sample_rows(test_df, n, seed=42)`: a fixed random subset, sorted by id. By default $`n`$ is `ZEROSHOT_N_TEST = 50` (Day 8 of the plan); `--n-test 0` means all 1,767 test rows (`data/splits/split_report.json`). `--mode eval` uses all test rows by default. If you pass `--n-test`, it applies to both stages. Then `src/fnsl/llm_lora.py` → `generate_labels`:

- pads each batch on the left (`tokenizer.padding_side = "left"`), so the last position of every row is the real end of its prompt, where generation continues;
- calls `model.generate` with `do_sample=False`, `max_new_tokens=5` (`MAX_NEW_TOKENS`: the longest label is 3 tokens, plus the end-of-turn token), `repetition_penalty=1.0`, `temperature=None`, `top_p=None`, `top_k=None` and `pad_token_id=tokenizer.pad_token_id`;
- decodes only the newly generated tokens, with special tokens removed;
- works in batches of `--batch-size` (default 8; the function's own default of 16 is not used by the script).

The metrics JSON records `"decoding": "greedy"` and, for zero-shot, `train_time_s = 0.0`.

Why the explicit settings: `transformers` fills every generation setting you do not pass from the model's `generation_config.json`, and explicit keyword arguments win (checked in transformers 5.17, `GenerationMixin._prepare_generation_config`). Qwen2.5-1.5B-Instruct's `generation_config.json` sets `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20` and `repetition_penalty=1.1` (checked in the cached `Qwen/Qwen2.5-1.5B-Instruct` files; the Unsloth repository's copy is not cached here), and the penalty applies even to greedy decoding (`DEBUG_LOG.md`, Day 8). A tokenizer check shows the harm. In the system prompt the label words are tokenised as ` Bear` + `ish`, ` Bull` + `ish` and ` Neutral`, with a leading space. So the token `ish`, which finishes both `Bearish` and `Bullish`, is already in the context, while the answer token `Neutral` (no leading space) is a different token from ` Neutral`. A penalty of 1.1 would push down the second half of two labels but not the third: a bias between classes. The sampling settings do nothing when `do_sample=False`; setting them to `None` stops the model's defaults from being carried along.

**Pitfall.** Fifty rows give a very noisy score. A random 50-row sample is expected to hold only about 7-8 Bearish and about 10 Bullish headlines (15.0 % and 20.2 % of the test split, `data/splits/split_report.json`; this is an expectation, and the actual seed-42 subset can differ), so per-class F1 swings with a few answers. Count the `label` column of `results/predictions/llm_zeroshot.csv` after the run to see the real class counts. The 50 rows also differ from the other models' test rows. `scripts/compare_models.py` → `align_pair` pairs only models scored on exactly the same ids, and `src/fnsl/reporting.py` → `comparable_note` marks such a row "not comparable". Run `--mode zero-shot --n-test 0` if you want McNemar's test on zero-shot.

### 8.7.4 The label parser and UNPARSEABLE

**Intuition.** The model writes free text. The parser turns it into a class id only when the text names exactly one class. Everything else is marked unparseable and counted as wrong.

**Maths.** Let $`W(o)`$ be the set of lower-case letter runs in the output $`o`$. Let $`f`$ map the accepted words to class ids: bearish and negative to 0 (Bearish), bullish and positive to 1 (Bullish), neutral to 2 (Neutral). Let $`S(o) = \{f(w) : w \in W(o),\ w \text{ accepted}\}`$. Then

```math
\text{parse}(o) = \begin{cases} \text{the single element of } S(o) & \text{if } |S(o)| = 1 \\ -1 \ (\text{UNPARSEABLE}) & \text{otherwise} \end{cases}
```

A prediction of −1 never equals a true label $`y \in \{0, 1, 2\}`$, so accuracy counts it wrong. Macro-F1 is computed over the classes 0, 1 and 2 only, so −1 adds a false negative for the true class (its recall drops) but no false positive to any class.

Toy example (checked with `fnsl.metrics.compute_metrics`): true labels [0, 1, 1, 2] and predictions [0, 1, −1, 2]. Accuracy is 0.75. Bullish has recall 0.5 and precision 1.0, so its F1 is 0.667, and macro-F1 is 0.889. The confusion matrix covers classes 0-2 only, so the unparseable answer does not appear in it; when there are unparseable answers, the saved figure's title says how many are not shown.

**In this code.** `src/fnsl/llm_lora.py` → `parse_label` (regex `[a-z]+` on the lower-cased output, the `LABEL_WORDS` dictionary, and `config.UNPARSEABLE = -1`). Markdown, punctuation and case are ignored (`**Neutral**` gives Neutral). Only whole words count (`Bullishness` and `Bull` are unparseable). Two different labels (`bullish or bearish`) are unparseable. Negation is not understood (`not bullish` gives Bullish). `scripts/train_unsloth.py` → `report_outputs` prints the number of unparseable answers, and `llm_lora.summarize_outputs` stores up to 10 distinct unparseable answers (`unparseable_examples`) and the 10 most common raw answers (`most_common_outputs`) in the metrics JSON; `n_unparseable` comes from `metrics.compute_metrics`, and `src/fnsl/reporting.py` → `results_section` adds "N unparseable outputs counted as wrong" to the model's row in the README table. Tests: `tests/test_llm_parser.py` → `test_parse_label` and `test_every_training_answer_parses_back_to_its_label`.

**Why it matters here.** Counting unparseable answers as wrong is the honest choice. Dropping them would score the model only on the answers it formatted well; mapping them to a default class would hide format failures. A standard alternative that this code does not use is to compare the model's probabilities for the three label words and take the largest. That is always parseable, but it measures something slightly different from what the model actually writes.

### 8.7.5 4-bit quantisation (NF4, bitsandbytes) and the QLoRA idea

**Intuition.** Store each weight in 4 bits (16 possible values) instead of 16 bits, plus one scale number per small block of weights. Weight memory drops about four-fold. Just before each matrix multiplication, the 4-bit weights are expanded back to 16-bit numbers ("dequantised on the fly"); no 16-bit copy of the whole model is kept. QLoRA freezes these 4-bit weights and trains only small LoRA adapters (8.7.6), which are kept in higher precision (bfloat16 in the QLoRA paper).

**Maths.** Block-wise absmax quantisation with the NF4 ("4-bit NormalFloat") code book. Split the weights into blocks of $`M`$ consecutive values ($`M = 64`$ in the QLoRA paper). For one block $`w_1, \dots, w_M`$:

```math
c = \max_{i} |w_i|, \qquad j_i = \arg\min_{j \in \{0, \dots, 15\}} \left| \frac{w_i}{c} - q_j \right|, \qquad \hat{w}_i = c \, q_{j_i}
```

$`c`$ is the block's absolute maximum (its scale), $`j_i`$ is the 4-bit index stored for weight $`i`$, and $`\hat{w}_i`$ is the dequantised weight used in computation. The 16 levels $`q_0 < \dots < q_{15}`$ lie in $`[-1, 1]`$, include an exact 0, and are quantiles of a standard normal distribution rescaled so that the largest is 1. They are not symmetric: 7 negative levels, 0 and 8 positive levels. Recomputed from the QLoRA construction, they are about −1, −0.696, −0.525, −0.395, −0.284, −0.185, −0.091, 0, 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723 and 1. Trained weights are roughly normally distributed, so normal quantiles put more levels where weights are dense than evenly spaced levels would.

Storage per weight, counting one 32-bit scale per block:

```math
4 + \frac{32}{64} = 4.5 \ \text{bits}, \qquad \text{with double quantisation: } 4 + \frac{8}{64} + \frac{32}{64 \cdot 256} \approx 4.127 \ \text{bits}
```

Double quantisation stores the block scales themselves in 8 bits, with one 32-bit scale per 256 blocks.

Toy example (a block of 4 instead of 64): $`w = [0.02, -0.05, 0.10, -0.01]`$, so $`c = 0.10`$ and $`w / c = [0.2, -0.5, 1.0, -0.1]`$. The nearest levels are $`[0.161, -0.525, 1.0, -0.091]`$, which gives $`\hat{w} = [0.0161, -0.0525, 0.1000, -0.0091]`$. The largest weight is exact; the others carry small errors.

For a quantised layer with a LoRA adapter the forward pass is

```math
y = \operatorname{dequant}(W_{\text{NF4}})\, x + \frac{\alpha}{r} B A x
```

Gradients flow through the dequantised $`W`$, so adapters in earlier layers get gradients, but $`W`$ itself is never updated.

**In this code.**
- Default run: `--backend unsloth` (`scripts/train_unsloth.py` → `parse_args`) with `DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit"` (`src/fnsl/llm_lora.py`). `resolve_settings` sets `load_in_4bit = not smoke`. `load_model` calls `FastLanguageModel.from_pretrained(..., dtype=None, load_in_4bit=True)`; per the code comment, Unsloth then picks float16 on a T4 and bfloat16 on newer GPUs. The checkpoint is already stored in 4 bits. Its `config.json` (cached copy of the Hub file) records bitsandbytes NF4 with double quantisation, a bfloat16 compute type, and a list of modules left unquantised (the output layer `lm_head` and a few sensitive attention and MLP blocks); this skip list is what Unsloth's "dynamic" 4-bit refers to. How Unsloth reconciles the bfloat16 compute type with a T4, which has no native bfloat16, was not checked.
- `hf` backend on a GPU (the fallback): the code passes `transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=gpu_dtype())`, without double quantisation. `gpu_dtype` returns bfloat16 only with native support (Ampere or newer GPUs), otherwise float16 (a T4). Deviation: for a checkpoint that is already quantised, transformers keeps the checkpoint's own quantisation settings and only prints a warning (checked in transformers 5.17: `get_hf_quantizer` → `merge_quantization_configs`). So with the default model this config has no effect, and the checkpoint's NF4, double-quantisation, bfloat16-compute and skip-list settings apply. The code's config matters only for a 16-bit model passed with `--model-name`.
- Without CUDA, `load_model` prints a message and loads float32 instead of 4 bits (the code treats 4-bit loading as GPU-only). That works for a 16-bit checkpoint such as `SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"`, which `--smoke` uses. It cannot work for the default checkpoint, whose weights are stored in 4 bits: loading it here raises `ImportError` because bitsandbytes is not installed (checked).
- Adapter precision: in the `hf` path, PEFT casts float16 or bfloat16 adapter weights to float32 by default (`autocast_adapter_dtype=True`, checked in PEFT 0.21.3). What Unsloth does could not be checked here.
- Status: code only. No 4-bit model has been loaded in this repo.

**Why it matters here.** Back-of-envelope arithmetic, not a measurement: 1.5 billion weights take about 3 GB in 16-bit and about 0.75 GB in 4-bit, plus the scales (about 0.13 extra bits per weight with double quantisation). Full fine-tuning with Adam in mixed precision needs roughly 16 bytes per parameter (a common rule of thumb: 16-bit weights and gradients, a 32-bit master copy and two 32-bit Adam moments), which is about 24 GB for 1.5 billion parameters, before activations. That is more than the 16 GB of a free T4. A frozen 4-bit base plus small adapters fits. The QLoRA paper (Dettmers et al., 2023) reports that NF4 plus LoRA matched 16-bit fine-tuning on its benchmarks. This project does not test that, so the guide makes no claim about what quantisation costs in accuracy here.

### 8.7.6 LoRA: what is trained and what is frozen

**Intuition.** Instead of changing a large weight matrix $`W`$, LoRA learns a small correction of low rank and adds it to $`W`$. All original weights stay frozen. Only the two small factor matrices of each correction are trained.

**Maths.** A linear layer computes $`y = W x`$ with $`W \in \mathbb{R}^{d \times k}`$: $`d`$ outputs and $`k`$ inputs (PyTorch stores the weight as outputs × inputs). LoRA replaces $`W`$ by

```math
W' = W + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)
```

so $`y = W x + \frac{\alpha}{r} B (A x)`$. Here $`r`$ is the rank of the update ($`\operatorname{rank}(BA) \le r`$) and $`\alpha`$ is a scaling constant. With dropout probability $`p > 0`$, $`x`$ passes through dropout before $`A`$. $`W`$ gets no gradient. Initialisation: $`A`$ is random and $`B = 0`$. Then $`BA = 0`$ and $`W' = W`$, so training starts exactly at the base model. With $`\delta = \partial \mathcal{L} / \partial y`$ for one input $`x`$, the gradients are

```math
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r}\, \delta \, (A x)^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r}\, B^{\top} \delta \, x^{\top}
```

As long as $`B = 0`$, $`A`$ gets no gradient (its gradient contains $`B`$), but $`B`$ does (its gradient contains $`A x`$, which is non-zero because $`A`$ is random). Once $`B`$ has moved away from zero, both receive gradients. If both started at zero, neither would ever move. (In this project the first update has learning rate 0 because of the warm-up, see 8.7.8, so $`B`$ first changes at the second update and $`A`$ at the third.)

Trainable parameters for one $`d \times k`$ matrix:

```math
\text{full fine-tuning: } d k, \qquad \text{LoRA: } r (d + k), \qquad \text{fraction: } \frac{r (d + k)}{d k}
```

Toy example: $`d = k = 1000`$ and $`r = 16`$ give 1,000,000 weights for full fine-tuning and 32,000 for LoRA (3.2 %). The model's total is this sum over every adapted matrix in every layer. The factor $`\alpha / r`$ keeps the size of the update roughly stable if you change $`r`$; the LoRA paper sets $`\alpha`$ once and does not tune it.

For the default model the same formula gives a concrete count. This is arithmetic from the model's `config.json`, not a results-file number: hidden size 1536, MLP size 8960, 2 key-value heads of size 128 (so `k_proj` and `v_proj` have 256 outputs) and 28 layers. Per layer, $`r(d + k)`$ summed over the seven matrices is $`16 \times (3072 + 1792 + 1792 + 3072 + 3 \times 10496) = 659{,}456`$, so 28 layers give 18,464,768 trainable numbers, about 1.2 % of the 1.54 billion base weights. PEFT on a weightless copy of the architecture gives the same count (checked while writing this guide). The training run writes the real count to `llm_lora_train.json` → `trainable_params`.

**In this code.** `src/fnsl/llm_lora.py` → `add_lora`, called by `scripts/train_unsloth.py` → `run_train` with the defaults from `parse_args`: `--lora-r 16`, `--lora-alpha 16` (so $`\alpha / r = 1`$), `--lora-dropout 0.0`, `bias="none"`, and `DEFAULT_TARGET_MODULES`: `q_proj`, `k_proj`, `v_proj`, `o_proj` (the attention projections) and `gate_proj`, `up_proj`, `down_proj` (the MLP projections), in every transformer layer, "as the Unsloth LoRA guide recommends" (code comment). Everything else is frozen: the embeddings, the norms, the output layer, the biases (`bias="none"`) and all original projection weights.
- Unsloth path: `FastLanguageModel.get_peft_model(..., use_gradient_checkpointing="unsloth", random_state=seed)`.
- `hf` path: `peft.LoraConfig(task_type="CAUSAL_LM")` and `get_peft_model`, after `torch.manual_seed(seed)` so the random $`A`$ matrices are reproducible. Checked in PEFT 0.21.3: scaling `lora_alpha / r`, `kaiming_uniform_` for $`A`$ (the default initialisation of a PyTorch linear layer), zeros for $`B`$, and the forward pass `base_layer(x) + lora_B(lora_A(dropout(x))) * scaling`. Deviation from the paper: the LoRA paper draws $`A`$ from a Gaussian; PEFT's default is Kaiming-uniform. Both are random, which is all the argument above needs.
- `trainable_parameters` uses PEFT's `get_nb_trainable_parameters` (which counts 4-bit weights correctly), and `llm_lora_train.json` records `lora`, `trainable_params`, `total_params` and `trainable_pct` (a percentage).
- Only the adapter is saved (`model.save_pretrained`). Evaluation attaches it to the base model again and never merges it into $`W`$.
- `tests/test_llm_smoke.py` → `test_lora_attach_train_generate_and_reload` (heavy; smoke model with $`r = 4`$, $`\alpha = 8`$) checks that every $`B`$ is zero before training, that at least one $`B`$ is non-zero after two training steps, and that fewer than 5 % of the weights are trainable.

**Why it is cheaper than full fine-tuning.** Memory: the frozen weights have no gradients and no Adam moments; only $`A`$ and $`B`$ have optimiser state. With a 4-bit base the frozen weights also take about a quarter of their 16-bit memory. Storage: the saved adapter holds only $`A`$, $`B`$ and a config, not a copy of the model. Compute per step falls only somewhat: the backward pass skips the weight-gradient products for every frozen $`W`$, but the forward pass and the gradient with respect to each layer's input must still go through every $`W`$, because gradients have to reach the adapters in earlier layers. The savings are mainly memory and storage.

**Pitfall.** The update is limited to rank $`r`$ per matrix. That is enough for a narrow task such as "write one of three words", but it is a real restriction. Separately, an adapter trained with the `--backend hf` fallback is LoRA but not Unsloth. `src/fnsl/reporting.py` → `trained_with_unsloth` reads `params.train_backend` in `llm_lora.json`; unless it is `unsloth`, `RESUME_NUMBERS.md` says "Unsloth: NO" and asks you to reword bullet 2.

### 8.7.7 Supervised fine-tuning with a completion-only loss (TRL)

**Intuition.** Show the model (prompt, correct answer) pairs and raise the probability of the answer tokens only. The prompt is the same boilerplate in every row; training on it would spend effort on predicting the instruction.

**Maths.** The labels are a copy of the token ids with every prompt position set to −100: $`\ell_t = x_t`$ for answer tokens and $`\ell_t = -100`$ otherwise (so $`\omega_t = 1`$ exactly when $`\ell_t \ne -100`$). The cross-entropy skips −100. For the examples $`n`$ in one optimiser step:

```math
\mathcal{L} = -\frac{1}{|C|} \sum_{(n, t) \in C} \log p_\theta\big(x^{(n)}_t \mid x^{(n)}_{<t}\big), \qquad C = \{(n, t) : \ell^{(n)}_t \ne -100\}
```

$`C`$ is the set of counted (example, position) pairs and $`|C|`$ its size. For Qwen2.5 the counted tokens of one row are the label word's tokens, `<|im_end|>` and the newline that the template puts after it (for `Bullish`: `B`, `ull`, `ish`, `<|im_end|>`, newline; checked with the tokenizer). Learning `<|im_end|>` teaches the model to stop after the label. Because the mean is over tokens, a Bullish row (5 counted tokens) weighs more than a Bearish row (4) or a Neutral row (3), and the easy tokens after the first one dilute the loss, although the class is decided at the first answer token.

**In this code.** `src/fnsl/llm_lora.py` → `train_lora` builds a `trl.SFTConfig` with these fields: `output_dir`, `max_steps`, `learning_rate`, `per_device_train_batch_size` and `per_device_eval_batch_size` (both `--batch-size`), `gradient_accumulation_steps`, `warmup_steps`, `lr_scheduler_type="linear"`, `logging_steps` (10; 1 in smoke runs), `eval_strategy="steps"`, `eval_steps`, `eval_on_start=True`, `save_strategy="no"`, `max_length=max_seq_length` (the TRL 1.x name; default `--max-seq-length 128`), `completion_only_loss=True`, `gradient_checkpointing=(backend == "unsloth")`, `fp16` / `bf16` chosen by GPU type, `dataloader_pin_memory=use_cuda`, `seed` and `report_to="none"`. This overrides TRL defaults that would hurt here: `bf16` defaults to True (fails on a T4, slow on a CPU), `gradient_checkpointing` defaults to True, `max_length` defaults to 1024, and the learning rate defaults to 2e-5 (checked in TRL 1.15, `SFTConfig` and `_BaseConfig`).

In TRL 1.15 (checked), the tokenised dataset gets a completion mask, labels outside it become −100, sequences longer than `max_length` are cut at the **end** (`truncation_mode="keep_start"`), and rows left with no answer token are dropped. This happens to the validation rows too. Cutting the end would remove the answer, so `count_too_long` counts such training rows, and `run_train` prints a warning and stores `n_train_too_long`. `count_too_long` does not check the validation rows: a validation row whose answer was cut off would be dropped silently, and the validation loss would cover fewer rows than `n_val_eval` says. A check made while writing this guide (cached Qwen2.5 tokenizer, default settings) found no train or test row longer than 128 tokens; the validation split does contain a longer row, but it is not among the default 200 validation rows. The margin is small: re-check if you lengthen the prompt or change `--n-val-eval`.

**Pitfall.** TRL supports a completion-only loss only for prompt-completion datasets (separate `prompt` and `completion` columns). That is why `to_sft_dataset` builds exactly that format. TRL also refuses a formatting function together with `completion_only_loss=True`.

### 8.7.8 Optimiser, learning-rate schedule, gradient accumulation and validation

**Intuition.** AdamW moves each trainable number using running averages of its gradient. The learning rate warms up, then falls linearly towards zero. Gradients from two small batches are added before each step, which acts like one batch twice as large. A validation loss on rows the model never trains on shows whether training helps.

**Maths.** In this subsection $`t`$ counts optimiser updates. For the trainable LoRA parameters $`\phi`$ (all $`A`$ and $`B`$ matrices) with gradient $`g_t`$ at update $`t`$, the transformers Trainer first clips the gradient to a global norm of at most $`\kappa`$:

```math
g_t \leftarrow g_t \cdot \min\!\left(1, \frac{\kappa}{\lVert g_t \rVert_2}\right)
```

where $`\lVert g_t \rVert_2`$ is the norm of all trainable gradients taken together (PyTorch adds $`10^{-6}`$ to it). Then AdamW updates:

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\phi_t &= \phi_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \, \phi_{t-1} \right)
\end{aligned}
```

$`m_t`$ and $`v_t`$ are running averages of the gradient and of the squared gradient (element by element), the hats correct their bias toward zero in the first updates, $`\eta_t`$ is the learning rate, $`\epsilon`$ avoids division by zero and $`\lambda`$ is the weight decay. "Decoupled" means the decay term $`\lambda \phi_{t-1}`$ is added outside the Adam ratio instead of being folded into the gradient. Linear warm-up and decay with peak rate $`\eta`$, $`t_w`$ warm-up steps and $`t_{\max}`$ steps in total (transformers' `get_linear_schedule_with_warmup`):

```math
\eta_t = \begin{cases} \eta \, \dfrac{t}{t_w} & 0 \le t < t_w \\[6pt] \eta \, \dfrac{t_{\max} - t}{t_{\max} - t_w} & t_w \le t < t_{\max} \end{cases}
```

Here $`t`$ is the number of updates already made, so $`\eta_t`$ is the rate used by update number $`t + 1`$. With the defaults below, the first update uses $`\eta_0 = 0`$, the peak $`\eta`$ is reached at update 31, and the last update (number 300) uses $`\eta / 270`$; the rate would reach 0 only after it (checked with a 300-step schedule).

Effective batch size $`= b \times a`$ for a per-device batch of $`b`$ rows and $`a`$ accumulation steps. The loss of each micro-batch is divided by the number of answer tokens over all $`a`$ micro-batches, so the summed gradient equals that of one batch of $`b \times a`$ rows (checked in transformers 5.17, `Trainer._get_num_items_in_batch`).

**In this code.** Defaults from `scripts/train_unsloth.py` → `parse_args` and `resolve_settings`: $`\eta`$ = `--lr 2e-4`; $`t_{\max}`$ = `--max-steps 300`; $`t_w`$ = `max(1, round(0.1 * max_steps))` = 30 (`WARMUP_FRACTION = 0.1`); $`b`$ = `--batch-size 8`; $`a`$ = `--grad-accum 2`; effective batch 16 (stored as `effective_batch_size`).
- Optimiser: the code does not set one, so the `transformers` default applies. In the installed transformers 5.17 that is `adamw_torch_fused` (the default for torch 2.8 or newer) with $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, $`\epsilon = 10^{-8}`$, weight decay $`\lambda = 0`$ and gradient clipping at $`\kappa = 1.0`$ (`TrainingArguments` defaults). With $`\lambda = 0`$ the decay term vanishes. Whether Unsloth changes these on the GPU was not checked.
- Budget: 300 × 16 = 4,800 training rows seen, about 0.58 of one pass over the 8,241 training rows (`data/splits/split_report.json`).
- Validation: `--n-val-eval 200` rows drawn from the 1,767-row validation split (`data/splits/split_report.json`) with `sample_rows` (seed 42). `eval_on_start=True` measures the untouched model's loss at step 0, then every `--eval-steps 50` steps, so at steps 0, 50, …, 300. Each `eval_loss` is the mean answer-token loss of each validation batch, averaged over the batches (weighted by rows; 25 batches of 8 at the defaults). Each point in `train_loss` is the average loss per update over the last `logging_steps` (10) updates. There is no early stopping and no choice of the best checkpoint (`save_strategy="no"`): the adapter after the last step is saved. The test split is never used.
- Precision: float16 mixed precision on a T4, bfloat16 on Ampere or newer GPUs (`gpu_dtype`), float32 on a CPU. With float16 the Trainer also scales the loss up before the backward pass, so that small gradients do not underflow, and unscales the gradients before clipping; bfloat16 has the exponent range of float32 and needs no scaling.

**Why it matters here.** Warm-up avoids large early steps while Adam's averages are still poor, and the linear decay lets training settle. **Pitfall:** validation loss is not accuracy. A lower loss means more probability on the correct label tokens, which usually, but not always, changes the arg max answer. Read `val_loss_start` and `val_loss_end` in `llm_lora_train.json`, then the test metrics; never tune anything on the test split.

### 8.7.9 CPU smoke path: replacing TRL's Triton loss

**Intuition.** TRL 1.15 computes the training loss with a GPU kernel written in Triton. A CPU build of torch has no Triton, so the first CPU smoke run crashed. On a CPU the code falls back to the ordinary `transformers` loss, which is the same objective.

**Maths.** Both compute the masked mean from 8.7.7. TRL: per-token loss $`-\log p`$, summed, divided by the number of counted tokens. `transformers`' `ForCausalLMLoss`: shift the labels by one position, `cross_entropy(ignore_index=-100, reduction="sum")`, divided by the number of counted tokens in the accumulated batch (or a plain mean when that number is not given, as in evaluation).

**In this code.** `src/fnsl/llm_lora.py` → `sft_trainer_class(use_cuda)`. With CUDA it returns TRL's `SFTTrainer` unchanged. Without CUDA it returns `CpuSFTTrainer`, whose `compute_loss` calls `transformers.Trainer.compute_loss`. TRL has already set the prompt labels to −100, so the loss is still answer-only. TRL's extra training metrics (token accuracy, entropy) are skipped on a CPU. `DEBUG_LOG.md`, Day 9, tells the story: TRL's guarded import set the kernel function to `None`, and `SFTConfig` has no switch to turn the kernel off.

**Pitfall.** The CPU smoke run checks the plumbing (data format, labels, saving, evaluation, file schemas). It does not exercise the GPU loss kernel or Unsloth, so it is not evidence that the GPU run works.

### 8.7.10 Unsloth input truncation and inference_max_seq_length

**Intuition.** According to the code comments, Unsloth cuts every input to the `max_seq_length` it was loaded with. Training uses a short limit on purpose (128 tokens, to save memory). If inference reused that limit, a long test prompt would lose its end. With left padding the end holds the last words of the headline and the `<|im_start|>assistant` line, so the model would answer a damaged prompt without any error.

**Maths.** For each batch the guard checks

```math
L_{\text{batch}} + n_{\text{new}} \le L_{\max}
```

where $`L_{\text{batch}}`$ is the padded batch length (the length of the longest prompt in the batch), $`n_{\text{new}}`$ is `max_new_tokens` (5) and $`L_{\max}`$ is the `max_seq_length` the model was loaded with.

**In this code.** `scripts/train_unsloth.py` sets `INFERENCE_MAX_SEQ_LENGTH = 512`, and `resolve_settings` sets `inference_max_seq_length = max(512, --max-seq-length)`. `run_zero_shot` and `run_eval` load the model with that limit; `run_train` loads it with `--max-seq-length` (default 128). `src/fnsl/llm_lora.py` → `generate_labels` reads `model.max_seq_length` (set by Unsloth; plain `transformers` models do not have it, so the `hf` path skips the check) and raises `ValueError` if a batch would not fit. Tests: `tests/test_llm_parser.py` → `test_inference_loads_use_a_limit_that_never_cuts_test_prompts` (CPU) and `tests/test_llm_smoke.py` → `test_generate_labels_refuses_prompts_longer_than_max_seq_length` (heavy; it sets the attribute by hand).

**Pitfall.** Silent truncation would not crash. It would only lower the score, and the score would no longer describe the model. Unsloth is not installed here, so its truncation behaviour is taken from the code comments, not observed.

### 8.7.11 Outputs

| Stage | Files (with `--smoke`: under `results/smoke/` and `checkpoints/smoke/` instead) |
|---|---|
| zero-shot | `results/metrics/llm_zeroshot.json`, `results/predictions/llm_zeroshot.csv`, `results/figures/llm_zeroshot_confusion.png` |
| train | adapter folder `checkpoints/llm_lora_adapter/` (PEFT `adapter_config.json` and `adapter_model.safetensors`, tokenizer files with `chat_template.jinja`, `train_info.json`; file names as written by the `hf` path in the smoke run); `results/metrics/llm_lora_train.json` (loss curves and LoRA settings, deliberately without `macro_f1`) |
| eval | `results/metrics/llm_lora.json`, `results/predictions/llm_lora.csv`, `results/figures/llm_lora_confusion.png` |

Prediction CSVs follow the project schema `id,text,label,pred`, with `pred = -1` for unparseable answers. `llm_lora_train.json` has no `macro_f1`, so `metrics.load_metrics` skips it and it can never appear in the results table as a test result. Non-smoke runs also log to MLflow (`mlruns/`, gitignored) unless `--no-mlflow` is given.

### 8.7.12 Interview answers (plan "Explain" questions, Days 8-10)

- **What is 4-bit quantisation?** Each weight is stored as a 4-bit index into 16 fixed levels, plus one scale per block of 64 weights. NF4 places the levels at quantiles of a normal distribution, because trained weights are roughly normal. At run time each block is expanded back to 16-bit numbers just before its matrix multiplication. Weight memory falls about four-fold, at the cost of a small rounding error in every weight. Here the default model is already stored in Unsloth's dynamic 4-bit format (NF4 with double quantisation, with some sensitive layers left unquantised).
- **Why must the prompt template match between training and inference?** The fine-tuned model learned $`p(\text{answer} \mid \text{exact prompt tokens})`$. Any change (other wording, a missing system message, other role markers, no generation prompt, even different whitespace) gives inputs it never saw, so its answers and their format can change, and the measured accuracy no longer describes the trained model. The code builds every prompt in one function, `build_messages`, uses the tokenizer's chat template for both training and inference, saves the tokenizer with the adapter, and refuses to evaluate an adapter whose `PROMPT_VERSION` differs.
- **What does LoRA train and what does it freeze?** It freezes every original weight. It adds a trainable low-rank pair $`B A`$ (rank 16 here) beside the seven attention and MLP projections of every layer and trains only those pairs. The adapted weight is $`W + (\alpha / r) B A`$ with $`B`$ starting at zero, so training starts from the base model.
- **Why is it cheaper than full fine-tuning?** For a $`d \times k`$ matrix it trains $`r(d + k)`$ numbers instead of $`d k`$, so gradients and Adam states exist only for a small fraction of the parameters (about 1.2 % for the default model, by the arithmetic in 8.7.6), and with QLoRA the frozen weights also sit in 4 bits. The adapter file is small. The forward and backward passes still run through the whole model, so the saving is mostly memory and storage, not compute per step.
- **Did the LLM beat DistilBERT, is the difference significant, and what is the cost trade-off? (Day 10)** This cannot be answered yet: neither `results/metrics/llm_lora.json` nor `results/metrics/distilbert.json` exists, and `results/metrics/comparison.json` has no pairs. Once both models have predictions on all 1,767 test rows, answer in three steps. (1) Scores: `macro_f1` in the two metrics files, with their bootstrap CIs (`comparison.json` → `models`). (2) Significance: the `distilbert` and `llm_lora` entry in `comparison.json` → `pairs` gives McNemar's p-value (from the rows where exactly one model is right), the Holm-adjusted p-value `p_holm` (significant if below `alpha`, 0.05) and a paired-bootstrap CI for the macro-F1 difference (`scripts/compare_models.py`; `src/fnsl/stats.py` → `mcnemar_test`, `holm_adjust`, `paired_bootstrap_diff`). If the difference is not significant, say so. (3) Cost: `train_time_s` in both files (note which GPU each run used; it is not recorded), `inference_time_s` (recorded only for the LLM runs), `trainable_params` in `llm_lora_train.json` against DistilBERT, which trains all its weights, and the fact that the LLM needs a GPU and generates up to 5 tokens per headline with a model of about 1.5 billion parameters. A higher F1 that is not significant does not justify the extra cost.

## 8.8 Retrieval-augmented generation (RAG) and hit@k

> **Status at HEAD 98a9d42.** The code is in `src/fnsl/rag/ingest.py`, `src/fnsl/rag/retriever.py`, `src/fnsl/rag/evaluate.py`, `scripts/build_rag.py` and `scripts/eval_rag.py`. Tests: `tests/test_rag.py` (CPU and offline, passes at HEAD; one heavy test for the real embedding model). Real run: `python scripts/build_rag.py` built the index over the real corpus with `sentence-transformers/all-MiniLM-L6-v2` and saved it to the gitignored `cache/rag_index/default/`. It measures nothing. `scripts/eval_rag.py` has only been smoke-tested (offline "hashing" embeddings and the three EXAMPLE rows; output in the gitignored `results/smoke/`). `results/metrics/rag_eval.json` does not exist, so **no hit@k has been measured**, and this section states none.
>
> **What you need to do.** `data/rag_eval.csv` holds only three `EXAMPLE` rows. Write 20-30 questions yourself (Day 12 of the plan), each answerable from exactly one file, following the "Watch out when writing `data/rag_eval.csv`" advice below the table in `data/rag_corpus/SOURCES.md`. Then run `python scripts/eval_rag.py` (it refuses fewer than 5 real questions) and `python scripts/make_results_table.py --update-readme`, run `pytest -m "not heavy"` and `ruff check .`, and commit. Optionally rebuild the demo index with the winning chunk size, `python scripts/build_rag.py --chunk-size <best_chunk_size>`, because `build_rag.py` defaults to 500, which is not one of the two sizes compared. Day 12 is done when hit@3 is reported for two chunk sizes. Fill the RAG paper entry in `reading_log.md` (Day 13).

Pipeline:

```text
build_rag.py  data/rag_corpus/*.txt -> load_corpus -> chunk_documents (500 / 50 characters)
              -> MiniLM embeddings (unit length) -> FAISS IndexFlatL2 -> cache/rag_index/default/
              -> answer_question(sample question, k=3): top chunks + retrieval-only answer with its source
eval_rag.py   data/rag_eval.csv -> validation gates -> for chunk size in (300, 800):
              chunk -> embed -> index -> top 3 chunks per question -> hit@1, hit@3
              -> best_chunk_size -> results/metrics/rag_eval.json
```

Library versions checked for this section (installed in `.venv`): langchain-core 1.8.0, langchain-community 0.4.2, langchain-huggingface 1.2.3, langchain-text-splitters 1.1.3, faiss-cpu 1.16.0, sentence-transformers 6.3.0.

**Retrieve, then generate.** RAG answers a question $`q`$ in two steps: retrieve chunks $`c_1, \dots, c_k`$ related to $`q`$, then generate an answer $`y`$ from them. The original RAG paper (Lewis et al., 2020) marginalises over the retrieved passages and fine-tunes the question encoder and the generator together, while the passage index stays fixed:

```math
p(y \mid q) \approx \sum_{j=1}^{k} p_\eta(c_j \mid q) \, p_\theta(y \mid q, c_j)
```

where $`p_\eta(c_j \mid q)`$ is the retriever's probability for passage $`c_j`$ (a softmax over the similarity scores of the top $`k`$) and $`p_\theta(y \mid q, c_j)`$ is the generator's probability for answer $`y`$ given the question and that passage. This project trains nothing. By default it has no generator (retrieval-only); with the optional generator it puts all $`k`$ retrieved chunks into one prompt and decodes greedily from $`p_\theta(y \mid q, c_1, \dots, c_k)`$. The plan evaluates only the retrieval step.

### 8.8.1 The corpus and its licences

**Intuition.** Ten short, public monetary-policy statements: small enough to read in full, similar enough to make retrieval non-trivial, and legally safe to commit.

**Maths.** None; this is data provenance.

**In this code.** `data/rag_corpus/` holds one `.txt` file per statement: seven Federal Reserve FOMC statements (2008-12-16, 2015-12-16, 2020-03-15, 2022-03-16, 2023-07-26, 2024-09-18, 2026-10-28) and three ECB "Monetary policy decisions" press releases (2022-07-21, 2024-06-06, 2026-10-29). Per `data/rag_corpus/SOURCES.md`, information on the Federal Reserve Board's website is in the public domain (the Board asks to be cited as the source), and ECB text may be reused if it is reproduced accurately with the ECB cited as the source. Each page was fetched once on 2026-11-14 and only the statement body was kept, with the wording unchanged (whitespace was normalised and two non-breaking hyphens became plain hyphens). Reserve Bank of India text was deliberately left out: its website terms reserve all rights and prohibit caching. Every file starts with `Source: <URL> (retrieved 2026-11-14)`, then `Title:`, `Date:` and `Publisher:` lines, a blank line and the text.

**Why it matters here.** FOMC statements reuse many sentences from meeting to meeting, so some files are near-duplicates. `SOURCES.md` lists the traps: 2015-12-16 and 2022-03-16 both raise the target range to 1/4 to 1/2 percent; 2008-12-16 and 2020-03-15 both set it to 0 to 1/4 percent; all three ECB releases mention the Transmission Protection Instrument, but only 2022-07-21 approves it. A question about such a fact needs an extra detail to have exactly one correct file.

### 8.8.2 Loading: header lines become metadata

**Intuition.** The URL, title, date and publisher describe a file; they are not its content. Kept in the text, they formed a chunk of their own that looked almost the same for every statement and won the search for questions that mention a date.

**Maths.** None.

**In this code.** `src/fnsl/rag/ingest.py` → `load_corpus` reads every `*.txt` file, sorted by name; `SOURCES.md` is not a `.txt` file, so it is skipped. If the first line starts with `Source:`, it is removed and its first word (the URL) goes to `metadata["url"]`. The following lines whose key before the colon is `Title`, `Date` or `Publisher` move to `metadata["title"]`, `metadata["date"]` and `metadata["publisher"]`. `metadata["source"]` is the file name: it is what answers cite and what hit@k compares. Empty files are skipped, and a folder without `.txt` files raises `FileNotFoundError`. `DEBUG_LOG.md`, Day 11, describes the bug behind this design: for "What did the Federal Reserve decide on September 18, 2024?" the whole retrieval-only answer was a header-only chunk, and it was the header of the wrong statement. Test: `tests/test_rag.py` → `test_committed_corpus_has_no_header_only_chunks`.

**Pitfall.** Dates are now metadata, not embedded text, and FOMC statements do not repeat their own date. A question such as "What did the Fed decide in October 2026?" gives the retriever nothing distinctive to match; ask about content instead (`SOURCES.md` gives examples). The code does not filter by metadata.

### 8.8.3 Chunking with RecursiveCharacterTextSplitter

**Intuition.** A whole statement is too long and mixes several topics, so it is cut into passages called chunks. Small chunks are specific but may split an answer in two. Large chunks keep more context but blur several topics into one vector. The recursive splitter prefers to cut at paragraph breaks, then at line breaks, then at spaces, and only then inside words.

**Maths.** Write $`S`$ for `chunk_size` and $`O`$ for `chunk_overlap`, both in characters. The algorithm:

1. Take the first separator in the list (paragraph break `\n\n`, line break `\n`, space, empty string) that occurs in the text, and split the text on it. The separator stays at the start of the following piece.
2. Merge pieces shorter than $`S`$ greedily, from left to right, into a chunk while the chunk stays within $`S`$.
3. When a chunk is full, emit it and carry its last pieces, at most $`O`$ characters in total, over as the start of the next chunk.
4. Split any piece of $`S`$ characters or more again, with the remaining separators (the recursion).

For consecutive chunks $`c_i`$ and $`c_{i+1}`$ of one document, with $`|c|`$ the number of characters of chunk $`c`$:

```math
|c_i| \le S, \qquad \operatorname{overlap}(c_i, c_{i+1}) \le O
```

The overlap is made of whole pieces, so chunks that end at a paragraph break usually share nothing; overlap appears when a cut falls inside a paragraph.

**In this code.** `src/fnsl/rag/ingest.py` → `chunk_documents(docs, chunk_size=500, chunk_overlap=50)` uses `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, add_start_index=True)` with LangChain's default separators and `length_function=len` (checked in langchain-text-splitters 1.1.3). Each chunk keeps its document's metadata and gets `chunk` (0, 1, 2, … within the document) and `start_index` (its character offset in the document). `scripts/build_rag.py` uses `--chunk-size 500 --chunk-overlap 50`; `scripts/eval_rag.py` compares `--chunk-sizes 300 800` with `--chunk-overlap 50`. An overlap larger than the chunk size raises `ValueError` (inside LangChain). Tests: `test_chunks_respect_size_and_keep_metadata` and `test_consecutive_chunks_overlap`.

**Pitfall.** `chunk_size` counts characters, not tokens. The embedding model reads at most 256 word pieces. Even the 800-character chunks of this corpus stay well under that limit (checked while writing this guide), so nothing is cut at the evaluated sizes, but much larger chunks would be truncated silently. Also read your chunks: a short divider line such as `***` can become a chunk of its own (it happens at chunk size 500), and such a chunk carries no meaning.

### 8.8.4 Embeddings: MiniLM sentence vectors of unit length

**Intuition.** An embedding model maps a text to a point in a 384-dimensional space, so that texts with similar meaning land close together, even when they use different words.

**Maths.** `all-MiniLM-L6-v2` is a 6-layer transformer encoder. For a text of $`n`$ word pieces (counting the `[CLS]` and `[SEP]` markers) it produces token vectors $`h_1, \dots, h_n \in \mathbb{R}^{384}`$, averages them (mean pooling over the real, non-padding tokens) and scales the result to length 1:

```math
e = \frac{1}{n} \sum_{i=1}^{n} h_i, \qquad u = \frac{e}{\lVert e \rVert_2}
```

$`\lVert e \rVert_2`$ is the Euclidean length of $`e`$. Cosine similarity measures the angle between two vectors $`u`$ and $`v`$; for unit vectors it is just the dot product:

```math
\cos(u, v) = \frac{u^{\top} v}{\lVert u \rVert_2 \, \lVert v \rVert_2} = u^{\top} v \quad \text{when } \lVert u \rVert_2 = \lVert v \rVert_2 = 1
```

**In this code.** `src/fnsl/rag/ingest.py` → `get_embeddings("sentence-transformers/all-MiniLM-L6-v2")` returns `langchain_huggingface.HuggingFaceEmbeddings(model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})`. In langchain-huggingface 1.2.3, `embed_query` falls back to `encode_kwargs` when no separate query settings are given (checked), so questions are normalised exactly like chunks. The model's own sentence-transformers pipeline also ends with a normalisation layer; the flag makes the guarantee explicit. The model's 384 dimensions, mean pooling, normalisation layer and 256-word-piece limit come from its sentence-transformers config files (`modules.json`, `1_Pooling/config.json`, `sentence_bert_config.json`). The index and the questions must use the same model; `build_rag.py` records the model in `info.json`.

For tests and smoke runs, `get_embeddings("hashing")` returns `HashingEmbeddings`, a hashed bag of words: each lower-cased run of letters and digits that is not a stop word adds 1 to bucket $`\text{crc32}(\text{word}) \bmod 1024`$, and the vector is then scaled to length 1 (a text with no usable words stays the zero vector). It uses `zlib.crc32` because Python's `hash()` of a string changes between processes.

**Why it matters here.** Dense embeddings can match paraphrases such as "raised interest rates" and "the policy rate was increased"; the heavy test `test_huggingface_embeddings_are_normalised` checks that such a pair is closer than an unrelated sentence. The hashing stand-in cannot do that ("cut" and "lower" share no word), which is why a real evaluation refuses it. **Pitfall:** MiniLM is a general English model, not trained specifically on central-bank text, and fractions such as "1/4 to 1/2 percent" carry little weight in a mean-pooled vector.

### 8.8.5 Vector search: FAISS IndexFlatL2 and the cosine identity

**Intuition.** To find the chunks nearest to a question, compare the question vector with every stored chunk vector and keep the closest ones. For unit vectors, "closest in straight-line distance" and "smallest angle" give the same order.

**Maths.** For unit vectors $`u`$ and $`v`$:

```math
\lVert u - v \rVert_2^2 = \lVert u \rVert_2^2 + \lVert v \rVert_2^2 - 2\, u^{\top} v = 2 - 2 \cos(u, v)
```

$`2 - 2 \cos`$ falls as $`\cos`$ rises, so sorting by ascending squared distance gives the same order as sorting by descending cosine. The squared distance runs from 0 (same direction) through 2 (orthogonal, unrelated) to 4 (opposite). Exact search for the question vector $`u`$ over $`M`$ stored chunk vectors $`v_1, \dots, v_M`$ computes every $`d_j = \lVert u - v_j \rVert_2^2`$ and returns the $`k`$ indices with the smallest $`d_j`$. That costs about $`M \times D`$ multiply-adds per question, with $`D = 384`$.

**In this code.** `src/fnsl/rag/ingest.py` → `build_index` calls `FAISS.from_documents(chunks, embeddings)`. In langchain-community 0.4.2 the default distance strategy is Euclidean, which builds a `faiss.IndexFlatL2` (checked). LangChain adds no normalisation by default (`normalize_L2=False`), which is why the embeddings must already have unit length. `src/fnsl/rag/retriever.py` → `retrieve(index, question, k=3)` calls `similarity_search_with_score` and returns `(chunk, score)` pairs, best first; the score is the squared L2 distance (lower is closer). If the index holds fewer than $`k`$ chunks, fewer come back. A toy index built while writing this guide confirmed that FAISS's score equals both $`\lVert u - v \rVert_2^2`$ and $`2 - 2 \cos(u, v)`$.

**Why it matters here.** Exact search over an index this small is instant and has no approximation error. Approximate indexes (IVF, HNSW) only pay off for much larger collections, and they can miss true neighbours. **Pitfall:** the score is a distance, not a similarity, and scores are only comparable within one embedding model and one index.

### 8.8.6 Top-k retrieval and the answer step

**Intuition.** Return the $`k`$ nearest chunks. Then either quote the best one with its file name (retrieval-only) or ask a small local language model to answer from those chunks.

**Maths.** For a question $`q`$ the retriever returns the ranked list $`(c_{(1)}, \dots, c_{(k)})`$ with distances $`d_{(1)} \le \dots \le d_{(k)}`$. The cited sources are the distinct files of that list, in rank order.

**In this code.** `src/fnsl/rag/retriever.py` → `answer_question(index, question, k=3, llm=None)`:
- Retrieval-only (the default): the answer is the top chunk's text, then `[source: <file>]`, then `Retrieved sources: <files>`. `unique_sources` keeps rank order and drops repeats.
- With a model: `build_chain(llm)` returns the LCEL chain `RAG_PROMPT | llm | StrOutputParser()`. The `|` operator joins LangChain "runnables", so each step's output is the next step's input: the dict `{"context": ..., "question": ...}` fills the prompt, the chat model replies, and the parser returns the reply as a string. `RAG_PROMPT`'s system message says: use only the context, say you do not know if the answer is not there, and cite the source file of every fact in square brackets. `format_context` puts `[<file>]` above each chunk. The code always appends `Retrieved sources: ...` itself, because small models often ignore the citation instruction (a code comment says Qwen2.5-0.5B did in a trial).
- `load_local_llm("Qwen/Qwen2.5-0.5B-Instruct")` wraps a `transformers` text-generation pipeline (`max_new_tokens=200`, `do_sample=False`, `return_full_text=False`) in `HuggingFacePipeline` and then in `ChatHuggingFace`, which applies the model's chat template. `scripts/build_rag.py --llm-model <id>` turns it on. The default is retrieval-only, and the default question is "How did the Federal Reserve respond to the coronavirus in March 2020?".
- Status: retrieval-only answering ran in the real `build_rag.py` run. The chain is unit-tested with LangChain's `FakeListChatModel` (`test_answer_with_llm_uses_lcel_chain_and_lists_sources`); no generated answer is recorded anywhere.

**Pitfall.** There is no distance threshold. The retriever always returns $`k`$ chunks, even for a question that the corpus cannot answer, and the retrieval-only answer will still quote something. The generated text is not evaluated anywhere in this project. Also, unlike the classifier in 8.7.3, `load_local_llm` does not override the model's `generation_config.json`, and the cached Qwen2.5-0.5B-Instruct file sets `repetition_penalty=1.1`. By the merge rule of 8.7.3 the generator's greedy answers are therefore penalised for repeating words that appear in the retrieved context, which works against quoting it.

### 8.8.7 Saving and loading the index (the pickle risk)

**Intuition.** Saving the index avoids embedding everything again. But part of the saved index is a Python pickle, and loading a pickle can run code.

**Maths.** None.

**In this code.** `src/fnsl/rag/ingest.py` → `save_index` writes `index.faiss` (the vectors), `index.pkl` (the chunk texts, metadata and id map, pickled by LangChain) and `info.json` (`embedding_model`, `chunk_size`, `chunk_overlap`, `n_documents`, `n_chunks`, `sources`) to `cache/rag_index/<name>/` (default name `default`, from `index_path`; `cache/` is gitignored). `load_index` calls `FAISS.load_local(..., allow_dangerous_deserialization=True)`. Unpickling can execute arbitrary code hidden in the file, so LangChain refuses unless you opt in. Opting in is safe here only because the project loads files it wrote itself; never load an `index.pkl` from anywhere else. In the current scripts only `build_rag.py` saves an index and nothing reloads it: `eval_rag.py` rebuilds an index in memory for each chunk size, which is cheap for ten documents. `load_index` is exercised by `test_save_and_load_index_round_trip`.

**Pitfall.** Loading an index with a different embedding model gives meaningless distances without any error. Check `info.json`.

### 8.8.8 The evaluation questions and their validation gates

**Intuition.** hit@k needs questions whose correct file is known. They are written by hand, so the loader checks their shape strictly instead of guessing.

**Maths.** None.

**In this code.** `data/rag_eval.csv` has the header `question,source_file`, one question per row, and the gold file name exactly as it appears in `data/rag_corpus/`. A question that contains a comma must be in double quotes. Rows whose question starts with `EXAMPLE` only show the format and are skipped. The gates, in the order `scripts/eval_rag.py` → `main` applies them:

1. `--k` must be at least 1 (an argparse error otherwise, in `parse_args`).
2. A real run refuses `--embedding-model hashing`.
3. `src/fnsl/rag/evaluate.py` → `load_eval_questions` reads the file with Python's `csv` module (it accepts the byte-order mark that Excel adds and skips blank lines). The header must be exactly `question,source_file`, and every row must have exactly two filled-in cells; otherwise it raises `ValueError` naming the line and suggesting double quotes, and the script exits with that message. This exists because `pandas.read_csv` silently turned "What did the ECB decide on 6 June 2024, and why?" into the question "and why?" (`DEBUG_LOG.md`, Day 12).
4. A real run needs at least `MIN_REAL_QUESTIONS = 5` non-EXAMPLE questions (the plan asks for 20-30); otherwise it exits and writes nothing.
5. No questions at all: exit (reachable only in smoke or `--include-examples` runs).
6. `unknown_sources` lists gold file names that are not in the corpus (usually typos): exit, instead of silently counting those questions as misses.
7. `--smoke` and `--include-examples` always write to `results/smoke/metrics/rag_eval.json`, never to the real file (`output_path`; `--out` is ignored); `--smoke` also defaults to the hashing embeddings.

**Why it matters here.** `README.md` and `RESUME_NUMBERS.md` are generated from `results/metrics/rag_eval.json`, so these gates stop a smoke run, a typo or a corrupted question from becoming a reported number. As a last line of defence, `scripts/make_results_table.py` → `load_result` ignores any results file whose `smoke` flag is true.

### 8.8.9 hit@k

**Intuition.** For each question, look at the files of the top-$`k`$ retrieved chunks. Score 1 if the correct file is among them, otherwise 0. Average over the questions.

**Maths.** For $`N`$ questions, with gold file $`g_i`$ for question $`i`$ and $`s_{i,j}`$ the source file of the chunk ranked $`j`$ for question $`i`$:

```math
\text{hit@}k = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\big[\, g_i \in \{ s_{i,1}, \dots, s_{i,k} \} \,\big]
```

$`\mathbf{1}[\cdot]`$ is 1 if the condition holds and 0 otherwise. Properties: hit@1 ≤ hit@3, because a larger $`k`$ can only add hits; each question moves the score by $`1/N`$; and the top $`k`$ are chunks, so several of them may come from the same file. Toy example (from `test_hit_at_k_matches_hand_computed_values`): retrieved file lists `[a, b, c]`, `[b, a, c]`, `[c, c, b]` and `[d, e, f]` with gold files `a`, `a`, `b` and `x` give hit@1 = 1/4, hit@2 = 2/4 and hit@3 = 3/4. The code reports no confidence interval. As a rough guide, if the questions are treated as independent, the binomial standard error is $`\sqrt{p(1 - p) / N}`$ for a true hit rate $`p`$; toy example: $`p = 0.8`$ and $`N = 25`$ give 0.08.

**In this code.** `src/fnsl/rag/evaluate.py` → `hit_at_k(retrieved_sources, gold_sources, k)` raises if $`k < 1`$, if the two lists differ in length, or if there are no questions. `evaluate_retrieval(index, questions_df, ks=(1, 3))` retrieves $`\max(ks) = 3`$ chunks per question and stores, per question, the retrieved files and whether each $`k`$ was hit. `scripts/eval_rag.py` uses `ks = sorted({1, --k})` with `--k 3`, so it reports hit@1 and hit@3.

### 8.8.10 Comparing two chunk sizes, and why the winner's score is optimistic

**Intuition.** The plan asks to try two chunk sizes and keep the better one. Picking the winner on the same questions that score it flatters the winner: part of its lead may be luck on these particular questions.

**Maths.** Write $`\hat{h}_3(S)`$ for the hit@3 measured on our questions with chunk size $`S`$, and $`h_3(S) = \mathbb{E}[\hat{h}_3(S)]`$ for its average over all question sets we could have written (the true hit rate). The code picks $`S^\ast = \arg\max_{S \in \{300, 800\}} \hat{h}_3(S)`$ (ties broken by hit@1, then by the smaller size), and the resume line reports $`\hat{h}_3(S^\ast) = \max_S \hat{h}_3(S)`$. For every size $`S'`$, $`\max_S \hat{h}_3(S) \ge \hat{h}_3(S')`$; taking averages of both sides gives

```math
\mathbb{E}\Big[\max_{S} \hat{h}_3(S)\Big] \ \ge\ \max_{S} h_3(S) \ \ge\ \mathbb{E}\big[h_3(S^\ast)\big]
```

The first inequality is also Jensen's inequality, because the maximum is a convex function. The right-hand side is what the chosen size scores, on average, on new questions. So the reported best score is, on average, at least as high as the chosen size will score on new questions, and with few questions the gap can be large. It is the same reason the project never tunes on the test split.

**In this code.** `scripts/eval_rag.py` → `main` loads the embedding model once, then for each size re-chunks the corpus, rebuilds the index and runs `evaluate_retrieval`. It saves both configurations (`chunk_size`, `n_chunks`, `hit@1`, `hit@3`), every per-question result, and `best_chunk_size`, chosen by `max(configs, key=(hit@k, hit@1, -chunk_size))` with $`k`$ = `--k` = 3. Its docstring and `src/fnsl/reporting.py` → `rag_section` (which writes the RAG block of `README.md`) both say that the best score is slightly optimistic, and `resume_numbers_md` labels it "picked on the same questions".

**Pitfall.** With 20-30 questions, a gap of one or two questions between the sizes is within noise, and no significance test is run here. If it matters, keep a held-out set of questions for the final number, or run a paired test on the per-question hits.

### 8.8.11 What hit@k does not measure, and why retrieval can fail

**What hit@k measures.** Whether retrieval puts the right document in front of the generator, judged at the level of files.

**What it does not measure.**
- Whether a final answer is correct or faithful to the chunks: a generator can misread or ignore them, and no generated answer is scored here.
- Whether the retrieved chunk of the right file actually contains the answer: the match is by file, not by passage.
- How high the right chunk ranks inside the top $`k`$: rank 1 and rank 3 count the same for hit@3 (mean reciprocal rank would not).
- Questions with more than one valid source: each question has exactly one gold file, so a near-duplicate statement that also answers it counts as a miss.
- Questions the corpus cannot answer: there is no "no answer" option to score.

**Why retrieval can fail** (from `src/fnsl/rag/retriever.py` and `SOURCES.md`): the question uses other words than the document ("cut" vs "lower"); the answer is split across two chunks; several statements say almost the same thing (reused FOMC sentences, repeated rate ranges); the distinguishing fact is a date, which lives only in metadata; the key detail is a number or a fraction that a mean-pooled embedding barely represents; a boilerplate or header chunk outranks real content (fixed for headers in 8.8.2); or a chunk is so long that its vector averages several topics.

### 8.8.12 Outputs

| Script | Files |
|---|---|
| `build_rag.py` | `cache/rag_index/default/index.faiss`, `index.pkl` and `info.json` (gitignored); the top-k chunks and the answer are printed to the terminal |
| `eval_rag.py` | `results/metrics/rag_eval.json` (real run; not produced yet) or `results/smoke/metrics/rag_eval.json` (smoke run) |

`scripts/make_results_table.py` reads `rag_eval.json` for the RAG block of `README.md` and the RAG line of `RESUME_NUMBERS.md`; until the file exists, both say the number is not available. A file with `smoke` set to true is ignored.

### 8.8.13 Interview answers (plan "Explain" questions, Days 11-12)

- **Chunking.** Each statement is cut into passages of at most 500 characters for the demo index (300 and 800 are compared in the evaluation), with up to 50 characters of overlap, cutting at paragraph, then line, then word boundaries. Smaller chunks are more specific; larger ones keep more context but blur topics together.
- **Embeddings.** MiniLM turns each chunk and each question into a vector of 384 numbers with length 1. Similar meaning gives a high dot product, which for unit vectors is the cosine similarity.
- **Vector search.** FAISS compares the question vector with every chunk vector (exact search) and returns the 3 chunks with the smallest squared distance. For unit vectors that is the same order as the highest cosine, because $`\lVert u - v \rVert_2^2 = 2 - 2 \cos(u, v)`$.
- **Why retrieval can fail.** Different wording in question and document, answers split across chunks, near-duplicate statements, dates that live only in metadata, numbers that embeddings barely encode, and boilerplate chunks that look similar to many questions.
- **What hit@k measures and what it does not.** It is the share of questions whose correct file appears among the files of the top-$`k`$ chunks. It says nothing about whether a generated answer is correct, whether the retrieved passage really contains the answer, or where inside the top $`k`$ the right chunk ranked. The reported best chunk size is also slightly optimistic, because it was chosen on the same questions.

---

## Files this part reads and writes

**LLM: `scripts/train_unsloth.py`** (with `--smoke` the same files go under `results/smoke/` and `checkpoints/smoke/`, both gitignored)

- Reads `data/splits/train.csv`, `val.csv`, `test.csv` (columns `id,text,label`, via `fnsl.data.load_run_splits`; with `--smoke`, the shared subsets of `config.SMOKE_SIZES`) and `data/splits/split_report.json` (dataset name written into the metrics JSON).
- `results/metrics/llm_zeroshot.json` (`run_zero_shot` → `metrics.save_results`). Standard per-model keys: `model` = "llm_zeroshot", `dataset`, `n_test` (rows actually scored: 50 by default, all with `--n-test 0`), `macro_f1`, `accuracy`, `per_class_f1` {`Bearish`, `Bullish`, `Neutral`}, `seed`, `train_time_s` (0.0, no training), `params`, `git_commit` (short hash, `-dirty` if `src/` or `scripts/` differ from HEAD, untracked files included), `timestamp` (UTC). Extra keys: `smoke`, `confusion_matrix` (rows = true label, columns = predicted, classes 0-2 only, so unparseable answers are missing), `n_unparseable`, `inference_time_s`, `unparseable_examples` (up to 10 distinct raw answers), `most_common_outputs` (up to 10 [answer, count] pairs). `params`: `model_name`, `backend`, `load_in_4bit`, `n_test`, `test_subset_seed`, `prompt_version`, `system_prompt`, `max_new_tokens`, `decoding` ("greedy"), `batch_size`, `inference_max_seq_length`.
- `results/predictions/llm_zeroshot.csv`: `id,text,label,pred`; `id` matches `data/splits/test.csv` so McNemar can pair rows; `pred` is 0, 1, 2 or -1 (UNPARSEABLE).
- `results/figures/llm_zeroshot_confusion.png`: confusion-matrix heatmap; when some answers are unparseable, its title says how many are not shown ("[SMOKE]" is added for smoke runs).
- `checkpoints/llm_lora_adapter/` (`--adapter-dir`, gitignored). As written by the `hf` path in the smoke run: `adapter_config.json` (PEFT: `base_model_name_or_path`, `r`, `lora_alpha`, `lora_dropout`, `target_modules` in arbitrary order, `bias`, `task_type`, ...), `adapter_model.safetensors` (the A and B matrices only), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, a PEFT `README.md` stub, and `train_info.json` (identical to `llm_lora_train.json`). The Unsloth backend's file list was not checked. `read_adapter_info` requires `adapter_config.json` and `train_info.json` and checks `prompt_version`. `checkpoints/llm_lora_trainer/` is TRL's `output_dir`; with `save_strategy="no"` it stays empty.
- `results/metrics/llm_lora_train.json` (`run_train` → `make_train_record`): a training log, `record_type` = "training_log", deliberately without `macro_f1`, so `metrics.load_metrics` skips it. Keys: `model` = "llm_lora_train", `record_type`, `base_model`, `backend`, `load_in_4bit`, `prompt_version`, `lora` {`r`, `alpha`, `dropout`, `target_modules`}, `trainable_params`, `total_params`, `trainable_pct` (percent, 3 decimals), `max_steps`, `learning_rate`, `lr_schedule` ("linear"), `warmup_steps`, `batch_size`, `grad_accum`, `effective_batch_size`, `max_seq_length`, `n_train`, `n_train_too_long`, `n_val_eval`, `eval_steps`, `train_loss` [{`step`, `loss`}] (each point = average loss per update over the last `logging_steps` updates), `val_loss` [{`step`, `eval_loss`}], `val_loss_start`, `val_loss_end`, `train_time_s`, `seed`, `smoke`, `adapter_dir` (repo-relative), `git_commit`, `timestamp`.
- `results/metrics/llm_lora.json` (`run_eval` → `metrics.save_results`): the standard keys with `model` = "llm_lora" and `train_time_s` = the LoRA training time from `train_info.json`; extra keys `smoke`, `confusion_matrix`, `n_unparseable`, `inference_time_s`, `val_loss_start`, `val_loss_end`, `unparseable_examples`, `most_common_outputs`. `params`: `base_model` (from `adapter_config.json`), `backend`, `train_backend`, `load_in_4bit`, `r`, `alpha`, `dropout`, `target_modules` (sorted), `max_steps`, `learning_rate`, `effective_batch_size`, `n_test`, `test_subset_seed`, `prompt_version`, `max_new_tokens`, `decoding`, `inference_max_seq_length`, `adapter_dir`. Check `backend` and `train_backend` before claiming Unsloth.
- `results/predictions/llm_lora.csv` and `results/figures/llm_lora_confusion.png`: same formats as for zero-shot.
- MLflow (non-smoke runs, unless `--no-mlflow`): runs named "llm_zeroshot", "llm_lora_train" and "llm_lora" in experiment "fin-news-sentiment-lab", stored in the SQLite file `mlruns/mlflow.db` (gitignored), with params plus test macro-F1/accuracy or per-step train and val loss (and the training log as an artifact).

**RAG: `src/fnsl/rag/`, `scripts/build_rag.py`, `scripts/eval_rag.py`**

- Input `data/rag_corpus/*.txt`: line 1 `Source: <URL> (retrieved <date>)`, then `Title:`, `Date:` and `Publisher:` lines with no blank line between them, then a blank line and the statement text. `load_corpus` returns one LangChain `Document` per file (files sorted by name; `SOURCES.md` is not a `.txt` file and is skipped): `page_content` = the text without the header lines; `metadata` = {`source`: file name, `url`, `title`, `date`, `publisher`}. Chunks add `chunk` (index within the file) and `start_index` (character offset). `data/rag_corpus/SOURCES.md` records URL, date and licence per file and the question-writing advice.
- Input `data/rag_eval.csv`: header exactly `question,source_file`; `source_file` must be a file name in `data/rag_corpus/`; a question that contains a comma must be in double quotes; rows starting with `EXAMPLE` are format demos (skipped unless `--include-examples`, which forces smoke output). It currently holds three EXAMPLE rows only.
- `cache/rag_index/<name>/` (default `default`, gitignored; written by `build_rag.py` → `ingest.save_index`): `index.faiss` (FAISS IndexFlatL2 vectors), `index.pkl` (LangChain docstore and id map, a pickle: load only files you wrote), `info.json` {`embedding_model`, `chunk_size`, `chunk_overlap`, `n_documents`, `n_chunks`, `sources`}.
- `results/metrics/rag_eval.json` (real `eval_rag.py` run only; does not exist yet): `kind` = "rag_eval", `n_questions`, `n_documents`, `embedding_model`, `chunk_overlap`, `k_values` (sorted {1, --k}, default [1, 3]), `configs` [{`chunk_size`, `n_chunks`, `hit@1`, `hit@3`}], `best_chunk_size`, `per_question` [{`chunk_size`, `question`, `gold`, `retrieved` (source file of each retrieved chunk, best first), `hit@1`, `hit@3`}], `smoke` (false), `include_examples`, `git_commit`, `timestamp`. It is not a per-model metrics file, so `metrics.load_metrics` skips it; `scripts/make_results_table.py` reads it through `load_result` (which ignores it if `smoke` is true) and passes it to `reporting.rag_section` and `reporting.resume_numbers_md`.
- `results/smoke/metrics/rag_eval.json`: same keys with `smoke` = true (gitignored); never used for README or resume numbers.
