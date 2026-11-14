# Mathematics, part 3: Keras BiLSTM and DistilBERT (section 8.5–8.6)

Part of the [project guide](../PROJECT_GUIDE.md#8-the-mathematics-part-by-part) for fin-news-sentiment-lab, Days 4-6 of `PROJECT_PLAN.md`. Checked against the code at commit `98a9d42`; the one later code change (`aaf142e`) makes the Unsloth resume claim require an adapter trained with `--backend unsloth`.

> **Numbers rule.** A *result* is copied from the results file named next to it. Every other number is a setting read from the code, a check you can rerun (named as such), or arithmetic shown in full. Models that have only had a `--smoke` run have no results, and none are stated.

Each concept follows the same pattern: the intuition, the mathematics (every symbol defined), how this code implements it, and why it matters or where it can go wrong. Formulas use GitHub's math rendering.

## Contents

- [8.5 Keras BiLSTM classifier](#85-keras-bilstm-classifier)
  - [8.5.1 Text standardisation (standardize_text)](#851-text-standardisation-standardize_text)
  - [8.5.2 Vocabulary, max_tokens and the unknown-word token](#852-vocabulary-max_tokens-and-the-unknown-word-token)
  - [8.5.3 Integer encoding, padding to seq_len, and masking](#853-integer-encoding-padding-to-seq_len-and-masking)
  - [8.5.4 Embedding lookup](#854-embedding-lookup)
  - [8.5.5 The LSTM cell](#855-the-lstm-cell)
  - [8.5.6 Bidirectional wrapper](#856-bidirectional-wrapper)
  - [8.5.7 Dropout](#857-dropout)
  - [8.5.8 Dense layer and softmax](#858-dense-layer-and-softmax)
  - [8.5.9 Loss: sparse categorical cross-entropy with balanced class weights](#859-loss-sparse-categorical-cross-entropy-with-balanced-class-weights)
  - [8.5.10 Adam optimiser](#8510-adam-optimiser)
  - [8.5.11 Early stopping on validation loss](#8511-early-stopping-on-validation-loss)
  - [8.5.12 Default hyperparameters (scripts/train_keras.py → parse_args)](#8512-default-hyperparameters-scriptstrain_keraspy--parse_args)
  - [8.5.13 Evaluation protocol, outputs and reproducibility](#8513-evaluation-protocol-outputs-and-reproducibility)
  - [8.5.14 Plan questions (Day 4)](#8514-plan-questions-day-4)
  - [8.5.15 What is left to do (Keras)](#8515-what-is-left-to-do-keras)
- [8.6 DistilBERT fine-tuning with an explicit PyTorch loop](#86-distilbert-fine-tuning-with-an-explicit-pytorch-loop)
  - [8.6.1 WordPiece tokenisation, special tokens and max_length](#861-wordpiece-tokenisation-special-tokens-and-max_length)
  - [8.6.2 Dynamic padding, the attention mask and the DataLoader](#862-dynamic-padding-the-attention-mask-and-the-dataloader)
  - [8.6.3 Token and position embeddings](#863-token-and-position-embeddings)
  - [8.6.4 Scaled dot-product attention](#864-scaled-dot-product-attention)
  - [8.6.5 Multi-head attention](#865-multi-head-attention)
  - [8.6.6 Feed-forward network](#866-feed-forward-network)
  - [8.6.7 Residual connections and LayerNorm](#867-residual-connections-and-layernorm)
  - [8.6.8 What DistilBERT is](#868-what-distilbert-is)
  - [8.6.9 Classification head](#869-classification-head)
  - [8.6.10 Cross-entropy loss](#8610-cross-entropy-loss)
  - [8.6.11 The explicit training loop, step by step (train_one_epoch)](#8611-the-explicit-training-loop-step-by-step-train_one_epoch)
  - [8.6.12 AdamW with decoupled weight decay](#8612-adamw-with-decoupled-weight-decay)
  - [8.6.13 Linear warmup, then linear decay of the learning rate](#8613-linear-warmup-then-linear-decay-of-the-learning-rate)
  - [8.6.14 Gradient clipping by global norm](#8614-gradient-clipping-by-global-norm)
  - [8.6.15 Mixed precision (float16) and dynamic loss scaling](#8615-mixed-precision-float16-and-dynamic-loss-scaling)
  - [8.6.16 Model selection on validation macro-F1, then one test evaluation](#8616-model-selection-on-validation-macro-f1-then-one-test-evaluation)
  - [8.6.17 --eval-only, training_info.json and reproducibility](#8617---eval-only-training_infojson-and-reproducibility)
  - [8.6.18 Model card generator and its refusal gates (scripts/push_to_hub.py)](#8618-model-card-generator-and-its-refusal-gates-scriptspush_to_hubpy)
  - [8.6.19 Default hyperparameters (scripts/train_distilbert.py → parse_args, src/fnsl/torch_model.py)](#8619-default-hyperparameters-scriptstrain_distilbertpy--parse_args-srcfnsltorch_modelpy)
  - [8.6.20 Plan questions (Days 5 and 6)](#8620-plan-questions-days-5-and-6)
  - [8.6.21 What is left to do (DistilBERT)](#8621-what-is-left-to-do-distilbert)
- [Files this part reads and writes](#files-this-part-reads-and-writes)

---

## 8.5 Keras BiLSTM classifier

> **Status: code complete, smoke-tested only.** `python scripts/train_keras.py --smoke` has run on CPU with the fixed tiny subsets (200 train, 64 val, 64 test rows; `SMOKE_SIZES` in `src/fnsl/config.py`) for 1 epoch. It wrote only to the gitignored `results/smoke/` and `checkpoints/smoke/`. The full run has not happened: `results/metrics/keras_bilstm.json` does not exist, and the README results table says "not run yet". Nothing in this section is a result of this model.

**Files.** `src/fnsl/keras_model.py` (text cleaning, vectoriser, model, class weights, prediction, loading), `scripts/train_keras.py` (command line, training with early stopping, one test prediction, results files), `tests/test_keras_model.py` (heavy tests: they need TensorFlow, so CI skips them). Versions in the project venv: TensorFlow 2.21.0, Keras 3.15.1. A run records both in `params`.

**Pipeline and shapes** for a batch of $`B`$ texts, with the defaults of `scripts/train_keras.py`:

```
raw strings                                   (B,)
 -> TextVectorization (standardize_text,
    whitespace split, vocabulary lookup)      (B, 64)        int ids, 0 = padding
 -> Embedding(V, 128, mask_zero=True)         (B, 64, 128)   plus a mask (B, 64)
 -> Bidirectional(LSTM(64)), concatenated     (B, 128)
 -> Dropout(0.3), training only               (B, 128)
 -> Dense(3, softmax)                         (B, 3)         class probabilities
```

$`V`$ is the vocabulary size. The whole pipeline, text cleaning included, sits inside one Keras model. It takes raw strings in and gives probabilities out.

### 8.5.1 Text standardisation (`standardize_text`)

**Intuition.** Before words are counted, every tweet is put into one standard form. "Beats", "beats!" and "beats" become the same token, and links disappear.

**Math.** Standardisation is a fixed function, applied in exactly the same way during training and prediction:

```math
s(x) = f_{\text{empty}}\Big(f_{\text{punct}}\big(f_{\text{url}}(\operatorname{lower}(x))\big)\Big)
```

$`x`$ is the raw text and $`\operatorname{lower}`$ lower-cases it. $`f_{\text{url}}`$ replaces every URL with a space, and $`f_{\text{punct}}`$ deletes a fixed set of ASCII punctuation characters. $`f_{\text{empty}}`$ replaces a result that is empty or only whitespace with the placeholder word `emptytext`.

**In this code.** `src/fnsl/keras_model.py` → `standardize_text` runs inside the model on `tf.string` tensors:
1. `tf.strings.lower`.
2. Replace every URL matching `URL_REGEX = r"https?://\S+"` with a space. This runs *before* punctuation removal. Otherwise `://` and `/` vanish first, and each link becomes one unique junk "word".
3. Delete the characters in `PUNCTUATION_REGEX`. This is the same ASCII set that Keras's built-in `lower_and_strip_punctuation` removes (compared with the installed Keras source). It includes the dollar sign, the percent sign and the apostrophe, so `$AAPL` becomes `aapl`. Emoji are not in the set and stay as tokens.
4. If nothing but whitespace is left (for example ":)"), return `EMPTY_TEXT = "emptytext"`.

The decorator `@keras.saving.register_keras_serializable(package="fnsl")` registers the function. A saved `.keras` file can then refer to it by name, and `load_model` (in the same module, so the registration has already run) can rebuild the layer.

Why the URL step exists (docstring; `DEBUG_LOG.md`, Day 4): 46% of the train tweets contain a URL. With Keras's default cleaning, 4,292 of the 21,034 distinct train tokens were link fragments such as `httpstcobd0xbfgjkt`. Both counts were re-checked on `data/splits/train.csv` while writing this guide.

Why the placeholder exists: an empty text would be all padding, which means a fully masked row. On a GPU, Keras uses the fast cuDNN LSTM kernel. Its TensorFlow backend then asserts that every row has at least one real token and has padding only at the end (`_assert_valid_mask` in the installed `keras/src/backend/tensorflow/rnn.py`). No training text becomes empty after cleaning (checked on `data/splits/train.csv`), so `emptytext` is not in the vocabulary. An empty text at prediction time is therefore encoded as one unknown-word id followed by padding (checked: ":)" gives ids `[1, 0, 0, …]`).

**Why it matters here.** Cleaning inside the model means training and prediction cannot drift apart (no "training/serving skew"), and the saved model cleans new text the same way. Interview line: "I read the learned vocabulary, found link junk, and fixed the preprocessing rather than the model."

### 8.5.2 Vocabulary, `max_tokens` and the unknown-word token

**Intuition.** The network can only use words that have an id. The vocabulary is a list of training words ranked by frequency. Every other word maps to one shared "unknown" id, `[UNK]`.

**Math.** Let $`\operatorname{count}(w)`$ be how often word $`w`$ occurs in the standardised train texts. Sort the words by $`\operatorname{count}(w)`$, largest first, and keep the first $`M - 2`$, where $`M`$ is `max_tokens`. Let $`r(w) = 0, 1, 2, \dots`$ be the position of a kept word in that list. Then:

```math
\operatorname{id}(w) = \begin{cases} 2 + r(w) & \text{if } w \text{ is a kept word} \\ 1 & \text{otherwise (the [UNK] id)} \end{cases}
```

Id 0 is reserved for padding. So the vocabulary size is $`V = \min(M,\ D + 2)`$, where $`D`$ is the number of distinct train words.

**In this code.** `src/fnsl/keras_model.py` → `build_text_vectorizer(train_texts, max_tokens=20000, seq_len=64)` creates `keras.layers.TextVectorization(max_tokens=max_tokens, standardize=standardize_text, split="whitespace", output_mode="int", output_sequence_length=seq_len)` and calls `adapt()` on the **train texts only**. It passes a NumPy object array (`to_string_array`). The code comments give the reasons: `adapt()` treats a pandas Series as a stream of one-string batches, and Keras 3 rejects plain lists. Vocabulary entry 0 is `""` (padding), entry 1 is `[UNK]`, and then come the words in order of falling frequency.

The comment above `DEFAULT_MAX_TOKENS` in `scripts/train_keras.py` records 16,742 distinct train words after cleaning. So the default cap of 20,000 keeps every one of them, and $`V = 16{,}742 + 2 = 16{,}744`$. This was re-checked on `data/splits/train.csv`, and a real run records it as `params.vocab_size`. Building the vocabulary from train only is a leakage guard: `test_vectorizer_is_adapted_on_train_texts_only` checks that val-only words are absent and map to id 1.

**Pitfall: `[UNK]` is never trained.** Every train word has its own id, so no training text contains id 1. The `[UNK]` embedding row therefore never receives a gradient and keeps its random starting value. That value comes from Keras's default `"uniform"` initialiser, uniform on $`[-0.05, 0.05]`$. Yet at prediction time every word first seen in val or test maps to this row. The same comment measures that 10.9% of the word occurrences in the val texts are unknown, and that 63.8% of val texts contain at least one unknown word. Both figures were re-checked.

The comment suggests a smaller cap instead. Then the rarest train words become `[UNK]` during training, and the network learns a vector for "unknown". For example, 9,747 train words occur only once, so `--max-tokens 6997` keeps exactly the $`16{,}742 - 9{,}747 = 6{,}995`$ words seen at least twice, plus ids 0 and 1. The comment says to try this only as a separately named run and to judge it on `val_macro_f1`.

### 8.5.3 Integer encoding, padding to `seq_len`, and masking

**Intuition.** A batch must be a rectangle, so every text becomes exactly 64 ids. Short texts are filled with 0s at the end, and long ones are cut. A mask tells the LSTM which positions are real, so the padding has no effect.

**Math.** A text with $`L`$ word ids $`(t_1, \dots, t_L)`$ and sequence length $`S`$ becomes

```math
\mathbf{x} = \begin{cases}
(t_1, \dots, t_L, \underbrace{0, \dots, 0}_{S - L}) & \text{if } L \le S \\
(t_1, \dots, t_S) & \text{if } L > S
\end{cases}
```

and the mask bit for position $`j = 1, \dots, S`$ is

```math
m_j = \begin{cases} 1 & \text{if } x_j \neq 0 \text{ (a real word)} \\ 0 & \text{if } x_j = 0 \text{ (padding)} \end{cases}
```

**In this code.** `output_sequence_length=seq_len` pads at the end and also cuts at the end, keeping the first $`S`$ words (both checked on a toy vocabulary). `src/fnsl/keras_model.py` → `build_bilstm_model` creates the `Embedding` with `mask_zero=True`, which produces $`m`$. The `Bidirectional(LSTM)` layer consumes the mask, and at a masked step the LSTM carries its previous state forward unchanged. Only id 0 is masked. An unknown word (id 1) is a real step, at which the LSTM reads the `[UNK]` vector.

Default `seq_len = 64`. The comment above `DEFAULT_SEQ_LEN` in `scripts/train_keras.py` records the train text lengths after cleaning: median 11 words, 95th percentile 20, 99th percentile 22, maximum 33. So no training text is cut (re-checked). Every run prints the number of cut texts and records it as `params.train_texts_truncated`. `test_padding_is_masked` copies identical weights into models with `seq_len` 8 and 16 and requires the same outputs, to within $`10^{-6}`$.

**Why it matters here.** Without the mask, a 5-word tweet would be followed by 59 padding steps. The forward LSTM would end on them, washing out what it read. With the mask, changing `seq_len` changes only speed, not predictions, as long as no text is cut (up to floating-point rounding).

### 8.5.4 Embedding lookup

**Intuition.** Each word id selects one row of a trainable table, and that row is the word's dense vector. Words used in similar ways tend to end up with similar vectors, because that lowers the loss.

**Math.** With table $`E \in \mathbb{R}^{V \times d}`$, the vector for position $`j`$ is row $`x_j`$ of $`E`$:

```math
\mathbf{e}_j = E_{x_j,:} = E^{\top}\mathbf{o}_{x_j}
```

$`\mathbf{o}_{x_j}`$ is the one-hot vector of length $`V`$ with a 1 at index $`x_j`$, and $`d`$ is the embedding size. Only the rows that occur in a batch receive a gradient.

**In this code.** `keras.layers.Embedding(input_dim=vectorizer.vocabulary_size(), output_dim=embed_dim, mask_zero=True)` with `embed_dim = 128`. It is randomly initialised (no pre-trained word vectors). With $`V = 16{,}744`$ the table holds $`16{,}744 \times 128 = 2{,}143{,}232`$ weights.

With the default settings on the current train split, the whole model has 2,242,435 parameters (`model.count_params()`, the total that `model.summary()` prints). That is 2,143,232 in the embedding, 98,816 in the BiLSTM and 387 in the dense layer; the vectoriser has none. So about 96% of the model is the embedding table. A real run records the total as `params.n_params`.

**Why it matters here.** Almost all parameters are word vectors, learned from 8,241 labelled tweets (`data/splits/split_report.json`), and many of them belong to words seen only once. That is the main overfitting risk, and the reason for the dropout and early stopping below.

### 8.5.5 The LSTM cell

**Intuition.** An LSTM reads one word vector at a time and keeps two running vectors: a cell state (long-term memory) and a hidden state (its current output). Three gates, each a number between 0 and 1 per unit, decide three things: how much old memory to keep, how much new information to write, and how much of the memory to output.

**Math.** At step $`t`$, with input $`\mathbf{x}_t \in \mathbb{R}^{d}`$ (the word vector), previous hidden state $`\mathbf{h}_{t-1} \in \mathbb{R}^{u}`$ and previous cell state $`\mathbf{c}_{t-1} \in \mathbb{R}^{u}`$:

```math
\begin{aligned}
\mathbf{i}_t &= \sigma(W_i \mathbf{x}_t + U_i \mathbf{h}_{t-1} + \mathbf{b}_i) && \text{input gate} \\
\mathbf{f}_t &= \sigma(W_f \mathbf{x}_t + U_f \mathbf{h}_{t-1} + \mathbf{b}_f) && \text{forget gate} \\
\mathbf{o}_t &= \sigma(W_o \mathbf{x}_t + U_o \mathbf{h}_{t-1} + \mathbf{b}_o) && \text{output gate} \\
\tilde{\mathbf{c}}_t &= \tanh(W_c \mathbf{x}_t + U_c \mathbf{h}_{t-1} + \mathbf{b}_c) && \text{candidate memory} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t && \text{new cell state} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) && \text{new hidden state}
\end{aligned}
```

The symbols:
- $`u`$ is the number of units.
- $`W_\ast \in \mathbb{R}^{u \times d}`$ and $`U_\ast \in \mathbb{R}^{u \times u}`$ are weight matrices, and $`\mathbf{b}_\ast \in \mathbb{R}^{u}`$ are biases, one set per gate.
- $`\sigma(z) = 1/(1 + e^{-z})`$ is the logistic sigmoid, with output between 0 and 1. $`\tanh`$ outputs values between −1 and 1.
- $`\odot`$ is element-wise multiplication, and $`\mathbf{h}_0 = \mathbf{c}_0 = \mathbf{0}`$.

With masking, if $`m_t = 0`$ then $`\mathbf{h}_t = \mathbf{h}_{t-1}`$ and $`\mathbf{c}_t = \mathbf{c}_{t-1}`$. One LSTM has $`4(ud + u^2 + u)`$ parameters.

**In this code.** `keras.layers.LSTM(lstm_units)` with `lstm_units = 64` and $`d = 128`$. It uses the Keras defaults (read with `get_config()`):
- `activation="tanh"` and `recurrent_activation="sigmoid"`;
- `use_bias=True` with `unit_forget_bias=True`: the forget-gate bias starts at 1 (checked on the built layer), so $`\mathbf{f}_t \approx \sigma(1) \approx 0.73`$ at the start of training and the cell keeps most of its memory;
- `kernel_initializer="glorot_uniform"` and `recurrent_initializer="orthogonal"`;
- no dropout inside the cell;
- `return_sequences=False`, so only the final hidden state is returned.

Parameters per direction: $`4(64 \cdot 128 + 64^2 + 64) = 49{,}408`$. These settings also let Keras use the fast cuDNN kernel on a GPU.

**Why it matters here.** In a plain RNN, the gradient that flows back through time is multiplied at every step by a weight matrix and by the derivative of a squashing function, so it tends to vanish. In an LSTM, the direct path through the cell state has $`\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1} = \operatorname{diag}(\mathbf{f}_t)`$ (ignoring the paths through the gates). While the forget gate stays near 1, the gradient passes back almost unchanged.

The tweets are short (median 11 words), so the bigger benefit here is that the LSTM reads words in order and keeps a memory. The TF-IDF baseline sees only single words and adjacent word pairs (`ngram_range` [1, 2] in `results/metrics/tfidf_logreg.json`). It cannot link words that are further apart, for example a negation and its verb in "did not, despite the rally, beat estimates".

### 8.5.6 Bidirectional wrapper

**Intuition.** One LSTM reads the tweet left to right. A second LSTM, with its own weights, reads it right to left. Their final states are joined, so the summary has read the text from both ends.

**Math.** Let $`\overrightarrow{\mathbf{h}}_t`$ be the forward LSTM's state after word $`t`$. Let $`\overleftarrow{\mathbf{h}}_t`$ be the backward LSTM's state after it has read words $`L, L-1, \dots, t`$. With $`L`$ real (unmasked) words, the text summary is

```math
\mathbf{s} = \big[\, \overrightarrow{\mathbf{h}}_{L} \,;\, \overleftarrow{\mathbf{h}}_{1} \,\big] \in \mathbb{R}^{2u}
```

where $`[\,\cdot\,;\,\cdot\,]`$ means concatenation. With $`u = 64`$, $`\mathbf{s}`$ has 128 numbers.

**In this code.** `keras.layers.Bidirectional(keras.layers.LSTM(lstm_units), name="bilstm")` with the default `merge_mode="concat"`. It has 98,816 parameters (two directions of 49,408), and the backward copy runs with `go_backwards=True`. Padding sits at the end and is masked. So the forward LSTM's final state is the one after the last real word. The backward LSTM effectively starts at the last real word, because the masked padding steps leave its zero state unchanged.

**Why it matters here.** In a headline the decisive word can be anywhere ("Shares fall after..." vs "...sending shares lower"). A single forward LSTM's final state tends to carry more of the end of the text than of its start. The backward pass balances that.

### 8.5.7 Dropout

**Intuition.** During training, each of the 128 summary numbers is set to 0 with probability 0.3, independently for each example (so about 30% on average). The classifier therefore cannot rely on any single number. At prediction time nothing is dropped.

**Math.** With dropout rate $`p`$, during training each component $`s_j`$ of $`\mathbf{s}`$ becomes

```math
\tilde{s}_j = \frac{z_j \, s_j}{1 - p}, \qquad z_j \sim \text{Bernoulli}(1 - p) \text{ independently}
```

Dividing by $`1 - p`$ keeps the expected value unchanged, $`\mathbb{E}[\tilde{s}_j] = s_j`$ ("inverted dropout"). At prediction time $`\tilde{s}_j = s_j`$.

**In this code.** `keras.layers.Dropout(dropout)` with `dropout = 0.3`. It sits only between the BiLSTM summary and the dense layer; there is no dropout inside the LSTM. Keras applies it only in training mode, inside `fit`. The validation pass inside `fit` and `predict_proba` (which calls `model.predict`) run in inference mode.

**Pitfall.** The logged training loss is measured with dropout on, and it is an average over the epoch while the weights are still improving. The validation loss is measured with dropout off, at the end of the epoch. So the training loss can be above the validation loss, especially in early epochs. That is not a bug.

### 8.5.8 Dense layer and softmax

**Intuition.** A linear layer turns the 128-number summary into 3 scores (logits). Softmax turns the scores into probabilities that sum to 1.

**Math.**

```math
\mathbf{z} = W \tilde{\mathbf{s}} + \mathbf{b}, \qquad \hat{p}_k = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}} \quad (k = 0, 1, \dots, K - 1)
```

$`W \in \mathbb{R}^{K \times 2u}`$ and $`\mathbf{b} \in \mathbb{R}^{K}`$ are learned. There are $`K = 3`$ classes, numbered as in `config.LABELS` (0 = Bearish, 1 = Bullish, 2 = Neutral). $`\hat{p}_k`$ is the predicted probability of class $`k`$, and the predicted label is $`\hat{y} = \arg\max_k \hat{p}_k`$.

**In this code.** `keras.layers.Dense(num_classes, activation="softmax", name="softmax")`, with $`3 \cdot 128 + 3 = 387`$ parameters. `scripts/train_keras.py` predicts with `predict_proba(...).argmax(axis=1)`. No decision thresholds are tuned.

**Why it matters here.** Macro-F1 is computed on these arg-max labels. No calibration step is applied, and the probabilities are used only for the arg-max.

### 8.5.9 Loss: sparse categorical cross-entropy with balanced class weights

**Intuition.** Cross-entropy charges the model $`-\log`$ of the probability it gave the true class. The charge is close to 0 when the model is confident and right, and large when it is confident and wrong. Class weights make a mistake on a rare class cost more, so the model cannot win by mostly predicting Neutral.

**Math.** For example $`i`$ with integer label $`y_i \in \{0, 1, 2\}`$ ("sparse" means the label is an integer, not a one-hot vector):

```math
\ell_i = -\log \hat{p}_{i, y_i}
```

The balanced weight of class $`k`$ is computed from the training labels:

```math
w_k = \frac{N}{K \, n_k}
```

$`N`$ is the number of training examples, $`n_k`$ the number in class $`k`$, and $`K = 3`$. Every class then carries the same total weight, $`n_k w_k = N / K`$. Also $`\sum_{i=1}^{N} w_{y_i} = N`$, so the weights average exactly 1 over the training set. Keras's loss for a batch of $`B`$ examples (its default reduction, "sum over batch size") is

```math
\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i \in \text{batch}} w_{y_i} \, \ell_i
```

The epoch value Keras logs is the batch-size-weighted mean of these batch losses. That equals $`\frac{1}{N}\sum_i w_{y_i}\ell_i`$ over the epoch, with each $`\ell_i`$ computed with the weights that were current at its batch.

**In this code.** `build_bilstm_model` compiles with `loss="sparse_categorical_crossentropy"`. `src/fnsl/keras_model.py` → `balanced_class_weights(labels)` computes `len(labels) / (num_classes * np.bincount(labels))`. This is the same formula as scikit-learn's `class_weight="balanced"`, and the function raises `ValueError` if a class has no examples. With `--class-weight balanced` (the default), `scripts/train_keras.py` → `main` passes these weights to `model.fit(class_weight=...)`, which Keras turns into one sample weight per example. It also attaches the same train-based weights to the validation data. So `val_loss`, the quantity early stopping watches, is the same weighted objective. With the train counts in `data/splits/split_report.json` (Bearish 1,235, Bullish 1,666, Neutral 5,340; $`N = 8{,}241`$):

```math
\begin{aligned}
w_{\text{Bearish}} &= \frac{8241}{3 \cdot 1235} \approx 2.224 \\
w_{\text{Bullish}} &= \frac{8241}{3 \cdot 1666} \approx 1.649 \\
w_{\text{Neutral}} &= \frac{8241}{3 \cdot 5340} \approx 0.514
\end{aligned}
```

Two details were checked against the installed Keras (its source and a toy run):
- The weighted sum is divided by the batch size $`B`$, not by the sum of the weights.
- The compiled metric `"accuracy"` ignores sample weights. So `accuracy` and `val_accuracy` in the saved `history` are plain (unweighted) accuracies, while `loss` and `val_loss` are weighted.

**Why it matters here.** Neutral is 64.8% of the training split (`data/splits/split_report.json`). With an unweighted loss, the network can lower its loss by leaning towards Neutral. That costs recall on Bearish and Bullish, and therefore macro-F1, the project's headline metric, in which all three classes count equally. The script's comment suggests checking the choice with `--class-weight none --run-name keras_bilstm_unweighted` and comparing **validation** macro-F1, never test scores.

### 8.5.10 Adam optimiser

**Intuition.** Adam is gradient descent in which every weight gets its own step size. It keeps a running mean of each weight's gradient (the direction, like momentum) and a running mean of its squared gradient (the typical size). It divides the first by the square root of the second.

**Math.** At step $`t = 1, 2, \dots`$ with gradient $`\mathbf{g}_t = \nabla_\theta \mathcal{L}_{\text{batch}}`$:

```math
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^{2} \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^{t}}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
```

The symbols:
- $`\theta`$ stands for all weights, and $`\eta`$ is the learning rate.
- $`\beta_1`$ and $`\beta_2`$ are the decay rates of the two running means, which start at $`\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}`$.
- Squares, square roots and the division act element by element.
- The factors $`1 - \beta_1^{t}`$ and $`1 - \beta_2^{t}`$ correct the bias towards 0 of the first few averages.
- $`\epsilon`$ prevents division by zero.

**In this code.** `keras.optimizers.Adam(learning_rate=lr)` with `lr = 1e-3` (`--lr`) and the Keras defaults: $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, $`\epsilon = 10^{-7}`$, no AMSGrad, no weight decay (read from `get_config()`). The learning rate is constant (no schedule), and early stopping decides how long training lasts. There is one update per batch of 64, so $`\lceil 8241 / 64 \rceil = 129`$ updates per epoch on the full train split.

Deviation from the textbook form: Keras's `Adam.update_step` computes

```math
\theta_t = \theta_{t-1} - \eta \, \frac{\sqrt{1 - \beta_2^{t}}}{1 - \beta_1^{t}} \cdot \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}
```

This is the textbook update with $`\epsilon`$ replaced by $`\epsilon / \sqrt{1 - \beta_2^{t}}`$. That value is about $`3 \times 10^{-6}`$ at $`t = 1`$ and falls towards $`10^{-7}`$, so the difference is negligible here.

**Why it matters here.** The rows of the embedding table that do not occur in a batch get a zero gradient. Adam starts every running average at 0, so a row whose gradient is always 0 keeps $`\mathbf{m}_t = \mathbf{v}_t = \mathbf{0}`$ and is never changed. That is exactly why the `[UNK]` row keeps its initial values.

### 8.5.11 Early stopping on validation loss

**Intuition.** Keep training while the model improves on data it does not train on. Stop once it stops improving, and roll the weights back to the best epoch.

**Math.** Let $`\mathcal{L}^{\text{val}}_e`$ be the class-weighted validation loss after epoch $`e`$, $`P`$ the patience and $`E_{\max}`$ the maximum number of epochs. Epoch $`e`$ is an improvement if $`\mathcal{L}^{\text{val}}_e < \min_{j < e} \mathcal{L}^{\text{val}}_j`$, with no minimum improvement size; the first epoch always counts as one. Training stops after the first epoch that completes $`P`$ consecutive epochs without improvement, or after epoch $`E_{\max}`$, whichever comes first. Call that epoch $`e_{\text{stop}}`$. The weights kept are those of

```math
e^{\ast} = \arg\min_{1 \le e \le e_{\text{stop}}} \mathcal{L}^{\text{val}}_e
```

with the earlier epoch kept if two are tied.

**In this code.** `scripts/train_keras.py` → `main` uses `keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)`, with at most `--epochs 10` (1 with `--smoke`). `min_delta` is the default 0. In the installed Keras 3.15 source, the best weights are restored in `on_train_end`. So they are restored even when all 10 epochs run without an early stop.

The script records `epochs_trained`, `stopped_early`, `best_epoch` (Keras counts from 0; the script adds 1) and the per-epoch `history`. It then computes `val_macro_f1` with the restored weights. `test_train_keras_keeps_the_best_val_epoch` runs a smoke training with `--epochs 4 --patience 1 --lr 0.03`, so that the validation loss rises after its minimum. It then checks that the saved model is the arg-min epoch, not the last one: the saved model's recomputed weighted validation loss must equal the history value at `best_epoch`, and its validation macro-F1 must equal `params.val_macro_f1`.

**Why it matters here.** Early stopping prevents overfitting (see the plan question below). **Pitfall:** the epoch is chosen on the validation split, so the validation score is slightly optimistic. That is why the test split is kept separate and predicted only once.

### 8.5.12 Default hyperparameters (`scripts/train_keras.py` → `parse_args`)

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 10 (1 with `--smoke`) | maximum epochs; early stopping can end sooner |
| `--batch-size` | 64 | examples per Adam update |
| `--max-tokens` | 20000 | vocabulary cap, including ids 0 and 1 |
| `--seq-len` | 64 | token ids per text after padding or cutting |
| `--embed-dim` | 128 | word-vector size $`d`$ |
| `--lstm-units` | 64 | units per direction $`u`$; summary size $`2u = 128`$ |
| `--dropout` | 0.3 | dropout rate $`p`$ before the dense layer |
| `--lr` | 1e-3 | Adam learning rate $`\eta`$ |
| `--patience` | 2 | epochs without a lower `val_loss` before stopping |
| `--class-weight` | `balanced` | or `none` |
| `--seed` | 42 (`config.SEED`) | run name `keras_bilstm`; other seeds give `keras_bilstm_seed<N>` |
| `--run-name` | none | give every non-default experiment its own name |
| `--no-mlflow` | off | MLflow logging is also off for `--smoke` |

### 8.5.13 Evaluation protocol, outputs and reproducibility

- **Model selection uses validation only.** The best epoch is chosen by the weighted `val_loss`. The macro-F1 of the restored weights on the validation split is stored as `params.val_macro_f1`.
- **The test split is predicted exactly once,** after training, with the best-epoch weights. `src/fnsl/metrics.py` → `save_results` writes three files:
  - `results/metrics/<run>.json`;
  - `results/predictions/<run>.csv` (`id,text,label,pred`, with the same ids and order as `data/splits/test.csv`, so McNemar's test can pair models);
  - `results/figures/<run>_confusion.png`.

  The model (vocabulary and weights) is saved to `checkpoints/<run>.keras` (gitignored). Reload it with `src/fnsl/keras_model.py` → `load_model`.
- **Seeds.** `keras.utils.set_random_seed(seed)` and `config.set_seed(seed)` run before the vocabulary and model are built. This fixes the weight initialisation, the dropout masks and the shuffle order of each epoch. The code comment says CPU reruns with the same seed repeat exactly, while the GPU cuDNN LSTM kernels are not bit-for-bit deterministic.
- **`train_time_s`** is the wall-clock time of `model.fit`, including the validation pass after each epoch.
- **Smoke runs** use the deterministic tiny subsets from `data.load_run_splits(smoke=True)` and 1 epoch. They write only under `results/smoke/` and `checkpoints/smoke/`, switch MLflow off, and print "[SMOKE: not a result]".

### 8.5.14 Plan questions (Day 4)

**What do the BiLSTM layers do?**
1. The vectoriser cleans the text and turns it into 64 word ids, with 0 for padding.
2. The embedding layer replaces each id with a learned 128-number vector.
3. The forward LSTM reads these vectors left to right. At every word its gates decide what to forget from its memory, what to write into it and what to output. Its final 64-number state summarises the text in order.
4. The backward LSTM does the same from right to left.
5. The two final states are concatenated into one 128-number summary of the tweet.
6. Dropout zeroes each of those numbers with probability 0.3 during training.
7. A dense layer with softmax turns the summary into probabilities for Bearish, Bullish and Neutral.

The padding is masked, so it has no effect.

**What does early stopping prevent?** Overfitting. After a few epochs, the network can keep lowering its training loss by memorising the training headlines; it has over 2 million embedding weights for 8,241 tweets. Meanwhile its loss on the validation split, which it never trains on, starts to rise. Early stopping ends training once the validation loss has not improved for 2 epochs, and it restores the weights of the best epoch. So the saved model is the one that did best on unseen text. It also saves compute.

### 8.5.15 What is left to do (Keras)

1. Run `python scripts/train_keras.py`. The README says a laptop CPU is enough, and `notebooks/colab_runner.ipynb` has a fallback cell. Check the printed best epoch and validation macro-F1.
2. Optional variants, each with its own `--run-name` and judged on `val_macro_f1` only:
   - `--class-weight none --run-name keras_bilstm_unweighted` (the script's own suggestion);
   - a smaller vocabulary cap that trains `[UNK]`, for example `--max-tokens 6997`.

   Each variant also predicts the test split and writes its own results files. `scripts/compare_models.py` pairs every predictions file in `results/predictions/`, and the README table lists every metrics file. So decide which variants to keep before step 3.
3. Run `python scripts/compare_models.py` and `python scripts/make_results_table.py --update-readme --count-tests`. Then run `pytest -m "not heavy"` and `ruff check .`, and commit the new files in `results/`. The heavy tests in `tests/test_keras_model.py` are deselected in CI. They need no downloads and pass locally; rerun them with `pytest -m heavy tests/test_keras_model.py` before the real run.

## 8.6 DistilBERT fine-tuning with an explicit PyTorch loop

> **Status: code complete, smoke-tested only.** `python scripts/train_distilbert.py --smoke` has run on CPU (200 train, 64 val, 64 test rows, 1 epoch, batch 16). It wrote only to the gitignored `results/smoke/` and `checkpoints/smoke/`. The full fine-tune is meant for a Colab or Kaggle GPU (on CPU the script warns that it is slow) and has not been run. So `results/metrics/distilbert.json` does not exist, and the float16 / `GradScaler` code path has never executed on a GPU. Nothing in this section is a DistilBERT result.

**Files.**
- `src/fnsl/torch_model.py`: device, autocast, model loading, dataset and loader, truncation count, optimizer, schedule, `train_one_epoch`, `predict`.
- `scripts/train_distilbert.py`: command line, epoch loop, best-checkpoint saving, test evaluation, `--eval-only`.
- `scripts/push_to_hub.py`: model card and guarded upload.
- Tests: `tests/test_torch_model.py` is heavy; it uses a tiny random 1-layer DistilBERT and an in-memory tokenizer, so nothing is downloaded. `tests/test_model_card.py` is CPU only and runs in CI.

Versions in the project venv: torch 2.14.1 (CPU build), transformers 5.17.0. A run records the versions it used in `params`.

**Pipeline and shapes** for a batch of $`B`$ texts, where $`L \le 64`$ is the length of the longest text in the batch and $`d = 768`$:

```
raw strings
 -> WordPiece tokenizer (uncased), cut at 64 tokens   input_ids, attention_mask (Python lists)
 -> DataLoader + DataCollatorWithPadding              (B, L) tensors, padded per batch
 -> token + position embeddings, LayerNorm, dropout   (B, L, 768)
 -> 6 x [multi-head self-attention -> add & LayerNorm
         -> feed-forward -> add & LayerNorm]          (B, L, 768)
 -> vector at position 0 ([CLS])                      (B, 768)
 -> pre_classifier 768->768, ReLU, dropout 0.2,
    classifier 768->3                                 (B, 3) logits
 -> cross-entropy with the labels (training), arg-max (prediction)
```

### 8.6.1 WordPiece tokenisation, special tokens and `max_length`

**Intuition.** The model has a fixed vocabulary of 30,522 word pieces. A common word is one token. A rare word is split into known pieces, so almost nothing is ever "unknown".

**Math (the algorithm).**
1. The text is normalised (lower-cased, accents stripped, control characters removed) and split on whitespace and punctuation.
2. Each word is then segmented greedily, longest match first: take the longest prefix of the word that is in the vocabulary, then repeat on the remainder. Pieces after the first carry the prefix `##`.
3. If at some point no vocabulary piece matches the start of what is left, or the word is longer than 100 characters, the whole word becomes `[UNK]`.

The model input is

```math
\mathbf{x} = \big(\text{[CLS]},\ t_1, \dots, t_n,\ \text{[SEP]}\big), \qquad n + 2 \le \ell_{\max}
```

where $`t_1, \dots, t_n`$ are the word-piece ids of the text and $`\ell_{\max}`$ is `max_length`.

Toy example (tokenizer output, not a result), using `EXAMPLE_TEXT` from `scripts/push_to_hub.py`:
- input: `$AAPL beats earnings estimates and raises full-year guidance`
- tokens: `[CLS] $ aa ##pl beats earnings estimates and raises full - year guidance [SEP]`

**In this code.** `scripts/train_distilbert.py` → `train` loads `AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")`. In transformers 5 this is a `BertTokenizer` with `do_lower_case=True`. Its special ids are `[PAD]` 0, `[UNK]` 100, `[CLS]` 101 and `[SEP]` 102. The saved `tokenizer.json` shows a `BertNormalizer` with `lowercase: true` and a `WordPiece` model with prefix `##` and `max_input_chars_per_word` 100.

`src/fnsl/torch_model.py` → `make_loader` tokenises the whole split once with `truncation=True, max_length=max_length, return_token_type_ids=False`. DistilBERT has no segment embeddings, but this tokenizer returns `token_type_ids` unless told not to. When a text is cut, `[SEP]` stays as the last token (`test_loader_truncates_long_texts_and_keeps_sep`).

Default `--max-length 64`. The comment above `DEFAULT_MAX_LENGTH` records the token lengths of the 8,241 train texts, with `[CLS]` and `[SEP]` counted: median 25, 95th percentile 50, 99th percentile 57, maximum 82. Only 13 texts (0.16%) are longer than 64, and what gets cut is almost always the tail of a t.co link. These figures, and the link tails, were re-checked with the tokenizer while writing this guide. `count_truncated` records the count as `params.train_texts_truncated`.

**Pitfall (a difference from the Keras model).** DistilBERT receives the raw text, with URLs still in it. In a toy check, `https://t.co/Ab12Cd` became 11 word pieces; the real links are longer, and the 4,311 links in the train split take 12 to 20 pieces each (median 15, a check you can rerun with the tokenizer). The Keras pipeline deletes links. So the two models do not see identical inputs; be ready to say so.

### 8.6.2 Dynamic padding, the attention mask and the DataLoader

**Intuition.** Each batch is padded only up to its own longest text. A 0/1 attention mask tells every attention layer to ignore the padding.

**Math.** In a batch whose longest text has $`L`$ tokens, shorter rows are filled with `[PAD]` (id 0) up to $`L`$. The mask $`a_j \in \{0, 1\}`$ is 1 for a real token and 0 for padding. Inside attention it acts as an additive bias for every query position $`q`$ and key position $`j`$:

```math
M_{qj} = \begin{cases} 0 & \text{if } a_j = 1 \\ -\infty & \text{if } a_j = 0 \end{cases}
```

After the softmax, a padded key therefore gets weight exactly 0.

**In this code.** `src/fnsl/torch_model.py` → `TextDataset` returns one example as a dict of plain lists plus `"labels"`, with no padding. `make_loader` builds `DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer), generator=..., num_workers=num_workers)`, where the generator is a `torch.Generator` seeded with `seed`. The collator's default `padding=True` means "pad to the longest text in the batch", with `attention_mask` set to 0 at the padded positions (`test_loader_pads_each_batch_only_to_its_longest_text`).

Training uses `shuffle=True`, and the seed fixes the order (`test_shuffle_order_is_fixed_by_the_seed`). Validation and test use `shuffle=False`, so predictions come out in row order.

In the installed transformers (5.17.0), DistilBERT computes attention with PyTorch's `scaled_dot_product_attention`. This is `attn_implementation="sdpa"`, which transformers picks by default; the code does not set it. `create_bidirectional_mask` turns the 0/1 `attention_mask` into a boolean tensor of shape (B, 1, L, L), True where a query may attend to a key. SDPA treats False exactly like adding $`-\infty`$, which is the bias $`M`$ above (checked on a toy batch).

**Why it matters here.** The tweets are short, so padding every batch to 64 would waste compute, and attention cost grows with $`L^2`$. Without the mask, padding tokens would leak into every real token's representation.

### 8.6.3 Token and position embeddings

**Intuition.** Each token id gets a learned vector. A second learned vector, for the token's position (0, 1, 2, ...), is added to it, because attention on its own has no sense of word order.

**Math.**

```math
\mathbf{h}^{(0)}_j = \operatorname{Dropout}_{0.1}\Big(\operatorname{LN}\big(E_{\text{tok}}[x_j] + E_{\text{pos}}[j]\big)\Big)
```

for positions $`j = 0, \dots, L - 1`$. The symbols:
- $`E_{\text{tok}} \in \mathbb{R}^{30522 \times 768}`$ is the token table.
- $`E_{\text{pos}} \in \mathbb{R}^{512 \times 768}`$ has one row per position, so at most 512 tokens fit.
- $`E[x]`$ means row $`x`$ of $`E`$, and $`\operatorname{LN}`$ is LayerNorm (defined below).
- $`\mathbf{h}^{(0)}_j`$ is the input of the first transformer layer at position $`j`$.

**In this code.** The `Embeddings` module of transformers uses learned positions (`sinusoidal_pos_embds: false` in the checkpoint's `config.json`), LayerNorm with $`\varepsilon = 10^{-12}`$, and dropout 0.1. There is no segment (token-type) embedding, which is why `make_loader` drops `token_type_ids`. Both tables are pre-trained and are fine-tuned along with everything else.

**Why it matters here.** The position embeddings make word order visible to the transformer at any distance. A unigram bag of words cannot tell "beats estimates" from "estimates beats". The project's TF-IDF baseline also uses adjacent word pairs (`ngram_range` [1, 2] in `results/metrics/tfidf_logreg.json`), so it does catch this two-word case, but it loses order beyond adjacent pairs.

### 8.6.4 Scaled dot-product attention

**Intuition.** Every token forms three vectors: a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I pass on?"). A token's new vector is an average of all values, weighted by how well its query matches each key.

**Math.** Stack the $`L`$ token vectors of a layer as the rows of $`H \in \mathbb{R}^{L \times d}`$. For one head:

```math
\begin{aligned}
Q &= H W_Q + \mathbf{b}_Q \\
K &= H W_K + \mathbf{b}_K \\
V &= H W_V + \mathbf{b}_V
\end{aligned}
```

```math
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V
```

The symbols:
- $`W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}`$ are learned projection matrices, and the biases are added to every row.
- $`d_k = d / h = 768 / 12 = 64`$ is the size of one head.
- $`M`$ is the padding bias above.
- The softmax acts on each row separately. So the weights $`\alpha_{qj}`$ of query $`q`$ are non-negative and sum to 1 over the keys $`j`$.

Why divide by $`\sqrt{d_k}`$: if the components of a query and a key are independent with mean 0 and variance 1, their dot product has variance $`d_k`$. Dividing by $`\sqrt{d_k}`$ brings the variance back to 1. Then the softmax does not saturate (put almost all weight on one key), and its gradients do not vanish.

**In this code.** In `DistilBertSelfAttention` (transformers), `q_lin`, `k_lin` and `v_lin` are `Linear(768, 768)` layers whose outputs are reshaped into 12 heads of 64. The scale is $`64^{-1/2}`$ (`self.scaling`). Attention dropout 0.1 is applied to the weights during training only.

**Why it matters here.** Each token's vector becomes context-dependent: "cut" next to "rates" and "cut" next to "jobs" end up as different vectors. A fixed word vector cannot do that; examples are TF-IDF features, or the Keras embedding before its LSTM.

### 8.6.5 Multi-head attention

**Intuition.** Twelve heads run attention in parallel with different projections, so different heads can follow different relations (for example a negation and its verb, or a ticker and its verb). Their outputs are concatenated and mixed.

**Math.**

```math
\begin{aligned}
\operatorname{head}_r &= \operatorname{Attention}\big(H W_Q^{(r)},\ H W_K^{(r)},\ H W_V^{(r)}\big), \qquad r = 1, \dots, h \\
\operatorname{MHA}(H) &= \big[\operatorname{head}_1;\ \dots;\ \operatorname{head}_h\big]\, W_O + \mathbf{b}_O
\end{aligned}
```

The symbols:
- $`h = 12`$ heads, and $`r`$ indexes them.
- Each head outputs an $`L \times 64`$ matrix. Concatenating them side by side gives $`L \times 768`$.
- $`W_O \in \mathbb{R}^{768 \times 768}`$ is the output projection.
- Biases inside the heads are omitted for brevity.

**In this code.** The reshape inside `DistilBertSelfAttention` creates the heads, and `out_lin` is $`W_O`$. The checkpoint config gives `n_heads=12` and `dim=768`.

**Why it matters here.** The cost is about that of one full-width head, but the model gets 12 different "views" of the sentence.

### 8.6.6 Feed-forward network

**Intuition.** After the tokens have exchanged information, each token's vector goes separately through the same small two-layer network. This adds non-linear processing power.

**Math.** Here $`\mathbf{v}`$ is a column vector:

```math
\begin{aligned}
\operatorname{FFN}(\mathbf{v}) &= W_2 \, \operatorname{GELU}(W_1 \mathbf{v} + \mathbf{b}_1) + \mathbf{b}_2 \\
\operatorname{GELU}(x) &= x \, \Phi(x)
\end{aligned}
```

$`\mathbf{v} \in \mathbb{R}^{768}`$ is one token's vector, $`W_1 \in \mathbb{R}^{3072 \times 768}`$ and $`W_2 \in \mathbb{R}^{768 \times 3072}`$. $`\Phi`$ is the cumulative distribution function of the standard normal distribution.

**In this code.** The `FFN` module of transformers runs `lin1`, then `activation="gelu"`, then `lin2`, with `hidden_dim=3072` and dropout 0.1 after `lin2`. The `gelu` here is `GELUActivation`, which calls `torch.nn.functional.gelu`: the exact erf-based GELU, not the tanh approximation.

**Why it matters here.** Attention only forms weighted averages of value vectors. The feed-forward network is where each token's vector is transformed non-linearly. It also holds about two-thirds of each block's weights: 4,722,432 of 7,087,872, counted on the loaded model.

### 8.6.7 Residual connections and LayerNorm

**Intuition.** Each sub-layer adds its output to its input (a "skip connection"), so information and gradients can pass straight through. LayerNorm then rescales each token's vector to a standard range, which keeps training stable.

**Math.**

```math
\begin{aligned}
\mu &= \frac{1}{d} \sum_{k=1}^{d} v_k, \qquad \sigma^2 = \frac{1}{d} \sum_{k=1}^{d} (v_k - \mu)^2 \\
\operatorname{LN}(\mathbf{v}) &= \boldsymbol{\gamma} \odot \frac{\mathbf{v} - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \boldsymbol{\beta}
\end{aligned}
```

```math
\begin{aligned}
U &= \operatorname{LN}_1\big(H + \operatorname{MHA}(H)\big) \\
H' &= \operatorname{LN}_2\big(U + \operatorname{FFN}(U)\big)
\end{aligned}
```

$`\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^{d}`$ are a learned scale and shift, and $`\varepsilon = 10^{-12}`$. LN and FFN act on each row (token) separately, and $`H'`$ is the output of one transformer block.

**In this code.** `TransformerBlock` in transformers computes `sa_layer_norm(attention_output + hidden_states)` and then `output_layer_norm(ffn_output + attention_output)`. This is the "post-LN" order of the original BERT. DistilBERT stacks 6 such blocks. Counted on the loaded model, the encoder (embeddings plus 6 blocks) has 66,362,880 parameters, and the classification head below adds 592,899.

**Why it matters here.** The residual paths let gradients reach the lower, pre-trained layers directly, and LayerNorm keeps each token's vector at a stable scale through 6 layers. The LayerNorm scales and shifts are also among the 1-D tensors that the optimizer excludes from weight decay (see AdamW below).

### 8.6.8 What DistilBERT is

**Intuition.** DistilBERT is a smaller BERT: 6 transformer layers instead of BERT-base's 12, with the same width (768), trained to imitate BERT.

**Math (background only; nothing like this is done in this repository).** Knowledge distillation trains a small *student* model to match the softened output distribution of a large *teacher*. With logits $`\mathbf{z}`$ and a temperature $`T > 1`$, the softened probabilities are $`p^{(T)}_k = e^{z_k / T} / \sum_j e^{z_j / T}`$. The distillation loss is the cross-entropy $`-\sum_k p^{(T),\text{teacher}}_k \log p^{(T),\text{student}}_k`$. DistilBERT's pre-training (Sanh et al., 2019) combined this loss with BERT's masked-word loss and with a cosine loss that aligns the two models' hidden states. It dropped BERT's next-sentence task, its token-type embeddings and its pooler.

**In this code.** The repository only uses the finished model: `DEFAULT_MODEL = "distilbert/distilbert-base-uncased"` in `scripts/train_distilbert.py`. Its config gives `n_layers=6`, `dim=768`, `n_heads=12`, `hidden_dim=3072`, `vocab_size=30522` and `max_position_embeddings=512`. The module docstring of `src/fnsl/torch_model.py` summarises the paper's claim: about 40% smaller and 60% faster than BERT, while keeping most of its accuracy. Fine-tuning here is ordinary supervised training of all weights on the labelled tweets; no teacher model is involved.

**Why it matters here.** It is small enough to fine-tune on a free Colab or Kaggle T4 GPU. The plan's risk table asks to keep each GPU run under 30 minutes; no DistilBERT run time has been measured yet. Its Apache-2.0 licence (`MODEL_LICENSE` in `scripts/push_to_hub.py`) allows publishing the fine-tuned model.

### 8.6.9 Classification head

**Intuition.** The final vector of the first token, `[CLS]`, serves as a summary of the whole tweet. A small two-layer network on top of it produces the 3 class scores.

**Math.**

```math
\begin{aligned}
\mathbf{c} &= H^{(6)}_{0,:} \in \mathbb{R}^{768} \\
\mathbf{z} &= W_c \, \operatorname{Dropout}_{0.2}\big(\operatorname{ReLU}(W_p \mathbf{c} + \mathbf{b}_p)\big) + \mathbf{b}_c \in \mathbb{R}^{3}
\end{aligned}
```

The symbols:
- $`H^{(6)}_{0,:}`$ is row 0 (the `[CLS]` position) of the last layer's output.
- $`W_p \in \mathbb{R}^{768 \times 768}`$ and $`W_c \in \mathbb{R}^{3 \times 768}`$ are weight matrices, with biases $`\mathbf{b}_p, \mathbf{b}_c`$.
- $`\operatorname{ReLU}(x) = \max(0, x)`$.
- $`\mathbf{z}`$ holds the 3 logits.

The predicted class is $`\arg\max_k z_k`$. This equals the arg-max of the softmax probabilities, because softmax never changes the order of the scores.

**In this code.** Verified in `DistilBertForSequenceClassification.forward` of the installed transformers 5.17.0: `hidden_state[:, 0]` → `pre_classifier` (`Linear(768, 768)`) → `nn.ReLU()` → `dropout` (`seq_classif_dropout = 0.2`) → `classifier` (`Linear(768, 3)`). `src/fnsl/torch_model.py` → `load_model` creates it with `num_labels=3`, the project's `id2label` / `label2id` from `config.py`, and `dtype=torch.float32`.

The head is new. Its weights are drawn from a normal distribution with standard deviation 0.02 (`initializer_range` in the checkpoint's `config.json`), and its biases start at 0 (transformers' `_init_weights`). `config.set_seed` runs before `load_model`, so the seed fixes this draw. Because the head is new, transformers' load report lists `pre_classifier` and `classifier` as missing. It lists the masked-word head's `vocab_*` weights as unexpected. Both are expected, as the docstring of `load_model` explains. The head has $`768 \cdot 768 + 768 + 768 \cdot 3 + 3 = 592{,}899`$ parameters, under 1% of the 66,955,779 in total.

**Why it matters here.** DistilBERT has no pooler and was not pre-trained on a sentence-level task. So the `[CLS]` vector only becomes a useful summary of a tweet through fine-tuning. That is one reason to train all the layers, not just the head.

### 8.6.10 Cross-entropy loss

**Intuition.** As in the Keras model, the loss is $`-\log`$ of the probability given to the correct class, averaged over the batch.

**Math.**

```math
\mathcal{L}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{z_{i, y_i}}}{\sum_{k=0}^{2} e^{z_{i, k}}}
```

$`\theta`$ stands for all weights, $`z_{i,k}`$ is the logit of class $`k \in \{0, 1, 2\}`$ for example $`i`$, $`y_i`$ is its label, and $`B`$ is the batch size.

**In this code.** Each batch contains `"labels"` as 64-bit integers, and the model has `num_labels = 3`. So transformers sets `problem_type="single_label_classification"` and computes `outputs.loss` with `torch.nn.CrossEntropyLoss()`: the mean over the batch, with no class weights and no label smoothing. `train_one_epoch` uses `model(**batch).loss`.

**Pitfall.** Unlike the Keras BiLSTM, the DistilBERT loss is **unweighted**: the code passes no class weights. The only macro-F1-oriented choice for DistilBERT is that its epoch is selected by validation macro-F1. Mention this asymmetry whenever you compare the two models.

### 8.6.11 The explicit training loop, step by step (`train_one_epoch`)

**Intuition.** One epoch is one pass over the shuffled training batches. For every batch, the loop does seven things:
1. clear old gradients;
2. predict;
3. measure the error;
4. work out how each weight should change;
5. cap the size of that change;
6. update the weights;
7. move the learning rate one step along its schedule.

**Math.** Let $`s = 0, 1, 2, \dots`$ count the optimizer steps already taken. For the batch used at step $`s`$:

```math
\begin{aligned}
\mathbf{g}_s &= \nabla_\theta \mathcal{L}_{\text{batch}}(\theta_s) && \text{backward pass} \\
\tilde{\mathbf{g}}_s &= \operatorname{clip}(\mathbf{g}_s) && \text{gradient clipping} \\
\theta_{s+1} &= \operatorname{AdamW}(\theta_s, \tilde{\mathbf{g}}_s, \eta_s) && \text{optimizer step} \\
\eta_{s+1} &= \eta_{\max} \, \phi(s + 1) && \text{scheduler step}
\end{aligned}
```

$`\theta_s`$ are the weights before step $`s`$ and $`\eta_s`$ is the learning rate used at step $`s`$. $`\operatorname{clip}`$, $`\operatorname{AdamW}`$ and the schedule factor $`\phi`$ are defined in the next subsections. The epoch loss that is returned is $`\sum_b B_b \mathcal{L}_b / \sum_b B_b`$, the mean loss per example, where $`B_b`$ and $`\mathcal{L}_b`$ are the size and mean loss of batch $`b`$.

**In this code.** `src/fnsl/torch_model.py` → `train_one_epoch` is called once per epoch by `scripts/train_distilbert.py` → `train`:

0. `model.train()`: training mode, dropout on.
1. Move `input_ids`, `attention_mask` and `labels` to the device.
2. `optimizer.zero_grad()`. PyTorch *adds* each backward pass's gradients to the stored ones, so they must be cleared before every batch. (PyTorch 2.x sets them to `None` by default.)
3. Forward pass inside `autocast_context(scaler is not None)`. Float16 autocast is used only when a `GradScaler` exists, which happens only on CUDA. On CPU this is a do-nothing context, and everything runs in float32.
4. Loss: `loss = model(**batch).loss`, the cross-entropy. It is computed inside the model because the batch contains `labels`.
5. Backward pass: `loss.backward()` computes $`\partial \mathcal{L} / \partial w`$ for every weight $`w`$ by backpropagation, which is the chain rule applied layer by layer from the loss back to the embeddings. It stores the result in `.grad`.
6. Clip: `torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)` with `MAX_GRAD_NORM = 1.0`.
7. Update: `optimizer.step()` applies AdamW with the current learning rate.
8. Schedule: `scheduler.step()` after **every batch**, because the schedule is defined in optimizer steps.
9. Bookkeeping: `total_loss += loss.item() * len(batch["labels"])`, so the shorter last batch counts only as much as it should.

On CUDA, steps 5 to 7 become `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → clip → `scaler.step(optimizer)` → `scaler.update()` (see mixed precision below).

`predict` is the evaluation counterpart:
- `model.eval()` turns dropout off.
- `torch.no_grad()` skips gradient bookkeeping, which saves memory and time.
- Autocast is used on CUDA.
- `logits.argmax(dim=-1)` is the prediction, and the function also returns the mean loss.

Tests:
- `test_training_loop_learns_toy_data`: after 10 epochs on toy data, the last epoch's training loss must be below half the first epoch's, and toy accuracy must be at least 90%.
- `test_predict_turns_dropout_off`.
- `test_grad_scaler_branch_gives_the_same_losses_as_plain_float32`: runs the scaler branch with a CPU `GradScaler`.

**Why it matters here.** These are classic interview bugs:
- Without `zero_grad`, gradients silently pile up across batches.
- With `scheduler.step()` once per epoch instead of once per batch, the learning rate would stay near 0 for the whole 3-epoch run; warmup alone lasts 77 steps with the defaults.
- Without `model.eval()`, dropout would stay on, so predictions would be noisy and would change from run to run.

### 8.6.12 AdamW with decoupled weight decay

**Intuition.** AdamW is Adam plus weight decay. Adam gives per-weight step sizes from running gradient statistics, as in the Keras section. Weight decay shrinks each weight slightly towards 0 at every step, which discourages large weights. "Decoupled" means the shrinking is applied to the weights directly. It is not added to the gradient, where Adam's per-weight scaling would distort it.

**Math.** Let $`t = s + 1 = 1, 2, \dots`$ count the updates, as PyTorch does for the bias correction. For each weight tensor $`\theta`$ with gradient $`\mathbf{g}_t`$ and learning rate $`\eta_t`$ (the value the scheduler set for this update), PyTorch's AdamW does:

```math
\begin{aligned}
\theta &\leftarrow \theta \, (1 - \eta_t \lambda) \\
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\, \mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\, \mathbf{g}_t^{2} \\
\theta &\leftarrow \theta - \eta_t \, \frac{\mathbf{m}_t / (1 - \beta_1^{t})}{\sqrt{\mathbf{v}_t / (1 - \beta_2^{t})} + \epsilon}
\end{aligned}
```

$`\lambda`$ is the weight-decay coefficient. $`\mathbf{m}_t`$ and $`\mathbf{v}_t`$ are the running means of the gradient and of the squared gradient, starting at 0. $`\beta_1, \beta_2`$ are their decay rates, and $`\epsilon`$ is a small constant, as in the Adam subsection of 8.5. Unlike Keras, PyTorch adds $`\epsilon`$ after the bias correction, exactly as in the textbook form.

Compare "Adam with L2 regularisation", which adds $`\lambda \theta`$ to $`\mathbf{g}_t`$. There the decay is also divided by $`\sqrt{\hat{\mathbf{v}}_t}`$, so weights with large gradients are decayed less. AdamW (Loshchilov and Hutter) removes that inconsistency.

**In this code.** `src/fnsl/torch_model.py` → `make_optimizer(model, lr, weight_decay)` returns `torch.optim.AdamW` with two parameter groups:
- tensors with `ndim >= 2` (every weight matrix, including the token and position embedding tables) get `weight_decay = 0.01` (`--weight-decay`);
- 1-D tensors (all biases, and the LayerNorm $`\boldsymbol{\gamma}`$ and $`\boldsymbol{\beta}`$) get 0.

The peak learning rate is `lr = 2e-5` (`--lr`), with PyTorch's defaults $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, $`\epsilon = 10^{-8}`$. On the full model this puts 40 tensors (66,893,568 numbers) in the decay group and 64 tensors (62,211 numbers) in the no-decay group.

The shape rule catches every bias and every LayerNorm parameter. A common string rule, "no decay if the name contains `bias` or `LayerNorm.weight`", would *not* give the same split for DistilBERT. Only the embedding LayerNorm is named `LayerNorm`, so the string rule would decay the 12 per-layer scales `sa_layer_norm.weight` and `output_layer_norm.weight`. Checked: the string rule matches 52 tensors, the shape rule 64.

PyTorch multiplies the decay by the scheduled $`\eta_t`$, so the decay also follows the warmup and the decay of the learning rate. `test_optimizer_skips_weight_decay_on_biases_and_layernorm` checks the groups.

**Why it matters here.** The docstring of `make_optimizer` gives the reason for the no-decay group. Biases and LayerNorm scales are few and hardly overfit, and pulling a LayerNorm scale towards 0 would just shrink that layer's output. This is the standard BERT fine-tuning recipe.

### 8.6.13 Linear warmup, then linear decay of the learning rate

**Intuition.** Start with a learning rate of 0. Raise it in a straight line to its peak over the first 10% of the steps. Then lower it in a straight line to exactly 0 at the last step.

**Math.** Let $`T`$ be the total number of optimizer steps, $`T_{\text{w}}`$ the number of warmup steps, and $`s`$ the number of steps already taken. Update number $`s + 1`$ uses the learning rate

```math
\eta_s = \eta_{\max} \, \phi(s)
```

with the schedule factor

```math
\phi(s) = \begin{cases}
\dfrac{s}{T_{\text{w}}} & \text{if } 0 \le s < T_{\text{w}} \\[6pt]
\max\!\Big(0,\ \dfrac{T - s}{T - T_{\text{w}}}\Big) & \text{if } T_{\text{w}} \le s \le T
\end{cases}
```

and the step counts

```math
T = E \cdot \Big\lceil \frac{N}{B} \Big\rceil, \qquad T_{\text{w}} = \lfloor r \, T \rfloor
```

The symbols:
- $`\eta_{\max}`$ is the peak learning rate, $`E`$ the number of epochs, $`N`$ the number of training examples, $`B`$ the batch size and $`r`$ the warmup ratio.
- $`\lceil\cdot\rceil`$ rounds up and $`\lfloor\cdot\rfloor`$ rounds down.
- transformers also guards both denominators against 0.
- In the AdamW notation above, update $`t`$ uses $`\eta_t = \eta_{\max}\,\phi(t - 1)`$.

**In this code.** `src/fnsl/torch_model.py` → `make_scheduler(optimizer, num_training_steps, warmup_ratio)` sets `num_warmup_steps = int(warmup_ratio * num_training_steps)`. It returns `transformers.get_linear_schedule_with_warmup`, a `LambdaLR` that multiplies each group's base learning rate by $`\phi(s)`$. `train` sets `num_training_steps = args.epochs * len(train_loader)`.

With the defaults ($`E = 3`$, $`B = 32`$, $`r = 0.1`$, $`\eta_{\max} = 2 \times 10^{-5}`$) and $`N = 8{,}241`$ (`data/splits/split_report.json`), there are $`\lceil 8241 / 32 \rceil = 258`$ steps per epoch, $`T = 774`$ and $`T_{\text{w}} = 77`$. Stepping the scheduler through these values confirms three things:
- The very first update uses $`\eta_0 = 0`$. It changes no weight; it only feeds AdamW's running averages.
- The peak $`2 \times 10^{-5}`$ is first used at $`s = 77`$.
- The rate is 0 after the last step.

A real run records `num_training_steps` and `num_warmup_steps` in `params`. `test_scheduler_warms_up_then_decays_linearly_to_zero` checks the shape on 20 steps.

**Why it matters here.** Warmup protects the pre-trained weights while the head is still random, and the decay lets training settle at the end; the plan answer below explains both. **Pitfall:** $`T`$ depends on the batch size and the number of epochs. So changing `--batch-size` or `--epochs` changes how many steps the warmup lasts, though it always stays 10% of the run.

### 8.6.14 Gradient clipping by global norm

**Intuition.** If the combined gradient of all weights is unusually long, shrink it to a fixed length but keep its direction. Then one odd batch cannot cause a huge step.

**Math.** Treat all gradients as one long vector $`\mathbf{g}`$:

```math
\begin{aligned}
\|\mathbf{g}\|_2 &= \sqrt{\sum_{p} \sum_{j} g_{p,j}^{2}} \\
\mathbf{g} &\leftarrow \mathbf{g} \cdot \min\!\Big(1,\ \frac{c}{\|\mathbf{g}\|_2 + 10^{-6}}\Big)
\end{aligned}
```

$`p`$ runs over the parameter tensors and $`j`$ over their entries. $`c`$ is the maximum allowed norm, and $`10^{-6}`$ is PyTorch's guard against division by 0.

**In this code.** `torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)` with `MAX_GRAD_NORM = 1.0`, a constant in `src/fnsl/torch_model.py` that is recorded as `params.max_grad_norm`. It runs after `backward()` (after `unscale_` on a GPU) and before `optimizer.step()`. The function also returns the norm before clipping, but the code does not log it.

**Pitfall.** With mixed precision, clipping must come after `scaler.unscale_(optimizer)`. Otherwise the threshold would be compared with gradients multiplied by the loss scale (initially $`2^{16}`$), and nearly every step would be clipped. The code uses the right order.

### 8.6.15 Mixed precision (float16) and dynamic loss scaling

**Intuition.** On a GPU, matrix multiplications in 16-bit floats are much faster and use half the memory. The catch is float16's small range: very small gradients round to 0. Multiplying the loss by a large factor before the backward pass lifts the gradients into range. Dividing by the same factor afterwards restores them.

**Math.** float16's largest finite value is 65,504. Its smallest positive normal value is $`2^{-14} \approx 6.1 \times 10^{-5}`$. Below that, precision drops (subnormal numbers), and magnitudes below about $`2^{-24} \approx 6 \times 10^{-8}`$ round to 0 ("underflow"). Backpropagation is linear in the loss, so with a scale factor $`S`$:

```math
\nabla_\theta (S \, \mathcal{L}) = S \, \nabla_\theta \mathcal{L}, \qquad \mathbf{g} = \frac{\nabla_\theta (S \, \mathcal{L})}{S}
```

Dynamic scaling starts at $`S_0`$. If any unscaled gradient is infinite or NaN, the update is skipped and $`S \leftarrow \rho_{\text{down}} \, S`$. After $`n_{\text{grow}}`$ consecutive good steps, $`S \leftarrow \rho_{\text{up}} \, S`$.

**In this code.**
- `scripts/train_distilbert.py` → `train` sets `use_amp = device.type == "cuda"` and creates `torch.amp.GradScaler("cuda")` only then. It uses PyTorch's defaults: $`S_0 = 2^{16} = 65{,}536`$, $`\rho_{\text{up}} = 2`$, $`\rho_{\text{down}} = 0.5`$, $`n_{\text{grow}} = 2000`$.
- `src/fnsl/torch_model.py` → `autocast_context` returns `torch.autocast(device_type="cuda", dtype=torch.float16)`. PyTorch keeps numerically sensitive operations (softmax, losses, normalisation) in float32.
- The weights stay float32 because `load_model` passes `dtype=torch.float32`. Autocast only makes temporary float16 copies for the fast operations, and `GradScaler` cannot unscale float16 gradients (code comment).
- `scaler.step` skips the update if an overflow was found, and `scaler.update` adjusts $`S`$. `scheduler.step()` still runs after a skipped update, so a skipped step uses up one step of the schedule. PyTorch may then warn that the scheduler stepped before the optimizer, which the code comment calls harmless.
- `predict` also uses autocast on CUDA, without a scaler, because no gradients are computed.

**Pitfall.** A default run has only 774 optimizer steps, fewer than the 2,000-step growth interval. So $`S`$ can only stay at $`2^{16}`$ or shrink after overflows. This path has never run, because there has been no GPU so far. On the first Colab run, check that the training loss is finite and falls.

### 8.6.16 Model selection on validation macro-F1, then one test evaluation

**Intuition.** Train for a fixed number of epochs and score the validation split after each one. Keep the epoch with the best validation macro-F1, and only then touch the test split, once.

**Math.**

```math
\begin{aligned}
e^{\ast} &= \arg\max_{e \in \{1, \dots, E\}} \text{F1}^{\text{val}}_{\text{macro}}(\theta_e) \\
\text{F1}_{\text{macro}} &= \frac{1}{3} \sum_{k=0}^{2} \frac{2 P_k R_k}{P_k + R_k}
\end{aligned}
```

$`\theta_e`$ are the weights after epoch $`e`$, and $`E`$ is the number of epochs. $`P_k`$ and $`R_k`$ are the precision and recall of class $`k`$ on the validation split. A class with $`P_k + R_k = 0`$ gets F1 0 (`zero_division=0` in `src/fnsl/metrics.py` → `macro_f1`). The test metrics are computed once, with $`\theta_{e^{\ast}}`$.

**In this code.** `scripts/train_distilbert.py` → `train`:
1. After each epoch it calls `predict(model, val_loader, device)` and `metrics.macro_f1`.
2. If the score is strictly better than the best so far (`val_f1 > best_val_f1`, which starts at −1, so epoch 1 is always saved and the earliest epoch wins a tie), it calls `model.save_pretrained(best_dir)` and `tokenizer.save_pretrained(best_dir)`. The default folder is `checkpoints/distilbert/best`.
3. After the last epoch it writes `best/training_info.json`. This holds the run name, seed, smoke flag, `git_commit`, `train_time_s`, every hyperparameter, `best_epoch`, `val_macro_f1` and the per-epoch `history` of `train_loss`, `val_loss` and `val_macro_f1`.
4. Then `evaluate_on_test` **reloads the best checkpoint from disk** and predicts the test split once, with `shuffle=False`.
5. `metrics.save_results` writes the metrics JSON (with extra keys `test_loss`, `checkpoint` and `train_git_commit`), the predictions CSV and the confusion-matrix figure.

Unlike the Keras script, there is no early stopping: all `--epochs` (default 3) run, and choosing the best epoch plays the same role. `test_train_script_reloads_the_best_validation_epoch_not_the_last` fakes validation scores of 0.2, 0.9 and 0.1 (toy values) and checks that epoch 2's weights are saved and used.

**Why it matters here.** Choosing the epoch on the test split would leak test information into the model choice and inflate the reported score. The test loss is recorded but never used for a decision.

### 8.6.17 `--eval-only`, `training_info.json` and reproducibility

**Intuition.** A saved checkpoint should reproduce its test predictions exactly. The metrics file should describe the run that produced the weights, not the run that re-evaluated them.

**Math.** No formula, but one precise point. The attention mask makes padding mathematically irrelevant. Yet floating-point results can still differ in the last bits when the padded length of a batch changes, because different kernel sizes can add numbers in a different order. Reusing the training batch size gives the same batches with the same padding, and hence the same arithmetic.

**In this code.** `python scripts/train_distilbert.py --eval-only --checkpoint checkpoints/distilbert/best`:
- `parse_args` refuses `--eval-only` without `--checkpoint`, and `eval_only` raises `FileNotFoundError` if the folder has no `training_info.json`.
- The hyperparameters (including `max_length`), seed, training time and run name come from `training_info.json`.
- The batch size defaults to the training run's (`info["params"]["batch_size"]`). The code comment says this is "so every batch is padded exactly as in that run and the predictions are reproduced exactly". `test_train_script_smoke_then_eval_only` checks that the predictions and test loss are identical and that the training batch size was used.
- `smoke = args.smoke or info["smoke"]`, so a smoke checkpoint can only write under `results/smoke/`.
- In the metrics file, `git_commit` is the code that ran the evaluation. It is marked `-dirty` if `src/` or `scripts/` has uncommitted changes. `train_git_commit` is copied from `training_info.json`: it is the code that trained the weights.
- `train` deletes an old `best/training_info.json` before training starts, so an interrupted run cannot leave the previous run's metadata next to new weights (`DEBUG_LOG.md`, Day 5; `test_interrupted_run_cannot_reuse_an_older_training_info`).
- Seeds: `config.set_seed(seed)` seeds Python, NumPy and torch (and CUDA when present) before the model and loaders are built. This fixes the head initialisation and the dropout masks, and with the seeded `torch.Generator` it also fixes the shuffle order. The code does not switch on PyTorch's deterministic-algorithms mode, so GPU reruns may still differ slightly.

**Why it matters here.** The optional Day 7 seed reruns (`--seed 43` and `--seed 44`, saved as `distilbert_seed43` and `distilbert_seed44`) are only meaningful if every result can be traced to its weights and its code.

### 8.6.18 Model card generator and its refusal gates (`scripts/push_to_hub.py`)

**Intuition.** The Hugging Face model card is generated from the results files, so it cannot contain a number that no script produced. The upload refuses weights that do not belong to the reported run.

**Math.** None of its own. From `comparison.json` the card shows macro-F1 with its bootstrap CI, the paired-bootstrap CI of each macro-F1 difference, and McNemar p-values with Holm's correction (see the statistics section).

**In this code.**
- `main` loads only `results/metrics/distilbert.json`, `results/metrics/comparison.json` and `data/splits/split_report.json`; the paths can be changed with `--metrics-dir` and `--split-report`. `render_card(distilbert, comparison, split_report, repo_id)` builds the card from those three dicts.
- A missing input is shown as "_Not available yet._", never as a number.
- The card starts with YAML front matter (language, licence, library, task, base model, dataset, metrics, tags). Then come task, base model, training data, hyperparameters with the per-epoch history, evaluation, comparison, limitations, intended use and usage code.
- `main` ignores a `distilbert.json` that has `"smoke": true`.
- `comparison_problem` hides the comparison in any of these cases:
  - `comparison.json` is missing, has an unexpected layout or comes from a smoke run;
  - `distilbert.json` is missing;
  - the comparison has no `distilbert` entry;
  - that entry's macro-F1 differs from `distilbert.json` by more than $`10^{-9}`$ (a stale comparison).
- `--push` requires `--repo-id USER/NAME` and refuses to run without a real `distilbert.json`. `checkpoint_problem` refuses a folder that has no `config.json`, has no `training_info.json`, or whose `training_info.json` has different `params` or `seed` from `distilbert.json` (weights from another run). The token comes from `HF_TOKEN` or `hf auth login` and is never stored.
- Tests in `tests/test_model_card.py` (CPU, run in CI) check that:
  - no score appears without results;
  - every 4-decimal number in a rendered card appears, up to sign, in an input file;
  - stale, smoke or oddly formatted comparisons are not shown;
  - smoke metrics never reach the card;
  - `--push` refuses without real results or with a mismatched checkpoint;
  - rendering imports no heavy libraries.

**Why it matters here.** The model card is the one artefact that leaves the repository. So the project's first rule, never invent numbers, is enforced in code and by a test rather than by care alone.

**Current state.** A dry run has written `results/model_card/README.md` (gitignored). `distilbert.json` does not exist, so the card says the model has not been evaluated yet and contains no scores. Nothing has been pushed.

### 8.6.19 Default hyperparameters (`scripts/train_distilbert.py` → `parse_args`, `src/fnsl/torch_model.py`)

| Setting | Default | Notes |
|---|---|---|
| `--model-name` | `distilbert/distilbert-base-uncased` | Hub id or local folder |
| `--epochs` | 3 (1 with `--smoke`) | all epochs run; the best by validation macro-F1 is kept |
| `--batch-size` | 32 (16 with `--smoke`) | with `--eval-only`: the training run's batch size |
| `--lr` | 2e-5 | peak learning rate $`\eta_{\max}`$ |
| `--weight-decay` | 0.01 | weight matrices only; 0 for biases and LayerNorm |
| `--warmup-ratio` | 0.1 | share of all steps used for warmup |
| `--max-length` | 64 | tokens, including `[CLS]` and `[SEP]` |
| `MAX_GRAD_NORM` | 1.0 | global L2 clipping threshold (module constant) |
| AdamW $`\beta_1, \beta_2, \epsilon`$ | 0.9, 0.999, 1e-8 | PyTorch defaults |
| Dropout | 0.1 (embeddings, attention, feed-forward), 0.2 (head) | from the checkpoint config |
| Mixed precision | float16 autocast and `GradScaler` on CUDA only | float32 on CPU |
| `--seed` | 42 | run name `distilbert`; other seeds give `distilbert_seed<N>` |
| `--output-dir` | `checkpoints/<run_name>` (smoke: `checkpoints/smoke/<run_name>`) | the best epoch goes to `best/` |
| `--num-workers` | 0 | DataLoader worker processes |

### 8.6.20 Plan questions (Days 5 and 6)

**Explain each step of the training loop.** For every batch:
1. `zero_grad` clears the gradients left by the previous batch, because PyTorch accumulates them.
2. The forward pass computes the logits and, because labels are passed, the cross-entropy loss. On a GPU this runs under float16 autocast.
3. `backward` computes the gradient of the loss with respect to every weight by backpropagation.
4. `clip_grad_norm_` rescales the whole gradient to length at most 1.0 if it is longer.
5. `optimizer.step` updates every weight with AdamW: Adam's adaptive step plus decoupled weight decay on the weight matrices.
6. `scheduler.step` moves the learning rate one step along the warmup and decay.

With mixed precision, the loss is multiplied by the scale factor before `backward`, the gradients are divided back before clipping, and the step is skipped if they overflowed. After each epoch the model is scored on validation in `eval` mode, and the best epoch is saved.

**What does the learning-rate schedule do?** It raises the learning rate from 0 to $`2 \times 10^{-5}`$ over the first 77 of the 774 default steps. Then it lowers it linearly to 0 at the last step.

Warmup matters because at the start the new head is random, its gradients are large and noisy, and AdamW's running averages are based on very few batches. Full-size steps then could damage the pre-trained weights. The decay makes the steps smaller as training ends, so the weights settle instead of bouncing around a minimum.

**Why do fine-tuned transformers usually beat bag-of-words?** This is an expectation to be tested here, not a result.

The TF-IDF + logistic regression baseline represents a text as weighted counts of single words and adjacent word pairs. Its settings, recorded in `results/metrics/tfidf_logreg.json`, are `ngram_range` [1, 2], `min_df` 2 and `sublinear_tf` true, giving 16,172 features. It scores those features with one linear weight each. So it:
- loses word order beyond adjacent pairs;
- gives a feature the same weight in every context;
- drops any word or pair seen in fewer than 2 training texts, and ignores words never seen in training.

A fine-tuned transformer has four advantages:
1. It builds context-dependent token vectors through self-attention. So "cut" in "Fed cuts rates" and in "cuts jobs", or "beat" after "failed to", can be read differently.
2. It uses position embeddings, so order counts at any distance.
3. It splits unseen tickers and rare words into known word pieces instead of discarding them.
4. It starts from weights pre-trained on large English corpora. So 8,241 labelled tweets only have to adapt existing knowledge instead of teaching the language from scratch.

Reasons the expectation might not hold here:
- The tweets are short and keyword-heavy, which suits bag-of-words.
- The pre-training text is English books and Wikipedia, not finance tweets with cashtags and links.
- The labels contain noise.
- The test split has 1,767 posts (`data/splits/split_report.json`), so a small gain may not be statistically significant.

The bar to beat is the TF-IDF baseline's test macro-F1 of 0.7389 (`results/metrics/tfidf_logreg.json`). Its 95% bootstrap CI is 0.7152 to 0.7619, from 2,000 resamples (`results/metrics/comparison.json`). After the GPU run, `scripts/compare_models.py` will test the difference two ways: McNemar's test (Holm-adjusted over all pairs) and a paired bootstrap CI of the macro-F1 difference. The per-class F1 scores and McNemar's discordant counts will show any case where DistilBERT did worse. The plan's follow-up, "any case here where they did not", cannot be answered until then.

### 8.6.21 What is left to do (DistilBERT)

1. Decide before the real run, on validation data only, two differences from the Keras model:
   - DistilBERT trains with unweighted cross-entropy, while Keras uses balanced class weights.
   - DistilBERT sees raw text with links, while Keras deletes links.

   Either keep both differences and state them in the README, or change the code first, judging the change on `val_macro_f1`. Also be ready to explain that Keras picks its epoch by weighted validation loss, while DistilBERT picks it by validation macro-F1.
2. In `notebooks/colab_runner.ipynb` on a GPU runtime, run the smoke cell, then `python scripts/train_distilbert.py`. Day 5 is done when this run shows a falling training loss. Optionally run `--seed 43` and `--seed 44`. The float16 path is new, so check that the loss is finite.
3. If you will publish the model, set `INCLUDE_DISTILBERT_CHECKPOINT = True` in the download cell (about 270 MB, per the notebook).
4. Unzip `results.zip` into the repo. Run `python scripts/compare_models.py` and `python scripts/make_results_table.py --update-readme --count-tests`, then `pytest -m "not heavy"` and `ruff check .`, and commit. The heavy tests in `tests/test_torch_model.py` need no downloads and can be run locally with `pytest -m heavy tests/test_torch_model.py`.
5. On Day 14, run `python scripts/push_to_hub.py` (a dry run) and read the card. Then run `python scripts/push_to_hub.py --push --repo-id USER/NAME --checkpoint checkpoints/distilbert/best`.

---

## Files this part reads and writes

**Keras BiLSTM (`scripts/train_keras.py`)**: run name `keras_bilstm`, or `keras_bilstm_seed<N>` for other seeds, or the `--run-name` value.
- `results/metrics/<run>.json`
  - Standard keys: `model, dataset, n_test, macro_f1, accuracy, per_class_f1, seed, train_time_s, params, git_commit, timestamp`.
  - Extra keys: `smoke`; `confusion_matrix` (rows = true, columns = predicted); `n_unparseable` (always 0 here); `checkpoint` (repo-relative path).
  - `params` keys:
    - `epochs` (the maximum);
    - `batch_size, max_tokens, seq_len, embed_dim, lstm_units, dropout, lr, optimizer, loss, patience, early_stopping`;
    - `class_weight` (`balanced` or `none`) and `class_weights` ({label: weight} or null);
    - `val_loss_weighted, vocab_size, n_params, n_train, n_val, train_texts_truncated, device, tensorflow_version, keras_version`;
    - `epochs_trained, stopped_early, best_epoch` (1-based), `val_macro_f1` (from the restored best-epoch weights);
    - `history`: a list of {epoch, loss, accuracy, val_loss, val_accuracy}. The losses are class-weighted; the accuracies are unweighted.
- `results/predictions/<run>.csv`: `id,text,label,pred`, with the same ids and order as `data/splits/test.csv`.
- `results/figures/<run>_confusion.png`.
- `checkpoints/<run>.keras` (gitignored): vectoriser vocabulary plus weights. Reload it with `src/fnsl/keras_model.py` -> `load_model`.

**DistilBERT (`scripts/train_distilbert.py`)**: run name `distilbert`, or `distilbert_seed<N>` for other seeds, or the `--run-name` value.
- `results/metrics/<run>.json`
  - The standard keys, plus `smoke, confusion_matrix, n_unparseable` and three more:
    - `test_loss`: unweighted mean cross-entropy on test; recorded, never used for decisions.
    - `checkpoint`.
    - `train_git_commit`.
  - `params` keys:
    - `model_name, epochs, batch_size, lr, weight_decay, warmup_ratio, num_training_steps, num_warmup_steps, max_length, max_grad_norm, optimizer, scheduler`;
    - `device, mixed_precision, n_train, n_val, train_texts_truncated, torch_version, transformers_version`;
    - `best_epoch, val_macro_f1`;
    - `history`: a list of {epoch, train_loss, val_loss, val_macro_f1}.
- `results/predictions/<run>.csv` (`id,text,label,pred`) and `results/figures/<run>_confusion.png`.
- `checkpoints/<run>/best/` (gitignored): `config.json` (includes id2label, label2id, problem_type), `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and `training_info.json`.
  - `training_info.json` holds `run_name, seed, smoke, git_commit` (the training code), `train_time_s` and `params` (the same dict as the metrics file's `params`). It is read by `--eval-only` and checked by `push_to_hub.py --push`.

**Model card (`scripts/push_to_hub.py`)**
- Reads three files:
  - `results/metrics/distilbert.json` (above).
  - `results/metrics/comparison.json`. The fields the card uses are `alpha, n_resamples, smoke`, `models[m].{macro_f1, ci_low, ci_high}`, `pairs[i].{model_a, model_b, mcnemar.{b, c, p_value}, macro_f1_diff.{diff, low, high}, p_holm, significant_holm}` and the optional `skipped_pairs[i].{model_a, model_b, reason}`.
  - `data/splits/split_report.json`, fields `dataset, revision, license, seed`, `cleaning.{n_raw, n_clean}`, `sizes.{train, val, test}` and `class_balance.all.{<Label>_%}`.
- Writes `results/model_card/README.md` (gitignored). `--push` uploads the checkpoint folder and the card to the Hub repo.

Smoke runs write the same files under `results/smoke/` and `checkpoints/smoke/`, both gitignored. Non-smoke runs also log to MLflow in `mlruns/mlflow.db` (gitignored): the flattened hyperparameters, the per-epoch metrics, the final metrics and the confusion figure. The per-epoch metrics are loss/accuracy/val_loss/val_accuracy for Keras and train_loss/val_loss/val_macro_f1 for DistilBERT. The final metrics are best_epoch, test macro-F1, accuracy and per-class F1, plus val_macro_f1 for Keras.
