# Mathematics, part 2: Embeddings, PCA and K-Means (section 8.4)

Part of the [project guide](../PROJECT_GUIDE.md#8-the-mathematics-part-by-part) for fin-news-sentiment-lab, Day 3 of `PROJECT_PLAN.md`. Checked against the code at commit `98a9d42`; the one later code change (`aaf142e`) makes the Unsloth resume claim require an adapter trained with `--backend unsloth`.

> **Numbers rule.** A *result* is copied from the results file named next to it. Every other number is a setting read from the code, a check you can rerun (named as such), or arithmetic shown in full. Models that have only had a `--smoke` run have no results, and none are stated.

Each concept follows the same pattern: the intuition, the mathematics (every symbol defined), how this code implements it, and why it matters or where it can go wrong. Formulas use GitHub's math rendering.

## Contents

- [8.4 Sentence embeddings, PCA via SVD and K-Means clustering](#84-sentence-embeddings-pca-via-svd-and-k-means-clustering)
  - [8.4.0 Pipeline at a glance](#840-pipeline-at-a-glance)
  - [8.4.1 What a sentence embedding is, and how all-MiniLM-L6-v2 makes one](#841-what-a-sentence-embedding-is-and-how-all-minilm-l6-v2-makes-one)
  - [8.4.2 The embedding cache](#842-the-embedding-cache)
  - [8.4.3 Cosine similarity versus Euclidean distance on unit vectors](#843-cosine-similarity-versus-euclidean-distance-on-unit-vectors)
  - [8.4.4 PCA from the covariance matrix](#844-pca-from-the-covariance-matrix)
  - [8.4.5 PCA via the SVD (how SVD relates to PCA)](#845-pca-via-the-svd-how-svd-relates-to-pca)
  - [8.4.6 The sign convention, and how the test compares with scikit-learn](#846-the-sign-convention-and-how-the-test-compares-with-scikit-learn)
  - [8.4.7 inverse_transform and the best rank-k approximation (Eckart–Young)](#847-inverse_transform-and-the-best-rank-k-approximation-eckartyoung)
  - [8.4.8 K-Means: the objective and Lloyd's algorithm](#848-k-means-the-objective-and-lloyds-algorithm)
  - [8.4.9 k-means++ initialisation, n_init and random_state](#849-k-means-initialisation-n_init-and-random_state)
  - [8.4.10 The silhouette coefficient (cluster structure, no labels)](#8410-the-silhouette-coefficient-cluster-structure-no-labels)
  - [8.4.11 The contingency table and the adjusted Rand index (ARI)](#8411-the-contingency-table-and-the-adjusted-rand-index-ari)
  - [8.4.12 Normalised mutual information (NMI)](#8412-normalised-mutual-information-nmi)
  - [8.4.13 Majority label and purity](#8413-majority-label-and-purity)
  - [8.4.14 Results (results/metrics/kmeans.json)](#8414-results-resultsmetricskmeansjson)
  - [8.4.15 What clustering agreement says, and does not say, about sentiment](#8415-what-clustering-agreement-says-and-does-not-say-about-sentiment)
  - [8.4.16 Tests that guard this part](#8416-tests-that-guard-this-part)
  - [8.4.17 Open items for this part](#8417-open-items-for-this-part)
- [Files this part reads and writes](#files-this-part-reads-and-writes)

---

## 8.4 Sentence embeddings, PCA via SVD and K-Means clustering

This part asks one question without using any labels: if every tweet is turned into a vector by a general-purpose sentence-embedding model, do the vectors fall into groups, and do those groups match Bearish / Bullish / Neutral? Labels are used only after clustering, to score the groups. This is Day 3 of the plan.

**Status: real run.** `scripts/run_kmeans.py` clustered the full train split (8,241 texts) and wrote `results/metrics/kmeans.json` (`git_commit` `9d28770`, `timestamp` `2026-11-14T10:52:31+00:00`, `smoke` false) and `results/figures/kmeans_pca.png`; both files were committed in `b363da6`. None of the code that produces these numbers (`src/fnsl/embeddings.py`, `src/fnsl/pca.py`, `scripts/run_kmeans.py`, `src/fnsl/data.py`, `src/fnsl/config.py`, `src/fnsl/metrics.py`, `src/fnsl/tracking.py`) has changed between `9d28770` and HEAD `98a9d42`. The non-heavy tests in `tests/test_pca.py`, `tests/test_kmeans.py` and `tests/test_embeddings.py` pass at HEAD. The one heavy test (it loads the real model) is deselected in CI by `pytest -m "not heavy"`; it passes locally with `pytest -m heavy`.

### 8.4.0 Pipeline at a glance

1. **Read texts.** `src/fnsl/data.py` → `load_split("train")` reads `data/splits/train.csv` (columns `id,text,label`) and checks the columns and the label ids. `--split val` is allowed. `--split test` is refused, because the test set is reserved for the final evaluation of the classifiers.
2. **Embed.** `src/fnsl/embeddings.py` → `embed_texts` turns every text into a vector of 384 numbers with length 1, cached on disk (8.4.1 to 8.4.3).
3. **Cluster and score.** For every k in `--ks` (default 3 and 5), `scripts/run_kmeans.py` → `evaluate_kmeans` runs scikit-learn K-Means on all 384 dimensions (8.4.8, 8.4.9). It then scores the clusters: the silhouette uses no labels (8.4.10); the contingency table, ARI, NMI and purity compare clusters with labels (8.4.11 to 8.4.13).
4. **Project for the picture.** `src/fnsl/pca.py` → `PCA(n_components=2)` is fit on the same embeddings (no labels) and gives the 2-D coordinates for `plot_pca` (8.4.4 to 8.4.7). The PCA is used only for the figure, never for the clustering.
5. **Save.** `kmeans.json`, `kmeans_pca.png`, and an MLflow run in `mlruns/` (skipped for `--smoke` and `--no-mlflow`). `--smoke` uses 300 texts of the chosen split (100 per label, picked by `data.smoke_subset`) and writes to the gitignored `results/smoke/`.

### 8.4.1 What a sentence embedding is, and how all-MiniLM-L6-v2 makes one

**Intuition.** A sentence embedding is a fixed-length vector for a whole text. The model is trained so that texts that belong together (same question, same subject) get vectors that point in similar directions. all-MiniLM-L6-v2 gives 384 numbers per text, whatever the text's length.

**Mathematics.** The downloaded model is a pipeline of three modules, listed in its `modules.json`: Transformer, then Pooling, then Normalize.

1. *Tokenise.* An uncased WordPiece tokenizer lowercases the text, splits it into word pieces from a vocabulary of 30,522 entries (model `config.json`), and adds `[CLS]` at the start and `[SEP]` at the end. Texts longer than 256 word pieces are cut (`max_seq_length` in the model's `sentence_bert_config.json`). Toy example (real tokenizer output): `Shares of Nvidia jump 5% after earnings beat` becomes `[CLS] shares of n ##vid ##ia jump 5 % after earnings beat [SEP]`. The rare word "Nvidia" is split into three known pieces, and `##` marks a piece that continues a word.
2. *Transformer.* A 6-layer BERT-style encoder (hidden size 384, 12 attention heads, feed-forward size 1,536; model `config.json`) turns the $`T`$ tokens into $`T`$ context-dependent vectors $`\mathbf{h}_1, \dots, \mathbf{h}_T \in \mathbb{R}^{384}`$. Self-attention, which lets each token's vector depend on every other token, is derived in section [8.6](3_keras_distilbert.md#86-distilbert-fine-tuning-with-an-explicit-pytorch-loop) (DistilBERT), under "Scaled dot-product attention".
3. *Mean pooling.* Average the token vectors, ignoring padding:

```math
\mathbf{e} = \frac{\sum_{t=1}^{T} m_t \, \mathbf{h}_t}{\max\left(\sum_{t=1}^{T} m_t,\ 10^{-9}\right)}
```

Here $`T`$ is the padded length of the batch, $`\mathbf{h}_t`$ is the last-layer vector of token $`t`$, and $`m_t \in \{0, 1\}`$ is the attention mask: 1 for real tokens (including `[CLS]` and `[SEP]`), 0 for padding (`1_Pooling/config.json` sets `pooling_mode_mean_tokens: true`). The $`10^{-9}`$ only prevents division by zero. Because padding is masked, a text's vector does not depend on which other texts share its batch (up to floating-point rounding).

4. *L2 normalisation.* Scale the vector to length 1:

```math
\hat{\mathbf{e}} = \frac{\mathbf{e}}{\lVert \mathbf{e} \rVert_2}, \qquad \lVert \mathbf{e} \rVert_2 = \sqrt{\sum_{f=1}^{384} e_f^2}
```

where $`f`$ indexes the 384 dimensions and $`e_f`$ is entry $`f`$ of $`\mathbf{e}`$.

5. *Why the vectors mean anything.* According to the model card, the encoder (initialised from `nreimers/MiniLM-L6-H384-uncased`) was fine-tuned with a contrastive objective on more than a billion text pairs: Reddit comments and their replies (the largest source), citing and cited papers, paper titles and abstracts, duplicate questions, questions and answers, forum titles and bodies, and more. The card describes the objective in words: compute the cosine similarity of every possible pair in a batch, then apply a cross-entropy loss that rewards the true pairs. One standard way to write this, for a batch of $`B`$ pairs, with $`\mathbf{a}_i`$ and $`\mathbf{p}_i`$ the embeddings of the two texts of pair $`i`$:

```math
\mathcal{L}_{\text{con}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\big(\cos(\mathbf{a}_i, \mathbf{p}_i)/\tau\big)}{\sum_{j=1}^{B} \exp\big(\cos(\mathbf{a}_i, \mathbf{p}_j)/\tau\big)}
```

$`\tau > 0`$ is a temperature (equivalently, a scale factor $`1/\tau`$), and $`\cos`$ is cosine similarity (8.4.3). The other partners $`\mathbf{p}_j`$ with $`j \ne i`$ act as negatives ("in-batch negatives"). Minimising $`\mathcal{L}_{\text{con}}`$ pulls true partners together and pushes the rest apart, measured by cosine. The card gives neither the value of $`\tau`$ nor whether the loss is also applied in the other direction (from $`\mathbf{p}`$ to $`\mathbf{a}`$); neither detail changes the idea.

**In this code.** `src/fnsl/embeddings.py` → `embed_texts` loads `SentenceTransformer(model_name)` with `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"` and calls `model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) >= 1000)`. It returns a float32 array of shape `(n, 384)`; row $`i`$ belongs to text $`i`$ (sentence-transformers sorts the texts by length internally for speed and then restores the original order). Details, checked against the installed sentence-transformers 6.3.0:

- The model's own Normalize module already outputs unit vectors. `normalize_embeddings=True` makes `encode` normalise once more, which changes nothing beyond floating-point rounding, because a vector of length 1 divided by 1 is unchanged. The `embed_texts` docstring says this, and every row of the cached train embeddings has length 1 up to float32 rounding.
- Texts are passed positionally because the parameter name changed between library versions (it is `inputs` in 6.x).
- `sentence_transformers` (and so torch) is imported only on a cache miss. The CPU tests use fakes, so they never load it, and CI does not install it (`requirements.txt`).
- No train tweet comes close to the 256-word-piece limit, or even to the 128-token length the model was fine-tuned with (checked with the model's tokenizer while writing this guide), so truncation never happens.
- The encoder is used frozen. It is not fine-tuned on this dataset and has never seen a sentiment label.

**Why it matters here.** The training pairs come mostly from conversations, question-and-answer sites and scientific papers, so the objective mainly rewards "these two texts belong together", which mostly means "about the same subject". It does not specifically reward separating "beats estimates" from "misses estimates" when the rest of a headline is the same. That is the most likely reason (a hypothesis; the repo does not test it) why the unsupervised clusters do not follow sentiment (8.4.15). The RAG index uses the same model name (`src/fnsl/rag/ingest.py` → `DEFAULT_EMBEDDING_MODEL`).

### 8.4.2 The embedding cache

**Intuition.** Embedding is the slowest step. The result is saved under a file name that changes whenever any input changes, so a rerun with the same inputs just loads the file.

**Mathematics.** The file name contains a 64-bit fingerprint, the first 16 hexadecimal digits of a SHA-256 hash:

```math
\text{key} = \text{first 16 hex digits of } \operatorname{SHA256}\big(\operatorname{JSON}(\{\text{model},\ \text{normalize},\ \text{texts}\})\big)
```

$`\operatorname{JSON}`$ writes the model name, the normalize flag and the list of texts as one string (encoded as UTF-8 before hashing), and $`\operatorname{SHA256}`$ maps that string to a fixed-length hash. Changing the model name, the flag, any text, or the order of the texts gives a different key. JSON-encoding the list keeps texts apart, so `["ab", "c"]` and `["a", "bc"]` get different keys. With $`N_{\text{files}}`$ cache files, the chance that two share a 64-bit key is about $`N_{\text{files}}^2 / 2^{65}`$ (the birthday bound), which is negligible.

**In this code.** `src/fnsl/embeddings.py` → `cache_path` builds `cache/embeddings/all-MiniLM-L6-v2_<key>.npy` (`config.CACHE_DIR / "embeddings"`; `cache/` is gitignored). `embed_texts` first converts every text with `str()`. It returns the cached array if the file exists and has one row per text; a file with the wrong number of rows is ignored and overwritten. Otherwise it embeds, saves to a per-process temporary file `<stem>.<pid>.tmp.npy`, and renames it into place with `Path.replace`. A rename within one file system is atomic, so an interrupted run can never leave a half-written cache file. The per-process name is the fix for the Day 3 bug in `DEBUG_LOG.md`: concurrent runs used one shared temporary name, and one process sometimes renamed the file away while another was still writing it (`FileNotFoundError`). Empty input raises `ValueError`.

**Pitfall.** The key ignores the batch size, which is fine: because padding is masked, the batch size changes the vectors only by floating-point rounding. It also ignores the model revision and the library versions, and `embed_texts` does not pin a model revision. After upgrading sentence-transformers, or if the model on the Hugging Face Hub changes, delete `cache/embeddings/` so that stale vectors cannot be served.

### 8.4.3 Cosine similarity versus Euclidean distance on unit vectors

**Intuition.** Cosine similarity compares directions and ignores length. Once all vectors have length 1, Euclidean distance is a fixed decreasing function of the cosine, so both rank neighbours in the same order.

**Mathematics.** For vectors $`\mathbf{u}, \mathbf{v} \in \mathbb{R}^{d}`$, with $`\mathbf{u}^\top \mathbf{v}`$ their dot product and $`\lVert \cdot \rVert`$ the Euclidean length:

```math
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert}
```

If $`\lVert \mathbf{u} \rVert = \lVert \mathbf{v} \rVert = 1`$, expanding the square gives

```math
\lVert \mathbf{u} - \mathbf{v} \rVert^2 = \lVert \mathbf{u} \rVert^2 + \lVert \mathbf{v} \rVert^2 - 2\,\mathbf{u}^\top \mathbf{v} = 2 - 2\cos(\mathbf{u}, \mathbf{v})
```

so the distance $`\sqrt{2 - 2\cos(\mathbf{u}, \mathbf{v})}`$ lies between 0 and 2. Toy example: $`\cos = 1`$ gives distance 0, $`\cos = 0.5`$ gives 1, $`\cos = 0`$ gives $`\sqrt{2}`$, and $`\cos = -1`$ gives 2.

**In this code.** `scripts/run_kmeans.py` → `main` calls `embed_texts(..., normalize=True)`. `evaluate_kmeans` runs K-Means (squared Euclidean distance) and `silhouette_score(..., metric="euclidean")`. `kmeans.json` records the choice as `"silhouette_distance": "euclidean on L2-normalised embeddings (same ranking as cosine distance)"` (the constant `SILHOUETTE_DISTANCE`), and `tests/test_kmeans.py` → `test_main_smoke_writes_kmeans_json_and_figure` checks that `normalize=True` is passed. Two honest deviations from "pure cosine":

- K-Means on unit vectors is not exactly spherical (cosine) K-Means. A centroid is an average of unit vectors, so its length is at most 1, and below 1 unless all its members are identical. With $`\boldsymbol{\mu}_j`$ the centroid of cluster $`j`$ and $`\lVert \mathbf{x} \rVert = 1`$, the assignment rule $`\arg\min_j \lVert \mathbf{x} - \boldsymbol{\mu}_j \rVert^2 = \arg\min_j \big(\lVert \boldsymbol{\mu}_j \rVert^2 - 2\,\mathbf{x}^\top \boldsymbol{\mu}_j\big)`$ depends on the centroid lengths. Spherical K-Means would rescale every centroid to length 1, and the rule would become "largest cosine".
- The silhouette averages distances, not cosines. It orders pairs exactly like cosine distance, but its value is not the same number as a cosine-distance silhouette.

**Why it matters here.** Interview question: "Why Euclidean K-Means on text embeddings?" Answer: after L2 normalisation, squared Euclidean distance equals $`2 - 2\cos`$, so K-Means works in the cosine geometry the model was trained for, up to the centroid-length detail above.

### 8.4.4 PCA from the covariance matrix

**Intuition.** PCA rotates the axes. The first new axis points where the data spread the most, the second where they spread the most at right angles to the first, and so on. Keeping the first two axes gives a 2-D picture.

**Mathematics.** Let $`X \in \mathbb{R}^{n \times d}`$ hold one sample per row (here $`n = 8{,}241`$ texts and $`d = 384`$ dimensions), with rows $`\mathbf{x}_1, \dots, \mathbf{x}_n`$. Let $`\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i`$ be the vector of column means, $`\mathbf{1}`$ a column of $`n`$ ones, and $`X_c = X - \mathbf{1}\bar{\mathbf{x}}^\top`$ the centred data. The sample covariance matrix is

```math
C = \frac{1}{n-1} X_c^\top X_c \in \mathbb{R}^{d \times d}
```

Projecting the data onto a unit vector $`\mathbf{w} \in \mathbb{R}^{d}`$ gives one number per sample, the vector $`X_c \mathbf{w}`$, whose variance is

```math
\operatorname{Var}(X_c \mathbf{w}) = \frac{1}{n-1} \lVert X_c \mathbf{w} \rVert^2 = \mathbf{w}^\top C \mathbf{w}
```

PCA maximises this subject to $`\mathbf{w}^\top \mathbf{w} = 1`$. With a Lagrange multiplier $`\lambda`$:

```math
\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^\top C \mathbf{w} - \lambda\,(\mathbf{w}^\top \mathbf{w} - 1), \qquad \nabla_{\mathbf{w}} \mathcal{L} = 2C\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} \;\Rightarrow\; C\mathbf{w} = \lambda \mathbf{w}
```

So $`\mathbf{w}`$ must be an eigenvector of $`C`$, and the variance along it is $`\mathbf{w}^\top C \mathbf{w} = \lambda`$. The best direction is the eigenvector with the largest eigenvalue $`\lambda_1`$. The second component maximises the same quantity while staying orthogonal to the first, which gives the eigenvector of the second-largest eigenvalue $`\lambda_2`$, and so on. Because $`C`$ is symmetric and positive semi-definite, its eigenvalues are real and non-negative, and its eigenvectors can be chosen orthonormal. The total variance is the trace of $`C`$ (the sum of its diagonal entries), $`\operatorname{tr}(C) = \sum_{f=1}^{d} \operatorname{Var}(\text{column } f) = \sum_{i} \lambda_i`$. The explained variance ratio of component $`i`$ is $`\lambda_i / \sum_j \lambda_j`$.

**In this code.** `src/fnsl/pca.py` never forms $`C`$; it gets the same eigenvectors from the SVD (8.4.5). The `PCA` class docstring states the link $`C = V (S^2/(n-1)) V^\top`$.

**Pitfall.** PCA is unsupervised: it finds the directions of largest variance, and those need not be the directions that separate the labels (this matters in 8.4.15). PCA is also sensitive to the scale of each feature. The 384 embedding dimensions share one scale, so `pca.py` only centres and does not standardise.

### 8.4.5 PCA via the SVD (how SVD relates to PCA)

**Intuition.** The singular value decomposition (SVD) factorises the centred data matrix itself. Its right singular vectors are exactly the eigenvectors of the covariance matrix, so PCA comes out without ever building that matrix.

**Mathematics.** In 8.4.5 to 8.4.7, $`k`$ is the number of principal components kept (2 for the figure); from 8.4.8 on, $`k`$ is the number of clusters. The thin SVD of the centred data is

```math
X_c = U S V^\top
```

with $`r = \min(n, d)`$; $`U \in \mathbb{R}^{n \times r}`$ has orthonormal columns ($`U^\top U = I`$, the identity matrix); $`S = \operatorname{diag}(s_1, \dots, s_r)`$ holds the singular values $`s_1 \ge s_2 \ge \dots \ge s_r \ge 0`$; and $`V \in \mathbb{R}^{d \times r}`$ has orthonormal columns ($`V^\top V = I`$). Put this into the covariance:

```math
C = \frac{1}{n-1} X_c^\top X_c = \frac{1}{n-1} V S U^\top U S V^\top = V \left(\frac{S^2}{n-1}\right) V^\top
```

This is an eigendecomposition of $`C`$, and it gives three results.

*Directions and variances.* The principal directions are the columns of $`V`$, which are the rows of $`V^\top`$. The variance along direction $`i`$ (the explained variance, equal to the eigenvalue $`\lambda_i`$) is $`s_i^2 / (n-1)`$.

*Scores.* Let $`V_k`$ be the first $`k`$ columns of $`V`$, and $`U_k`$, $`S_k`$ the matching parts of $`U`$ and $`S`$. The coordinates (scores) of the samples on the first $`k`$ directions are

```math
Z = X_c V_k = U S V^\top V_k = U_k S_k
```

because $`V^\top V_k`$ is the first $`k`$ columns of the identity matrix. Column $`i`$ of $`Z`$ has mean 0 and squared length $`s_i^2`$, and different columns are orthogonal, so the new coordinates are uncorrelated.

*Ratios.* The explained variance ratio of component $`i`$ is

```math
\rho_i = \frac{s_i^2 / (n-1)}{\sum_{j=1}^{r} s_j^2 / (n-1)} = \frac{s_i^2}{\sum_{j=1}^{r} s_j^2}
```

Why use the SVD instead of computing $`C`$ and its eigenvectors? Forming $`X_c^\top X_c`$ squares the condition number (the ratio of the largest to the smallest singular value), so directions with small variance lose numerical precision. The SVD works on $`X_c`$ directly.

**In this code.** `src/fnsl/pca.py` → `PCA.fit`:

1. `_as_2d_float` converts the input to a 2-D float64 array (the embeddings are float32 on disk) so that the SVD is accurate.
2. `mean_ = X.mean(axis=0)` and `X_centred = X - mean_`.
3. `_, S, Vt = np.linalg.svd(X_centred, full_matrices=False)` is the thin SVD; $`U`$ is not needed.
4. `Vt = flip_signs(Vt)` fixes the signs (8.4.6).
5. `variance = S**2 / (n_samples - 1)` for all $`r`$ directions, and `total_variance = variance.sum()`, which equals $`\operatorname{tr}(C)`$.
6. Keep the first $`k`$: `components_ = Vt[:k]` (shape $`k \times d`$), `singular_values_ = S[:k]`, `explained_variance_ = variance[:k]`, `explained_variance_ratio_ = variance[:k] / total_variance`.

`PCA.transform` returns `(X - mean_) @ components_.T`, which equals $`U_k S_k`$ on the training data. The constructor refuses `n_components` below 1; `fit` refuses `n_components` above $`\min(n, d)`$, fewer than 2 samples, and data with zero total variance. `scripts/run_kmeans.py` → `main` calls `PCA(n_components=2).fit_transform(embeddings)`: fit on the embeddings only, without labels, and used only for the plot.

**Pitfall.** Forgetting to centre is the classic PCA bug: the first singular vector of uncentred data points towards the mean of the data instead of along its largest spread. The PCA tests use data with a clearly non-zero mean, so a missing centring step would fail them.

> **Explain: how does SVD relate to PCA?** PCA needs the eigenvectors and eigenvalues of the covariance matrix $`C = X_c^\top X_c / (n-1)`$. The SVD of the centred data, $`X_c = U S V^\top`$, gives $`C = V\,(S^2/(n-1))\,V^\top`$. So the right singular vectors (the rows of $`V^\top`$) are the principal directions, $`s_i^2/(n-1)`$ is the variance along direction $`i`$, and $`U_k S_k`$ are the coordinates of the samples on the first $`k`$ directions. Keeping the first $`k`$ singular values and vectors also gives the best rank-$`k`$ approximation of the centred data (8.4.7), so the sum of the first $`k`$ explained variance ratios is the share of the total squared spread that the best $`k`$-dimensional summary keeps. The SVD is preferred because it never forms $`C`$, which would square the condition number and lose accuracy.

### 8.4.6 The sign convention, and how the test compares with scikit-learn

**Intuition.** A singular vector is defined only up to its sign: $`\mathbf{v}`$ and $`-\mathbf{v}`$ describe the same axis. Without a fixed rule, the same data could produce a mirrored plot on another computer.

**Mathematics.** The SVD is a sum of rank-1 terms, $`X_c = \sum_{i=1}^{r} s_i \mathbf{u}_i \mathbf{v}_i^\top`$, where $`\mathbf{u}_i`$ and $`\mathbf{v}_i`$ are column $`i`$ of $`U`$ and $`V`$. Replacing $`(\mathbf{u}_i, \mathbf{v}_i)`$ by $`(-\mathbf{u}_i, -\mathbf{v}_i)`$ leaves every term unchanged, because $`s_i(-\mathbf{u}_i)(-\mathbf{v}_i)^\top = s_i \mathbf{u}_i \mathbf{v}_i^\top`$, so both choices are valid. The rule used here: in row $`i`$ of $`V^\top`$, find the entry with the largest absolute value, at column $`f_i^{*} = \arg\max_f \lvert (V^\top)_{if} \rvert`$, and multiply the row by the sign of $`(V^\top)_{i f_i^{*}}`$, so that this entry becomes positive. Flipping component $`i`$ only changes the sign of score column $`\mathbf{z}_i`$ (the plot is mirrored along that axis). The variances $`s_i^2/(n-1)`$, the distances between projected points, and the reconstruction do not change, because $`(-\mathbf{z}_i)(-\mathbf{v}_i)^\top = \mathbf{z}_i \mathbf{v}_i^\top`$.

**In this code.** `src/fnsl/pca.py` → `flip_signs` implements this rule; a zero sign, possible only for an all-zero row, is treated as +1. scikit-learn 1.9.1 uses the same rule: its PCA calls `svd_flip(U, Vt, u_based_decision=False)`. In a toy check while writing this guide, the two libraries' components agreed even before any sign alignment. The test does not rely on that. In `tests/test_pca.py`:

- `_sign_alignment` takes, for every component, the sign of the dot product between our row and scikit-learn's row. It fails if a dot product is 0, which would mean the two vectors are orthogonal, that is, really different and not just flipped.
- `_assert_matches_sklearn` compares with `SklearnPCA(n_components=k, svd_solver="full")`. `"full"` is the exact solver; `"auto"` may pick an approximate or covariance-based one. It checks `mean_`, `explained_variance_`, `explained_variance_ratio_` and `singular_values_` with `np.testing.assert_allclose` at its default relative tolerance $`10^{-7}`$, then the sign-aligned components (absolute tolerance $`10^{-10}`$ on top of that relative tolerance) and the sign-aligned projections `transform(X)` (absolute tolerance $`10^{-8}`$).
- It runs for `n_components` = 1, 3 and 10 on correlated random data with a non-zero mean ($`200 \times 10`$), and for 3 components on data that lie exactly on a shifted 3-dimensional plane ($`150 \times 12`$), where the three ratios must sum to 1 and a fourth ratio must be below $`10^{-12}`$.
- `test_components_are_orthonormal_and_signs_are_fixed` checks that the rows of `components_` are orthonormal and that each row's largest-magnitude entry is positive. `test_projection_is_u_times_s_with_the_explained_variances` checks $`Z = U_k S_k`$: the column lengths of $`Z`$ equal the singular values, and the covariance of $`Z`$ is diagonal with the explained variances on the diagonal.

**Pitfall.** Never read meaning into the sign of a principal component ("positive PC1 means bullish") unless a sign convention is fixed. The plan suggested comparing absolute values to allow for sign flips; the test is stricter, because it aligns the signs and then compares the vectors themselves.

### 8.4.7 inverse_transform and the best rank-k approximation (Eckart–Young)

**Intuition.** Mapping $`k`$ coordinates back to $`d`$ dimensions gives each point's closest point on the $`k`$-dimensional plane through the mean. No other $`k`$-dimensional plane loses less squared error.

**Mathematics.** The reconstruction is

```math
\hat{X} = Z V_k^\top + \mathbf{1}\bar{\mathbf{x}}^\top = U_k S_k V_k^\top + \mathbf{1}\bar{\mathbf{x}}^\top
```

The Eckart–Young (–Mirsky) theorem says that, among all $`n \times d`$ matrices $`M`$ of rank at most $`k`$ (at most $`k`$ linearly independent columns),

```math
\min_{\operatorname{rank}(M) \le k} \lVert X_c - M \rVert_F^2 = \lVert X_c - U_k S_k V_k^\top \rVert_F^2 = \sum_{i=k+1}^{r} s_i^2
```

where $`\lVert M \rVert_F^2 = \sum_{i,j} M_{ij}^2`$ is the squared Frobenius norm (the sum of all squared entries). Since $`\lVert X_c \rVert_F^2 = \sum_{i=1}^{r} s_i^2`$, dividing gives

```math
\frac{\lVert X_c - U_k S_k V_k^\top \rVert_F^2}{\lVert X_c \rVert_F^2} = 1 - \sum_{i=1}^{k} \rho_i
```

So the explained variance ratios also measure how much of the squared spread a $`k`$-dimensional summary keeps.

**In this code.** `src/fnsl/pca.py` → `PCA.inverse_transform` returns `Z @ components_ + mean_`. No script uses it; two tests do. `test_reconstruction_with_all_components_recovers_x` checks that with $`k = d`$ (here $`d = 8`$) it returns $`X`$ to within $`10^{-10}`$, and `test_rank_k_data_is_recovered_from_k_components` checks that rank-3 data come back from 3 components to within $`10^{-10}`$.

**Why it matters here.** For the figure, $`\rho_1 + \rho_2 = 0.0383 + 0.0278 \approx 0.066`$ (`results/metrics/kmeans.json` → `pca_explained_variance_ratio_2d`). The 2-D view therefore leaves about 93% of the squared spread of the embeddings out. Overlap in the scatter plot is weak evidence on its own; the silhouette, ARI and NMI were all computed in the full 384-dimensional space.

### 8.4.8 K-Means: the objective and Lloyd's algorithm

**Intuition.** K-Means places $`k`$ centres and gives each point to its nearest centre. It moves the centres until the points are, in total, as close as possible to their own centre.

**Mathematics.** Take points $`\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^{d}`$, an assignment $`c_i \in \{1, \dots, k\}`$ for every point (the code numbers clusters from 0), and centres $`\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_k \in \mathbb{R}^{d}`$. The objective is the within-cluster sum of squares (scikit-learn calls its final value *inertia*):

```math
J(c, \boldsymbol{\mu}) = \sum_{i=1}^{n} \lVert \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \rVert^2
```

Lloyd's algorithm alternates two steps:

1. **Assign:** $`c_i \leftarrow \arg\min_{j} \lVert \mathbf{x}_i - \boldsymbol{\mu}_j \rVert^2`$ for every point.
2. **Update:** $`\boldsymbol{\mu}_j \leftarrow \frac{1}{\lvert G_j \rvert} \sum_{i \in G_j} \mathbf{x}_i`$, where $`G_j = \{i : c_i = j\}`$ is the set (group) of points in cluster $`j`$ and $`\lvert G_j \rvert`$ is their number.

Neither step can increase $`J`$. Step 1 picks the smallest term for each point. In step 2, for a fixed cluster, $`\sum_{i \in G_j} \lVert \mathbf{x}_i - \boldsymbol{\mu} \rVert^2`$ has gradient $`-2\sum_{i \in G_j} (\mathbf{x}_i - \boldsymbol{\mu})`$ with respect to $`\boldsymbol{\mu}`$. This is zero exactly at the mean, and the function is convex, so the mean is the minimiser. $`J`$ cannot go below 0 and there are finitely many ways to partition the points, so the algorithm stops, but at a local minimum: finding the global minimum is NP-hard in general. One iteration costs $`O(nkd)`$ operations.

scikit-learn stops a run when the assignments no longer change, or when the total squared centre movement is small,

```math
\sum_{j=1}^{k} \lVert \boldsymbol{\mu}_j^{\text{new}} - \boldsymbol{\mu}_j^{\text{old}} \rVert^2 \le \texttt{tol} \times \frac{1}{d} \sum_{f=1}^{d} \sigma_f^2
```

where $`\sigma_f^2`$ is the variance of column $`f`$ of the data (`np.var`, which divides by $`n`$) and `tol = 1e-4`, or after `max_iter = 300` iterations.

**In this code.** `scripts/run_kmeans.py` → `evaluate_kmeans` calls `KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(embeddings)` with `N_INIT = 10` and `seed = 42` (`--seed`, default `config.SEED`). Everything else is the scikit-learn 1.9.1 default: `init="k-means++"`, `algorithm="lloyd"`, `max_iter=300`, `tol=1e-4`. `--ks` defaults to `[3, 5]`; `parse_args` accepts only $`2 \le k \le 8`$ (8 is the number of plot colours, `len(CLUSTER_COLORS)`) and then sorts and de-duplicates the list.

**Pitfall.** K-Means always returns $`k`$ clusters, even when the data have no cluster structure at all. It also favours round clusters of similar spread. Check a structure score such as the silhouette (8.4.10) before reading meaning into the clusters.

### 8.4.9 k-means++ initialisation, n_init and random_state

**Intuition.** Lloyd's algorithm only finds a local minimum, and which one depends on the starting centres. k-means++ spreads the starting centres out. Repeating the whole run several times and keeping the best one lowers the risk of a poor local minimum.

**Mathematics.** k-means++ (Arthur and Vassilvitskii, 2007):

1. Choose the first centre uniformly at random from the data points.
2. Choose each next centre from the data points with probability

```math
P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}
```

where $`D(\mathbf{x})`$ is the distance from $`\mathbf{x}`$ to the nearest centre chosen so far, and the sum runs over all data points $`\mathbf{x}'`$. Points far from every existing centre are likely to be picked. For this original version the expected objective is at most $`8(\ln k + 2)`$ times the optimum, where $`\ln`$ is the natural logarithm.

scikit-learn implements *greedy* k-means++: at each step it samples $`2 + \lfloor \ln k \rfloor`$ candidates with the probabilities above ($`\lfloor \cdot \rfloor`$ rounds down; this gives 3 candidates for both $`k = 3`$ and $`k = 5`$) and keeps the one that lowers $`\sum_{\mathbf{x}} D(\mathbf{x})^2`$ the most. The guarantee above is proved for the original version, not the greedy one. `n_init` repeats initialisation plus Lloyd's algorithm that many times and keeps the run with the lowest $`J`$.

**In this code.** `scripts/run_kmeans.py` sets `N_INIT = 10`; scikit-learn 1.9.1's default `n_init="auto"` would do only 1 run with k-means++. `random_state=seed` (42) fixes all 10 starts. The run is reproducible: re-running the PCA and both K-Means fits on the cached train embeddings while writing this guide (scikit-learn 1.9.1, NumPy 2.5.4, one CPU thread) reproduced both contingency tables and every stored silhouette, ARI and NMI value in `kmeans.json` exactly, and the PCA ratios to 15 significant digits. Other library versions or BLAS builds may change the last digits, or in rare cases the clustering.

**Why it matters here.** Cluster numbers are arbitrary names. Without a fixed `random_state`, "cluster 1 is the most Neutral" could become "cluster 2" on the next run, and memberships could shift.

### 8.4.10 The silhouette coefficient (cluster structure, no labels)

**Intuition.** For each text, ask: is it clearly closer to the texts of its own cluster than to the texts of the nearest other cluster? Then average the answer over all texts.

**Mathematics.** Let text $`i`$ be in cluster $`c_i`$, with $`G_j`$ the set of texts in cluster $`j`$ (8.4.8), and let $`\operatorname{dist}(i, l) = \lVert \mathbf{x}_i - \mathbf{x}_l \rVert`$ be the Euclidean distance (not squared) between texts $`i`$ and $`l`$:

```math
a(i) = \frac{1}{\lvert G_{c_i} \rvert - 1} \sum_{l \in G_{c_i},\ l \ne i} \operatorname{dist}(i, l), \qquad b(i) = \min_{j \ne c_i} \frac{1}{\lvert G_j \rvert} \sum_{l \in G_j} \operatorname{dist}(i, l)
```

```math
s(i) = \frac{b(i) - a(i)}{\max\{a(i),\ b(i)\}}, \qquad \bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)
```

$`a(i)`$ is the mean distance to the other members of its own cluster. $`b(i)`$ is the mean distance to the members of the nearest other cluster. Each $`s(i)`$ lies between −1 and 1, and $`s(i) = 0`$ for a text that is alone in its cluster (scikit-learn sets it to 0). $`s(i)`$ near 1 means the text sits well inside its cluster, near 0 means on a border, and negative means it is closer to another cluster. $`\bar{s}`$ is the score reported. (This $`s(i)`$ has nothing to do with the singular values $`s_i`$ of 8.4.5.) Toy example: $`a(i) = 0.8`$ and $`b(i) = 1.0`$ give $`s(i) = 0.2 / 1.0 = 0.2`$.

**In this code.** `scripts/run_kmeans.py` → `evaluate_kmeans` calls `silhouette_score(embeddings, clusters, metric="euclidean")` on all texts (no sub-sampling: `sample_size` keeps its default `None`). On unit vectors $`\operatorname{dist}(i, l) = \sqrt{2 - 2\cos(\mathbf{x}_i, \mathbf{x}_l)}`$ (8.4.3). The cost grows with $`n^2`$ because every pair of texts is compared; scikit-learn computes the distances in chunks.

**Pitfall.** The silhouette uses no labels: it says whether there is cluster structure, not whether that structure is sentiment. In high dimensions, pairwise distances often lie in a narrow range, which can push silhouettes towards 0. Compare settings with each other rather than with textbook thresholds. For scale, `tests/test_kmeans.py` → `test_separated_groups_with_matching_labels_score_perfectly` requires a silhouette above 0.8 for three tight, well-separated 2-D blobs.

### 8.4.11 The contingency table and the adjusted Rand index (ARI)

**Notation for 8.4.11 to 8.4.13.** The contingency table counts $`n_{gy}`$, the number of texts in cluster $`g`$ with label $`y`$. Row sums $`a_g = \sum_y n_{gy}`$ are the cluster sizes, column sums $`b_y = \sum_g n_{gy}`$ are the label counts, and $`n = \sum_{g,y} n_{gy}`$ is the number of texts. (These counts $`a_g`$, $`b_y`$ are unrelated to the silhouette's $`a(i)`$, $`b(i)`$.) $`\binom{x}{2} = x(x-1)/2`$ is the number of pairs that can be formed from $`x`$ items.

**Intuition.** Look at every pair of texts. The clustering and the labels agree on a pair if both put the two texts together, or both keep them apart. Count the agreements, then subtract the count that random clusters of the same sizes would get.

**Mathematics.** $`\sum_{g,y}\binom{n_{gy}}{2}`$ counts the pairs that are together in both groupings, $`\sum_g \binom{a_g}{2}`$ the pairs in the same cluster, and $`\sum_y \binom{b_y}{2}`$ the pairs with the same label. The ARI has the form (index − expected index) / (maximum index − expected index):

```math
\text{ARI} = \frac{\sum_{g,y} \binom{n_{gy}}{2} - E}{\frac{1}{2}\left[\sum_g \binom{a_g}{2} + \sum_y \binom{b_y}{2}\right] - E}, \qquad E = \frac{\sum_g \binom{a_g}{2} \, \sum_y \binom{b_y}{2}}{\binom{n}{2}}
```

$`E`$ is the expected number of together-in-both pairs if the texts were shuffled randomly between clusters while all cluster sizes and label counts stay fixed (the hypergeometric model). ARI is 1 for identical groupings (cluster names do not matter), close to 0 for random clusters, and negative for worse-than-random agreement; scikit-learn's docstring gives −0.5 as the lower bound. Toy example: labels `[0, 0, 1, 1]` and clusters `[1, 1, 0, 0]` give ARI = 1, because renaming clusters does not matter.

**In this code.** `scripts/run_kmeans.py` → `contingency_table` builds the table with `np.add.at(table, (clusters, labels), 1)`, which adds 1 for every text even when the same (cluster, label) pair repeats. Rows are clusters; columns follow `kmeans.json` → `labels` (Bearish, Bullish, Neutral). `evaluate_kmeans` calls `adjusted_rand_score(labels, clusters)`. scikit-learn 1.9.1 computes it from a 2×2 table of pair counts (`pair_confusion_matrix`), which is algebraically the same formula: recomputing the formula above from the contingency tables stored in `kmeans.json` reproduces the stored ARI values up to floating-point rounding.

**Why it matters here.** The plain Rand index (the share of pairs on which both groupings agree) has no fixed chance level: its value for random clusters depends on the group sizes and is far above 0. Here the classes are unbalanced (15.0%, 20.2% and 64.8% in `data/splits/split_report.json`), so two texts drawn at random share a label with probability about $`0.150^2 + 0.202^2 + 0.648^2 \approx 0.48`$, and a clustering with a few large clusters "agrees" with the labels on many pairs by chance. ARI subtracts the expected agreement $`E`$ for random clusters of the same sizes, so 0 is the chance level for every $`k`$ and every set of cluster sizes. That is why ARI, and not the Rand index or NMI, is the score for comparing $`k = 3`$ with $`k = 5`$.

### 8.4.12 Normalised mutual information (NMI)

**Intuition.** How much does knowing a text's cluster reduce the uncertainty about its label? NMI scales the answer to lie between 0 and 1.

**Mathematics.** Draw one of the $`n`$ texts at random; let $`G`$ be its cluster and $`Y`$ its label. With the notation of 8.4.11, $`\log`$ the natural logarithm, and terms with $`n_{gy} = 0`$ counted as 0:

```math
H(Y) = -\sum_{y} \frac{b_y}{n} \log \frac{b_y}{n}, \qquad H(G) = -\sum_{g} \frac{a_g}{n} \log \frac{a_g}{n}, \qquad I(G; Y) = \sum_{g,y} \frac{n_{gy}}{n} \log \frac{n \, n_{gy}}{a_g \, b_y}
```

```math
\text{NMI} = \frac{I(G; Y)}{\frac{1}{2}\left[H(G) + H(Y)\right]}
```

$`H(Y)`$ is the entropy (uncertainty) of the label, $`H(G)`$ the entropy of the cluster, and $`I(G; Y)`$ the mutual information, which is 0 exactly when clusters and labels are independent. NMI is 1 when the two groupings are the same up to renaming. The base of the logarithm cancels in the ratio.

**In this code.** `evaluate_kmeans` calls `normalized_mutual_info_score(labels, clusters)` with scikit-learn's default `average_method="arithmetic"`, which is the denominator above (checked in the scikit-learn 1.9.1 source). Recomputing from the tables in `kmeans.json` with the arithmetic mean reproduces the stored NMI values; geometric or max normalisation would give different numbers. NMI with the arithmetic mean equals the V-measure (`v_measure_score` gives the same values on these clusterings).

**Pitfall.** NMI is not corrected for chance, and for random clusters it tends to grow with $`k`$. `kmeans.json` is consistent with this, although it does not prove it: from $`k = 3`$ to $`k = 5`$, NMI rises (0.051 to 0.054) while ARI falls (0.050 to 0.043). Compare different $`k`$ with ARI; the `run_kmeans.py` docstring says the same.

### 8.4.13 Majority label and purity

**Intuition.** Name each cluster after its most common label, then ask what share of the cluster carries that label.

**Mathematics.** With the notation of 8.4.11:

```math
\text{majority}(g) = \arg\max_{y} n_{gy}, \qquad \text{purity}(g) = \frac{\max_y n_{gy}}{a_g}, \qquad \text{overall purity} = \frac{1}{n} \sum_{g} \max_y n_{gy}
```

**In this code.** `scripts/run_kmeans.py` → `evaluate_kmeans` stores `cluster_majority_label`, `cluster_purity` (cluster sizes are clipped to at least 1 to avoid 0/0 for an empty cluster) and `overall_purity`.

**Pitfall.** If every cluster's majority is the biggest class, overall purity equals that class's share, whatever the clusters are. Splitting a cluster can never lower purity, and with one text per cluster purity is 1, so purity rewards large $`k`$. Always compare it with the share of the biggest class, here Neutral: 5,340 / 8,241 = 0.648 (`kmeans.json` → `label_counts` and `n`). `tests/test_kmeans.py` → `test_purity_is_high_for_free_when_one_label_dominates` demonstrates this with labels that are unrelated to the clusters.

### 8.4.14 Results (results/metrics/kmeans.json)

Run facts copied from `results/metrics/kmeans.json`: dataset `zeroshot/twitter-financial-news-sentiment`, split `train`, $`n = 8{,}241`$ texts (Bearish 1,235; Bullish 1,666; Neutral 5,340), embedding model `sentence-transformers/all-MiniLM-L6-v2`, `seed` 42, `n_init` 10, `git_commit` `9d28770`, `smoke` false.

| k | Silhouette | ARI | NMI | Majority label of each cluster | Overall purity |
|---|---|---|---|---|---|
| 3 | 0.0117 | 0.0498 | 0.0513 | Neutral, Neutral, Neutral | 0.648 |
| 5 | 0.0114 | 0.0427 | 0.0545 | Neutral for all 5 | 0.648 |

The 2-D PCA, fit on the same 8,241 embeddings, gives explained variance ratios of 0.0383 for PC1 and 0.0278 for PC2 (`pca_explained_variance_ratio_2d`).

Contingency table for k = 3 (counts from `runs[0].contingency`; size is the row sum; purity from `cluster_purity`):

| Cluster | Bearish | Bullish | Neutral | Size | Majority | Purity |
|---|---|---|---|---|---|---|
| 0 | 463 | 936 | 2,014 | 3,413 | Neutral | 0.590 |
| 1 | 263 | 219 | 2,355 | 2,837 | Neutral | 0.830 |
| 2 | 509 | 511 | 971 | 1,991 | Neutral | 0.488 |
| All | 1,235 | 1,666 | 5,340 | 8,241 | Neutral | 0.648 (overall) |

Contingency table for k = 5 (`runs[1]`):

| Cluster | Bearish | Bullish | Neutral | Size | Majority | Purity |
|---|---|---|---|---|---|---|
| 0 | 74 | 156 | 1,073 | 1,303 | Neutral | 0.823 |
| 1 | 315 | 507 | 846 | 1,668 | Neutral | 0.507 |
| 2 | 350 | 245 | 858 | 1,453 | Neutral | 0.591 |
| 3 | 204 | 145 | 1,773 | 2,122 | Neutral | 0.836 |
| 4 | 292 | 613 | 790 | 1,695 | Neutral | 0.466 |
| All | 1,235 | 1,666 | 5,340 | 8,241 | Neutral | 0.648 (overall) |

**The figure** (`results/figures/kmeans_pca.png`). Both panels show the same 2-D PCA coordinates. The left panel colours the k = 3 clusters: they are three adjacent regions of one continuous cloud, with no empty space between them, and cluster 1 lies at positive PC1. The right panel colours the true labels: Bearish (red) and Bullish (blue) points are mixed together wherever they occur, with Bullish leaning a little to the left (negative PC1), the side of cluster 0, which holds 936 of the 1,666 Bullish texts. Neutral (grey) points are more common at positive PC1, the region of cluster 1, which matches its purity of 0.830. The note under the figure says that this view keeps only 7% of the variance (0.066, rounded).

### 8.4.15 What clustering agreement says, and does not say, about sentiment

**What the numbers say.**

1. *There is no clear cluster structure.* Silhouettes of 0.0117 (k = 3) and 0.0114 (k = 5) mean that, on average, a text is barely closer to its own cluster than to the next one: K-Means cut one continuous cloud into slices. The left panel of the figure looks the same way (adjacent regions, no gaps), but that picture alone proves little, because a 2-D view can hide gaps; the silhouette, computed in 384 dimensions, is the evidence.
2. *Sentiment is not the main structure of the embedding space.* ARI is 0.050 (k = 3) and 0.043 (k = 5), far from 1 and close to the chance level of 0, and NMI is about 0.05 for both. No permutation test was run, so this guide makes no claim about whether these small values are statistically above 0; the point is how far they are from 1.
3. *Naming clusters by their majority label is no better than always answering Neutral.* Every cluster, for both k, is majority Neutral, so overall purity (0.648) equals the Neutral share of the train split (5,340 / 8,241).
4. *There is a weak tilt, but no sentiment split.* For k = 3, cluster 1 is more Neutral (purity 0.830) than the split as a whole (0.648), and clusters 0 and 2 hold most of the polar texts (936 + 511 of the 1,666 Bullish texts, and 463 + 509 of the 1,235 Bearish texts). Cluster 0 has about twice as many Bullish (936) as Bearish (463) texts. But both clusters are still majority Neutral, and cluster 2 holds almost equal numbers of Bearish (509) and Bullish (511) texts: no cluster has a Bullish or Bearish majority. The small positive ARI reflects this tilt.

**What the numbers do not say.**

1. *Not that the embeddings carry no sentiment information.* K-Means groups by total squared distance, which is dominated by the directions in which the embeddings vary most. If sentiment moves the vectors along a direction with little variance, K-Means largely ignores it, while a supervised classifier (for example logistic regression on the 384 numbers) could still use it. That supervised probe has not been run in this repo, so the question is open.
2. *Nothing about the supervised models.* Clustering never sees a label. The TF-IDF baseline and the other planned classifiers learn from labels and are judged separately.
3. *A pure cluster is not a sentiment detector.* It can be a topic in which one sentiment happens to be common.
4. *Only for this setup.* The result holds for this embedding model, these values of k, Euclidean K-Means and the train split. Another model, another k or another algorithm could group the texts differently.
5. *Not from the picture.* The 2-D plot keeps about 6.6% of the variance; the conclusions rest on scores computed in 384 dimensions.

**What do the clusters follow, then?** No script in the repo measures this. The training objective (8.4.1) suggests topic and wording. While preparing this guide, a quick look at sample texts and over-represented words per cluster for k = 3 pointed the same way: cluster 0 looked like company news (earnings, EPS, dividends, share offerings, price targets, call transcripts; "beats" and "misses" were both over-represented there), cluster 1 like general, world and political news, and cluster 2 like macro and market news (the Fed, interest rates, oil, futures, the economy, the coronavirus). This look was informal and is not saved in any results file, so treat it as a hypothesis. A short script that saves the top words and ten sample texts per cluster would turn it into evidence.

> **Explain: what does clustering agreement say (and not say) about sentiment?** Clustering agreement measures whether the groups that an unsupervised algorithm finds in the embedding space coincide with the sentiment labels. Here they barely do: ARI is 0.050 and 0.043 and NMI is 0.051 and 0.054 for k = 3 and k = 5; every cluster is majority Neutral; and overall purity (0.648) equals the Neutral share, so the clusters predict sentiment no better than always guessing Neutral. The near-zero silhouettes add that there are no well-separated groups at all. This says that sentiment is not the dominant structure of a general-purpose sentence-embedding space, which most likely organises texts by topic. It does not say that the embeddings lack sentiment information, because a supervised classifier can use a low-variance direction that K-Means ignores. It also says nothing about the supervised models.

### 8.4.16 Tests that guard this part

All of these run in CI (`pytest -m "not heavy"`) except the one marked heavy.

- `tests/test_pca.py`: the scikit-learn comparison and sign checks described in 8.4.6; full reconstruction and recovery of rank-3 data (8.4.7); $`Z = U_k S_k`$ with a diagonal covariance (8.4.5); `flip_signs` on a hand-made matrix; `fit_transform` equals `fit` followed by `transform`; invalid use raises (more components than features, `n_components=0`, `transform` before `fit`, wrong number of features).
- `tests/test_kmeans.py`: `contingency_table` counts a toy case correctly; three tight, well-separated blobs with matching labels score ARI = NMI = 1, a silhouette above 0.8 and purity 1 in every cluster; labels unrelated to the clusters give an ARI near 0 (below 0.05 in absolute value) but an overall purity at least as high as the biggest class's share; `parse_args` sorts and de-duplicates `--ks` and rejects `k = 1` and `--split test`. A smoke run of `main` with fake random 16-dimensional unit vectors (no model, no torch) checks that `normalize=True`, the batch size and the model name reach `embed_texts`; the JSON keys, `model` and `kind`; the absence of `macro_f1` (so `metrics.is_model_metrics` is false and `load_metrics` skips the file); that the runs are for k = 3 and 5; table shapes and sums (column sums equal `label_counts`); the silhouette range; $`1 \ge \rho_1 \ge \rho_2 > 0`$ for the PCA ratios; that the figure exists; and that the smoke run creates no non-smoke `metrics/` folder. Every results path points to a temporary directory (`tests/conftest.py` → `tmp_results`), so the test never touches the real `results/`.
- `tests/test_embeddings.py`: the cache key changes with the texts, their order, the model and the flag, and keeps `["ab", "c"]` apart from `["a", "bc"]`; a cache hit works even when `sentence_transformers` cannot be imported; a cache miss loads the model once, saves float32 and leaves no temporary file behind, and a second call is served from the file; `normalize=False` gets its own file; empty input raises. Heavy (deselected in CI; passes locally): `test_real_model_embeds_normalises_and_caches` loads the real model and checks the shape $`(3, 384)`$, unit length, that two earnings headlines are more similar to each other than to an interest-rate headline, and the caching.

### 8.4.17 Open items for this part

Day 3's "done when" (the PCA test passes and a cluster plot is saved) is met, and the Day 3 commit exists (`1c775c0`, "add kmeans clustering and numpy pca with test"). Nothing else is required by the plan. Items for you, then optional improvements:

- For you: the Day 3 entry in `DEBUG_LOG.md` is marked "(skeleton build)". The plan asks for one real bug per day (deliverable 5 and the daily close-out); decide whether to keep it or add a bug from your own runs.
- Optional: save the top words and sample texts per cluster from a script, so the topic reading in 8.4.15 rests on a file instead of an informal look.
- Optional: pin the embedding model revision in `embed_texts`, and record it (with the sentence-transformers version) in `kmeans.json`. The dataset revision is already pinned in `src/fnsl/config.py`.
- Optional: to test whether the embeddings contain sentiment information, fit a logistic regression on the train embeddings and score it on the validation split only; never tune on the test split.
- Optional: the README `KMEANS` block shows only silhouette, ARI and NMI. Adding the majority labels and overall purity next to the Neutral share would show the most telling fact; `notebooks/kaggle_report.ipynb` already shows overall purity.

---

## Files this part reads and writes

**Reads**
- `data/splits/train.csv` (or `val.csv` with `--split val`; `test` is refused): columns `id,text,label`; label ids 0 = Bearish, 1 = Bullish, 2 = Neutral (`config.LABELS`), validated by `data.load_split`. `--smoke` keeps 300 rows, 100 per label (`data.smoke_subset`).
- `data/splits/split_report.json` → `dataset` (via `data.current_dataset_name`), copied into `kmeans.json`.
- The embedding cache file for exactly these texts, if present (below).

**Writes**
- `cache/embeddings/<model short name>_<16 hex>.npy` (gitignored): float32 array of shape (n, 384), one row per input text in input order, unit length. Key = first 16 hex digits of SHA-256 over the UTF-8 JSON of `{"model", "normalize", "texts"}`. Written as `<stem>.<pid>.tmp.npy`, then renamed into place with `Path.replace`. A full train run gives an 8,241-row file; a smoke run a 300-row file.
- `results/metrics/kmeans.json` (smoke: `results/smoke/metrics/kmeans.json`), own schema, keys in file order:
  - `model` = "kmeans_minilm"; `kind` = "clustering"
  - `dataset`; `split` (train/val); `n` (texts clustered); `embedding_model`
  - `seed` (used for `config.set_seed` and as K-Means `random_state`)
  - `git_commit` (short hash, "-dirty" if src/ or scripts/ had modified or untracked files); `timestamp` (UTC, ISO 8601, seconds); `smoke` (bool)
  - `n_init` (10); `silhouette_distance` (text describing the distance used)
  - `labels` (column order of every contingency table: Bearish, Bullish, Neutral)
  - `label_counts` ({label: count} over the clustered texts)
  - `pca_explained_variance_ratio_2d` ([PC1, PC2] ratios of the NumPy PCA fit on the same embeddings)
  - `runs`: one object per k (sorted) with `k`, `silhouette`, `ari`, `nmi`, `contingency` (k x 3 counts; rows = clusters, columns = labels), `cluster_majority_label` (k names), `cluster_purity` (k shares), `overall_purity`
  - Not a per-model metrics record: it lacks `n_test`, `macro_f1`, `accuracy`, `per_class_f1`, `train_time_s` and `params`, so `metrics.is_model_metrics` is false and `metrics.load_metrics` skips it.
  - Readers: `scripts/make_results_table.py` → `reporting.kmeans_section` renders `embedding_model`, `split`, `n` and, per run, k, silhouette, ARI and NMI into the README `KMEANS` block; `notebooks/kaggle_report.ipynb` (section 5) shows k, silhouette, ARI, NMI and overall purity and displays the PNG.
- `results/figures/kmeans_pca.png` (smoke: `results/smoke/figures/kmeans_pca.png`), saved at 150 dpi: two panels sharing the 2-D PCA coordinates; left coloured by the k = 3 clustering (`PLOT_K`; if 3 is not in `--ks`, the smallest k), right by true label; axis labels show each PC's explained-variance %, and the caption the total share kept.
- MLflow (gitignored `mlruns/`, SQLite store `mlruns/mlflow.db`, experiment `fin-news-sentiment-lab`; skipped for `--smoke` and `--no-mlflow`): run `kmeans_minilm` with params `embedding_model, split, n, ks, n_init, seed`; metrics `silhouette, ari, nmi, overall_purity` logged with `step = k`; artifacts `kmeans.json` and the PNG.
