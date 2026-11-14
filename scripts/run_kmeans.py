"""K-Means clustering of sentence embeddings, plotted with our NumPy PCA (Day 3).

    python scripts/run_kmeans.py              # train split, k = 3 and 5
    python scripts/run_kmeans.py --smoke      # 300 texts, writes to results/smoke/

Every text is embedded with all-MiniLM-L6-v2 (384 numbers, scaled to length 1). K-Means
then splits the vectors into k groups: assign each point to its nearest centre, move
each centre to the mean of its points, repeat until the centres stop moving. The result
depends on the starting centres, so scikit-learn picks them 10 times with k-means++
(random, but spread apart) and keeps the run with the smallest inertia (sum of squared
distances from each point to its centre). The sentiment labels are NOT used for
clustering, only afterwards, to check whether the groups line up with them:

- Silhouette (-1 to 1, uses no labels): for each text, a = mean distance to the other
  texts in its cluster, b = mean distance to the texts of the nearest other cluster, and
  s = (b - a) / max(a, b); the score is the mean of s over all texts. Near 1: tight,
  well separated clusters; near 0: overlapping. Distance: Euclidean on the length-1
  embeddings. For unit vectors |x - y|^2 = 2 - 2 cos(x, y), so it orders pairs exactly
  like cosine distance (the usual measure for sentence embeddings), and squared
  Euclidean distance is what K-Means itself minimises. (The silhouette value is still
  not the same number as a cosine-distance silhouette, because it averages distances.)
- Adjusted Rand index (ARI): over all pairs of texts, how often clusters and labels agree
  on "same group" vs "different groups", corrected for chance (0 = random, 1 = identical).
- Normalised mutual information (NMI): the mutual information between cluster and label
  (how much knowing the cluster reduces the uncertainty about the label), divided by the
  average entropy of the two groupings. 0 = independent, 1 = the same grouping up to
  renaming; splitting one label over several pure clusters already scores below 1. NMI
  is not corrected for chance and creeps up as k grows, so compare different k with ARI.
- Contingency table (cluster x label counts), each cluster's majority label and its
  purity (share of the cluster that has that label). Compare purity with the share of
  the biggest class (Neutral): clusters that are all mostly Neutral look "pure" for free.

What agreement does and does not say about sentiment: K-Means groups texts that are
close in embedding space, and a general-purpose sentence embedding places texts by
everything they say, so clusters can follow topic or wording (analyst rating changes,
earnings, macro news, ...) instead of sentiment. Low ARI/NMI means sentiment is not the
dominant structure of the embedding space, not that the embeddings lack sentiment
information (a supervised classifier on them could still do well). A pure cluster can
be a topic that happens to be mostly one sentiment rather than a sentiment detector.

Writes results/metrics/kmeans.json (its own schema with no macro_f1, so model tables
skip it) and results/figures/kmeans_pca.png, and logs the run to MLflow (mlruns/).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from fnsl import config, data, metrics
from fnsl.embeddings import DEFAULT_MODEL, embed_texts
from fnsl.pca import PCA
from fnsl.tracking import mlflow_run

MODEL_NAME = "kmeans_minilm"
N_INIT = 10  # 10 k-means++ starts; the run with the lowest inertia is kept
SMOKE_N = 300  # texts in a --smoke run
PLOT_K = 3  # the figure shows the k=3 clustering (one cluster per sentiment class)
SILHOUETTE_DISTANCE = "euclidean on L2-normalised embeddings (same ranking as cosine distance)"

# Colours from a palette checked for colour-blind safety. The first three stay
# distinguishable in a scatter plot; more clusters reuse further slots.
CLUSTER_COLORS = [
    "#eb6834",
    "#1baf7a",
    "#6250d6",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#2a78d6",
    "#e34948",
]
# Labels are polar (Bearish < Neutral < Bullish): two opposite hues and a grey middle.
LABEL_COLORS = {"Bearish": "#e34948", "Bullish": "#2a78d6", "Neutral": "#898781"}
INK, MUTED_INK, AXIS_GREY, GRID_GREY = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Means on sentence embeddings + NumPy PCA")
    # The test split is reserved for the final evaluation of the classifiers.
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 5], help="cluster counts")
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_N} texts; results/smoke/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="sentence-transformers model")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args(argv)
    if any(k < 2 or k > len(CLUSTER_COLORS) for k in args.ks):
        parser.error(f"every k must be between 2 and {len(CLUSTER_COLORS)}, got {args.ks}")
    args.ks = sorted(set(args.ks))
    return args


def contingency_table(clusters: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Counts with one row per cluster and one column per label (config.LABELS order)."""
    table = np.zeros((k, config.NUM_LABELS), dtype=int)
    np.add.at(table, (clusters, labels), 1)  # table[c, y] += 1 for every text
    return table


def evaluate_kmeans(
    embeddings: np.ndarray, labels: np.ndarray, k: int, seed: int
) -> tuple[dict, np.ndarray]:
    """Cluster with K-Means and score the result against the labels.

    Returns the run record for kmeans.json and the cluster id of every text.
    """
    kmeans = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)
    clusters = kmeans.fit_predict(embeddings)

    table = contingency_table(clusters, labels, k)
    majority_counts = table.max(axis=1)
    cluster_sizes = np.maximum(table.sum(axis=1), 1)  # max(.., 1) avoids 0/0 if one is empty
    run = {
        "k": k,
        "silhouette": float(silhouette_score(embeddings, clusters, metric="euclidean")),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "contingency": table.tolist(),
        "cluster_majority_label": [config.ID2LABEL[int(i)] for i in table.argmax(axis=1)],
        "cluster_purity": (majority_counts / cluster_sizes).tolist(),
        # Share of all texts that carry their cluster's majority label.
        "overall_purity": float(majority_counts.sum() / table.sum()),
    }
    return run, clusters


def plot_pca(
    coords: np.ndarray,
    clusters: np.ndarray,
    labels: np.ndarray,
    run: dict,
    variance_ratio: list[float],
    path: Path,
    title: str,
) -> None:
    """Two scatter plots of the same 2-D PCA coordinates: coloured by K-Means cluster
    (left) and by true sentiment label (right)."""
    import matplotlib.pyplot as plt  # imported here so importing this script stays cheap

    fig, (ax_c, ax_y) = plt.subplots(
        1, 2, figsize=(11, 5.6), sharex=True, sharey=True, layout="constrained"
    )
    dot = {"s": 5, "alpha": 0.45, "linewidths": 0}

    for c in range(run["k"]):
        mask = clusters == c
        purity, majority = run["cluster_purity"][c], run["cluster_majority_label"][c]
        name = f"cluster {c}: {mask.sum():,} texts, {purity:.0%} {majority}"
        ax_c.scatter(coords[mask, 0], coords[mask, 1], color=CLUSTER_COLORS[c], label=name, **dot)
    for c in range(run["k"]):  # cluster number at the centre of its points
        cx, cy = coords[clusters == c].mean(axis=0)
        box = {"boxstyle": "round,pad=0.25", "fc": "white", "ec": AXIS_GREY, "lw": 0.6}
        ax_c.annotate(str(c), (cx, cy), ha="center", va="center", fontsize=9, color=INK, bbox=box)

    # Neutral (about two thirds of the texts) is drawn first, so it cannot hide the others.
    for name in ["Neutral", "Bearish", "Bullish"]:
        mask = labels == config.LABEL2ID[name]
        label = f"{name}: {mask.sum():,} texts"
        ax_y.scatter(coords[mask, 0], coords[mask, 1], color=LABEL_COLORS[name], label=label, **dot)

    ax_c.set_title(f"K-Means clusters (k={run['k']}), found without labels", fontsize=10)
    ax_y.set_title("True sentiment labels", fontsize=10)
    ax_c.set_ylabel(f"PC2 ({variance_ratio[1]:.1%} of variance)", color=MUTED_INK)
    for ax in (ax_c, ax_y):
        ax.set_xlabel(f"PC1 ({variance_ratio[0]:.1%} of variance)", color=MUTED_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS_GREY)
        ax.grid(color=GRID_GREY, linewidth=0.5)
        ax.set_axisbelow(True)
        legend = ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=8
        )
        for handle in legend.legend_handles:  # opaque, larger legend dots
            handle.set_alpha(1.0)
            handle.set_sizes([30])

    fig.suptitle(title, fontsize=11, color=INK)
    # supxlabel (not fig.text) so the constrained layout keeps it clear of the legends.
    fig.supxlabel(
        f"K-Means used all embedding dimensions; this 2-D view keeps only "
        f"{sum(variance_ratio):.0%} of the variance. Cluster colours and label colours "
        "are unrelated.",
        fontsize=8,
        color=MUTED_INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config.set_seed(args.seed)

    df = data.load_split(args.split)
    if args.smoke:
        df = data.smoke_subset(df, SMOKE_N)
    texts, labels = df["text"].tolist(), df["label"].to_numpy(dtype=int)
    print(f"{args.split}: {len(df)} texts, smoke={args.smoke}, ks={args.ks}")

    start = time.perf_counter()
    # normalize=True gives unit-length rows, which the silhouette distance relies on.
    embeddings = embed_texts(
        texts, model_name=args.model_name, batch_size=args.batch_size, normalize=True
    )
    print(f"embeddings {embeddings.shape} ready in {time.perf_counter() - start:.1f}s")

    dirs = config.output_dirs(args.smoke)
    json_path = dirs["metrics"] / "kmeans.json"
    fig_path = dirs["figures"] / "kmeans_pca.png"
    params = {
        "embedding_model": args.model_name,
        "split": args.split,
        "n": len(df),
        "ks": args.ks,
        "n_init": N_INIT,
        "seed": args.seed,
    }
    with mlflow_run(MODEL_NAME, params, enabled=not (args.smoke or args.no_mlflow)) as run:
        runs, clusters_by_k = [], {}
        for k in args.ks:
            result, clusters_by_k[k] = evaluate_kmeans(embeddings, labels, k, args.seed)
            runs.append(result)
            print(
                f"  k={k}: silhouette={result['silhouette']:.4f} ari={result['ari']:.4f} "
                f"nmi={result['nmi']:.4f} overall purity={result['overall_purity']:.4f} "
                f"majority labels={result['cluster_majority_label']}"
            )
            scores = ("silhouette", "ari", "nmi", "overall_purity")
            run.log_metrics({name: result[name] for name in scores}, step=k)

        # Our NumPy PCA, fit on the embeddings only (no labels), for the 2-D picture.
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        variance_ratio = [float(r) for r in pca.explained_variance_ratio_]
        print(f"PCA 2-D explained variance ratio: {variance_ratio}")

        plot_k = PLOT_K if PLOT_K in clusters_by_k else args.ks[0]
        title = (
            f"{args.model_name.split('/')[-1]} embeddings of the {args.split} split "
            f"(n={len(df):,}), projected to 2-D with NumPy PCA" + (" [SMOKE]" if args.smoke else "")
        )
        plot_run = next(r for r in runs if r["k"] == plot_k)
        plot_pca(coords, clusters_by_k[plot_k], labels, plot_run, variance_ratio, fig_path, title)

        label_counts = np.bincount(labels, minlength=config.NUM_LABELS)
        record = {
            "model": MODEL_NAME,
            "kind": "clustering",
            "dataset": data.current_dataset_name(),
            "split": args.split,
            "n": len(df),
            "embedding_model": args.model_name,
            "seed": args.seed,
            "git_commit": metrics.git_commit(),
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "smoke": args.smoke,
            "n_init": N_INIT,
            "silhouette_distance": SILHOUETTE_DISTANCE,
            "labels": list(config.LABELS),  # column order of every contingency table
            "label_counts": {name: int(label_counts[i]) for i, name in config.ID2LABEL.items()},
            "pca_explained_variance_ratio_2d": variance_ratio,
            "runs": runs,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        run.log_artifact(json_path)
        run.log_artifact(fig_path)

    print(f"wrote {json_path} and {fig_path}")
    return record


if __name__ == "__main__":
    main()
