"""Ground-truth composition pies on CORE-SG / GraphHDBSCAN* condensed trees.

This module adds a plotting utility that overlays a ground-truth composition
pie chart at every cluster node of a condensed tree.

The function accepts, transparently, any of:

* a fitted :class:`GraphCoreSGHDBSCAN` / :class:`CoreSGHDBSCAN` estimator
  (in which case ``m`` selects the per-``min_samples`` tree),
* a per-``m`` :class:`CoreSGModel` (which carries a single ``condensed_tree_``),
* a bare ``hdbscan.plots.CondensedTree`` object,
* or a plain fitted ``hdbscan.HDBSCAN`` estimator.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Patch, Wedge


__all__ = ["plot_condensed_tree_ground_truth_pies", "resolve_condensed_tree"]


def resolve_condensed_tree(tree_or_clusterer, m=None):
    """Return the ``hdbscan`` ``CondensedTree`` to plot.

    This is the adaptive resolution step that makes a single plotting routine
    work for both single-tree objects and this package's per-``min_samples``
    estimators.

    Parameters
    ----------
    tree_or_clusterer
        One of:
          * a fitted ``GraphCoreSGHDBSCAN`` / ``CoreSGHDBSCAN`` estimator
            (holds ``condensed_trees_`` keyed by ``min_samples``);
          * a ``CoreSGModel`` (holds a single ``condensed_tree_``);
          * an ``hdbscan.plots.CondensedTree``;
          * a fitted ``hdbscan.HDBSCAN`` (holds a single ``condensed_tree_``).

    m : int, optional
        The ``min_samples`` value selecting which condensed tree to use.
        Required when ``tree_or_clusterer`` carries several trees
        (``condensed_trees_``); ignored otherwise. If the object carries
        several trees and exactly one is present, that one is used when ``m``
        is omitted.

    Returns
    -------
    hdbscan.plots.CondensedTree
    """

    # Case 1: a multi-tree estimator from this package.
    trees = getattr(tree_or_clusterer, "condensed_trees_", None)
    models = getattr(tree_or_clusterer, "models_", None)

    if trees is not None or models is not None:
        trees = trees or {}
        models = models or {}

        available = sorted(set(trees) | set(models))
        if not available:
            raise ValueError(
                "The estimator has no fitted condensed trees. "
                "Call fit(...) first."
            )

        if m is None:
            if len(available) == 1:
                m = available[0]
            else:
                raise ValueError(
                    "This estimator holds condensed trees for "
                    f"min_samples={available}. Pass m=<min_samples> to "
                    "choose which one to plot."
                )

        m = int(m)
        if m in trees and trees[m] is not None:
            return trees[m]
        if m in models and getattr(models[m], "condensed_tree_", None) is not None:
            return models[m].condensed_tree_
        raise KeyError(
            f"m={m} not found. Available min_samples: {available}."
        )

    # Case 2: a single-tree object (CoreSGModel or hdbscan.HDBSCAN),
    # or already a CondensedTree.
    condensed_tree = getattr(
        tree_or_clusterer,
        "condensed_tree_",
        tree_or_clusterer,
    )

    if not hasattr(condensed_tree, "to_pandas"):
        raise TypeError(
            "Could not resolve a condensed tree from the given object. "
            "Pass a fitted GraphCoreSGHDBSCAN/CoreSGHDBSCAN estimator (with "
            "an `m` argument), a CoreSGModel, an hdbscan CondensedTree, or a "
            "fitted hdbscan.HDBSCAN."
        )

    return condensed_tree


def plot_condensed_tree_ground_truth_pies(
    tree_or_clusterer,
    y_true,
    *,
    m=None,
    min_node_size=2,
    min_pie_diameter=12,
    max_pie_diameter=38,
    show_node_ids=False,
    tree_cmap="Greys",
    label_cmap="tab20",
    leaf_separation=1,
    max_rectangles_per_icicle=20,
    figsize=(16, 10),
    ax=None,
):
    """Plot a condensed tree with a ground-truth composition pie per cluster node.

    Parameters
    ----------
    tree_or_clusterer
        A fitted ``GraphCoreSGHDBSCAN`` / ``CoreSGHDBSCAN`` estimator, a
        ``CoreSGModel``, an ``hdbscan`` ``CondensedTree``, or a fitted
        ``hdbscan.HDBSCAN``. See :func:`resolve_condensed_tree`.

    y_true : array-like, shape (n_samples,)
        Ground-truth labels aligned row-for-row with the data used to fit.

    m : int, optional
        ``min_samples`` value selecting which condensed tree to draw. Required
        only when ``tree_or_clusterer`` holds several trees.

    min_node_size : int
        Only draw pies for cluster nodes containing at least this many samples.

    min_pie_diameter, max_pie_diameter : float
        Pie diameter range in display points (area scales with cluster size).

    show_node_ids : bool
        Annotate each pie with the internal node ID and node size.

    tree_cmap, label_cmap : str
        Colormaps for the icicle bars and the ground-truth classes.

    leaf_separation, max_rectangles_per_icicle : int
        Passed through to the underlying condensed-tree plot so the pies line
        up with the drawn icicles.

    figsize : tuple
        Figure size, used only when ``ax`` is not provided.

    ax : matplotlib.axes.Axes, optional
        Draw on an existing axis instead of creating a new figure.

    Returns
    -------
    fig, ax
    """

    condensed_tree = resolve_condensed_tree(tree_or_clusterer, m=m)

    tree = condensed_tree.to_pandas().copy()

    required_columns = {"parent", "child", "lambda_val", "child_size"}
    if not required_columns.issubset(tree.columns):
        raise ValueError(
            f"Condensed-tree data must contain {required_columns}."
        )

    tree["parent"] = tree["parent"].astype(int)
    tree["child"] = tree["child"].astype(int)
    tree["child_size"] = tree["child_size"].astype(int)

    y_true = np.asarray(y_true, dtype=object)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional label array.")

    n_samples = len(y_true)

    # The minimum parent node is the condensed-tree root.
    root = int(tree["parent"].min())

    # Direct children of the root partition the original dataset.
    tree_n_samples = int(
        tree.loc[tree["parent"] == root, "child_size"].sum()
    )

    if n_samples != tree_n_samples:
        raise ValueError(
            f"y_true has {n_samples} labels, but the condensed tree "
            f"contains {tree_n_samples} samples."
        )

    # Make missing ground-truth labels a visible category.
    true_labels = pd.Series(y_true, dtype="object")
    true_labels = true_labels.where(
        true_labels.notna(),
        "<missing>",
    )

    label_codes, label_values = pd.factorize(
        true_labels,
        sort=False,
    )

    n_classes = len(label_values)

    if n_classes == 0:
        raise ValueError("No ground-truth classes were found.")

    # Build parent -> direct children lookup using the entire condensed tree,
    # including singleton point nodes.
    children = {}

    for row in tree.itertuples(index=False):
        parent = int(row.parent)
        child = int(row.child)
        children.setdefault(parent, []).append(child)

    @lru_cache(maxsize=None)
    def node_label_counts(node):
        """Ground-truth class counts among all descendant samples of a node."""

        # IDs below n_samples correspond to original sample indices.
        if node < n_samples:
            counts = np.zeros(n_classes, dtype=int)
            counts[label_codes[node]] = 1
            return counts

        counts = np.zeros(n_classes, dtype=int)

        for child in children.get(node, []):
            counts += node_label_counts(child)

        return counts

    # Coordinates used internally by the HDBSCAN icicle plot.
    plot_data = condensed_tree.get_plot_data(
        leaf_separation=leaf_separation,
        log_size=False,
        max_rectangle_per_icicle=max_rectangles_per_icicle,
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    condensed_tree.plot(
        axis=ax,
        cmap=tree_cmap,
        colorbar=False,
        select_clusters=False,
        label_clusters=False,
        leaf_separation=leaf_separation,
        log_size=False,
        max_rectangles_per_icicle=max_rectangles_per_icicle,
    )

    # Size of each cluster at the moment it appears.
    node_sizes = {root: n_samples}

    for row in tree.loc[tree["child_size"] > 1].itertuples(index=False):
        node_sizes[int(row.child)] = int(row.child_size)

    cluster_bounds = plot_data["cluster_bounds"]

    # Singleton point nodes are excluded: their pies would be one class at 100%.
    cluster_nodes = [
        int(node)
        for node in cluster_bounds
        if int(node) >= n_samples
        and node_sizes.get(int(node), 0) >= min_node_size
    ]

    if not cluster_nodes:
        raise ValueError(
            "No cluster nodes satisfy the requested min_node_size."
        )

    visible_sizes = np.array(
        [node_sizes[node] for node in cluster_nodes],
        dtype=float,
    )

    sqrt_min = np.sqrt(visible_sizes.min())
    sqrt_max = np.sqrt(visible_sizes.max())

    def pie_diameter(node_size):
        """Scale pie area approximately with cluster size."""

        if sqrt_max == sqrt_min:
            return max_pie_diameter

        normalized = (
            np.sqrt(node_size) - sqrt_min
        ) / (sqrt_max - sqrt_min)

        return (
            min_pie_diameter
            + normalized * (max_pie_diameter - min_pie_diameter)
        )

    color_map = plt.get_cmap(label_cmap, n_classes)
    class_colors = [color_map(i) for i in range(n_classes)]

    for node in cluster_nodes:
        left, right, bottom, top = cluster_bounds[node]

        # Center of this cluster's icicle branch.
        x = (left + right) / 2.0

        # The bottom is the lambda value at which this cluster is born.
        y = bottom

        counts = node_label_counts(node)
        total = counts.sum()

        if total == 0:
            continue

        diameter = pie_diameter(node_sizes[node])
        radius = diameter / 2.0

        drawing = DrawingArea(
            diameter,
            diameter,
            clip=False,
        )

        start_angle = 90.0

        for count, color in zip(counts, class_colors):
            if count == 0:
                continue

            angle = 360.0 * count / total

            wedge = Wedge(
                center=(radius, radius),
                r=radius,
                theta1=start_angle,
                theta2=start_angle + angle,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
            )

            drawing.add_artist(wedge)
            start_angle += angle

        # Outline makes each pie distinguishable from the tree bars.
        drawing.add_artist(
            Circle(
                (radius, radius),
                radius,
                fill=False,
                edgecolor="black",
                linewidth=0.7,
            )
        )

        annotation = AnnotationBbox(
            drawing,
            (x, y),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0,
            annotation_clip=False,
            zorder=10,
        )

        annotation.set_clip_on(False)
        ax.add_artist(annotation)

        if show_node_ids:
            ax.annotate(
                f"{node}\nn={node_sizes[node]}",
                xy=(x, y),
                xytext=(0, -(diameter / 2.0 + 3)),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                zorder=11,
            )

    legend_handles = [
        Patch(
            facecolor=class_colors[i],
            edgecolor="black",
            label=str(label),
        )
        for i, label in enumerate(label_values)
    ]

    ax.legend(
        handles=legend_handles,
        title="Ground truth",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )

    title = "CORE-SG condensed tree with ground-truth composition"
    if m is not None:
        title += f" (min_samples = {int(m)})"
    ax.set_title(title)

    # Start the lambda axis at 1 (= 1/max_eps) and dash the synthetic
    # weight-1 joins that appear when the graph is disconnected.
    floor = 1.0
    for line in ax.get_lines():
        yd = np.asarray(line.get_ydata(), float)
        if yd.size == 2 and np.all(np.isclose(yd, floor, atol=1e-9)):
            line.set_linestyle((0, (5, 3)))
            line.set_color("0.35")
            line.set_linewidth(1.3)

    lower, upper = ax.get_ylim()
    span = abs(lower - upper) or 1.0
    ax.set_ylim(lower + 0.04 * span, floor)  # inverted axis: top = floor

    return fig, ax
