"""graphhdbscan_optimized

Optimized CoreSG-HDBSCAN density clustering.

This is a drop-in, output-preserving optimization of the original
``GraphHDBSCAN`` package. The public API is unchanged; the internals avoid the
dense O(N^2) hotspots (full-row argsort, ``triu_indices`` scans, eager dense
fill-1 matrix and NetworkX MST) and add a sparse CORE-SG hand-off.

Public classes
--------------
GraphCoreSGHDBSCAN
    Graph-based CoreSG-HDBSCAN estimator (the main entry point).
CoreSGHDBSCAN
    Generic CoreSG-HDBSCAN over a distance matrix / sparse graph.
CoreSGModel
    Lightweight HDBSCAN-like per-``min_samples`` result wrapper.
"""

from .core import CoreSGHDBSCAN, CoreSGModel
from .graph import GraphCoreSGHDBSCAN
from .ground_truth_pies import (
    plot_condensed_tree_ground_truth_pies,
    resolve_condensed_tree,
)

__all__ = [
    "GraphCoreSGHDBSCAN",
    "CoreSGHDBSCAN",
    "CoreSGModel",
    "plot_condensed_tree_ground_truth_pies",
    "resolve_condensed_tree",
]

__version__ = "0.2.0-optimized"
