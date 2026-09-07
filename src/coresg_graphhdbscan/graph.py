"""Graph-based wrapper around CoreSG-HDBSCAN (optimized).

Speed/memory notes vs. the previous revision
---------------------------------------------
* The initial similarity graph is kept in scipy-sparse form end to end.
  The old pipeline built it as a scipy matrix, wrapped it in a NetworkX
  graph, then looped over ``graph.edges(data=True)`` in Python to rebuild
  a scipy matrix again -- an O(E) scipy->networkx->scipy round trip that
  allocates a dict-of-dicts for every edge. That round trip is gone;
  ``_wss_similarity_sparse`` works directly on the sparse adjacency.
* ``similarity_graph_`` / ``similarity_graph_WSS`` / ``dissimilarity_graph_``
  / ``connected_graph_`` are lazily-built NetworkX views (properties). They
  are debug/inspection only and never touched by ``fit``.
* The MST used for optional noise reassignment is computed with scipy
  (C-level) on the sparse matrix; only the tiny (n-1 edge) result becomes
  a NetworkX graph.
* Component bridging is done entirely in scipy-sparse form.
* In the ``heuristic_connect`` search the full distance matrix is computed
  once and reused across iterations (it does not depend on n_neighbors).

Result equivalence
------------------
For the default ``add_neighbor=True`` path the weighted structural
similarity is bit-identical to the previous implementation (same ``A@A.T``
on the same adjacency ``A``). The only non-deterministic choices left are
(a) which minimum spanning tree is returned among equal-weight MSTs and
(b) which representative nodes are bridged when the graph is disconnected;
neither changes the clustering result (bridge weight 1 equals the
non-edge fill value, so ``dist_matrix_`` is unaffected), and any MST-tie
difference only affects individual noise-point reassignment, which is
inherently ambiguous at ties.
"""

from .core import CoreSGHDBSCAN

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
#import fast_hdbscan as hdbscan
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import scipy.sparse as sp
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors as NN, kneighbors_graph
import heapq
import time
import inspect
import warnings
from functools import wraps
from collections.abc import Iterable
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    njit = None
    prange = range
    _HAS_NUMBA = False


# Second-order ("distance-of-distances") metrics. Each point is represented by
# its full distance profile, and the second metric is applied to those profiles.
#   name -> (metric for the full distance matrix, metric applied to its ROWS)
_SECOND_ORDER_METRICS = {
    'hybrid_euclidean_cosine': ('euclidean', 'cosine'),
    'euclidean_ii':            ('euclidean', 'euclidean'),
    'cosine_ii':               ('cosine',    'cosine'),
}


def _optional_import(module_name, package_name=None):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pkg = package_name or module_name
        raise ImportError(f"Optional dependency '{pkg}' is required for this functionality. Please install it.") from e


def _timeit(func):
    """Decorator that prints how long the wrapped function/method took to run."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[TIMER] {func.__qualname__}: {elapsed:.4f}s")
    return wrapper


def _get_scanpy_modules():
    sc = _optional_import("scanpy")
    sce = _optional_import("scanpy.external")
    sc_neighbors_connectivity = _optional_import("scanpy.neighbors._connectivity")
    sc_neighbors_common = _optional_import("scanpy.neighbors._common")
    return (
        sc,
        sce,
        sc_neighbors_connectivity.gauss,
        sc_neighbors_connectivity.umap,
        sc_neighbors_common._get_indices_distances_from_dense_matrix,
    )


if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _directed_jaccard_weights_numba(knn_idx, sorted_knn_idx):
        n, k = knn_idx.shape
        weights = np.zeros((n, k), dtype=np.float64)

        for i in prange(n):
            ni = sorted_knn_idx[i]

            for t in range(k):
                j = knn_idx[i, t]
                nj = sorted_knn_idx[j]

                a = 0
                b = 0
                shared = 0

                while a < k and b < k:
                    va = ni[a]
                    vb = nj[b]

                    if va == vb:
                        shared += 1
                        a += 1
                        b += 1
                    elif va < vb:
                        a += 1
                    else:
                        b += 1

                if shared > 0:
                    weights[i, t] = shared / ((2.0 * k) - shared)

        return weights


class GraphCoreSGHDBSCAN(CoreSGHDBSCAN):
    """
    Graph-based CoreSG + HDBSCAN interface.

    See the module docstring for the speed/memory characteristics of this
    optimized revision. The public API (parameters, attributes, methods) is
    unchanged relative to the previous version.
    """
    def __init__(
                self,
                min_samples=list(range(2, 31)),
                sim_graph_method='sc_umap',
                metric='euclidean',
                metric_kwds=None,
                add_neighbor=True,
                no_noise=True,
                n_neighbors=15,
                heuristic_connect=False,
                min_cluster_size=None,
                save_models=False,
                similarity_backend="auto",
                cluster_selection_method="eom",
                foscx_settings={},
                use_sparse_fit=True,
                **kwargs,
            ):

        # store graph params
        valid_graph_methods = {'sc_gauss', 'sc_umap', 'jaccard_phenograph', 'precomputed'}
        if sim_graph_method not in valid_graph_methods:
            raise ValueError(
                f"Unsupported sim_graph_method '{sim_graph_method}'. "
                f"Use one of {sorted(valid_graph_methods)}."
            )
        if metric is None:
            metric = 'euclidean'

        valid_metrics = {
            'precomputed',
            'cityblock',
            'cosine',
            'euclidean',
            'l1',
            'l2',
            'manhattan',
            'braycurtis',
            'canberra',
            'chebyshev',
            'correlation',
            'dice',
            'hamming',
            'jaccard',
            'mahalanobis',
            'minkowski',
            'rogerstanimoto',
            'russellrao',
            'seuclidean',
            'sokalmichener',
            'sokalsneath',
            'sqeuclidean',
            'yule',
            'hybrid_euclidean_cosine',
            'euclidean_ii',
            'cosine_ii',
        }

        if not isinstance(metric, str) and not callable(metric):
            raise TypeError(
                "metric must be a string metric name, a callable distance function, or None."
            )

        if isinstance(metric, str) and metric not in valid_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Use one of {sorted(valid_metrics)}, or pass a callable metric."
            )

        if sim_graph_method == "precomputed" and metric == "precomputed":
            raise ValueError(
                "Do not use metric='precomputed' together with "
                "sim_graph_method='precomputed'. "
                "metric='precomputed' means the input is a distance matrix; "
                "sim_graph_method='precomputed' means the input is already "
                "a similarity graph."
            )
              
        if sim_graph_method == 'sc_gauss' and metric == 'yule':
            raise ValueError(
                "metric='yule' is not supported with sim_graph_method='sc_gauss' "
                "because this combination can produce non-finite graph weights. "
                "Use sim_graph_method='sc_umap' or sim_graph_method='jaccard_phenograph' "
                "with metric='yule', or choose a different metric with sim_graph_method='sc_gauss'."
            )
        valid_similarity_backends = {"auto", "default", "numba"}
        if similarity_backend not in valid_similarity_backends:
            raise ValueError(
                "similarity_backend must be one of "
                f"{sorted(valid_similarity_backends)}, got {similarity_backend!r}."
            )

        self.similarity_backend = similarity_backend
        self.sim_graph_method = sim_graph_method
        self.metric = metric
        self.metric_kwds = {} if metric_kwds is None else dict(metric_kwds)
        self.add_neighbor = add_neighbor
        self.no_noise = no_noise
        self.n_neighbors = n_neighbors
        self.cluster_selection_method = cluster_selection_method
        self.foscx_settings = foscx_settings
        if 'mst_approx' in kwargs:
            heuristic_connect = kwargs.pop('mst_approx')
        self.heuristic_connect = bool(heuristic_connect)
        self.save_models = bool(save_models)
        self.models_ = {}
        self.condensed_trees_ = {}
        self.labels_by_m_ = {}

        # Prefer the sparse CORE-SG hand-off (no dense N x N) when the bound
        # core.py exposes it; falls back to the dense fill-1 path otherwise.
        self.use_sparse_fit = bool(use_sparse_fit)

        # Lazy NetworkX-view caches (built on demand by the properties below).
        self._similarity_sparse_ = None
        self._precomputed_nx_ = None
        self._similarity_graph_cache = None
        self._connected_graph_cache = None
        self._similarity_graph_WSS_cache = None
        self._dissimilarity_graph_cache = None
        # Lazy dense-matrix / MST caches (only materialised when actually used).
        self._dist_matrix_cache = None
        self._mst_graph_cache = None
        self._connected_sparse_ = None

        # Backward-compatible handling of removed parameters.
        kwargs.pop('force_connected', None)
        kwargs.pop('gamma', None)
        kwargs.pop('min_dist', None)

        # ``m_list`` is now internal rather than a public hyperparameter.
        # Keep a legacy escape hatch through kwargs only.
        legacy_m_list = kwargs.pop('m_list', None)
        if legacy_m_list is not None:
            resolved_m_list = list(legacy_m_list)
        elif isinstance(min_samples, Iterable) and not isinstance(min_samples, (str, bytes, np.str_)):
            resolved_m_list = list(min_samples)
        else:
            resolved_m_list = [int(min_samples)]

        if len(resolved_m_list) == 0:
            raise ValueError("min_samples must define at least one value.")

        self.m_list = [int(m) for m in resolved_m_list]
        self.min_samples = list(self.m_list) if len(self.m_list) > 1 else int(self.m_list[0])

        resolved_min_cluster_size = None if min_cluster_size is None else int(min_cluster_size)

        if callable(metric):
            core_metric = 'euclidean'
        else:
            # Second-order metrics cluster on their base (first-order) distances.
            core_metric = _SECOND_ORDER_METRICS.get(metric, (metric,))[0]
        super().__init__(
            min_samples_list=self.m_list,
            metric=core_metric,
            min_cluster_size=resolved_min_cluster_size,
            save_models=self.save_models,
            **kwargs
        )
        # CoreSGHDBSCAN is a @dataclass: its generated __init__ assigns
        # self.metric = core_metric, clobbering the user-facing metric set
        # above. Restore it -- _create_similarity_sparse dispatches on it.
        self.metric = metric
        self.core_metric_ = core_metric
        self.min_cluster_size = resolved_min_cluster_size

    def __repr__(self):
        fitted = hasattr(self, "coresg_") and self.coresg_ is not None

        if fitted:
            n_models = len(getattr(self.coresg_, "models_", {}))
            n_trees = len(getattr(self.coresg_, "condensed_trees_", {}))
            n_label_sets = len(getattr(self.coresg_, "labels_by_m_", {}))
        else:
            n_models = 0
            n_trees = 0
            n_label_sets = 0

        return (
            f"GraphCoreSGHDBSCAN("
            f"min_samples={list(self.m_list)}, "
            f"sim_graph_method={self.sim_graph_method!r}, "
            f"metric={self.metric!r}, "
            f"n_neighbors={self.n_neighbors}, "
            f"min_cluster_size={self.min_cluster_size}, "
            f"save_models={self.save_models}, "
            f"fitted={fitted}, "
            f"n_models={n_models}, "
            f"n_condensed_trees={n_trees}, "
            f"n_label_sets={n_label_sets}"
            f")"
        )

    # ------------------------------------------------------------------
    # Lazy NetworkX views (debug/inspection only -- never used by fit)
    # ------------------------------------------------------------------
    @property
    def similarity_graph_(self):
        """Initial similarity graph as a NetworkX graph (lazy)."""
        if self.sim_graph_method == "precomputed" and self._precomputed_nx_ is not None:
            return self._precomputed_nx_
        if getattr(self, "_similarity_sparse_", None) is None:
            raise AttributeError(
                "similarity_graph_ is not available until the model has been fit."
            )
        if getattr(self, "_similarity_graph_cache", None) is None:
            g = nx.from_scipy_sparse_array(self._similarity_sparse_, edge_attribute="weight")
            g.add_nodes_from(range(self.n_obs_))
            self._similarity_graph_cache = g
        return self._similarity_graph_cache

    @property
    def connected_graph_(self):
        """Final connected graph used by the clustering stage (lazy)."""
        if getattr(self, "_connected_sparse_", None) is None:
            raise AttributeError(
                "connected_graph_ is not available until the model has been fit."
            )
        if getattr(self, "_connected_graph_cache", None) is None:
            g = nx.from_scipy_sparse_array(self._connected_sparse_, edge_attribute="weight")
            g.add_nodes_from(range(self.n_obs_))
            self._connected_graph_cache = g
        return self._connected_graph_cache

    @property
    def similarity_graph_WSS(self):
        """Weighted structural similarity graph (lazy, debug/inspection only)."""
        if getattr(self, "similarity_graph_WSS_sparse_", None) is None:
            raise AttributeError(
                "similarity_graph_WSS is not available until the model has been fit."
            )
        if getattr(self, "_similarity_graph_WSS_cache", None) is None:
            g = nx.from_scipy_sparse_array(self.similarity_graph_WSS_sparse_, edge_attribute="weight")
            g.add_nodes_from(range(self.n_obs_))
            self._similarity_graph_WSS_cache = g
        return self._similarity_graph_WSS_cache

    @property
    def dissimilarity_graph_(self):
        """WSS dissimilarity graph (lazy, debug/inspection only)."""
        if getattr(self, "dissimilarity_graph_sparse_", None) is None:
            raise AttributeError(
                "dissimilarity_graph_ is not available until the model has been fit."
            )
        if getattr(self, "_dissimilarity_graph_cache", None) is None:
            g = nx.from_scipy_sparse_array(self.dissimilarity_graph_sparse_, edge_attribute="weight")
            g.add_nodes_from(range(self.n_obs_))
            self._dissimilarity_graph_cache = g
        return self._dissimilarity_graph_cache

    @property
    def dist_matrix_(self):
        """Dense fill-1 edge-distance matrix (lazy).

        Materialised on first access from the connected sparse graph. The
        sparse fit path (``use_sparse_fit=True``) never touches this, so the
        O(N^2) dense matrix is only built when explicitly requested or when the
        dense fallback path is used.
        """
        if getattr(self, "_connected_sparse_", None) is None:
            raise AttributeError(
                "dist_matrix_ is not available until the model has been fit."
            )
        if getattr(self, "_dist_matrix_cache", None) is None:
            self._dist_matrix_cache = self.dense_from_sparse_edges_fill1(self._connected_sparse_)
        return self._dist_matrix_cache

    @property
    def mst_graph_(self):
        """MST of the connected WSS graph as a NetworkX graph (lazy).

        Only needed for optional noise reassignment (``no_noise=True``); built
        on first access so the common path never pays for the NetworkX round
        trip.
        """
        if getattr(self, "_connected_sparse_", None) is None:
            raise AttributeError(
                "mst_graph_ is not available until the model has been fit."
            )
        if getattr(self, "_mst_graph_cache", None) is None:
            _t0 = time.perf_counter()
            mst_sparse = minimum_spanning_tree(self._connected_sparse_)
            mst_sparse = mst_sparse + mst_sparse.T  # symmetrize for nx.Graph
            g = nx.from_scipy_sparse_array(mst_sparse, edge_attribute="weight")
            g.add_nodes_from(range(self.n_obs_))
            self._mst_graph_cache = g
            print(f"[TIMER] mst_graph_ (lazy build): {time.perf_counter() - _t0:.4f}s")
        return self._mst_graph_cache

    def _min_cluster_size_for(self, m):
        m = int(m)
        return m if self.min_cluster_size is None else int(self.min_cluster_size)

    # ------------------------------------------------------------------
    # Weighted structural similarity (sparse, no NetworkX round trip)
    # ------------------------------------------------------------------
    @_timeit
    def _wss_similarity_sparse(self, S_sim) -> sp.csr_matrix:
        """Weighted structural similarity directly from a sparse adjacency.

        ``S_sim`` is the initial similarity adjacency (symmetric, or a
        single triangle -- both are accepted). This reproduces the old
        ``compute_similarity_sparse`` exactly for the default
        ``add_neighbor=True`` path: it forms the same symmetric adjacency
        ``A`` (edges on both sides + an explicit self-loop of weight 1),
        cosine-normalizes, and returns ``A @ A.T`` scaled by the inverse
        norms with a zero diagonal.
        """
        S_sim = sp.csr_matrix(S_sim)
        n = S_sim.shape[0]
        if n == 0:
            return sp.csr_matrix((0, 0))

        # symmetric adjacency, each undirected edge on both sides, no self loops
        M = S_sim.maximum(S_sim.T).tocsr()
        M.setdiag(0.0)
        M.eliminate_zeros()

        # explicit self-loop of weight 1 for every node (matches original A)
        A = (M + sp.eye(n, format="csr", dtype=np.float64)).tocsr()
        A.eliminate_zeros()

        norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        norms[norms == 0.0] = 1.0
        inv = 1.0 / norms

        if not self.add_neighbor:
            A_norm = A.multiply(inv[:, None]).tocsr()
            Mu = sp.triu(M, k=1).tocoo()
            rows, cols = Mu.row, Mu.col
            if rows.size:
                sims = np.asarray(
                    A_norm[rows].multiply(A_norm[cols]).sum(axis=1)
                ).ravel()
                data = np.concatenate([sims, sims])
                r = np.concatenate([rows, cols])
                c = np.concatenate([cols, rows])
                S = sp.csr_matrix((data, (r, c)), shape=(n, n))
            else:
                S = sp.csr_matrix((n, n))
            S.eliminate_zeros()
            return S

        numerators = (A @ A.T).tocsr()
        S = numerators.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
        S.setdiag(0.0)
        S.eliminate_zeros()
        return S

    @_timeit
    def compute_similarity_sparse(self, graph) -> sp.csr_matrix:
        """Backward-compatible wrapper: accepts a NetworkX graph.

        Converts the graph to a sparse adjacency once (integer node labels
        ``0..n-1`` are assumed, as everywhere in this pipeline) and delegates
        to :meth:`_wss_similarity_sparse`.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return sp.csr_matrix((0, 0))
        S_sim = nx.to_scipy_sparse_array(
            graph, nodelist=list(range(n)), weight="weight", format="csr"
        )
        return self._wss_similarity_sparse(S_sim)

    @_timeit
    def compute_similarity(self, graph):
        """Backward-compatible NetworkX wrapper over the sparse implementation."""
        S = self.compute_similarity_sparse(graph)
        if self.add_neighbor:
            S = triu(S, k=1).tocsr()
            if S.nnz:
                mask = S.data > 0.0
                if not np.all(mask):
                    S = sp.csr_matrix((S.data[mask], S.indices[mask], S.indptr.copy()), shape=S.shape)
                    S.eliminate_zeros()
        out = nx.from_scipy_sparse_array(S, edge_attribute='weight')
        out.add_nodes_from(range(graph.number_of_nodes()))
        return out

    @staticmethod
    @_timeit
    def similarity_to_dissimilarity_sparse(similarity_matrix: sp.csr_matrix) -> sp.csr_matrix:
        D = similarity_matrix.copy().tocsr()
        D.data = 1.0 - D.data
        D.setdiag(0.0)
        D.eliminate_zeros()
        return D

    @staticmethod
    @_timeit
    def similarity_to_dissimilarity(similarity_graph):
        dissimilarity_graph = nx.Graph()
        for u, v, data in similarity_graph.edges(data=True):
            dissimilarity_graph.add_edge(u, v, weight=1 - data['weight'])
        return dissimilarity_graph

    @staticmethod
    def is_graph_connected(graph):
        return nx.is_connected(graph)

    @staticmethod
    @_timeit
    def _coerce_precomputed_graph(graph_like):
        """Convert a supported precomputed graph representation into a NetworkX graph."""
        if isinstance(graph_like, nx.Graph):
            graph = nx.convert_node_labels_to_integers(
                graph_like,
                first_label=0,
                ordering="default",
                label_attribute="original_label",
            )
        elif hasattr(graph_like, 'tocoo'):
            graph = nx.from_scipy_sparse_array(graph_like, edge_attribute='weight')
        else:
            arr = np.asarray(graph_like)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    "For sim_graph_method='precomputed', input must be a NetworkX graph, "
                    "a scipy sparse adjacency matrix, or a square dense adjacency matrix."
                )
            graph = nx.from_numpy_array(arr)

        graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d.get('weight', 0) == 0])
        return graph

    @_timeit
    def _fast_phenograph_jaccard_from_knn_graph(self, knn_graph):
        """
        Fast replacement for PhenoGraph's Jaccard graph construction.

        Input is the same sparse kNN graph that the old code passed to:

            sce.tl.phenograph(knn_dist, directed=False, clustering_algo=None)

        Output matches PhenoGraph's default undirected Jaccard graph.
        """
        if not _HAS_NUMBA:
            raise ImportError(
                "Fast jaccard_phenograph requires numba. "
                "Install it with `pip install numba`."
            )

        knn_graph = knn_graph.tocsr()
        n = knn_graph.shape[0]

        if n <= 1:
            return sp.csr_matrix((n, n), dtype=np.float64)

        # The old branch passes a kNN graph with exactly self.n_neighbors - 1
        # neighbors per row.
        k = int(self.n_neighbors) - 1

        if k < 1:
            raise ValueError("n_neighbors must be at least 2 for jaccard_phenograph.")

        indptr = knn_graph.indptr
        indices = knn_graph.indices

        row_counts = np.diff(indptr)
        if not np.all(row_counts == k):
            raise ValueError(
                "Expected the kNN graph to have exactly "
                f"{k} neighbors per row, but got row counts from "
                f"{row_counts.min()} to {row_counts.max()}."
            )

        knn_idx = indices.reshape(n, k).astype(np.int32, copy=False)
        sorted_knn_idx = np.sort(knn_idx, axis=1).astype(np.int32, copy=False)

        weights = _directed_jaccard_weights_numba(
            knn_idx,
            sorted_knn_idx,
        )

        rows = np.repeat(np.arange(n, dtype=np.int32), k)
        cols = knn_idx.ravel()
        data = weights.ravel()

        mask = data > 0.0

        directed = sp.csr_matrix(
            (data[mask], (rows[mask], cols[mask])),
            shape=(n, n),
            dtype=np.float64,
        )
        directed.eliminate_zeros()

        conn = (directed + directed.T).multiply(0.5)
        conn = sp.tril(conn, k=-1).tocsr()
        conn.eliminate_zeros()

        return conn

    # ------------------------------------------------------------------
    # Initial similarity graph -- sparse (fast path) and nx (compat)
    # ------------------------------------------------------------------


    def _second_order_knn_distances(self, distances_full, knn_metric):
        """``knn_metric`` distances between the ROWS of ``distances_full``.

        Used by the metrics in ``_SECOND_ORDER_METRICS``. This is exactly the
        space ``kneighbors_graph(distances_full, metric=knn_metric)`` searches
        in the ``sc_gauss`` / ``jaccard_phenograph`` branches, materialised
        densely because ``sc_umap`` needs a full matrix.

        Cached: it depends only on ``distances_full`` and ``knn_metric``, not on
        ``n_neighbors``, so the ``heuristic_connect`` loop pays for it once
        rather than once per iteration.
        """
        cached = getattr(self, '_second_order_knn_', None)
        if (cached is not None
                and cached[0] == knn_metric
                and cached[1].shape == distances_full.shape):
            return cached[1]

        knn_source = pairwise_distances(distances_full, metric=knn_metric)
        # sklearn already zeroes the diagonal when X is Y; be explicit so that
        # _get_indices_distances_from_dense_matrix always finds self in col 0.
        np.fill_diagonal(knn_source, 0.0)
        self._second_order_knn_ = (knn_metric, knn_source)
        return knn_source

    @_timeit
    def _create_similarity_sparse(self, data, distances_full=None):
        """Build the initial similarity graph as a scipy sparse matrix.

        Returns ``(S_sparse_csr, n_obs)``. ``distances_full`` may be passed
        in to avoid recomputing the (n_neighbors-independent) full distance
        matrix during the ``heuristic_connect`` search.

        For precomputed input the coerced NetworkX graph is cached so that
        the ``similarity_graph_`` property can return it with its original
        node labels intact.
        """
        if self.sim_graph_method == 'precomputed':
            g_nx = self._coerce_precomputed_graph(data)
            n = g_nx.number_of_nodes()
            S = nx.to_scipy_sparse_array(
                g_nx, nodelist=list(range(n)), weight='weight', format='csr'
            ).astype(np.float64)
            self._precomputed_nx_ = g_nx
            return S, n

        sc, sce, sc_gauss, sc_umap, _get_indices_distances_from_dense_matrix = _get_scanpy_modules()

        X = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
        
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array-like object.")
        
        
        # ---------------------------------------------------------
        # NEW: support a precomputed DISTANCE MATRIX
        # ---------------------------------------------------------
        if self.metric == "precomputed":
        
            if distances_full is None:
        
                distances_full = np.asarray(X, dtype=np.float64)
        
                # Must be square
                if (
                    distances_full.ndim != 2
                    or distances_full.shape[0] != distances_full.shape[1]
                ):
                    raise ValueError(
                        "With metric='precomputed', X must be a square "
                        "distance matrix."
                    )
        
                # Must contain finite values
                if not np.all(np.isfinite(distances_full)):
                    raise ValueError(
                        "Precomputed distance matrix contains non-finite values."
                    )
        
                # Distances cannot be negative
                if np.any(distances_full < 0):
                    raise ValueError(
                        "Precomputed distance matrix cannot contain negative distances."
                    )
        
                # Must be symmetric
                if not np.allclose(
                    distances_full,
                    distances_full.T,
                    rtol=1e-7,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Precomputed distance matrix must be symmetric."
                    )
        
                # Work with our own copy
                distances_full = distances_full.copy()
        
                # Remove tiny numerical errors on diagonal
                np.fill_diagonal(distances_full, 0.0)
        
            knn_metric = None
        
            # The graph routines below should treat distances_full
            # directly as distances.
            use_precomputed_knn = True
        
        
        # ---------------------------------------------------------
        # Normal feature matrices
        # ---------------------------------------------------------
        else:
        
            second_order = _SECOND_ORDER_METRICS.get(self.metric)
        
            if second_order is not None:
        
                base_metric, knn_metric = second_order
        
                if distances_full is None:
                    distances_full = pairwise_distances(
                        X,
                        metric=base_metric,
                    )
        
                    self._second_order_knn_ = None
        
                use_precomputed_knn = False
        
            else:
        
                knn_metric = None
        
                if distances_full is None:
                    distances_full = pairwise_distances(
                        X,
                        metric=self.metric,
                        **self.metric_kwds
                    )
        
                use_precomputed_knn = True

        self.distances_full_ = distances_full
        n = distances_full.shape[0]

        if self.sim_graph_method == 'jaccard_phenograph':
            knn_dist = kneighbors_graph(
                distances_full,
                n_neighbors=self.n_neighbors - 1,
                mode='distance',
                metric='precomputed' if use_precomputed_knn else knn_metric,
                include_self=False,
            )

            if self.similarity_backend == "numba":
                conn = self._fast_phenograph_jaccard_from_knn_graph(knn_dist)
            elif self.similarity_backend == "default":
                _, conn, _ = sce.tl.phenograph(knn_dist, directed=False, clustering_algo=None)
            else:  # "auto"
                if _HAS_NUMBA:
                    conn = self._fast_phenograph_jaccard_from_knn_graph(knn_dist)
                else:
                    _, conn, _ = sce.tl.phenograph(knn_dist, directed=False, clustering_algo=None)

            return sp.csr_matrix(conn).astype(np.float64), n

        if self.sim_graph_method == 'sc_gauss':
            knn_dist = kneighbors_graph(
                distances_full,
                n_neighbors=self.n_neighbors - 1,
                mode='distance',
                metric='precomputed' if use_precomputed_knn else knn_metric,
                include_self=False,
            )
            conn = sc_gauss(knn_dist, n_neighbors=self.n_neighbors, knn=True)
            return sp.csr_matrix(conn).astype(np.float64), n

        if self.sim_graph_method == 'sc_umap':
            knn_source = (
                distances_full
                if use_precomputed_knn
                else self._second_order_knn_distances(distances_full, knn_metric)
            )
            idx, dists = _get_indices_distances_from_dense_matrix(
                knn_source, self.n_neighbors
            )
            conn = sc_umap(idx, dists, n_obs=n, n_neighbors=self.n_neighbors)
            return sp.csr_matrix(conn).astype(np.float64), n

        raise ValueError(
            "Unsupported sim_graph_method. Use one of 'sc_gauss', 'sc_umap', 'jaccard_phenograph', or 'precomputed'."
        )

    @_timeit
    def create_similarity_graph(self, data):
        """Public wrapper returning the initial similarity graph as NetworkX.

        The internal pipeline uses :meth:`_create_similarity_sparse` and never
        materializes this NetworkX object; it is provided for backward
        compatibility / inspection only.
        """
        if self.sim_graph_method == 'precomputed':
            return self._coerce_precomputed_graph(data)
        S, n = self._create_similarity_sparse(data)
        g = nx.from_scipy_sparse_array(S, edge_attribute='weight')
        g.add_nodes_from(range(n))
        return g

    # ------------------------------------------------------------------
    # Connectivity / bridging
    # ------------------------------------------------------------------
    @staticmethod
    @_timeit
    def _connect_sparse_heuristically(D_sparse, n_obs):
        """Sparse equivalent of ``connect_graph_heuristically``.

        Adds weight=1 bridge edges between disconnected components entirely
        in scipy-sparse form, avoiding any NetworkX materialization. Since a
        bridge weight of 1 equals the non-edge fill value used when building
        the dense distance matrix, the specific representatives chosen here do
        not affect ``dist_matrix_`` or the clustering result.
        """
        D_sparse = D_sparse.tocsr()
        n_components, comp_labels = sp.csgraph.connected_components(
            D_sparse, directed=False
        )
        if n_components <= 1:
            return D_sparse

        # one representative node per component, in first-seen order
        reps = []
        seen = set()
        for node, comp in enumerate(comp_labels):
            if comp not in seen:
                seen.add(comp)
                reps.append(node)

        bridge_rows, bridge_cols, bridge_data = [], [], []
        for i in range(len(reps) - 1):
            u, v = reps[i], reps[i + 1]
            bridge_rows.extend((u, v))
            bridge_cols.extend((v, u))
            bridge_data.extend((1.0, 1.0))

        coo = D_sparse.tocoo()
        rows = np.concatenate([coo.row, np.asarray(bridge_rows, dtype=coo.row.dtype)])
        cols = np.concatenate([coo.col, np.asarray(bridge_cols, dtype=coo.col.dtype)])
        data = np.concatenate([coo.data, np.asarray(bridge_data, dtype=coo.data.dtype)])

        out = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
        out.sum_duplicates()
        return out

    @_timeit
    def connect_graph_heuristically(self, graph, n_obs):
        """Connect disconnected components with synthetic bridge edges (NetworkX).

        Kept for backward compatibility; the fast path uses
        :meth:`_connect_sparse_heuristically`.
        """
        new_graph = graph.copy()
        new_graph.add_nodes_from(range(n_obs))

        if nx.is_connected(new_graph):
            return new_graph

        components = list(nx.connected_components(new_graph))

        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            new_graph.add_edge(u, v, weight=1)

        return new_graph

    @staticmethod
    @_timeit
    def compute_full_distance_matrix(graph):
        """Full dense matrix of shortest path distances via Floyd-Warshall."""
        return np.array(nx.floyd_warshall_numpy(graph, weight='weight'))

    @staticmethod
    @_timeit
    def compute_sparse_distance_dict(graph):
        """Dict-of-dicts of shortest path distances (single-source Dijkstra per node)."""
        distance_dict = {}
        for node in graph.nodes():
            distance_dict[node] = nx.single_source_dijkstra_path_length(graph, node, weight='weight')
        return distance_dict

    def graph_metric(self, u, v):
        """Custom distance metric backed by the precomputed sparse distance dict."""
        idx_u = self._point_to_index.get(tuple(u))
        idx_v = self._point_to_index.get(tuple(v))
        try:
            return self.distance_dict_[idx_u][idx_v]
        except KeyError:
            return self.distance_dict_[idx_v][idx_u]

    @staticmethod
    @_timeit
    def compute_custom_distance_matrix(graph):
        """Dense pairwise distance matrix from a NetworkX graph (non-edges = 1)."""
        n = graph.number_of_nodes()
        dist = np.full((n, n), 1, dtype=np.float64)
        np.fill_diagonal(dist, 0)
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            dist[u, v] = weight
            dist[v, u] = weight
        return dist

    @staticmethod
    @_timeit
    def dense_from_sparse_edges_fill1(D_sparse: sp.csr_matrix) -> np.ndarray:
        """Create the dense edge-distance matrix expected by CoreSG/HDBSCAN.

        Non-edges are filled with 1, diagonal with 0, and sparse entries
        overwrite the corresponding distances.
        """
        D_sparse = D_sparse.tocsr()
        n = D_sparse.shape[0]
        D = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(D, 0.0)
        coo = D_sparse.tocoo()
        D[coo.row, coo.col] = coo.data
        return np.minimum(D, D.T)

    @staticmethod
    @_timeit
    def reassign_noise_via_mst(mst_graph, labels0, c=5):
        """
        Reassign noise labels by propagating labels over a precomputed MST.

        Parameters
        ----------
        mst_graph : networkx.Graph
            Minimum spanning tree of the final connected WSS graph.
        labels0 : ndarray
            Initial labels with noise marked as -1.
        c : int, default=5
            Number of largest edge weights to keep in the lexicographic path
            signature during propagation.
        """
        if not isinstance(mst_graph, nx.Graph):
            raise TypeError("mst_graph must be a networkx.Graph.")

        n = len(labels0)
        labels = np.asarray(labels0).copy()
        if mst_graph.number_of_nodes() != n:
            raise ValueError("mst_graph and labels0 must have the same number of nodes.")

        # adjacency of the tree
        adj = [[] for _ in range(n)]
        for u, v, data in mst_graph.edges(data=True):
            w = float(data.get('weight', 1.0))
            adj[int(u)].append((int(v), w))
            adj[int(v)].append((int(u), w))

        # Multi-source propagation from labeled vertices.
        pq = []
        paths = [None] * n
        for u in range(n):
            if labels[u] != -1:
                paths[u] = [0.0] * c
                for v, w in adj[u]:
                    if labels[v] == -1:
                        heapq.heappush(pq, (w, u, v))

        while pq:
            w, u, v = heapq.heappop(pq)
            if labels[v] != -1:
                continue

            same = [(u, v)]
            while pq and pq[0][0] == w and pq[0][2] == v:
                _, u2, _ = heapq.heappop(pq)
                same.append((u2, v))

            def top_c_path(u_idx):
                vec = list(paths[u_idx]) + [w]
                return sorted(vec, reverse=True)[:c]

            candidates = [(top_c_path(u_idx), labels[u_idx]) for u_idx, _ in same]
            best_path, best_label = min(candidates, key=lambda x: tuple(x[0]))

            labels[v] = best_label
            paths[v] = best_path

            for nbr, w2 in adj[v]:
                if labels[nbr] == -1:
                    heapq.heappush(pq, (w2, v, nbr))

        return labels

    # ------------------------------------------------------------------
    # ------------------------ GRAPH PREPROCESSING ---------------------
    # ------------------------------------------------------------------
    @_timeit
    def _build_graph_distance(self, X):
        """Build graph-derived dense distances using the sparse fast path.

        Pipeline (all in scipy-sparse until the final dense matrix):
            data / precomputed graph
            -> initial similarity adjacency (sparse)
            -> weighted structural similarity (sparse)
            -> WSS dissimilarity (sparse)
            -> connected sparse matrix
            -> dense precomputed distance matrix
        """
        self.data_ = (
            X if self.sim_graph_method == "precomputed"
            else (np.array(X) if isinstance(X, pd.DataFrame) else X)
        )

        # Reset lazy nx caches / precomputed handle from any previous fit.
        self._precomputed_nx_ = None
        self._similarity_graph_cache = None
        self._connected_graph_cache = None
        self._similarity_graph_WSS_cache = None
        self._dissimilarity_graph_cache = None

        # ------------------------------------------------------------
        # 1. Initial similarity adjacency (sparse)
        # ------------------------------------------------------------
        S_sim, n_obs = self._create_similarity_sparse(self.data_)
        self._similarity_sparse_ = S_sim
        self.n_obs_ = n_obs

        # ------------------------------------------------------------
        # 2. WSS similarity (sparse), then dissimilarity (sparse)
        # ------------------------------------------------------------
        self.similarity_graph_WSS_sparse_ = self._wss_similarity_sparse(S_sim)
        self.dissimilarity_graph_sparse_ = self.similarity_to_dissimilarity_sparse(
            self.similarity_graph_WSS_sparse_
        )

        # ------------------------------------------------------------
        # 3. Connectivity check
        # ------------------------------------------------------------
        _t0 = time.perf_counter()
        n_components, _ = sp.csgraph.connected_components(
            self.dissimilarity_graph_sparse_, directed=False
        )
        print(f"[TIMER] _build_graph_distance:connectivity_check: {time.perf_counter() - _t0:.4f}s")

        # ------------------------------------------------------------
        # 4. Optionally grow n_neighbors until connected (non-precomputed).
        #    The full distance matrix does not depend on n_neighbors, so it
        #    is computed once (in step 1) and reused here.
        # ------------------------------------------------------------
        self.n_neighbors_initial_ = self.n_neighbors
        self.n_neighbors_used_ = self.n_neighbors

        if (
            n_components > 1
            and self.heuristic_connect
            and self.sim_graph_method != "precomputed"
        ):
            original_n_neighbors = self.n_neighbors
            new_n_neighbors = self.n_neighbors
            max_neighbors = n_obs
            cached_distances = getattr(self, "distances_full_", None)

            while n_components > 1 and new_n_neighbors < max_neighbors:
                new_n_neighbors += 1
                print("Trying n_neighbors =", new_n_neighbors)

                self.n_neighbors = new_n_neighbors

                S_sim, _ = self._create_similarity_sparse(
                    self.data_, distances_full=cached_distances
                )
                self._similarity_sparse_ = S_sim
                self.similarity_graph_WSS_sparse_ = self._wss_similarity_sparse(S_sim)
                self.dissimilarity_graph_sparse_ = self.similarity_to_dissimilarity_sparse(
                    self.similarity_graph_WSS_sparse_
                )

                n_components, _ = sp.csgraph.connected_components(
                    self.dissimilarity_graph_sparse_, directed=False
                )

            self.n_neighbors_used_ = self.n_neighbors

            if n_components > 1:
                raise RuntimeError(
                    "Could not build a connected WSS dissimilarity graph even after "
                    f"increasing n_neighbors from {original_n_neighbors} "
                    f"to {max_neighbors}."
                )

        # ------------------------------------------------------------
        # 5/6. Connected sparse matrix (bridge in sparse form if needed).
        #      This is the single source of truth handed to CORE-SG; the dense
        #      fill-1 matrix (dist_matrix_) and the noise-reassignment MST
        #      (mst_graph_) are now BUILT LAZILY from it, only when accessed.
        # ------------------------------------------------------------
        if n_components <= 1:
            self._connected_sparse_ = self.dissimilarity_graph_sparse_
        else:
            self._connected_sparse_ = self._connect_sparse_heuristically(
                self.dissimilarity_graph_sparse_, n_obs
            )

        # Invalidate lazy views/caches built from stale sparse data.
        self._connected_graph_cache = None
        self._similarity_graph_WSS_cache = None
        self._dissimilarity_graph_cache = None
        self._similarity_graph_cache = None
        self._dist_matrix_cache = None
        self._mst_graph_cache = None

    # ------------------------------------------------------------------
    # ------------------------- FIT ------------------------------------
    # ------------------------------------------------------------------
    def _call_coresg_run(self):
        """Call ``self.coresg_.run(...)`` forwarding only supported kwargs.

        Different ``core.py`` revisions expose different ``run`` signatures:
        the published repo version accepts ``cluster_selection_method`` but
        not ``foscx_settings``; a newer revision accepts both. Instead of
        hard-coding one, inspect the signature and pass only what fits, so
        this file runs unmodified against either core.
        """
        desired = {
            "cluster_selection_method": self.cluster_selection_method,
            "foscx_settings": self.foscx_settings,
        }
        try:
            params = inspect.signature(self.coresg_.run).parameters
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            params, accepts_var_kw = {}, False

        if accepts_var_kw:
            kwargs = dict(desired)
        else:
            kwargs = {k: v for k, v in desired.items() if k in params}

        # Warn only if a *non-default* option is being silently dropped
        # (an empty foscx_settings dict is falsy and thus never reported).
        dropped = [k for k, v in desired.items() if k not in kwargs and v]
        if dropped:
            warnings.warn(
                f"CoreSGHDBSCAN.run() does not accept {dropped}; these "
                "options are ignored. Update core.py to enable them.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.coresg_.run(**kwargs)

    def _coresg_fit_graph(self):
        """Hand the connected graph to CORE-SG, preferring the sparse path.

        When ``use_sparse_fit`` is on and the bound ``core.py`` exposes
        ``fit_from_sparse_graph``, feed the sparse graph directly (no dense
        N x N ever materialised). Otherwise fall back to the dense fill-1
        matrix (built lazily on first ``dist_matrix_`` access) so this file
        keeps working against an older core.
        """
        use_sparse = (
            getattr(self, "use_sparse_fit", True)
            and hasattr(self.coresg_, "fit_from_sparse_graph")
        )
        _t0 = time.perf_counter()
        if use_sparse:
            self.coresg_.fit_from_sparse_graph(self._connected_sparse_)
            label = "fit:coresg_.fit_from_sparse_graph"
        else:
            self.coresg_.fit_from_distance_matrix(self.dist_matrix_)
            label = "fit:coresg_.fit_from_distance_matrix"
        print(f"[TIMER] {label}: {time.perf_counter() - _t0:.4f}s")

    @_timeit
    def fit(self, X, y=None):
        """Fit the model on feature data or a precomputed graph."""
        self._build_graph_distance(X)

        self.coresg_ = CoreSGHDBSCAN(
            min_samples_list=self.m_list,
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            save_models=self.save_models,
        )

        self.coresg_.nn = self.n_neighbors
        self.coresg_.similarity_graph_WSS_sparse_ = self.similarity_graph_WSS_sparse_

        self._coresg_fit_graph()

        _t0 = time.perf_counter()
        self._call_coresg_run()
        print(f"[TIMER] fit:coresg_.run: {time.perf_counter() - _t0:.4f}s")

        self.models_ = self.coresg_.models_
        self.condensed_trees_ = self.coresg_.condensed_trees_
        self.labels_by_m_ = self.coresg_.labels_by_m_
        return self

    @_timeit
    def fit_predict(self, X, y=None, m=None, c=5, **fit_params):
        """Fit the model and return cluster labels."""
        self.fit(X, y, **fit_params)

        if m is None:
            if len(self.m_list) != 1:
                raise ValueError(
                    "fit_predict requires `m` when m_list contains multiple values. "
                    "Use labels_for(m) or pass m=... explicitly."
                )
            m = self.m_list[0]

        labels = self.coresg_.labels_by_m_[int(m)]

        if self.no_noise:
            if isinstance(labels[0], np.int64):
                labels = self.reassign_noise_via_mst(self.mst_graph_, labels, c=c)
            else:
                for i, labs in enumerate(labels):
                    labels[i] = self.reassign_noise_via_mst(self.mst_graph_, labs, c=c)

        return labels

    @_timeit
    def fit_coresg(self, X, m_list, coresg_kwargs=None):
        """Build graph-derived distances and run CoreSGHDBSCAN on them."""
        self._build_graph_distance(X)

        if coresg_kwargs is None:
            coresg_kwargs = {}

        self.coresg_ = CoreSGHDBSCAN(
            min_samples_list=list(m_list),
            min_cluster_size=self.min_cluster_size,
            save_models=self.save_models,
            **coresg_kwargs
        )
        self._coresg_fit_graph()
        self.coresg_.run()
        self.models_ = self.coresg_.models_
        self.condensed_trees_ = self.coresg_.condensed_trees_
        self.labels_by_m_ = self.coresg_.labels_by_m_
        return self

    # ------------------------------------------------------------------
    # -------------------------- ACCESSORS -----------------------------
    # ------------------------------------------------------------------
    @_timeit
    def labels_for(self, m, no_noise=None, c=5):
        """Return labels for a selected ``min_samples`` value."""
        labels = self.coresg_.labels_by_m_[int(m)]

        if no_noise is None:
            no_noise = self.no_noise

        if no_noise:
            if isinstance(labels[0], np.int64):
                labels = self.reassign_noise_via_mst(self.mst_graph_, labels, c=c)
            else:
                for i, labs in enumerate(labels):
                    labels[i] = self.reassign_noise_via_mst(self.mst_graph_, labs, c=c)

        return labels

    def plot_condensed_tree(self, m, figsize=(10, 6), lambda_floor=1.0, **kwargs):
        """Plot the condensed tree for a selected ``min_samples`` value."""
        import matplotlib.pyplot as plt
        from .core import plot_condensed_tree_bounded

        if not hasattr(self, "coresg_") or self.coresg_ is None:
            raise ValueError("Model is not fitted yet. Call fit(...) first.")

        m = int(m)

        if m in getattr(self.coresg_, "condensed_trees_", {}):
            ct = self.coresg_.condensed_trees_[m]
        elif m in getattr(self.coresg_, "models_", {}):
            ct = self.coresg_.models_[m].condensed_tree_
        else:
            raise KeyError(f"m={m} not found in CORE-SG results.")

        if ct is None or not hasattr(ct, "plot"):
            print(f"No condensed tree for CORE-SG m={m}")
            return

        plt.figure(figsize=figsize)
        ax = plot_condensed_tree_bounded(
            ct,
            lambda_floor=lambda_floor,
            select_clusters=False, label_clusters=False,
            **kwargs,
        )
        ax.set_title(f"CORE-SG Condensed Tree (min_samples = {m})")
        plt.show()
        return ax



    def plot_condensed_tree_ground_truth_pies(
        self, m, y_true, *, figsize=(16, 10), **kwargs
    ):
        """Condensed tree for ``min_samples=m`` with ground-truth pies.

        Overlays a pie chart of the ground-truth class composition at every
        cluster node of the condensed tree fitted for the selected
        ``min_samples`` value.

        Parameters
        ----------
        m : int
            The ``min_samples`` value selecting which condensed tree to draw.
        y_true : array-like, shape (n_samples,)
            Ground-truth labels aligned row-for-row with the data passed to
            ``fit(...)``.
        figsize : tuple, default=(16, 10)
            Figure size.
        **kwargs
            Forwarded to
            :func:`~coresg_graphhdbscan.plot_condensed_tree_ground_truth_pies`
            (e.g. ``min_node_size``, ``label_cmap``, ``show_node_ids``).

        Returns
        -------
        fig, ax
        """
        from .ground_truth_pies import (
            plot_condensed_tree_ground_truth_pies as _pies,
        )

        if not hasattr(self, "coresg_") or self.coresg_ is None:
            raise ValueError("Model is not fitted yet. Call fit(...) first.")

        return _pies(
            self.coresg_,
            y_true,
            m=int(m),
            figsize=figsize,
            **kwargs,
        )

    
    def interactive_condensed_tree(self, figsize=(10, 6)):
        """Interactive condensed tree explorer across fitted ``min_samples`` values."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError as e:
            raise ImportError(
                "ipywidgets is required for interactive plotting. "
                "Install it with `pip install ipywidgets`."
            ) from e

        import matplotlib.pyplot as plt

        if not hasattr(self, "coresg_") or self.coresg_ is None:
            raise RuntimeError("Call fit(...) before interactive_condensed_tree().")

        m_list = sorted(
            set(getattr(self.coresg_, "condensed_trees_", {}).keys()) |
            set(getattr(self.coresg_, "models_", {}).keys())
        )

        if len(m_list) == 0:
            raise ValueError("No condensed trees are available.")

        output = widgets.Output()

        slider = widgets.SelectionSlider(
            options=m_list,
            value=m_list[0],
            description="min_samples",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

        def redraw(m):
            with output:
                clear_output(wait=True)
                self.plot_condensed_tree(m=int(m), figsize=figsize)

        def on_change(change):
            if change["name"] == "value":
                redraw(change["new"])

        slider.observe(on_change, names="value")
        display(widgets.VBox([slider, output]))
        redraw(m_list[0])

        return slider
