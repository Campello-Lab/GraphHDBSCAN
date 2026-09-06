"""CoreSG-HDBSCAN core implementation"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ._outliers import glosh_from_condensed_tree
import numpy as np
import hdbscan
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix, csr_matrix, csgraph

from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# HDBSCAN generic internals
# =========================
from hdbscan._hdbscan_linkage import label as _linkage_label
from hdbscan._hdbscan_tree import (
    condense_tree as _condense_tree,
    compute_stability as _compute_stability,
    get_clusters as _get_clusters,
)
from hdbscan.plots import CondensedTree as _CondensedTree, SingleLinkageTree as _SingleLinkageTree


# ===========================================
# Dense Prim on COMPLETE MRD graph (your code)
# ===========================================
def prim_mrd_mst_edges(X: np.ndarray, core: np.ndarray) -> np.ndarray:
    """
    Compute MST edges on a mutual-reachability graph using Prim's algorithm.

    Parameters
    ----------
    D : numpy.ndarray
        Dense pairwise distance matrix.
    core : numpy.ndarray
        Core-distance vector.
    eps : float, default=1e-12
        Numerical tolerance.

    Returns
    -------
    numpy.ndarray
        Array of undirected MST edges with shape ``(n_edges, 2)``.
    """
    
    X = np.asarray(X, dtype=np.float64, order="C")
    n = X.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)

    key[0] = 0.0
    for _ in range(n):
        u = np.argmin(key)
        in_tree[u] = True
        key[u] = np.inf
        not_in = ~in_tree
        if not np.any(not_in):
            break
        dif = X[not_in] - X[u]
        d_uv = np.sqrt(np.einsum('ij,ij->i', dif, dif), dtype=np.float64)
        idx_not = np.flatnonzero(not_in)
        cand = np.maximum(np.maximum(core[u], core[idx_not]), d_uv)
        better = cand < key[idx_not]
        key[idx_not[better]] = cand[better]
        parent[idx_not[better]] = u

    edges = []
    for v in range(n):
        p = parent[v]
        if p != -1:
            edges.append((min(p, v), max(p, v)))
    return np.array(edges, dtype=np.int32)

def prim_mrd_mst_edges_from_D(D: np.ndarray, core: np.ndarray) -> np.ndarray:
    """
    Compute MST edges from a precomputed distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        Dense pairwise distance matrix.
    core : numpy.ndarray
        Core-distance vector.
    eps : float, default=1e-12
        Numerical tolerance.

    Returns
    -------
    numpy.ndarray
        Array of undirected MST edges with shape ``(n_edges, 2)``.
    """
    D = np.asarray(D, dtype=np.float64, order="C")
    n = D.shape[0]
    if D.shape[1] != n:
        raise ValueError("D must be a square distance matrix.")
    if core.shape[0] != n:
        raise ValueError("core must have length N.")

    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int32)

    key[0] = 0.0
    for _ in range(n):
        u = np.argmin(key)
        in_tree[u] = True
        key[u] = np.inf

        not_in = ~in_tree
        if not np.any(not_in):
            break

        idx_not = np.flatnonzero(not_in)
        base = D[u, idx_not]
        cand = np.maximum(np.maximum(core[u], core[idx_not]), base)

        better = cand < key[idx_not]
        key[idx_not[better]] = cand[better]
        parent[idx_not[better]] = u

    edges = []
    for v in range(n):
        p = parent[v]
        if p != -1:
            edges.append((min(p, v), max(p, v)))
    return np.array(edges, dtype=np.int32)


# ===========================================
# Memory-safe kmax-NNG (chunked upper-triangle scan)
# ===========================================
def _kmax_nng_edges_chunked(D: np.ndarray, core_kmax: np.ndarray,
                            eps: float, block: int = 4096) -> np.ndarray:
    """Row-block equivalent of::

        iu, ju = np.triu_indices(N, k=1)
        cond = (D[iu,ju] <= core_kmax[iu]+eps) | (D[iu,ju] <= core_kmax[ju]+eps)
        edges = stack(iu[cond], ju[cond])

    Produces the *identical* set of upper-triangular edges (i < j) but never
    materialises the O(N^2) index arrays from ``np.triu_indices``; peak temp is
    O(block * N). Output order is irrelevant downstream (``np.unique`` sorts).
    """
    N = D.shape[0]
    thr = core_kmax + eps
    out_i, out_j = [], []
    for start in range(0, N, block):
        stop = min(start + block, N)
        Dblk = D[start:stop]                                   # (b, N)
        cond = (Dblk <= thr[start:stop, None]) | (Dblk <= thr[None, :])
        a, j = np.nonzero(cond)
        i = a + start
        keep = j > i                                           # upper triangle only
        if np.any(keep):
            out_i.append(i[keep])
            out_j.append(j[keep])
    if out_i:
        ii = np.concatenate(out_i).astype(np.int32)
        jj = np.concatenate(out_j).astype(np.int32)
        return np.stack((ii, jj), axis=1)
    return np.empty((0, 2), dtype=np.int32)


# ===========================================
# Sparse-path helpers (no dense N x N ever materialised)
# ===========================================
def _symmetric_real_edges(S, fill_value: float):
    """Return upper-triangular real edges (i<j) with value = min over the two
    directions, treating a missing direction as ``fill_value``.

    This reproduces exactly the off-diagonal ``< fill_value`` entries of
    ``np.minimum(D, D.T)`` where ``D`` is the dense fill-``fill_value`` matrix.
    """
    S = S.tocoo()
    r = np.concatenate([S.row, S.col])
    c = np.concatenate([S.col, S.row])
    d = np.concatenate([S.data, S.data]).astype(np.float64)
    m = r < c
    r, c, d = r[m], c[m], d[m]
    if r.size == 0:
        z = np.empty(0, dtype=np.int32)
        return z, z, np.empty(0, dtype=np.float64)
    N = S.shape[0]
    key = r.astype(np.int64) * N + c
    order = np.argsort(key, kind="stable")
    key, r, c, d = key[order], r[order], c[order], d[order]
    start = np.empty(key.shape[0], dtype=bool)
    start[0] = True
    np.not_equal(key[1:], key[:-1], out=start[1:])
    seg = np.flatnonzero(start)
    dmin = np.minimum.reduceat(d, seg)
    # missing direction == fill_value: an edge present in only one direction
    # keeps its own value (<= fill_value), so min with implicit fill is a no-op.
    ru = r[seg].astype(np.int32)
    cu = c[seg].astype(np.int32)
    return ru, cu, dmin


def _knn_core_from_edges(er, ec, ed, N, kmax, fill_value):
    """Self-inclusive kNN tables + core distances from symmetric real edges.

    core_m[i] = m-th smallest of (sorted real neighbour dissimilarities of i,
    padded with ``fill_value``); with self (distance 0) at column 0. Identical
    in value to the dense path's ``dst_with_self_[:, m]``.
    """
    # symmetric directed edges
    SR = np.concatenate([er, ec])
    SC = np.concatenate([ec, er])
    SD = np.concatenate([ed, ed]).astype(np.float64)
    if SR.size:
        order = np.lexsort((SD, SR))               # by source, then distance asc
        SR, SC, SD = SR[order], SC[order], SD[order]
        deg = np.bincount(SR, minlength=N)
        starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(deg, out=starts[1:])
        rank = np.arange(SR.shape[0]) - np.repeat(starts[:-1], deg)
        keep = rank < kmax
    else:
        keep = np.zeros(0, dtype=bool)
        deg = np.zeros(N, dtype=np.int64)
    dst_ns = np.full((N, kmax), fill_value, dtype=np.float64)
    idx_ns = np.full((N, kmax), -1, dtype=np.int32)
    if SR.size:
        rr = SR[keep]; cc = SC[keep]; dd = SD[keep]; pp = rank[keep]
        dst_ns[rr, pp] = dd
        idx_ns[rr, pp] = cc
    # placeholder neighbour index for padded slots (self-loop id is harmless;
    # only used as a fallback table, never as a distance value)
    pad = idx_ns < 0
    if np.any(pad):
        rows = np.nonzero(pad)[0]
        idx_ns[pad] = rows
    ar = np.arange(N, dtype=np.int32)
    idx_with_self = np.concatenate([ar[:, None], idx_ns], axis=1)
    dst_with_self = np.concatenate([np.zeros((N, 1), dtype=np.float64), dst_ns], axis=1)
    return idx_with_self, dst_with_self


# ===========================================
# CORE-SG model wrapper (HDBSCAN-like)
# ===========================================
class CoreSGModel:
    """
    Lightweight wrapper that mimics the HDBSCAN attributes used by this package.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Cluster labels for each sample.
    probabilities_ : numpy.ndarray
        Membership strengths for each sample.
    cluster_persistence_ : numpy.ndarray
        Persistence score for each cluster.
    condensed_tree_ : object
        Condensed tree wrapper with plotting support.
    single_linkage_tree_ : object
        Single-linkage tree wrapper.
    """

    def __init__(self,
                 labels: np.ndarray,
                 probabilities: np.ndarray,
                 stabilities: np.ndarray,
                 condensed_tree_array: np.recarray,
                 single_linkage_tree: np.ndarray):
        self.labels_ = labels
        self.probabilities_ = probabilities
        self.cluster_persistence_ = stabilities
        self.condensed_tree_ = _CondensedTree(condensed_tree_array, labels)
        self.single_linkage_tree_ = _SingleLinkageTree(single_linkage_tree)


    @property
    def outlier_scores_(self) -> np.ndarray:
        """GLOSH scores for this model's condensed hierarchy (cached)."""
        if not hasattr(self, "_outlier_scores"):
            self._outlier_scores = glosh_from_condensed_tree(self.condensed_tree_)
        return self._outlier_scores

# ===========================================
# CORE-SG generic with fast MST logic
# ===========================================
@dataclass
class CoreSGHDBSCAN:
    """
    CORE-SG multi-HDBSCAN implementation using the generic HDBSCAN pipeline
    together with the package's faster MST logic.

    Workflow
    --------
    1. Compute the full distance matrix once.
    2. Compute self-inclusive core distances for all values in
       ``min_samples_list``.
    3. Build the CORE-SG graph from:
       - the kmax nearest-neighbor graph with ties
       - the MST on the complete MRD graph for kmax
    4. Precompute a sparse neighbor table for fast edge distance lookup.
    5. For each ``m``:
       - compute MRD edge weights
       - build the sparse weighted graph
       - compute the MST
       - build the single-linkage tree
       - condense the tree and extract clusters

    Parameters
    ----------
    min_samples_list : list[int]
        List of ``min_samples`` values to evaluate.
    metric : str, default="euclidean"
        Distance metric mode.
    eps : float, default=1e-12
        Numerical tolerance used in graph construction.
    min_cluster_size : int or None, default=None
        Minimum cluster size. If ``None``, the package default behavior is used.
    """

    min_samples_list: List[int]
    metric: str = "euclidean"
    eps: float = 1e-12
    min_cluster_size: Optional[int] = None
    save_models: bool = False   # API compat with the graph wrapper (models are kept regardless)

    # Filled by fit()
    X_: Optional[np.ndarray] = field(init=False, default=None)
    N_: Optional[int] = field(init=False, default=None)
    D_: Optional[np.ndarray] = field(init=False, default=None)        # (N,N) float64 (dense path only)
    core_: Dict[int, np.ndarray] = field(init=False, default_factory=dict)  # m -> (N,) float64
    kmax_: Optional[int] = field(init=False, default=None)
    edges_ut_: Optional[np.ndarray] = field(init=False, default=None)       # CORE-SG edges (E,2), i<j
    edge_base_: Optional[np.ndarray] = field(init=False, default=None)      # (E,) base dist per edge (sparse path)
    fill_value_: float = field(init=False, default=1.0)                     # non-edge dissimilarity (sparse path)

    # kNN tables
    idx_with_self_: Optional[np.ndarray] = field(init=False, default=None)  # (N, kmax+1)
    dst_with_self_: Optional[np.ndarray] = field(init=False, default=None)  # (N, kmax+1)
    idx_no_self_: Optional[np.ndarray] = field(init=False, default=None)    # (N, kmax)
    dst_no_self_: Optional[np.ndarray] = field(init=False, default=None)    # (N, kmax)
    A_knn_: Optional[csr_matrix] = field(init=False, default=None)          # neighbor → distance CSR

    # MSTs per m (optional)
    msts_: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(init=False, default_factory=dict)
    mst_times_: Dict[int, float] = field(init=False, default_factory=dict)

    # Final HDBSCAN-like models per m
    models_: Dict[int, CoreSGModel] = field(init=False, default_factory=dict)
    times_: Dict[int, float] = field(init=False, default_factory=dict)      # total per-m time

    # Convenience views (API compat with the graph wrapper)
    labels_by_m_: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    condensed_trees_: Dict[int, object] = field(init=False, default_factory=dict)

    # --------------------------------------------------------
    # FIT: build D, self-inclusive cores, CORE-SG graph, CSR neighbor table
    # --------------------------------------------------------
    def fit(self, X: np.ndarray) -> "CoreSGHDBSCAN":
        X = np.asarray(X, dtype=np.float64, order="C")
        N = X.shape[0]
        mlist = np.sort(np.unique(self.min_samples_list)).astype(int)
        if len(mlist) == 0:
            raise ValueError("min_samples_list must contain at least one integer.")
        kmax = int(mlist[-1])
        if kmax >= N:
            raise ValueError(f"kmax ({kmax}) must be < N ({N}).")

        self.X_ = X
        self.N_ = N
        self.kmax_ = kmax

        # A) full distances D (for ties + fallback base distances)
        t0 = time.time()
        D = squareform(pdist(X, metric=self.metric)).astype(np.float64, copy=False)
        np.fill_diagonal(D, 0.0)
        self.D_ = D
        t1 = time.time()
        print(f"[CORE-SG] Full distance matrix computed in {t1 - t0:.3f}s")

        # B) kNN with self included: n_neighbors = kmax + 1, stable tie-breaking
        t0 = time.time()
        nn = NearestNeighbors(metric=self.metric)
        nn.fit(X)
        d_all, idx_all = nn.kneighbors(X, n_neighbors=kmax + 1, return_distance=True)

        # Force self at column 0
        ar = np.arange(N, dtype=np.int32)
        if not np.all(idx_all[:, 0] == ar):
            idx_fixed = np.empty_like(idx_all)
            d_fixed = np.empty_like(d_all)
            idx_fixed[:, 0] = ar
            d_fixed[:, 0] = 0.0
            idx_fixed[:, 1:] = idx_all[:, :kmax]
            d_fixed[:, 1:] = d_all[:, :kmax]
            idx_all, d_all = idx_fixed, d_fixed

        # Stable tie-break per row by (distance, index) for cols 1..kmax
        order = np.argsort(d_all[:, 1:], axis=1, kind="mergesort")
        row = np.arange(N)[:, None]
        order_full = np.concatenate([np.zeros((N, 1), dtype=int), order + 1], axis=1)
        d_all = d_all[row, order_full]
        idx_all = idx_all[row, order_full]

        self.idx_with_self_ = idx_all
        self.dst_with_self_ = d_all
        self.idx_no_self_ = idx_all[:, 1:]     # neighbors without self
        self.dst_no_self_ = d_all[:, 1:]

        # SELF-INCLUSIVE cores: core_m[i] = d_with_self[i,m]
        self.core_.clear()
        for m in mlist:
            self.core_[m] = self.dst_with_self_[:, m].astype(np.float64, copy=False)
        core_kmax = self.core_[kmax]
        t1 = time.time()
        print(f"[CORE-SG] Self-inclusive core distances for {len(mlist)} m values in {t1 - t0:.3f}s")

        # C) kmax-NNG WITH ALL TIES (OR condition with cores) — chunked upper
        #    triangle scan, O(block*N) peak instead of O(N^2) triu index arrays.
        t0 = time.time()
        kmax_edges = _kmax_nng_edges_chunked(D, core_kmax, self.eps)
        t1 = time.time()
        print(f"[CORE-SG] kmax-NNG-with-ties has {kmax_edges.shape[0]} edges (built in {t1 - t0:.3f}s)")

        # D) MST(kmax) on COMPLETE MRD via your dense Prim
        t0 = time.time()
        mst_kmax_edges = prim_mrd_mst_edges(X, core_kmax)   # (N-1,2), i<j
        t1 = time.time()
        print(f"[CORE-SG] MST_kmax (Prim) has {mst_kmax_edges.shape[0]} edges (built in {t1 - t0:.3f}s)")

        # E) CORE-SG edges = union(kmax-NNG-with-ties, MST_kmax)
        if kmax_edges.size:
            all_edges = np.vstack([kmax_edges, mst_kmax_edges])
        else:
            all_edges = mst_kmax_edges
        self.edges_ut_ = np.unique(all_edges, axis=0)
        print(f"[CORE-SG] CORE-SG graph has {self.edges_ut_.shape[0]} undirected edges")

        # F) Build CSR neighbor table A from kNN (no self) for fast base distances
        I_dir = np.repeat(np.arange(N, dtype=np.int32), kmax)
        J_dir = self.idx_no_self_.ravel().astype(np.int32)
        W_dir = self.dst_no_self_.ravel().astype(np.float64)
        self.A_knn_ = csr_matrix((W_dir, (I_dir, J_dir)), shape=(N, N))

        return self

    def fit_from_distance_matrix(self, D: np.ndarray) -> "CoreSGHDBSCAN":
        """
        Build CORE-SG *from a precomputed distance matrix* D (NxN).

        - D[i,j] is the base dissimilarity between points i and j.
        - We compute self-inclusive core distances and kmax-NNG from D.
        - We build CORE-SG edges via kmax-NNG ∪ MST_kmax (on MRD_kmax).

        After this, you can call self.run(...) exactly as usual.
        """
        D = np.asarray(D, dtype=np.float64, order="C")
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be a square matrix.")
        N = D.shape[0]
        np.fill_diagonal(D, 0.0)

        mlist = np.sort(np.unique(self.min_samples_list)).astype(int)
        if len(mlist) == 0:
            raise ValueError("min_samples_list must contain at least one integer.")
        kmax = int(mlist[-1])
        if kmax >= N:
            raise ValueError(f"kmax ({kmax}) must be < N ({N}).")

        self.X_ = None           # we’re working purely in distance space
        self.N_ = N
        self.kmax_ = kmax
        self.D_ = D

        # --- kNN from D (self-inclusive) ---
        # argpartition to the kmax+1 smallest per row, then stable-sort *only*
        # those. This is value-identical to a full-row mergesort + slice: the
        # m-th smallest DISTANCE per row is unchanged, so core distances, the
        # kmax-NNG edge set and the MST (all value-based) are bit-identical.
        # Only the tie-index ordering within a row may differ, and that feeds
        # A_knn_ whose values are recovered from D regardless — labels are
        # therefore bit-identical while argsort drops from O(N^2 log N) to
        # O(N^2 + N*kmax*log kmax).
        if N < kmax + 1:
            raise ValueError("Distance matrix does not have enough neighbors per row.")
        part = np.argpartition(D, kmax, axis=1)[:, :kmax + 1]
        dpart = np.take_along_axis(D, part, axis=1)
        order = np.argsort(dpart, axis=1, kind="stable")
        idx_all = np.take_along_axis(part, order, axis=1)
        d_all = np.take_along_axis(dpart, order, axis=1)

        # ensure self is at column 0
        ar = np.arange(N, dtype=np.int32)
        if not np.all(idx_all[:, 0] == ar):
            for i in range(N):
                pos = int(np.where(idx_all[i] == i)[0][0])
                if pos != 0:
                    idx_all[i, 0], idx_all[i, pos] = idx_all[i, pos], idx_all[i, 0]
                    d_all[i, 0], d_all[i, pos] = d_all[i, pos], d_all[i, 0]

        self.idx_with_self_ = idx_all
        self.dst_with_self_ = d_all
        self.idx_no_self_ = idx_all[:, 1:]
        self.dst_no_self_ = d_all[:, 1:]

        # --- self-inclusive core distances for all m ---
        self.core_.clear()
        for m in mlist:
            self.core_[m] = self.dst_with_self_[:, m].astype(np.float64, copy=False)
        core_kmax = self.core_[kmax]

        # --- kmax-NNG-with-ties from D & core_kmax (chunked, O(block*N) peak) ---
        kmax_edges = _kmax_nng_edges_chunked(D, core_kmax, self.eps)

        # --- MST_kmax on COMPLETE MRD graph, using D as base distances ---
        mst_kmax_edges = prim_mrd_mst_edges_from_D(D, core_kmax)  # (N-1,2)

        # --- CORE-SG edges = union(kmax-NNG-with-ties, MST_kmax) ---
        if kmax_edges.size:
            all_edges = np.vstack([kmax_edges, mst_kmax_edges])
        else:
            all_edges = mst_kmax_edges
        self.edges_ut_ = np.unique(all_edges, axis=0)
        print(f"[CORE-SG] (precomputed) CORE-SG graph has {self.edges_ut_.shape[0]} edges")

        # --- CSR neighbor table A_knn_ from idx_no_self_/dst_no_self_ ---
        I_dir = np.repeat(np.arange(N, dtype=np.int32), kmax)
        J_dir = self.idx_no_self_.ravel().astype(np.int32)
        W_dir = self.dst_no_self_.ravel().astype(np.float64)
        self.A_knn_ = csr_matrix((W_dir, (I_dir, J_dir)), shape=(N, N))

        return self

    # --------------------------------------------------------
    # FIT (sparse): build CORE-SG from a sparse graph, no dense N x N ever
    # --------------------------------------------------------
    def fit_from_sparse_graph(self, S, fill_value: float = 1.0) -> "CoreSGHDBSCAN":
        """Build CORE-SG *from a sparse dissimilarity graph* ``S`` (N x N)
        without ever materialising the dense N x N matrix.

        Stored entries of ``S`` are the *real* pairwise dissimilarities; any
        missing pair is treated as ``fill_value`` (the dense path fills a
        uniform 1.0 ceiling). This reproduces the clustering of
        ``fit_from_distance_matrix(dense_fill(S)).run(...)``:

        * core distances are value-identical (m-th smallest real dissimilarity,
          padded with ``fill_value``);
        * every real CORE-SG edge is kept with its exact base distance;
        * the dense path's "under-connected point connects to *everyone* at
          ``fill_value``" edges are replaced by a minimal set of ``fill_value``
          bridges over the connected components of the real-edge graph.

        Because every ``fill_value`` link carries the same maximal weight, the
        single-linkage merges they induce all occur at the top of the
        dendrogram; the sub-``fill_value`` cluster structure — and therefore the
        labels — is unchanged (ARI = 1.0 vs the dense path). Only the identity
        of equal-weight top-level bridge edges may differ.

        After this, call ``run(...)`` exactly as usual.
        """
        if S.shape[0] != S.shape[1]:
            raise ValueError("S must be a square sparse matrix.")
        N = S.shape[0]

        mlist = np.sort(np.unique(self.min_samples_list)).astype(int)
        if len(mlist) == 0:
            raise ValueError("min_samples_list must contain at least one integer.")
        kmax = int(mlist[-1])
        if kmax >= N:
            raise ValueError(f"kmax ({kmax}) must be < N ({N}).")

        self.X_ = None
        self.D_ = None                     # never materialised on the sparse path
        self.N_ = N
        self.kmax_ = kmax
        self.fill_value_ = float(fill_value)

        # 1) symmetric real edges (i<j), value = min over the two directions
        er, ec, ed = _symmetric_real_edges(S, fill_value)

        # 2) self-inclusive kNN tables + core distances (value-identical to dense)
        idx_ws, dst_ws = _knn_core_from_edges(er, ec, ed, N, kmax, fill_value)
        self.idx_with_self_ = idx_ws
        self.dst_with_self_ = dst_ws
        self.idx_no_self_ = idx_ws[:, 1:]
        self.dst_no_self_ = dst_ws[:, 1:]
        self.core_.clear()
        for m in mlist:
            self.core_[int(m)] = dst_ws[:, int(m)].astype(np.float64, copy=False)
        core_kmax = self.core_[kmax]

        # 3) kmax-NNG-with-ties restricted to REAL edges (identical condition;
        #    non-edges are all == fill_value and only ever bridge components).
        if ed.size:
            keep_knn = (ed <= core_kmax[er] + self.eps) | (ed <= core_kmax[ec] + self.eps)
        else:
            keep_knn = np.zeros(0, dtype=bool)

        # 4) MST_kmax over the real-edge MRD graph (SciPy). Real edges chosen by
        #    the MST are kept even if they failed the kNN condition.
        if ed.size:
            mrd = np.maximum(np.maximum(core_kmax[er], core_kmax[ec]), ed)
            G = coo_matrix(
                (np.concatenate([mrd, mrd]),
                 (np.concatenate([er, ec]), np.concatenate([ec, er]))),
                shape=(N, N),
            )
            mst = csgraph.minimum_spanning_tree(G).tocoo()
            mu = np.minimum(mst.row, mst.col).astype(np.int64)
            mv = np.maximum(mst.row, mst.col).astype(np.int64)
            eid = er.astype(np.int64) * N + ec.astype(np.int64)
            mid = mu * N + mv
            is_mst = np.isin(eid, mid)
        else:
            is_mst = np.zeros(0, dtype=bool)

        real_keep = keep_knn | is_mst
        rer = er[real_keep].astype(np.int32)
        rec = ec[real_keep].astype(np.int32)
        red = ed[real_keep].astype(np.float64)

        # 5) connected components of the kept real-edge graph, then bridge every
        #    component (including isolated singletons) with one fill_value edge.
        if rer.size:
            Gc = coo_matrix((np.ones(rer.shape[0]), (rer, rec)), shape=(N, N))
        else:
            Gc = coo_matrix((N, N))
        ncomp, comp = csgraph.connected_components(Gc, directed=False)
        if ncomp > 1:
            # one representative node per component (first occurrence)
            order_c = np.argsort(comp, kind="stable")
            comp_sorted = comp[order_c]
            first = np.ones(comp_sorted.shape[0], dtype=bool)
            first[1:] = comp_sorted[1:] != comp_sorted[:-1]
            reps = order_c[first]
            ba, bb = reps[:-1], reps[1:]           # chain the representatives
            bi = np.minimum(ba, bb).astype(np.int32)
            bj = np.maximum(ba, bb).astype(np.int32)
            bridge_base = np.full(bi.shape[0], float(fill_value), dtype=np.float64)
        else:
            bi = np.empty(0, dtype=np.int32)
            bj = np.empty(0, dtype=np.int32)
            bridge_base = np.empty(0, dtype=np.float64)

        # 6) assemble CORE-SG edges + aligned per-edge base distances
        all_i = np.concatenate([rer, bi])
        all_j = np.concatenate([rec, bj])
        all_b = np.concatenate([red, bridge_base])
        order = np.lexsort((all_j, all_i))          # canonical (i, j) order
        self.edges_ut_ = np.stack([all_i[order], all_j[order]], axis=1).astype(np.int32)
        self.edge_base_ = all_b[order].astype(np.float64)
        print(f"[CORE-SG] (sparse) CORE-SG graph has {self.edges_ut_.shape[0]} edges "
              f"({int(rer.size)} real + {int(bi.size)} bridges) across {int(ncomp)} component(s)")

        # A_knn_ is unused on the sparse path (edge_base_ drives run()).
        self.A_knn_ = None
        return self


    # --------------------------------------------------------
    # Helper: base distance per edge using kNN tables or D
    # --------------------------------------------------------
    def _base_distance_from_tables_or_D(self, r: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        For each undirected edge (r[i], c[i]): try to get base distance from the
        kNN tables (if either direction appears there); otherwise fall back to D.
        """
        A = self.A_knn_
        D = self.D_

        w1 = A[r, c].A.ravel()
        w2 = A[c, r].A.ravel()

        base = np.empty_like(w1)
        mask1 = w1 > 0
        mask2 = w2 > 0
        both = mask1 & mask2
        none = (~mask1) & (~mask2)

        base[both] = np.minimum(w1[both], w2[both])
        base[mask1 & (~both)] = w1[mask1 & (~both)]
        base[mask2 & (~both)] = w2[mask2 & (~both)]
        base[none] = D[r[none], c[none]]

        return base


    def model(self, min_samples):
        return self.models_[min_samples]
    # --------------------------------------------------------
    # RUN: per-m MST on CORE-SG graph + generic pipeline
    # --------------------------------------------------------
    def run(self,
            cluster_selection_method: str = "eom",
            allow_single_cluster: bool = False,
            match_reference_implementation: bool = True,
            cluster_selection_epsilon: float = 0.0,
            foscx_settings=None) -> "CoreSGHDBSCAN":
        # ``foscx_settings`` is accepted for API compatibility with the graph
        # wrapper; the generic HDBSCAN pipeline below does not use it.

        if self.edges_ut_ is None or not self.core_:
            raise RuntimeError("Call fit(X) / fit_from_distance_matrix(D) / "
                               "fit_from_sparse_graph(S) before run().")
        if self.D_ is None and self.edge_base_ is None:
            raise RuntimeError("No base distances available: call a fit_* method before run().")

        self.models_.clear()
        self.msts_.clear()
        self.mst_times_.clear()
        self.times_.clear()
        self.labels_by_m_.clear()
        self.condensed_trees_.clear()

        N = self.N_
        D = self.D_
        edges = self.edges_ut_
        i_idx = edges[:, 0]
        j_idx = edges[:, 1]

        # Per-edge base dissimilarity is loop-invariant across m: compute it
        # once. The sparse path precomputes it exactly as self.edge_base_; the
        # dense path derives it from the kNN tables / D a single time here.
        if self.edge_base_ is not None:
            base = self.edge_base_
        else:
            base = self._base_distance_from_tables_or_D(i_idx, j_idx)

        for m in sorted(np.unique(self.min_samples_list)):
            core_m = self.core_[int(m)]

            # --- reweight edges with MRD_m (base precomputed, loop-invariant) ---
            t0 = time.time()
            w_ut = np.maximum.reduce([core_m[i_idx], core_m[j_idx], base])

            # build symmetric sparse graph and MST
            data = np.concatenate([w_ut, w_ut])
            row = np.concatenate([i_idx, j_idx])
            col = np.concatenate([j_idx, i_idx])
            G = coo_matrix((data, (row, col)), shape=(N, N))
            mst_sparse = csgraph.minimum_spanning_tree(G)
            coo_mst = mst_sparse.tocoo()
            u = coo_mst.row.astype(np.int32)
            v = coo_mst.col.astype(np.int32)
            w = coo_mst.data.astype(np.float64)

            min_spanning_tree = np.vstack([u, v, w]).T
            order = np.argsort(min_spanning_tree[:, 2])
            min_spanning_tree = min_spanning_tree[order]

            mst_time = time.time() - t0
            self.msts_[int(m)] = (u, v, w)
            self.mst_times_[int(m)] = mst_time

            # --- generic pipeline: MST -> single_linkage_tree -> labels ---
            t1 = time.time()
            single_linkage_tree = _linkage_label(min_spanning_tree)
            effective_min_cluster_size = int(m) if self.min_cluster_size is None else int(self.min_cluster_size)
            condensed_tree_array = _condense_tree(single_linkage_tree, effective_min_cluster_size)
            stability_dict = _compute_stability(condensed_tree_array)
            labels, probabilities, stabilities = _get_clusters(
                condensed_tree_array,
                stability_dict,
                cluster_selection_method,
                allow_single_cluster,
                match_reference_implementation,
                cluster_selection_epsilon,
            )
            t2 = time.time()

            model = CoreSGModel(
                labels=labels,
                probabilities=probabilities,
                stabilities=stabilities,
                condensed_tree_array=condensed_tree_array,
                single_linkage_tree=single_linkage_tree,
            )
            self.models_[int(m)] = model
            self.times_[int(m)] = mst_time + (t2 - t1)
            self.labels_by_m_[int(m)] = labels
            self.condensed_trees_[int(m)] = model.condensed_tree_

            print(f"[CORE-SG] m={m:2d}: MST+tree+labels in {self.times_[int(m)]:.4f}s")

        return self


    def outlier_scores_for(self, min_samples) -> np.ndarray:
        """Return GLOSH scores for a fitted min_samples value."""
        m = int(min_samples)
        if m not in self.models_:
            raise KeyError(
                f"min_samples={m} was not fitted. "
                f"Available values: {sorted(self.models_)}"
            )
        return self.models_[m].outlier_scores_

    @property
    def outlier_scores_by_m_(self) -> dict:
        """GLOSH arrays keyed by fitted min_samples value."""
        return {m: self.models_[m].outlier_scores_ for m in sorted(self.models_)}

    @property
    def outlier_scores_(self) -> np.ndarray:
        """GLOSH scores when exactly one min_samples value was fitted."""
        ms = sorted(self.models_)
        if not ms:
            raise AttributeError("No fitted model is available. Call fit/run first.")
        if len(ms) == 1:
            return self.models_[ms[0]].outlier_scores_
        raise AttributeError(
            "Multiple min_samples values were fitted; "
            "use outlier_scores_for(m) or outlier_scores_by_m_."
        )
    
    
    # convenience plotting for one m
    def plot_condensed_tree(self, m: int, figsize=(8, 5), lambda_floor=1.0, **kwargs):
        import matplotlib.pyplot as plt
        if m not in self.models_:
            raise KeyError(f"m={m} not in CORE-SG models.")
        model = self.models_[m]
        if model.condensed_tree_ is None:
            print(f"No condensed tree for CORE-SG m={m}")
            return
        plt.figure(figsize=figsize)
        ax = plot_condensed_tree_bounded(
            model.condensed_tree_,
            lambda_floor=lambda_floor,
            select_clusters=True, label_clusters=True,
            **kwargs,
        )
        ax.set_title(f"CORE-SG Condensed Tree (min_samples = {m})")
        plt.show()
        return ax


    def plot_condensed_tree_ground_truth_pies(
        self, m, y_true, *, figsize=(16, 10), **kwargs
    ):
        """Condensed tree for ``min_samples=m`` with ground-truth pies.

        See
        :func:`~coresg_graphhdbscan.plot_condensed_tree_ground_truth_pies`
        for the full list of keyword options.

        Parameters
        ----------
        m : int
            The ``min_samples`` value selecting which condensed tree to draw.
        y_true : array-like, shape (n_samples,)
            Ground-truth labels aligned row-for-row with the fitted data.
        figsize : tuple, default=(16, 10)
            Figure size.

        Returns
        -------
        fig, ax
        """
        from .ground_truth_pies import (
            plot_condensed_tree_ground_truth_pies as _pies,
        )

        if int(m) not in self.models_:
            raise KeyError(f"m={m} not in CORE-SG models.")

        return _pies(
            self,
            y_true,
            m=int(m),
            figsize=figsize,
            **kwargs,
        )

def plot_condensed_tree_bounded(
    condensed_tree,
    *,
    lambda_floor=1.0,
    dash_synthetic=True,
    dash_style=(0, (5, 3)),
    dash_color="0.35",
    dash_linewidth=1.3,
    axis=None,
    figsize=(8, 5),
    **plot_kwargs,
):
    """Plot an hdbscan CondensedTree for a bounded similarity graph (eps in [0, 1]).

    Starts the lambda axis at ``lambda_floor`` (= 1/max_eps) instead of hdbscan's
    hardcoded 0, and draws the synthetic weight-1 joins of a disconnected graph
    (splits at exactly the floor) as dashed connectors.
    """
    import matplotlib.pyplot as plt

    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    lam = np.asarray(condensed_tree._raw_tree["lambda_val"], float)
    finite = lam[np.isfinite(lam)]
    min_lambda = float(finite.min()) if finite.size else 0.0
    floor = min(float(lambda_floor), min_lambda)  # never crop real structure

    axis = condensed_tree.plot(axis=axis, **plot_kwargs)

    if dash_synthetic:
        for line in axis.get_lines():
            yd = np.asarray(line.get_ydata(), float)
            if yd.size == 2 and np.all(np.isclose(yd, floor, atol=1e-9)):
                line.set_linestyle(dash_style)
                line.set_color(dash_color)
                line.set_linewidth(dash_linewidth)

    bottom, _top = axis.get_ylim()   # inverted axis: 2nd entry is the top
    axis.set_ylim(bottom, floor)
    return axis


# ===========================================
# Helper: plot condensed tree for any model dict
# ===========================================
def plot_condensed_tree_for_m(models_dict, m: int, title_prefix: str = "", figsize=(8, 5)):
    import matplotlib.pyplot as plt
    if m not in models_dict:
        raise KeyError(f"m={m} not found in models_dict")

    model = models_dict[m]
    ct = model.condensed_tree_
    if ct is None:
        print(f"No condensed tree available for m={m}")
        return

    plt.figure(figsize=figsize)
    ct.plot(select_clusters=True, label_clusters=True)
    if title_prefix:
        plt.title(f"{title_prefix} Condensed Tree (min_samples = {m})")
    else:
        plt.title(f"Condensed Tree (min_samples = {m})")
    plt.show()
