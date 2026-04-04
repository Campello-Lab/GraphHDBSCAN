"""CoreSG-HDBSCAN core implementation"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


# ===========================================
# CORE-SG generic with fast MST logic
# ===========================================
@dataclass
class CoreSGHDBSCAN:
    """
    CoreSG-based hierarchical density clustering backend.

    This class implements the lower-level CoreSG-HDBSCAN pipeline operating on
    feature vectors or distance representations.

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
    save_models: bool = False

    # Filled by fit()
    X_: Optional[np.ndarray] = field(init=False, default=None)
    N_: Optional[int] = field(init=False, default=None)
    D_: Optional[np.ndarray] = field(init=False, default=None)        # (N,N) float64
    core_: Dict[int, np.ndarray] = field(init=False, default_factory=dict)  # m -> (N,) float64
    kmax_: Optional[int] = field(init=False, default=None)
    edges_ut_: Optional[np.ndarray] = field(init=False, default=None)       # CORE-SG edges (E,2), i<j

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
    condensed_trees_: Dict[int, object] = field(init=False, default_factory=dict)
    labels_by_m_: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    times_: Dict[int, float] = field(init=False, default_factory=dict)

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

        # C) kmax-NNG WITH ALL TIES (OR condition with cores)
        t0 = time.time()
        iu, ju = np.triu_indices(N, k=1)
        cond = (D[iu, ju] <= core_kmax[iu] + self.eps) | (D[iu, ju] <= core_kmax[ju] + self.eps)
        kmax_edges = np.stack((iu[cond].astype(np.int32), ju[cond].astype(np.int32)), axis=1)
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
        # sort each row by distance (stable, so ties broken by index)
        idx_all = np.argsort(D, axis=1, kind="mergesort")
        d_all = np.take_along_axis(D, idx_all, axis=1)

        # keep self + kmax neighbors
        if idx_all.shape[1] < kmax + 1:
            raise ValueError("Distance matrix does not have enough neighbors per row.")
        idx_all = idx_all[:, :kmax + 1]
        d_all = d_all[:, :kmax + 1]

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

        # --- kmax-NNG-with-ties from D & core_kmax (same condition as in fit) ---
        iu, ju = np.triu_indices(N, k=1)
        cond = (D[iu, ju] <= core_kmax[iu] + self.eps) | (
            D[iu, ju] <= core_kmax[ju] + self.eps
        )
        if np.any(cond):
            kmax_edges = np.stack(
                (iu[cond].astype(np.int32), ju[cond].astype(np.int32)), axis=1
            )
        else:
            kmax_edges = np.empty((0, 2), dtype=np.int32)

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
        if not self.save_models:
            raise ValueError(
                "Models were not saved. Initialize with save_models=True to access models_[min_samples]."
            )
        if min_samples not in self.models_:
            raise KeyError(f"min_samples={min_samples} not found in saved models.")
        return self.models_[min_samples]
        
    # --------------------------------------------------------
    # RUN: per-m MST on CORE-SG graph + generic pipeline
    # --------------------------------------------------------
    def run(self,
            cluster_selection_method: str = "eom",
            allow_single_cluster: bool = False,
            match_reference_implementation: bool = False,
            cluster_selection_epsilon: float = 0.0) -> "CoreSGHDBSCAN":

        if self.D_ is None or self.edges_ut_ is None or not self.core_:
            raise RuntimeError("Call fit(X) before run().")

        self.models_.clear()
        self.msts_.clear()
        self.mst_times_.clear()
        self.times_.clear()
        self.condensed_trees_.clear()
        self.labels_by_m_.clear()

        N = self.N_
        D = self.D_
        edges = self.edges_ut_
        i_idx = edges[:, 0]
        j_idx = edges[:, 1]

        for m in sorted(np.unique(self.min_samples_list)):
            core_m = self.core_[int(m)]

            # --- reweight edges with MRD_m via your base-distance scheme ---
            t0 = time.time()
            base = self._base_distance_from_tables_or_D(i_idx, j_idx)
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
            
            self.labels_by_m_[int(m)] = labels
            self.condensed_trees_[int(m)] = model.condensed_tree_
            
            if self.save_models:
                self.models_[int(m)] = model
            
            self.times_[int(m)] = mst_time + (t2 - t1)

            print(f"[CORE-SG] m={m:2d}: MST+tree+labels in {self.times_[int(m)]:.4f}s")

        return self


    # convenience plotting for one m

    def plot_condensed_tree(self, m: int, figsize=(8, 5)):
        import matplotlib.pyplot as plt
    
        if m in self.condensed_trees_:
            ct = self.condensed_trees_[m]
        elif m in self.models_:
            ct = self.models_[m].condensed_tree_
        else:
            raise KeyError(f"m={m} not found in CORE-SG results.")
    
        if ct is None:
            print(f"No condensed tree for CORE-SG m={m}")
            return
    
        plt.figure(figsize=figsize)
        ct.plot(select_clusters=True, label_clusters=True)
        plt.title(f"CORE-SG Condensed Tree (min_samples = {m})")
        plt.show()


# ===========================================
# Helper: plot condensed tree for any model dict
# ===========================================
def plot_condensed_tree_for_m(models_dict, m: int, title_prefix: str = "", figsize=(8, 5)):
    """
    Plot the condensed tree for a selected fitted ``min_samples`` value.

    Parameters
    ----------
    model : CoreSGHDBSCAN or GraphCoreSGHDBSCAN
        Fitted clustering object.
    m : int
        Selected ``min_samples`` value.

    Returns
    -------
    None
    """
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
