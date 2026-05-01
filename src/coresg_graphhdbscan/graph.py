"""Graph-based wrapper around CoreSG-HDBSCAN."""

from .core import CoreSGHDBSCAN

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
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
from collections.abc import Iterable
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    njit = None
    prange = range
    _HAS_NUMBA = False

def _optional_import(module_name, package_name=None):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pkg = package_name or module_name
        raise ImportError(f"Optional dependency '{pkg}' is required for this functionality. Please install it.") from e


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

    This class constructs a similarity graph from feature data or accepts a
    precomputed graph representation, transforms that graph into a
    graph-derived distance representation, and then runs the CoreSG-HDBSCAN
    clustering pipeline.

    Parameters
    ----------
    min_samples : int or iterable of int, default=10
        Main clustering hyperparameter. A single integer gives one fitted
        solution, while an iterable allows fitting multiple values in one run.

    sim_graph_method : {"sc_umap", "sc_gauss", "jaccard_phenograph", "precomputed"}, default="sc_umap"
        Graph-construction backend.

    metric : {"euclidean", "cosine", "hybrid_euclidean_cosine"}, default="euclidean"
        Metric strategy used during graph construction.

    add_neighbor : bool, default=True
        Controls how weighted structural similarity is expanded into graph edges.

    no_noise : bool, default=True
        If True, points initially labeled as noise are reassigned after
        clustering.

    n_neighbors : int, default=15
        Number of neighbors used during graph construction.

    heuristic_connect : bool, default=False
        If True, increase ``n_neighbors`` until the WSS dissimilarity graph becomes connected,
        except in precomputed mode, where bridge edges are used instead.
        If False, disconnected components are connected with synthetic bridge
        edges.

    min_cluster_size : int or None, default=None
        Minimum cluster size used in the clustering stage. If None, the package
        follows the selected ``min_samples`` value for each run.

    save_models : bool, default=False
        If True, save hdbscan models for different min_samples which can add some memory overhead.
        If False, just save labels and condensed trees for each min_samples.

    **kwargs
        Additional keyword arguments passed to internal graph-construction
        helpers.

    Attributes
    ----------
    similarity_graph_ : networkx.Graph
        Initial similarity graph.

    similarity_graph_WSS : networkx.Graph
        Weighted structural similarity graph.

    dissimilarity_graph_ : networkx.Graph
        Graph after conversion from similarity to dissimilarity.

    connected_graph_ : networkx.Graph
        Final connected graph used by the clustering stage.

    dist_matrix_ : numpy.ndarray
        Dense matrix used by the CoreSG-HDBSCAN pipeline.

    coresg_ : CoreSGHDBSCAN
        Internal fitted CoreSG-HDBSCAN object.

    models_ : dict
        Dictionary of saved per-``min_samples`` models. Populated only when
        ``save_models=True``.
        
    condensed_trees_ : dict
        Dictionary of condensed tree objects keyed by fitted
        ``min_samples`` value.
        
    labels_by_m_ : dict
        Dictionary of stored labels keyed by fitted ``min_samples`` value.
    """
    def __init__(self,
                 min_samples=10,
                 sim_graph_method='sc_umap',
                 metric='euclidean',
                 add_neighbor=True,
                 no_noise=True,
                 n_neighbors=15,
                 heuristic_connect=False,
                 min_cluster_size=None,
                 save_models=False,
                 similarity_backend="auto",
                 **kwargs):

        # store graph params
        valid_graph_methods = {'sc_gauss', 'sc_umap', 'jaccard_phenograph', 'precomputed'}
        if sim_graph_method not in valid_graph_methods:
            raise ValueError(
                f"Unsupported sim_graph_method '{sim_graph_method}'. "
                f"Use one of {sorted(valid_graph_methods)}."
            )
        valid_metrics = {'euclidean', 'cosine', 'hybrid_euclidean_cosine'}
        if metric not in valid_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Use one of {sorted(valid_metrics)}."
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
        self.add_neighbor = add_neighbor
        self.no_noise = no_noise
        self.n_neighbors = n_neighbors
        if 'mst_approx' in kwargs:
            heuristic_connect = kwargs.pop('mst_approx')
        self.heuristic_connect = bool(heuristic_connect)
        self.save_models = bool(save_models)
        self.models_ = {}
        self.condensed_trees_ = {}
        self.labels_by_m_ = {}
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

        core_metric = 'euclidean' if metric == 'hybrid_euclidean_cosine' else metric
        super().__init__(
            min_samples_list=self.m_list,
            metric=core_metric,
            min_cluster_size=resolved_min_cluster_size,
            save_models=self.save_models,
            **kwargs
        )
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
    
    def _min_cluster_size_for(self, m):
        m = int(m)
        return m if self.min_cluster_size is None else int(self.min_cluster_size)

    def compute_similarity_sparse(self, graph) -> sp.csr_matrix:
        """Fast weighted structural similarity as a sparse matrix.

        This is algebraically equivalent to the original ``compute_similarity``
        implementation, but avoids Python-level all-pairs iteration. The
        weighted adjacency vector for each node includes an explicit self-loop
        of weight 1 before cosine normalization.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return sp.csr_matrix((0, 0))

        u_list, v_list, w_list = [], [], []
        for u, v, d in graph.edges(data=True):
            u_list.append(u)
            v_list.append(v)
            w_list.append(d.get("weight", 1.0))

        if u_list:
            u = np.asarray(u_list, dtype=np.int32)
            v = np.asarray(v_list, dtype=np.int32)
            w = np.asarray(w_list, dtype=np.float64)
            rows = np.concatenate([u, v, np.arange(n, dtype=np.int32)])
            cols = np.concatenate([v, u, np.arange(n, dtype=np.int32)])
            data = np.concatenate([w, w, np.ones(n, dtype=np.float64)])
        else:
            rows = np.arange(n, dtype=np.int32)
            cols = np.arange(n, dtype=np.int32)
            data = np.ones(n, dtype=np.float64)

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        A.eliminate_zeros()

        norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        norms[norms == 0.0] = 1.0
        inv = 1.0 / norms

        if not self.add_neighbor:
            A_norm = A.multiply(inv[:, None]).tocsr()
            out_rows, out_cols, out_data = [], [], []
            for u, v in graph.edges():
                sim = float(A_norm[u].multiply(A_norm[v]).sum())
                out_rows.extend((u, v))
                out_cols.extend((v, u))
                out_data.extend((sim, sim))
            S = sp.csr_matrix((out_data, (out_rows, out_cols)), shape=(n, n))
            S.eliminate_zeros()
            return S

        numerators = (A @ A.T).tocsr()
        S = numerators.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
        S.setdiag(0.0)
        S.eliminate_zeros()
        return S

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
    def similarity_to_dissimilarity_sparse(similarity_matrix: sp.csr_matrix) -> sp.csr_matrix:
        D = similarity_matrix.copy().tocsr()
        D.data = 1.0 - D.data
        D.setdiag(0.0)
        D.eliminate_zeros()
        return D

    @staticmethod
    def similarity_to_dissimilarity(similarity_graph):
        dissimilarity_graph = nx.Graph()
        for u, v, data in similarity_graph.edges(data=True):
            dissimilarity_graph.add_edge(u, v, weight=1 - data['weight'])
        return dissimilarity_graph

    @staticmethod
    def is_graph_connected(graph):
        return nx.is_connected(graph)

    @staticmethod
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
    
    def create_similarity_graph(self, data):
        if self.sim_graph_method == 'precomputed':
            return self._coerce_precomputed_graph(data)

        sc, sce, sc_gauss, sc_umap, _get_indices_distances_from_dense_matrix = _get_scanpy_modules()

        X = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array-like object.")

        if self.metric == 'hybrid_euclidean_cosine':
            distances_full = pairwise_distances(X, metric='euclidean')
            knn_metric = 'cosine'
            use_precomputed_knn = False
        else:
            distances_full = pairwise_distances(X, metric=self.metric)
            knn_metric = 'precomputed'
            use_precomputed_knn = True

        self.distances_full_ = distances_full

        if self.sim_graph_method == 'jaccard_phenograph':
            if use_precomputed_knn:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='precomputed',
                    include_self=False,
                )
            else:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='cosine',
                    include_self=False,
                )
        
            if self.similarity_backend == "numba":
                conn = self._fast_phenograph_jaccard_from_knn_graph(knn_dist)
        
            elif self.similarity_backend == "default":
                _, conn, _ = sce.tl.phenograph(
                    knn_dist,
                    directed=False,
                    clustering_algo=None,
                )
        
            else:  # similarity_backend == "auto"
                if _HAS_NUMBA:
                    conn = self._fast_phenograph_jaccard_from_knn_graph(knn_dist)
                else:
                    _, conn, _ = sce.tl.phenograph(
                        knn_dist,
                        directed=False,
                        clustering_algo=None,
                    )
        
            return nx.from_scipy_sparse_array(conn.tocsr(), edge_attribute='weight')

        if self.sim_graph_method == 'sc_gauss':
            if use_precomputed_knn:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='precomputed',
                    include_self=False,
                )
            else:
                knn_dist = kneighbors_graph(
                    distances_full,
                    n_neighbors=self.n_neighbors - 1,
                    mode='distance',
                    metric='cosine',
                    include_self=False,
                )
            conn = sc_gauss(knn_dist, n_neighbors=self.n_neighbors, knn=True)
            return nx.from_scipy_sparse_array(conn, edge_attribute='weight')

        if self.sim_graph_method == 'sc_umap':
            if use_precomputed_knn:
                idx, dists = _get_indices_distances_from_dense_matrix(
                    distances_full, self.n_neighbors
                )
            else:
                nn = NN(n_neighbors=self.n_neighbors, metric=knn_metric)
                nn.fit(X)
                dists, idx = nn.kneighbors(X, return_distance=True)
            conn = sc_umap(
                idx,
                dists,
                n_obs=distances_full.shape[0],
                n_neighbors=self.n_neighbors,
            )
            return nx.from_scipy_sparse_array(conn, edge_attribute='weight')

        raise ValueError(
            "Unsupported sim_graph_method. Use one of 'sc_gauss', 'sc_umap', 'jaccard_phenograph', or 'precomputed'."
        )

    def connect_graph_heuristically(self, graph, n_obs):
        """Connect disconnected components with synthetic bridge edges.
    
        This function assumes `graph` is already a dissimilarity graph:
    
            smaller weight = closer
            larger weight = farther
    
        It does not rebuild the similarity graph.
        It only adds bridge edges of distance weight 1 between disconnected components.
        """
        new_graph = graph.copy()
        new_graph.add_nodes_from(range(n_obs))
    
        if nx.is_connected(new_graph):
            return new_graph
    
        components = list(nx.connected_components(new_graph))
    
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
    
            # weight=1 means weakest / maximum-distance bridge
            new_graph.add_edge(u, v, weight=1)
    
        return new_graph

    @staticmethod
    def compute_full_distance_matrix(graph):
        """
        Compute the full dense matrix of shortest path distances using Floyd–Warshall.
        """
        return np.array(nx.floyd_warshall_numpy(graph, weight='weight'))

    @staticmethod
    def compute_sparse_distance_dict(graph):
        """
        Compute a dictionary-of-dictionaries of shortest path distances.
        For each node, run single_source_dijkstra_path_length and store the results.
        """
        distance_dict = {}
        for node in graph.nodes():
            # Compute shortest path lengths from 'node' to all others.
            distance_dict[node] = nx.single_source_dijkstra_path_length(graph, node, weight='weight')
        return distance_dict

    def graph_metric(self, u, v):
        """
        Custom distance metric that uses the precomputed sparse distance dictionary.
        The data points are mapped to node indices using self._point_to_index.
        """
        idx_u = self._point_to_index.get(tuple(u))
        idx_v = self._point_to_index.get(tuple(v))
        try:
            return self.distance_dict_[idx_u][idx_v]
        except KeyError:
            # Since the graph is connected, this should rarely happen.
            # Check the reverse ordering as a fallback.
            return self.distance_dict_[idx_v][idx_u]

    @staticmethod
    def compute_custom_distance_matrix(graph):
        """Compute the pairwise distance matrix used by the graph-based pipeline.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
    
        Returns
        -------
        numpy.ndarray
            Pairwise distance matrix.
        """
        n = graph.number_of_nodes()
        dist = np.full((n, n), 1, dtype=np.float64)
        np.fill_diagonal(dist, 0)
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            dist[u, v] = weight
            dist[v, u] = weight
        return dist


    @staticmethod
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

    def _build_graph_distance(self, X):
        """Build graph-derived dense distances using the sparse fast path.
    
        Pipeline:
            data / precomputed graph
            -> initial similarity graph
            -> weighted structural similarity graph
            -> WSS dissimilarity graph
            -> connected graph
            -> dense precomputed distance matrix
        """
        self.data_ = (
            X if self.sim_graph_method == "precomputed"
            else (np.array(X) if isinstance(X, pd.DataFrame) else X)
        )
    
        # ------------------------------------------------------------
        # 1. Build initial similarity graph
        # ------------------------------------------------------------
        self.similarity_graph_ = self.create_similarity_graph(self.data_)
    
        # Use the graph size, not len(self.data_).
        # This is safer for precomputed NetworkX graphs and scipy sparse matrices.
        n_obs = self.similarity_graph_.number_of_nodes()
        self.n_obs_ = n_obs
    
        self.similarity_graph_.add_nodes_from(range(n_obs))
    
        # ------------------------------------------------------------
        # 2. Compute WSS similarity, then convert to dissimilarity
        # ------------------------------------------------------------
        self.similarity_graph_WSS_sparse_ = self.compute_similarity_sparse(
            self.similarity_graph_
        )
    
        self.dissimilarity_graph_sparse_ = self.similarity_to_dissimilarity_sparse(
            self.similarity_graph_WSS_sparse_
        )
    
        # ------------------------------------------------------------
        # 3. Check whether WSS dissimilarity graph is connected
        # ------------------------------------------------------------
        n_components, _ = sp.csgraph.connected_components(
            self.dissimilarity_graph_sparse_,
            directed=False
        )
    
        # ------------------------------------------------------------
        # 4. If disconnected and NOT precomputed, optionally increase
        #    n_neighbors until the WSS dissimilarity graph is connected.
        #
        #    In precomputed mode, n_neighbors cannot change the graph,
        #    so this block is skipped.
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
    
            while n_components > 1 and new_n_neighbors < max_neighbors:
                new_n_neighbors += 1
                print("Trying n_neighbors =", new_n_neighbors)
    
                self.n_neighbors = new_n_neighbors
    
                # Rebuild the full correct pipeline:
                # similarity graph -> WSS similarity -> WSS dissimilarity
                self.similarity_graph_ = self.create_similarity_graph(self.data_)
                self.similarity_graph_.add_nodes_from(range(n_obs))
    
                self.similarity_graph_WSS_sparse_ = self.compute_similarity_sparse(
                    self.similarity_graph_
                )
    
                self.dissimilarity_graph_sparse_ = self.similarity_to_dissimilarity_sparse(
                    self.similarity_graph_WSS_sparse_
                )
    
                n_components, _ = sp.csgraph.connected_components(
                    self.dissimilarity_graph_sparse_,
                    directed=False
                )
    
            self.n_neighbors_used_ = self.n_neighbors
    
            if n_components > 1:
                raise RuntimeError(
                    "Could not build a connected WSS dissimilarity graph even after "
                    f"increasing n_neighbors from {original_n_neighbors} "
                    f"to {max_neighbors}."
                )
    
        # ------------------------------------------------------------
        # 5. If connected, use WSS dissimilarity graph directly
        # ------------------------------------------------------------
        if n_components <= 1:
            self.dist_matrix_ = self.dense_from_sparse_edges_fill1(
                self.dissimilarity_graph_sparse_
            )
    
            self.connected_graph_ = nx.from_scipy_sparse_array(
                self.dissimilarity_graph_sparse_,
                edge_attribute="weight"
            )
            self.connected_graph_.add_nodes_from(range(n_obs))
    
        # ------------------------------------------------------------
        # 6. If still disconnected, connect components with bridge
        #    edges of distance 1.
        #
        #    This is used when:
        #      - heuristic_connect=False
        #      - or sim_graph_method="precomputed"
        # ------------------------------------------------------------
        else:
            sparse_nx = nx.from_scipy_sparse_array(
                self.dissimilarity_graph_sparse_,
                edge_attribute="weight"
            )
            sparse_nx.add_nodes_from(range(n_obs))
    
            self.connected_graph_ = self.connect_graph_heuristically(
                sparse_nx,
                n_obs
            )
    
            self.dist_matrix_ = self.compute_custom_distance_matrix(
                self.connected_graph_
            )
    
        # ------------------------------------------------------------
        # 7. MST used later for optional noise reassignment
        # ------------------------------------------------------------
        self.mst_graph_ = nx.minimum_spanning_tree(
            self.connected_graph_,
            weight="weight"
        )
    
        # ------------------------------------------------------------
        # 8. Store NetworkX versions for inspection/debugging
        # ------------------------------------------------------------
        self.similarity_graph_WSS = nx.from_scipy_sparse_array(
            self.similarity_graph_WSS_sparse_,
            edge_attribute="weight"
        )
        self.similarity_graph_WSS.add_nodes_from(range(n_obs))
    
        self.dissimilarity_graph_ = nx.from_scipy_sparse_array(
            self.dissimilarity_graph_sparse_,
            edge_attribute="weight"
        )
        self.dissimilarity_graph_.add_nodes_from(range(n_obs))

    # ------------------------------------------------------------------
    # ------------------------- FIT ------------------------------------
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit the model on feature data or a precomputed graph.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or graph-like
            Input feature matrix when ``sim_graph_method`` is not ``"precomputed"``.
            In ``"precomputed"`` mode, this may be a ``networkx.Graph``, a SciPy
            sparse adjacency matrix, or a square dense adjacency matrix.
    
        Returns
        -------
        self : GraphCoreSGHDBSCAN
            Fitted estimator.
        """
        self._build_graph_distance(X)

        self.coresg_ = CoreSGHDBSCAN(
            min_samples_list=self.m_list,
            metric="precomputed",
            min_cluster_size=self.min_cluster_size,
            save_models=self.save_models,
        )
        self.coresg_.fit_from_distance_matrix(self.dist_matrix_)
        self.coresg_.run()
        self.models_ = self.coresg_.models_
        self.condensed_trees_ = self.coresg_.condensed_trees_
        self.labels_by_m_ = self.coresg_.labels_by_m_
        return self




    def fit_predict(self, X, y=None, m=None, c=5, **fit_params):
        """
        Fit the model and return cluster labels.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or graph-like
            Input feature matrix or supported precomputed graph representation.
    
        Returns
        -------
        numpy.ndarray
            Cluster labels for the fitted solution.
        """
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
            return self.reassign_noise_via_mst(
                self.mst_graph_,
                labels,
                c=c,
            )
        return labels

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
        ).fit_from_distance_matrix(self.dist_matrix_)
        self.coresg_.run()
        self.models_ = self.coresg_.models_
        self.condensed_trees_ = self.coresg_.condensed_trees_
        self.labels_by_m_ = self.coresg_.labels_by_m_
        return self


    # ------------------------------------------------------------------
    # -------------------------- ACCESSORS -----------------------------
    # ------------------------------------------------------------------

    def labels_for(self, m, no_noise=None, c=5):
        """
        Return labels for a selected ``min_samples`` value.
    
        Parameters
        ----------
        m : int
            Selected ``min_samples`` value.
        no_noise : bool or None, optional
            If ``True``, apply MST-based noise reassignment. If ``None``, use
            the instance-level ``no_noise`` setting.
        c : int, optional
            Tie-breaking path length used during noise reassignment.
    
        Returns
        -------
        numpy.ndarray
            Cluster labels for the requested fitted solution.

        Notes
        -----
        ``labels_by_m_[m]`` stores the directly fitted labels.
        ``labels_for(m)`` may additionally apply noise reassignment.
        """
        labels = self.coresg_.labels_by_m_[int(m)]
    
        if no_noise is None:
            no_noise = self.no_noise
    
        if no_noise:
            labels = self.reassign_noise_via_mst(
                self.mst_graph_,
                labels,
                c=c,
            )
    
        return labels

        
    def plot_condensed_tree(self, m, figsize=(10, 6), **kwargs):
        """
        Plot the condensed tree for a selected ``min_samples`` value.
    
        Parameters
        ----------
        m : int
            The ``min_samples`` value whose condensed tree should be displayed.
        figsize : tuple of float, optional
            Figure size passed to Matplotlib, by default ``(8, 5)``.
        **kwargs
            Additional keyword arguments forwarded to
            ``CondensedTree.plot()``.
    
        Returns
        -------
        None
            Displays the condensed tree plot.
    
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        KeyError
            If the requested ``m`` is not available in the stored results.
    
        Notes
        -----
        This method first looks for the condensed tree in
        ``self.coresg_.condensed_trees_``. If it is not found there, it falls
        back to ``self.coresg_.models_[m].condensed_tree_`` when full models
        have been saved.
    
        Examples
        --------
        >>> g.fit(X)
        >>> g.plot_condensed_tree(10)
        """
        import matplotlib.pyplot as plt
    
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
        ct.plot(select_clusters=False, label_clusters=False, **kwargs)
        plt.title(f"CORE-SG Condensed Tree (min_samples = {m})")
        plt.show()


    def interactive_condensed_tree(self, figsize=(10, 6)):
        """
        Create an interactive condensed tree explorer across fitted
        ``min_samples`` values.
    
        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size passed to Matplotlib for each displayed condensed
            tree, by default ``(10, 6)``.
    
        Returns
        -------
        ipywidgets.Widget
            A selection slider widget for browsing condensed trees across
            available ``min_samples`` values.
    
        Raises
        ------
        ImportError
            If ``ipywidgets`` is not installed.
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If no condensed trees are available.
    
        Notes
        -----
        This method is intended for use in an interactive Jupyter
        environment. It uses the stored condensed trees in
        ``self.coresg_.condensed_trees_`` and falls back to any available
        entries in ``self.coresg_.models_``.
    
        Examples
        --------
        >>> g.fit(X)
        >>> widget = g.interactive_condensed_tree()
        """
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
