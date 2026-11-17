# GraphHDBSCAN*

GraphHDBSCAN* is a Python package for hierarchical density-based clustering using similarity graphs. The package provides an implementation of CoreSG + GraphHDBSCAN*, allowing users to construct similarity graphs from input data or use precomputed similarity graphs.

The package supports multiple similarity graph construction methods, different distance metrics, and multiple `min_samples` values. It also provides methods for obtaining clustering labels and visualizing condensed trees.

## Documentation

Full documentation is available here:

[GraphHDBSCAN* Documentation](https://graphhdbscan.readthedocs.io/en/latest/)

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/Campello-Lab/GraphHDBSCAN.git
```

### Install locally

```bash
pip install -e .
```

## Minimal usage

```python
from coresg_graphhdbscan import GraphCoreSGHDBSCAN

model = GraphCoreSGHDBSCAN(
    min_samples=[5, 10, 15],
    sim_graph_method="sc_gauss",
    metric="euclidean",
    n_neighbors=10,
    no_noise=True,
    heuristic_connect=False,
)

model.fit(X)

labels_5 = model.labels_for(5)
labels_10 = model.labels_for(10)
labels_15 = model.labels_for(15)

model.plot_condensed_tree(10)
```

When `min_cluster_size` is omitted, it defaults to the corresponding `min_samples` value.

## Graph options

The package provides the following built-in similarity graph construction methods:

- `sc_gauss`
- `sc_umap`
- `jaccard_phenograph`

Users can also pass their own similarity graph by setting `sim_graph_method="precomputed"`.

## Distance metrics

The `metric` parameter controls the distance measure used during similarity graph construction.

GraphHDBSCAN* supports multiple distance metrics, including Euclidean, cosine, correlation, Manhattan, Minkowski, and Mahalanobis.

Additional metric-specific parameters can be provided through `metric_kwds`.

For the complete list of supported metrics and their requirements, see the [documentation](https://graphhdbscan.readthedocs.io/en/latest/).

## Precomputed graph input

GraphHDBSCAN* also supports precomputed similarity graphs through `sim_graph_method="precomputed"`.

The input to `fit(...)` may be:

- a NetworkX graph
- a SciPy sparse adjacency matrix
- a square dense adjacency matrix

When a precomputed similarity graph is provided, the package uses it to construct the internal CoreSG / WSS representation used by GraphHDBSCAN*.

## Related paper

The associated paper is available on bioRxiv:

[GraphHDBSCAN* paper](https://www.biorxiv.org/content/10.64898/2026.03.24.713924)

To reproduce the results, first generate the input data by following the preprocessing instructions provided in the [Clustering-benchmarking-for-scRNAseq](https://github.com/Campello-Lab/Clustering-benchmarking-for-scRNAseq) repository.

## Third-party software

This package uses third-party open-source dependencies, including `hdbscan`, `scanpy`, and `PhenoGraph`.

See `THIRD_PARTY_NOTICES.md` for details.

## 📬 Contact

For questions or support, please open an [issue](https://github.com/Campello-Lab/GraphHDBSCAN/issues) or contact:

`ghoreishi@imada.sdu.dk`
