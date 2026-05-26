Parameter selection
===================

This page explains how to choose the main parameters of
``GraphCoreSGHDBSCAN`` in practice.

For most users, the most important decisions are:

- ``min_samples``
- ``sim_graph_method``
- ``metric``
- ``n_neighbors``
- ``heuristic_connect``
- ``no_noise``
- ``min_cluster_size``
- ``save_models``
- ``similarity_backend``

Advanced users may also use ``metric_kwds`` when the selected distance
metric requires additional arguments.

Start here
----------

A good default starting point for many datasets is:

.. code-block:: python

   from coresg_graphhdbscan import GraphCoreSGHDBSCAN

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=15,
       no_noise=True,
       heuristic_connect=False,
   )

A simple way to think about the main settings is:

- use ``min_samples`` to control the smoothness of the density estimates
- use ``sim_graph_method`` to choose how the similarity graph is built
- use ``metric`` to choose the geometry used during graph construction
- use ``n_neighbors`` to control local graph density
- use ``heuristic_connect`` to decide how disconnected graphs are handled
- use ``no_noise`` to decide whether noise points should be reassigned
- use ``save_models`` to decide whether full per-``min_samples`` model objects should be stored
- use ``similarity_backend`` to choose whether accelerated graph-construction backends are used when available

Constructor
-----------

The public constructor is:

.. code-block:: python

   GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="euclidean",
       metric_kwds=None,
       add_neighbor=True,
       no_noise=True,
       n_neighbors=15,
       heuristic_connect=False,
       min_cluster_size=None,
       save_models=False,
       similarity_backend="auto",
       **kwargs
   )

At-a-glance reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Parameter
     - Default
     - Practical meaning
   * - ``min_samples``
     - ``10``
     - Controls the smoothness of the density estimates as a single level or within an entire range.
   * - ``sim_graph_method``
     - ``"sc_umap"``
     - Chooses how the similarity graph is built.
   * - ``metric``
     - ``"euclidean"``
     - Chooses the distance metric used during similarity graph construction.
   * - ``metric_kwds``
     - ``None``
     - Optional keyword arguments passed to the selected distance metric.
   * - ``add_neighbor``
     - ``True``
     - Controls how weighted structural similarity is expanded into graph edges.
   * - ``no_noise``
     - ``True``
     - Reassigns points initially labeled ``-1`` after clustering.
   * - ``n_neighbors``
     - ``15``
     - Controls local graph density.
   * - ``heuristic_connect``
     - ``False``
     - Chooses how originally disconnected graphs are handled.
   * - ``min_cluster_size``
     - ``None``
     - (Optional) Minimum cluster size in the clustering stage.
   * - ``save_models``
     - ``False``
     - Stores full saved models for each fitted ``min_samples`` value.
   * - ``similarity_backend``
     - ``"auto"``
     - Chooses the backend used for similarity graph construction when alternative implementations are available.

How to choose each parameter
----------------------------

``min_samples``
^^^^^^^^^^^^^^^

Default: ``10``

This is the main clustering hyperparameter. It may be:

- a single integer, such as ``10``
- an iterable of integers, such as ``[5, 10, 15]`` or ``range(2, 10)``

Internally, the package converts it into an internal list of values used by
CoreSGHDBSCAN.

Examples:

- ``min_samples=10`` gives ``[10]``
- ``min_samples=7`` gives ``[7]``
- ``min_samples=[5, 10, 15]`` gives ``[5, 10, 15]``
- ``min_samples=range(2, 10)`` gives ``[2, 3, 4, 5, 6, 7, 8, 9]``

Practical interpretation:

- smaller values usually produce finer, more local cluster structure
- larger values usually produce more conservative and more stable clusters
- multiple values are useful when you want to compare density settings in one run

Recommended workflow:

1. Start with ``10``.
2. If clusters seem too coarse, try smaller values.
3. If clusters seem unstable or fragmented, try larger values.
4. When in doubt, fit several values and compare the condensed trees.

Example:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(min_samples=[5, 10, 15])
   model.fit(X)

   labels_5 = model.labels_for(5)
   labels_10 = model.labels_for(10)
   labels_15 = model.labels_for(15)

``sim_graph_method``
^^^^^^^^^^^^^^^^^^^^

Default: ``"sc_umap"``

This parameter chooses the graph-construction backend.

Supported values are:

- ``"sc_gauss"``
- ``"sc_umap"``
- ``"jaccard_phenograph"``
- ``"precomputed"``

Choosing a method:

``sc_umap``
   Good default choice. Uses Scanpy's UMAP-style connectivity routine.

``sc_gauss``
   Useful when you want Scanpy's Gaussian connectivity construction.

``jaccard_phenograph``
   Useful when you want a PhenoGraph-style Jaccard neighborhood graph. The
   backend used for this graph can be controlled with
   ``similarity_backend``. With ``similarity_backend="auto"``, the package uses
   the accelerated ``numba`` backend when available and otherwise falls back to
   the default PhenoGraph-based path.

``precomputed``
   Use this when you already have a graph or adjacency representation and do
   not want the package to build a graph from raw features.

Supported inputs in ``precomputed`` mode:

- a ``networkx.Graph``
- a SciPy sparse adjacency matrix
- a square dense adjacency matrix

When using ``"precomputed"``, the input to ``fit(...)`` is treated as an
already constructed graph representation rather than raw feature data.

Practical recommendation:

- start with ``"sc_umap"``
- try ``"sc_gauss"`` if you prefer Gaussian connectivity
- use ``"jaccard_phenograph"`` for PhenoGraph-style neighborhood structure
- use ``"precomputed"`` when your graph is part of the experimental design

``similarity_backend``
^^^^^^^^^^^^^^^^^^^^^^

Default: ``"auto"``

This parameter controls which backend is used for similarity graph construction
when alternative implementations are available.

Supported values are:

- ``"auto"``
- ``"default"``
- ``"numba"``

Currently, this option mainly affects ``sim_graph_method="jaccard_phenograph"``.

``auto``
   Uses the accelerated ``numba`` implementation when ``numba`` is available.
   If ``numba`` is not available, the package falls back to the default
   implementation.

``default``
   Uses the original default implementation. For
   ``sim_graph_method="jaccard_phenograph"``, this means using the
   Scanpy/PhenoGraph graph-construction path.

``numba``
   Uses the ``numba``-accelerated implementation when available. For
   ``sim_graph_method="jaccard_phenograph"``, this computes the
   PhenoGraph-style Jaccard graph using a compiled implementation. If
   ``numba`` is not installed, an import error is raised.

For ``jaccard_phenograph``, the ``numba`` backend is designed to reproduce the
same PhenoGraph-style undirected Jaccard graph as the default backend, while
reducing the time spent in Jaccard graph construction.

The undirected graph is constructed in the same style as PhenoGraph: directed
Jaccard weights are computed first, both directions are averaged, and the
lower-triangular sparse graph is retained internally before conversion to the
package graph representation.

Practical recommendation:

- keep ``similarity_backend="auto"`` for normal use
- use ``similarity_backend="default"`` when you want the original backend for
  comparison or debugging
- use ``similarity_backend="numba"`` when you specifically want the accelerated
  implementation and want an error if ``numba`` is unavailable

Example:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="jaccard_phenograph",
       similarity_backend="auto",
       n_neighbors=15,
       metric="euclidean",
   )

To force the accelerated backend:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="jaccard_phenograph",
       similarity_backend="numba",
       n_neighbors=15,
   )

To force the original PhenoGraph-based backend:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       sim_graph_method="jaccard_phenograph",
       similarity_backend="default",
       n_neighbors=15,
   )

``metric``
^^^^^^^^^^

Default: ``"euclidean"``

This controls the distance metric used during similarity graph
construction.

Supported distance metrics are:

- ``"cityblock"``
- ``"cosine"``
- ``"euclidean"``
- ``"l1"``
- ``"l2"``
- ``"manhattan"``
- ``"braycurtis"``
- ``"canberra"``
- ``"chebyshev"``
- ``"correlation"``
- ``"dice"``
- ``"hamming"``
- ``"jaccard"``
- ``"mahalanobis"``
- ``"minkowski"``
- ``"rogerstanimoto"``
- ``"russellrao"``
- ``"seuclidean"``
- ``"sokalmichener"``
- ``"sokalsneath"``
- ``"sqeuclidean"``
- ``"yule"``
- ``"hybrid_euclidean_cosine"``

Choosing a metric:

``euclidean``
    Default choice for standard continuous feature spaces.

``cosine``
    Useful when angular similarity is more meaningful than raw magnitude.

``correlation``
    Useful when similarity should depend on the shape or pattern of the
    feature vector rather than absolute scale.

``manhattan`` or ``l1``
    Useful when L1 geometry is preferred.

``jaccard`` and other binary metrics
    Useful for binary or boolean feature representations.

``minkowski``
    Supports custom ``p`` values through ``metric_kwds``.

``mahalanobis``
    Requires an inverse covariance matrix ``VI`` through ``metric_kwds``.

``seuclidean``
    Requires a variance vector ``V`` through ``metric_kwds``.

``hybrid_euclidean_cosine``
    Package-specific mode. Full pairwise distances remain Euclidean, but
    neighborhood graph construction uses cosine geometry.

Practical recommendation:

- use ``"euclidean"`` as a default starting point
- use ``"cosine"`` or ``"correlation"`` when direction or pattern matters more than magnitude
- use ``"minkowski"``, ``"mahalanobis"``, or ``"seuclidean"`` only when their assumptions match your data
- use ``"hybrid_euclidean_cosine"`` when you want Euclidean full distances but cosine-based local neighborhoods

The metric ``"kulsinski"`` is not supported because it is not available
in current versions of ``scikit-learn``'s ``pairwise_distances``.

The combination ``metric="yule"`` with ``sim_graph_method="sc_gauss"``
is intentionally not supported because it can produce non-finite graph
weights. Use ``metric="yule"`` with ``sim_graph_method="sc_umap"`` or
``sim_graph_method="jaccard_phenograph"`` instead.

Examples:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="correlation",
       n_neighbors=15,
   )

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="minkowski",
       metric_kwds={"p": 1.5},
       n_neighbors=15,
   )

.. code-block:: python

   import numpy as np

   VI = np.linalg.pinv(np.cov(X, rowvar=False))

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="mahalanobis",
       metric_kwds={"VI": VI},
       n_neighbors=15,
   )

``metric_kwds``
^^^^^^^^^^^^^^^

Default: ``None``

This optional dictionary is passed to the selected distance metric during
similarity graph construction.

It is mainly needed for metrics that require additional parameters.

Examples:

- use ``metric_kwds={"p": 1.5}`` with ``metric="minkowski"``
- use ``metric_kwds={"VI": VI}`` with ``metric="mahalanobis"``
- use ``metric_kwds={"V": V}`` with ``metric="seuclidean"``

Example:

.. code-block:: python

   import numpy as np

   V = np.var(X, axis=0, ddof=1)

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="seuclidean",
       metric_kwds={"V": V},
       n_neighbors=15,
   )

``n_neighbors``
^^^^^^^^^^^^^^^

Default: ``15``

This is the number of neighbors used during similarity graph construction.

Practical interpretation:

- smaller values make the graph more local and sparse
- larger values make the graph denser and may improve connectivity
- increasing this value is often the first thing to try when the graph is too fragmented

Practical recommendation:

- start with ``15``
- increase it if connectivity is poor
- decrease it if the graph becomes overly broad or too smoothed

Example:

.. code-block:: python

   GraphCoreSGHDBSCAN(
       sim_graph_method="sc_gauss",
       n_neighbors=20,
   )

``add_neighbor``
^^^^^^^^^^^^^^^^

Default: ``True``

This controls how weighted structural similarity is computed.

When enabled, an edge may still be added even when two nodes do not already
share a direct edge, as long as their weighted structural similarity is greater
than zero.

Practical recommendation:

- keep the default unless you are specifically studying graph-construction behavior
- change it only when you want to examine the effect of this edge-expansion step

``heuristic_connect``
^^^^^^^^^^^^^^^^^^^^^

Default: ``False``

The final graph used for clustering must be connected. This parameter
controls how originally disconnected graphs are handled.

``heuristic_connect=False``
   Default behavior. If the graph has multiple connected components, the
   package connects consecutive components by adding edges with maximum
   distance, equivalent to weight ``1`` in the dissimilarity graph.

``heuristic_connect=True``
   The package repeatedly increases ``n_neighbors`` until the graph becomes
   connected.

Example fitting log:

.. code-block:: text

   Trying n_neighbors = 16
   Trying n_neighbors = 17

Practical recommendation:

- use ``False`` when you want a simple and predictable fallback
- use ``True`` when you prefer connectivity to come from a denser neighborhood graph
  rather than from synthetic bridge edges

``no_noise``
^^^^^^^^^^^^

Default: ``True``

If enabled, points initially labeled as ``-1`` are reassigned by an
MST-based label propagation step after clustering.

Conceptually, this post-processing step:

1. builds a mutual-reachability view from graph-derived distances and core distances
2. computes an MST
3. propagates labels from labeled points to unlabeled points in increasing edge-weight order
4. resolves competition using a top-``c`` path comparison rule

Practical recommendation:

- use ``True`` if you prefer a full assignment with no final noise labels
- use ``False`` if you want to preserve the original HDBSCAN*-style noise behavior

``min_cluster_size``
^^^^^^^^^^^^^^^^^^^^

Default: ``None``

This is the minimum cluster size used in the clustering stage.

When left as ``None``, the package uses the selected ``min_samples`` value
for each run, so the effective minimum cluster size becomes ``m`` for each
fitted ``min_samples = m``.

If you set ``min_cluster_size`` explicitly, that fixed value is used for all
selected ``min_samples`` values.

Practical recommendation:

- leave it as ``None`` if you want cluster size to track ``min_samples``
- set it explicitly if you want a fixed minimum cluster size independent of
  the selected ``min_samples`` values


``save_models``
^^^^^^^^^^^^^^^

Default: ``False``

This controls whether full per-``min_samples`` model objects are stored after
fitting.

``save_models=False``
   The package still stores labels and condensed trees for each fitted
   ``min_samples`` value, but it does not keep full saved model objects.

``save_models=True``
   The package also stores full per-``m`` models in ``models_``.

Practical recommendation:

- use ``False`` if you mainly want labels and condensed trees with lower
  memory usage
- use ``True`` if you want direct access to saved per-``m`` model objects

Example:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=range(2, 20),
       sim_graph_method="sc_gauss",
       metric="euclidean",
       save_models=True,
   )
   model.fit(X)

   labels_10 = model.labels_by_m_[10]
   tree_10 = model.condensed_trees_[10]
   model_10 = model.models_[10]

Notes:

- ``labels_by_m_`` and ``condensed_trees_`` are available after fitting
  regardless of ``save_models``.
- ``models_`` is mainly useful when you want to inspect the full saved result
  object for a specific ``min_samples`` value.

Practical selection workflow
----------------------------

A useful tuning order is:

1. choose ``sim_graph_method`` based on how you want the graph to be built
2. choose ``metric`` based on the geometry that makes sense for your data
3. start with ``n_neighbors=15``
4. tune ``min_samples``
5. decide whether you want ``no_noise=True``
6. only then adjust ``heuristic_connect`` and ``add_neighbor`` if needed

A good exploratory run looks like:

.. code-block:: python

   g = GraphCoreSGHDBSCAN(
       min_samples=range(2, 20),
       sim_graph_method="sc_gauss",
       n_neighbors=16,
       no_noise=True,
       metric="euclidean",
       heuristic_connect=True,
       save_models=True,
   )
   g.fit(X)

After fitting several values, the package stores results by
``min_samples`` value. Labels are available in ``labels_by_m_``, condensed
trees are available in ``condensed_trees_``, and full saved models are
available in ``models_`` when ``save_models=True``.

Then inspect the hierarchy and choose a specific solution:

.. code-block:: python

   g.plot_condensed_tree(4)
   labels_18 = g.labels_for(18)
   tree_18 = g.condensed_trees_[18]
   model_18 = g.models_[18]


Ready-to-use presets
--------------------

Default baseline
^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN()
   model.fit(X)
   labels = model.fit_predict(X)

More conservative clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=20,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=20,
   )

Finer local structure
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=5,
       sim_graph_method="sc_umap",
       metric="euclidean",
       n_neighbors=12,
   )

Cosine-based graph construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=[5, 10],
       sim_graph_method="sc_gauss",
       metric="cosine",
       n_neighbors=20,
   )
   model.fit(X)
   labels_10 = model.labels_for(10)

Correlation-based graph construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=[5, 10],
       sim_graph_method="sc_umap",
       metric="correlation",
       n_neighbors=20,
   )

   model.fit(X)
   labels_10 = model.labels_for(10)

Minkowski distance with custom p
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="sc_umap",
       metric="minkowski",
       metric_kwds={"p": 1.5},
       n_neighbors=15,
   )

   model.fit(X)

Hybrid Euclidean-cosine mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=range(2, 10),
       sim_graph_method="sc_umap",
       metric="hybrid_euclidean_cosine",
       n_neighbors=16,
   )
   model.fit(X)

Precomputed graph input
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="precomputed",
       no_noise=True,
   )
   model.fit(my_graph)

PhenoGraph-style Jaccard graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="jaccard_phenograph",
       similarity_backend="auto",
       metric="euclidean",
       n_neighbors=15,
   )
   model.fit(X)

For reproducibility checks against the original backend, use:

.. code-block:: python

   model = GraphCoreSGHDBSCAN(
       min_samples=10,
       sim_graph_method="jaccard_phenograph",
       similarity_backend="default",
       n_neighbors=15,
   )

Troubleshooting by symptom
--------------------------

Too many tiny clusters
^^^^^^^^^^^^^^^^^^^^^^

Try:

- increasing ``min_samples``
- increasing ``n_neighbors``
- using ``metric="euclidean"`` if cosine-based neighborhoods are too fine

Clusters are too coarse
^^^^^^^^^^^^^^^^^^^^^^^

Try:

- decreasing ``min_samples``
- decreasing ``n_neighbors``
- checking whether ``no_noise=True`` is absorbing points you would rather keep as noise

Graph is disconnected
^^^^^^^^^^^^^^^^^^^^^

Try:

- increasing ``n_neighbors``
- setting ``heuristic_connect=True``
- checking whether the selected metric is making neighborhoods too sparse

Too many noise points
^^^^^^^^^^^^^^^^^^^^^

Try:

- lowering ``min_samples``
- increasing ``n_neighbors``
- using ``no_noise=True`` if a full assignment is desired

Jaccard graph construction is slow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Try:

- using ``similarity_backend="auto"`` or ``similarity_backend="numba"``
- reducing ``n_neighbors`` if the neighborhood graph is unnecessarily dense
- using ``similarity_backend="default"`` only when you need the original
  PhenoGraph-based backend for comparison or debugging

Practical notes
---------------

- If the graph is disconnected and ``heuristic_connect=False``, the package
  connects components with synthetic edges of weight ``1``. This is simple and
  effective, but it is a design choice worth reporting in experiments.
- ``min_cluster_size=None`` means that the package matches cluster size to each
  selected ``min_samples`` value.
- When several ``min_samples`` values are passed, fit once and retrieve labels
  later for the requested value.
- Some graph builders depend on optional packages and will raise a clear import
  error if those packages are not installed.
- ``labels_by_m_[m]`` stores the directly fitted labels for a selected
  ``min_samples`` value.
- ``labels_for(m)`` may additionally apply noise reassignment depending on
  the ``no_noise`` setting.
- ``condensed_trees_[m]`` gives direct access to the condensed tree for a
  selected ``min_samples`` value.
- ``models_[m]`` is available when ``save_models=True``.
- ``similarity_backend="auto"`` uses accelerated similarity-graph construction
  when available. Currently this mainly affects
  ``sim_graph_method="jaccard_phenograph"``.
- The ``numba`` backend may have a one-time compilation cost on first use, but
  can substantially reduce the time spent constructing PhenoGraph-style
  Jaccard graphs for larger datasets.
