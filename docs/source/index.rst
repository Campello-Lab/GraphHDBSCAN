GraphHDBSCAN* documentation
===========================

GraphHDBSCAN* is a graph-based, density-based hierarchical clustering method that can effectively work on very high-dimensional data. It builds on the well-known HDBSCAN* algorithm by operating on a sparse graph representation of the data.

A key advantage is its ability to recover interpretable density-based hierarchies that support visualization without requiring dimensionality reduction techniques like UMAP, t-SNE, or PCA.

GraphHDBSCAN* can also produce high-quality flat partitions with clusters of arbitrary shapes and varying densities, while detecting noise and outliers. It includes an optional label-propagation approach to assign cluster labels to noise points.

It leverages the theoretical CORE-SG graph sparsification machinery, enabling efficient simultaneous computation of multiple hierarchies for exploration, effectively eliminating the need for manual hyperparameter tuning of the clustering algorithm itself.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   parameters
   usage
   api
   examples
   references
   third_party_notices
