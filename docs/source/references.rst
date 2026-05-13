References
==========

If you use ``coresg-graphhdbscan`` in academic work, reports, or benchmark
studies, please cite the original paper for this package together with the
relevant foundational methods and software.

Main package reference
----------------------

``coresg-graphhdbscan`` is a package implementation of the GraphHDBSCAN*
method described in the following original paper:

Ghoreishi, S. A., Szmigiel, A. S., Nagai, J., Costa, I. G., Zimek, A., and Campello, R. J. G. B. (2026)
*GraphHDBSCAN\*: Graph-based Hierarchical Clustering on High Dimensional Single-cell RNA Sequencing Data*.
*bioRxiv preprint*, 2026.
Available at bioRxiv: ``10.64898/2026.03.24.713924v1``

This is the primary reference for the package and should be cited when the
package itself or the GraphHDBSCAN* method is used.

Foundational HDBSCAN* and hierarchical density estimation
--------------------------------------------------------

Campello, R. J., Moulavi, D., and Sander, J. (2013).
*Density-based clustering based on hierarchical density estimates*.
In *Pacific-Asia Conference on Knowledge Discovery and Data Mining*,
160--172. Springer Berlin Heidelberg.

Campello, R. J., Moulavi, D., Zimek, A., and Sander, J. (2015).
*Hierarchical density estimates for data clustering, visualization, and outlier detection*.
*ACM Transactions on Knowledge Discovery from Data (TKDD)*, 10(1), 5.

Neto, A. C. A., Naldi, M. C., Campello, R. J. G. B., and Sander, J. (2022).
*Core-SG: efficient computation of multiple MSTs for density-based methods*.
In *2022 IEEE 38th International Conference on Data Engineering (ICDE)*,
951--964. IEEE.

Software and ecosystem references
---------------------------------

McInnes, Leland, Healy, John, and Astels, Steve (2017).
*hdbscan: Hierarchical density based clustering*.
*Journal of Open Source Software*, 2(11), 205.

Wolf, F. A., Angerer, P., and Theis, F. J. (2018).
*SCANPY: Large-scale single-cell gene expression data analysis*.
*Genome Biology*, 19, 15.
DOI: 10.1186/s13059-017-1382-0

Li, Y., Nguyen, J., Anastasiu, D. C., and Arriaga, E. A.
*CosTaL: An accurate and scalable graph-based clustering algorithm for high-dimensional single-cell data analysis*.
*Briefings in Bioinformatics*, 24, bbad157.
DOI: 10.1093/bib/bbad157

Citation guidance
-----------------

When citing this package, the main GraphHDBSCAN* paper should be used as the
primary reference.

Depending on the workflow, it may also be appropriate to cite:

- the foundational HDBSCAN* and hierarchical density estimation papers
- the ``hdbscan`` software paper
- ``SCANPY`` when using Scanpy-based graph construction
- ``CosTaL`` when using PhenoGraph-based graph construction

BibTeX
------

.. code-block:: bibtex

   @article{ghoreishi2026graphhdbscan,
     title={GraphHDBSCAN*: Graph-based Hierarchical Clustering on High Dimensional Single-cell RNA Sequencing Data},
     author={Ghoreishi, Seyed Ardalan and Szmigiel, Aleksandra Weronika and Nagai, James Shiniti and Gesteira Costa Filho, Ivan and Zimek, Arthur and Campello, Ricardo Jose Gabrielli Barreto},
     journal={bioRxiv},
     pages={2026--03},
     year={2026},
     publisher={Cold Spring Harbor Laboratory}
   }

   @inproceedings{campello2013density,
     title={Density-based clustering based on hierarchical density estimates},
     author={Campello, Ricardo J. G. B. and Moulavi, Davoud and Sander, J{"o}rg},
     booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
     pages={160--172},
     year={2013},
     publisher={Springer Berlin Heidelberg}
   }

   @article{campello2015hierarchical,
     title={Hierarchical density estimates for data clustering, visualization, and outlier detection},
     author={Campello, Ricardo J. G. B. and Moulavi, Davoud and Zimek, Arthur and Sander, J{"o}rg},
     journal={ACM Transactions on Knowledge Discovery from Data},
     volume={10},
     number={1},
     pages={5},
     year={2015}
   }

  @inproceedings{neto2022core,
     title={Core-SG: efficient computation of multiple MSTS for density-based methods},
     author={Neto, Antonio Cavalcante Araujo and Naldi, Murilo Coelho and Campello, Ricardo JGB and Sander, J{\"o}rg},
     booktitle={2022 IEEE 38th International Conference on Data Engineering (ICDE)},
     pages={951--964},
     year={2022},
     organization={IEEE}
   }
   @article{mcinnes2017hdbscan,
     title={hdbscan: Hierarchical density based clustering},
     author={McInnes, Leland and Healy, John and Astels, Steve},
     journal={Journal of Open Source Software},
     volume={2},
     number={11},
     pages={205},
     year={2017}
   }

   @article{wolf2018scanpy,
     title={SCANPY: Large-scale single-cell gene expression data analysis},
     author={Wolf, F. Alexander and Angerer, Philipp and Theis, Fabian J.},
     journal={Genome Biology},
     volume={19},
     pages={15},
     year={2018},
     doi={10.1186/s13059-017-1382-0}
   }

   @article{li2023costal,
     title={CosTaL: An accurate and scalable graph-based clustering algorithm for high-dimensional single-cell data analysis},
     author={Li, Y. and Nguyen, J. and Anastasiu, D. C. and Arriaga, E. A.},
     journal={Briefings in Bioinformatics},
     volume={24},
     pages={bbad157},
     year={2023},
     doi={10.1093/bib/bbad157}
   }
