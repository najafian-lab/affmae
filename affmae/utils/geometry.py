# just backward compatibility imports for old code

from affmae.ops.clustering import (  # noqa: F401  (re-export)
    calculate_hilbert_order,
    calculate_peano_order,
    space_filling_cluster,
)
from affmae.ops.knn_keops import (  # noqa: F401  (re-export)
    have_keops,
    knn_keops,
)

__all__ = [
    "calculate_hilbert_order",
    "calculate_peano_order",
    "space_filling_cluster",
    "have_keops",
    "knn_keops",
]
