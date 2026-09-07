"""Built-in model specs.

Importing this package registers every model the release branch ships. The
registry imports it lazily on first lookup, so nothing here runs until a driver
actually resolves a ``model_type``.

Baseline comparison models (mixmae, hiera, hivit, greenmim, swin) are not part
of the release. They live on the ``rebuttals`` branch and register themselves by
adding modules under ``affmae/models/contrib/``, which the registry imports when
present. No edit to this package is needed to add one.
"""

from affmae.models.specs import affmae  # noqa: F401
from affmae.models.specs import vit  # noqa: F401
