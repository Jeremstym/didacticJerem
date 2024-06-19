from autogluon.common.utils.log_utils import _add_stream_handler

from . import constants, metrics
from .dataset import TabularDataset
from .space import Bool, Categorical, Int, Real, Space

_add_stream_handler()

from .version import __version__
