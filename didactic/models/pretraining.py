from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
from lightly.models.modules import SimCLRProjectionHead

from vital.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, filter_time_series_attributes
from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from didactic.models.layers import CLSToken, PositionalEncoding, SequencePooling
from didactic.models.tabular import TabularEmbedding
from didactic.models.time_series import TimeSeriesEmbedding

logger = logging.getLogger(__name__)
CardiacAttribute = TabularAttribute | Tuple[ViewEnum, TimeSeriesAttribute]



class MultimodalSimCLR(torch.nn.Module):
    
    def __init__(self,
        tabular_tokenizer: Optional[TabularEmbedding | DictConfig],
        time_series_tokenizer: Optional[TimeSeriesEmbedding | DictConfig],
    ) -> None:
        
        super().__init__()

        # Categorise the tabular attributes in terms of their type (numerical vs categorical)
        self.tabular_num_attrs = [
            attr for attr in self.hparams.tabular_attrs if attr in TabularAttribute.numerical_attrs()
        ]
        self.tabular_cat_attrs = [
            attr for attr in self.hparams.tabular_attrs if attr in TabularAttribute.categorical_attrs()
        ]
        self.tabular_cat_attrs_cardinalities = [
            len(TABULAR_CAT_ATTR_LABELS[cat_attr]) for cat_attr in self.tabular_cat_attrs
        ]

        assert isinstance(tabular_tokenizer, DictConfig), 'Tabular tokenizer must be provided for transformer models'
        self.tabular_tokenizer = tabular_tokenizer = hydra.utils.instantiate(
                    tabular_tokenizer,
                    n_num_features=len(self.tabular_num_attrs),
                    cat_cardinalities=self.tabular_cat_attrs_cardinalities,
                )
        self.tabular_encoder = hydra.utils.instantiate(self.hparams.pretrainer.tabular_encoder)

        assert isinstance(time_series_tokenizer, DictConfig), 'Time series tokenizer must be provided for transformer models'
        self.time_series_tokenizer = hydra.utils.instantiate(time_series_tokenizer)
        
        self.projector = SimCLRProjectionHead(self.hparams.embed_dim, self.hparams.embed_dim, self.hparams.projection_dim)

    def forward(self, x_imaging: torch.Tensor, x_tabular: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        # z_imaging = self.imaging_encoder(x_imaging)
        # z_tabular = self.tabular_encoder(x_tabular)
        # z_imaging_projected = self.projector(z_imaging)
        # z_tabular_projected = self.projector(z_tabular)
        # return z_imaging_projected, z_tabular_projected
        pass