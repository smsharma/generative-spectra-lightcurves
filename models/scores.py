import dataclasses

import jax
import jax.numpy as np
import flax.linen as nn

from models.transformer import Transformer
from models.mlp import MLP

from models.diffusion_utils import get_timestep_embedding

from functools import partial


class TransformerScoreNet(nn.Module):
    """Transformer score network."""

    d_t_embedding: int = 32
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
        }
    )
    adanorm: bool = False

    @nn.compact
    def __call__(self, z, t, conditioning, mask):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        t = t * np.ones(z.shape[0])  # Ensure t is a vector

        t_embedding = get_timestep_embedding(t, self.d_t_embedding)
        t_embedding = nn.Dense(self.score_dict["d_model"])(t_embedding)

        if conditioning is not None:
            # Project conditioning to embedding space of transformer
            conditioning = nn.Dense(self.score_dict["d_model"])(conditioning)
            cond = t_embedding[:, None, :] + conditioning
        else:
            cond = t_embedding[:, None, :]

        # Make copy of score dict since original cannot be in-place modified; remove `score` argument before passing to Net
        score_dict = dict(self.score_dict)
        score_dict.pop("score", None)

        h = Transformer(n_input=z.shape[-1], **score_dict)(z, cond, mask)

        return z + h
