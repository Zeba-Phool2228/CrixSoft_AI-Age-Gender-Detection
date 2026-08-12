import torch
import torch.nn as nn

from transformers import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class AgeGenderViTModel(ViTPreTrainedModel):
    """
    Local Age + Gender ViT model.

    Architecture:

        ViT backbone
             |
        CLS token representation (raw, no pooler)
             |
        +----------------+
        |                |
      age_head       gender_head
        |                |
       age       female probability

    Output:
        logits[:, 0] = age
        logits[:, 1] = female probability

    IMPORTANT — VERIFIED BY TESTING:
    An earlier version of this file switched to using outputs.pooler_output
    (on the theory that the checkpoint's vit.pooler.dense.weight/bias meant
    the pooler was used during training). That was tested empirically and
    produced clearly broken results (age collapsed to 0-2 across every test
    image, weak gender confidence). The RAW CLS TOKEN approach below is the
    one confirmed to give sensible results (verified: ~11, ~42, ~66 on known
    test photos of a child, adult woman, and elderly man respectively).
    Do NOT switch to pooler_output again without new evidence.
    """

    def __init__(self, config):
        super().__init__(config)

        # add_pooling_layer=False: we don't use the pooler in forward(),
        # so it doesn't need to be instantiated. The checkpoint's pooler
        # weights (if present) are simply ignored on load — this is safe
        # and expected (shows as an "unexpected keys" notice, not an error).
        self.vit = ViTModel(
            config,
            add_pooling_layer=False
        )

        # ------------------------------------------------------
        # Age head: 768 -> 256 -> 64 -> 1
        # ------------------------------------------------------
        self.age_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

        # ------------------------------------------------------
        # Gender head: 768 -> 256 -> 64 -> 1 -> Sigmoid
        # Output: probability of Female
        # ------------------------------------------------------
        self.gender_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.post_init()

    def forward(
        self,
        pixel_values=None,
        labels=None,
        **kwargs
    ):
        outputs = self.vit(
            pixel_values=pixel_values,
            **kwargs
        )

        # RAW CLS TOKEN — verified correct approach.
        # outputs.last_hidden_state: [batch_size, sequence_length, hidden_size]
        # First token (index 0) is CLS. Shape: [batch_size, 768]
        pooled_output = outputs.last_hidden_state[:, 0]

        age_output = self.age_head(pooled_output)
        gender_output = self.gender_head(pooled_output)

        # [age, female_probability] -> shape [batch_size, 2]
        logits = torch.cat(
            [age_output, gender_output],
            dim=1
        )

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )