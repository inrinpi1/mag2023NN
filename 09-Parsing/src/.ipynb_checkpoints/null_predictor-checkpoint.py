import numpy as np

import torch
from torch import nn
from torch import Tensor, BoolTensor, LongTensor

import sys
sys.path.append("..")
from common.token import Token, CLS_TOKEN

from mlp_classifier import MLPClassifier
from encoder import PretrainedTransformerMismatchedEncoder


class NullPredictor(nn.Module):
    """A pipeline to restore ellipted tokens."""
    def __init__(
        self,
        encoder: PretrainedTransformerMismatchedEncoder,
        hidden_dim: int,
        activation: str,
        dropout: float,
        consecutive_null_limit: int,
        class_weights: list[float] = None
    ):
        super().__init__()

        self.encoder = encoder
        self.null_classifier = MLPClassifier(
            input_dim=self.encoder.get_output_dim(),
            hidden_dim=hidden_dim,
            n_classes=consecutive_null_limit + 1,
            activation=activation,
            dropout=dropout,
            class_weights=class_weights
        )

    def forward(self, tokens: list[list[Token]], is_inference: bool) -> dict[str, Tensor]:
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        tokens_with_cls = self._prepend_cls_token(tokens)
        # Delete nulls.
        tokens_with_cls_and_no_nulls = self._del_nulls(tokens_with_cls)

        embeddings_with_cls_and_no_nulls, mask_no_nulls = self.encoder(tokens_with_cls_and_no_nulls)

        if is_inference:
            gold_counting_mask = self._build_counting_mask(tokens_with_cls)
        else:
            gold_counting_mask = None

        # Predict counting mask.
        classifier_output = self.null_classifier(
            embeddings_with_cls_and_no_nulls,
            gold_counting_mask,
            mask_no_nulls
        )

        # Add predicted nulls to the original sentences.
        pred_counting_mask = classifier_output["preds"]
        tokens_with_nulls = self._add_nulls(tokens, pred_counting_mask)
        return {
            "tokens": tokens_with_nulls,
            "loss": classifier_output["loss"]
        }

    @staticmethod
    def _build_counting_mask(sentences: list[list[Token]]) -> LongTensor:
        """
        Count the number of nulls following each non-null token for a bunch of sentences.
        output[i, j] = N means j-th non-null token in i-th sentence in followed by N nulls.

        Example:
        >>> sentences = [
        ...     ['Iraq', 'are', 'reported', 'dead', 'and', '500', '#NULL', '#NULL', '#NULL', 'wounded']
        ... ]
        >>> _build_counting_mask(sentences)
        [0, 0, 0, 0, 0, 0, 3, 0]
        """
        counting_masks: list[LongTensor] = []

        for sentence in sentences:
            nonnull_tokens_indices = [i for i, token in enumerate(sentence) if not token.is_null()]
            nonnull_tokens_indices.append(len(sentence))
            nonnull_tokens_indices = torch.LongTensor(nonnull_tokens_indices)
            counting_mask = torch.diff(nonnull_tokens_indices) - 1
            counting_masks.append(counting_mask)

        counting_masks_batched = torch.nn.utils.rnn.pad_sequence(
            counting_masks,
            batch_first=True,
            # Use -1 to make sure it is never attended to
            # (if it is, the CUDA will terminate with an error that classes must be positive).
            padding_value=-1
        )
        return counting_masks_batched.long()

    @staticmethod
    def _prepend_cls_token(sentences: list[list[Token]]) -> list[list[Token]]:
        """
        Return a copy of sentences with [CLS] tokens prepended.
        """
        return [[CLS_TOKEN, *sentence] for sentence in sentences]

    @staticmethod
    def _del_nulls(sentences: list[list[Token]]) -> list[list[Token]]:
        """
        Return a copy of sentences with nulls removed.
        """
        return [[token for token in sentence if not token.is_null()] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: list[list[Token]], counting_masks: LongTensor) -> list[list[Token]]:
        """
        Return a copy of sentences with nulls restored according to .
        """
        sentences_with_nulls = []
        for sentence, counting_mask in zip(sentences, counting_masks):
            sentence_with_nulls = []
            for token, n_nulls_to_insert in zip(sentence, counting_mask):
                sentence_with_nulls.append(token)
                for i in range(1, n_nulls_to_insert + 1):
                    sentence_with_nulls.append(Token.create_null(id=f"{token.id}.{i}"))
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls

