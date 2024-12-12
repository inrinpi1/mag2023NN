from torch import LongTensor
from torch import nn

import sys
sys.path.append("..")
from common.token import Token

from null_predictor import NullPredictor
from tagger import MultiHeadTagger
from encoder import PretrainedTransformerMismatchedEncoder


class MorphoSyntaxSemanticsParser(nn.Module):
    """Morpho-Syntax-Semantic Parser."""

    def __init__(
        self,
        encoder_args: dict,
        null_predictor_args: dict,
        tagger_args: dict
    ):
        super().__init__()

        self.encoder = PretrainedTransformerMismatchedEncoder(**encoder_args)
        self.null_predictor = NullPredictor(encoder=encoder, **null_predictor_args)
        self.tagger = MultiHeadTagger(encoder=encoder, **tagger_args)

    def forward(
        self,
        tokens: list[list[Token]],
        lemma_rules: LongTensor = None,
        joint_pos_feats: LongTensor = None,
        deps_ud: LongTensor = None,
        deps_eud: LongTensor = None,
        miscs: LongTensor = None,
        deepslots: LongTensor = None,
        semclasses: LongTensor = None,
        metadata: list[dict] = None
    ) -> dict[str, any]:

        # If all labels are empty, we are at inference.
        is_inference = all(
            x is None for x in [lemma_rules, joint_pos_feats, deps_ud, deps_eud, miscs, deepslots, semclasses]
        )
        # Restore nulls.
        null_out = self.null_classifier(tokens, is_inference)
        # Tokens with predicted nulls.
        tokens_with_nulls = null_out['tokens']

        # Teacher forcing: during training, pass the original tokens (with gold nulls)
        # to the tagger, so that the latter is trained upon correct sentences.
        if self.training:
            tokens_with_nulls = tokens

        # Predict morphological, syntactic and semantic tags.
        tagger_out = self.tagger(
            tokens_with_nulls,
            lemma_rules,
            joint_pos_feats,
            deps_ud,
            deps_eud,
            miscs,
            deepslots,
            semclasses,
        )

        # Add up null predictor and tagger losses.
        loss = null_out['loss'] + tagger_out['loss']

        return {
            'tokens': null_out['tokens'],
            'lemma_rule_ids': tagger_out['lemma_rule_ids'],
            'joint_pos_feat_ids': tagger_out['joint_pos_feat_ids'],
            'deps_ud_ids': tagger_out['deps_ud_ids'],
            'deps_eud_ids': tagger_out['deps_eud_ids'],
            'misc_ids': tagger_out['misc_ids'],
            'deepslot_ids': tagger_out['deepslot_ids'],
            'semclass_ids': tagger_out['semclass_ids'],
            'loss': loss
        }

