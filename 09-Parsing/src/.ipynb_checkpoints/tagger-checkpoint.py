from torch import LongTensor
from torch import nn

import sys
sys.path.append("..")
from common.token import Token, CLS_TOKEN

from mlp_classifier import MLPClassifier
from dependency_classifier import DependencyClassifier
from encoder import PretrainedTransformerMismatchedEncoder


class MultiHeadTagger(nn.Module):
    """Morpho-Syntax-Semantic tagger."""

    def __init__(
        self,
        encoder: PretrainedTransformerMismatchedEncoder,
        lemma_rule_classifier_args: dict,
        pos_feats_classifier_args: dict,
        depencency_classifier_args: dict,
        misc_classifier_args: dict,
        deepslot_classifier_args: dict,
        semclass_classifier_args: dict
    ):
        super().__init__()

        self.encoder = encoder
        embedding_dim = self.encoder.get_output_dim()

        # Heads.
        self.lemma_rule_classifier = MLPClassifier(
            input_dim=embedding_dim,
            **lemma_rule_classifier_args
        )
        self.joint_pos_feats_classifier = MLPClassifier(
            input_dim=embedding_dim,
            **pos_feats_classifier_args
        )
        self.dependency_classifier = DependencyClassifier(
            input_dim=embedding_dim,
            **depencency_classifier_args
        )
        self.misc_classifier = MLPClassifier(
            input_dim=embedding_dim,
            **misc_classifier_args
        )
        self.deepslot_classifier = MLPClassifier(
            input_dim=embedding_dim,
            **deepslot_classifier_args
        )
        self.semclass_classifier = MLPClassifier(
            input_dim=embedding_dim,
            **semclass_classifier_args
        )

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

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.encoder(tokens)

        # Padding mask.
        padding_mask = None
        assert padding_mask is not None, "TODO"
        # Some classifiers must not attend to nulls (e.g. basic UD dependency classifier),
        # so we additionaly build mask with nulls excluded.
        null_mask = build_null_mask(tokens)

        lemma_out = self.lemma_rule_classifier(
            embeddings,
            lemma_rules,
            # Mask nulls, since they have trivial lemmas.
            mask=(padding_mask & ~null_mask)
        )
        joint_pos_feats_out = self.joint_pos_feats_classifier(
            embeddings,
            joint_pos_feats,
            padding_mask
        )
        deps_out = self.dependency_classifier(
            embeddings,
            deps_ud,
            deps_eud,
            # Mask nulls for basic UD and don't mask for E-UD.
            mask_ud=(padding_mask & ~null_mask),
            mask_eud=padding_mask
        )
        misc_out = self.misc_classifier(embeddings, miscs, padding_mask)
        deepslot_out = self.deepslot_classifier(embeddings, deepslots, padding_mask)
        semclass_out = self.semclass_classifier(embeddings, semclasses, padding_mask)

        loss = lemma_out['loss'] \
            + joint_pos_feats_out['loss'] \
            + deps_out['loss_ud'] \
            + deps_out['loss_eud'] \
            + misc_out['loss'] \
            + deepslot_out['loss'] \
            + semclass_out['loss']

        return {
            'lemma_rule_ids': lemma_out['preds'],
            'joint_pos_feat_ids': joint_pos_feats_out['preds'],
            'deps_ud_ids': deps_out['preds_ud'],
            'deps_eud_ids': deps_out['preds_eud'],
            'misc_ids': misc_out['preds'],
            'deepslot_ids': deepslot_out['preds'],
            'semclass_ids': semclass_out['preds'],
            'loss': loss
        }

