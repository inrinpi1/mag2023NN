from typing import Iterable
from datasets import Dataset, DatasetDict, Features, Sequence, Value

import sys
sys.path.append("..")
from common.token import Token
from common.sentence import Sentence
from common.parse_conllu import parse_conllu_incr


def convert_sentence_to_dict(sentence: Sentence) -> dict:
    return {
        "words": sentence.words,
        "lemmas": sentence.lemmas,
        "upos": sentence.upos,
        "xpos": sentence.xpos,
        "feats": sentence.feats,
        "heads": sentence.heads,
        "deprels": sentence.deprels,
        "deps": sentence.deps,
        "miscs": sentence.miscs,
        "deepslots": sentence.semslots,
        "semclasses": sentence.semclasses
    }


def convert_conllu_to_hf(file_path: str) -> Iterable[dict]:
    with open(file_path, "r") as file:
        for sentence in parse_conllu_incr(file):
            yield convert_sentence_to_dict(sentence)


def build_raw_dataset(file_paths: dict[str, str]) -> DatasetDict:
    features = Features({
        "words": Sequence(Value("string")),
        "lemmas": Sequence(Value("string")),
        "upos": Sequence(Value("string")),
        "xpos": Sequence(Value("string")),
        "feats": Sequence(Value("string")),
        "heads": Sequence(Value("int32")),
        "deprels": Sequence(Value("string")),
        "deps": Sequence(Value("string")),
        "miscs": Sequence(Value("string")),
        "deepslots": Sequence(Value("string")),
        "semclasses": Sequence(Value("string"))
    })

    splits = {}
    for split_name, file_path in file_paths.items():
        splits[split_name] = Dataset.from_generator(convert_conllu_to_hf, gen_kwargs={"file_path": file_path}, features=features)

    return DatasetDict(splits)


def create_dataset():
    file_paths = {
        "train": "../data/train.conllu",
        "validation": "../data/validation.conllu",
        "test": "../data/test.conllu",
    }
    dataset = build_raw_dataset(file_paths)
    return dataset


#def extract_unique_labels(dataset_column) -> list[str]:
#    """Extract unique labels from a specific column in the dataset, e.g. dataset_dict['train']['xpos']."""
#    all_labels = itertools.chain.from_iterable(dataset_column)
#    return sorted(set(all_labels))  # Ensure consistent ordering of labels


# def build_lemma_rules(words: list[str], lemmas: list[str]) -> list[str]:
#     return [
#         str(predict_lemma_rule(word if word is not None else '', lemma if lemma is not None else ''))
#         for word, lemma in zip(words, lemmas)
#     ]

# def build_joint_pos_feats(upos: list[str], xpos: list[str], feats: list[str]) -> list[str]
#     return [
#         f"{upos_tag}#{xpos_tag}#" + ('|'.join([f"{k}={v}" for k, v in feats_tag.items()]) if 0 < len(feats_tag) else '_')
#         for upos_tag, xpos_tag, feats_tag in zip(upos, xpos, feats)
#     ]


# def map_function()
#     matrix = torch.full((seq_len, seq_len), self._padding_value, dtype=torch.long)


# def update_schema_with_class_labels(dataset_dict: DatasetDict):
#     """Update the schema to use ClassLabel for specified columns."""

#     # Extract labels from train dataset only, since all the labels must be present in training data.
#     train_dataset = dataset_dict['train']

#     # Extract unique labels for each column that needs to be ClassLabel.
#     lemma_rules = sorted(set(build_lemma_rules(train_dataset['words'], train_dataset['lemmas'])))

#     # Joint (upos + xpos + feats) tags.
#     pos_feats = sorted(set(build_joint_pos_feats(train_dataset['upos'], train_dataset['xpos'], train_dataset['feats'])))

#     deprels = sorted(set(train_dataset['deprels']))
#     deps_labels = for head, relation in train_dataset['deps']

#     deepslots = sorted(set(train_dataset['deepslots']))
#     semclasses = sorted(set(train_dataset['semclasses']))

#     # Define updated features schema
#     features = Features({
#         "words": Sequence(Value("string")),
#         "lemma_rules": Sequence(ClassLabel(names=lemma_rules)),
#         "pos_feats": Sequence(ClassLabel(names=pos_feats)),
#         "deprels": Sequence(ClassLabel(names=deprels_labels)),
#         "deps": Sequence(Value("string")),
#         "miscs": Sequence(Value("string")),
#         "deepslots": Sequence(ClassLabel(names=deepslots)),
#         "semclasses": Sequence(ClassLabel(names=semclasses))
#     })
#     return features

# def encode_dataset(dataset_dict: DatasetDict, updated_features: Features) -> DatasetDict:
#     """Re-encode the dataset with the updated features schema."""
#     return dataset_dict.cast(features=updated_features)


    #lemma_rules = None
    #if sentence.lemmas is not None:
    #    lemma_rules = [
    #        str(predict_lemma_rule(word if word is not None else '', lemma if lemma is not None else ''))
    #        for word, lemma in zip(words, lemmas)
    #    ]

    #joint_pos_feats = None
    #if sentence.upos_tags is not None and sentence.xpos_tags is not None and sentence.feats_tags is not None:
    #    joint_pos_feats = [
    #        f"{upos_tag}#{xpos_tag}#" + ('|'.join([f"{k}={v}" for k, v in feats_tag.items()]) if 0 < len(feats_tag) else '_')
    #        for upos_tag, xpos_tag, feats_tag in zip(upos_tags, xpos_tags, feats_tags)
    #    ]

    #deprels = None
    #if heads is not None and deprels is not None:
    #    ud_edges: List[Typle[int, int]] = []
    #    ud_edges_labels: List[str] = []
    #    for index, (head, relation) in enumerate(zip(heads, deprels)):
    #        # Skip nulls.
    #        if head == -1:
    #            continue
    #        assert 0 <= head
    #        # Hack: start indexing at 0 and replace ROOT with self-loop.
    #        # It makes parser implementation much easier.
    #        if head == 0:
    #            # Replace ROOT with self-loop.
    #            head = index
    #        else:
    #            # If not ROOT, shift token left.
    #            head -= 1
    #            assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
    #        edge = (index, head)
    #        ud_edges.append(edge)
    #        ud_edges_labels.append(relation)

    #if deps is not None:
    #    eud_edges: List[Typle[int, int]] = []
    #    eud_edges_labels: List[str] = []
    #    for index, token_deps in enumerate(deps):
    #        assert 0 < len(token_deps), f"Deps must not be empty"
    #        for head, relation in token_deps.items():
    #            assert 0 <= head
    #            # Hack: start indexing at 0 and replace ROOT with self-loop.
    #            # It makes parser implementation much easier.
    #            if head == 0:
    #                # Replace ROOT with self-loop.
    #                head = index
    #            else:
    #                # If not ROOT, shift token left.
    #                head -= 1
    #                assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
    #            edge = (index, head)
    #            eud_edges.append(edge)
    #            edu_edges_labels.append(relation)
