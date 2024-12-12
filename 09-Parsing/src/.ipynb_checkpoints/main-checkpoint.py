import argparse
import sys

from torch.utils.data import DataLoader

from dataset import CobaldJointDataset, NO_ARC_LABEL
from vocabulary import Vocabulary
from parser import MorphoSyntaxSemanticsParser


def train(train_conllu_path, val_conllu_path, batch_size):

    # Create raw training dataset to build vocabulary upon.
    raw_train_dataset = CobaldJointDataset(train_conllu_path)
    # Build training vocabulary that maps string labels into integers.
    vocab = Vocabulary(
        raw_train_dataset,
        # Namespaces to encode.
        namespaces=[
            "lemma_rules",
            "joint_pos_feats",
            "deps_ud",
            "deps_eud",
            "miscs",
            "deepslots",
            "semclasses"
        ]
    )
    # Make sure NO_ARC_LABEL has a class of 0, because
    # we will rely on operations like deps_ud.nonzero() in a model.
    vocab.replace_index(NO_ARC_LABEL, 0, namespace="deps_ud")
    vocab.replace_index(NO_ARC_LABEL, 0, namespace="deps_eud")

    # Create actual training and validation datasets.
    transform = lambda sample: vocab.encode(sample)
    train_dataset = CobaldJointDataset(train_conllu_path, transform)
    val_dataset = CobaldJointDataset(val_conllu_path, transform)

    # Create dataloaders.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=CobaldJointDataset.collate_fn,
        shuffle=False
    )

    # Create model.
    model_args = {
        "encoder_args": {
            "model_name": "distilbert-base-uncased",
            "train_parameters": True,
            "sub_token_mode": "avg"
        },
        "null_predictor_args": {
            "hidden_dim": 512,
            "activation": "relu",
            "dropout": 0.1,
            "consecutive_null_limit": 4
        },
        "tagger_args": {
            "lemma_rule_classifier_args": {
                "hidden_dim": 512,
                "n_classes": vocab.get_namespace_size("lemma_rules"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "pos_feats_classifier_args": {
                "hidden_dim": 512,
                "n_classes": vocab.get_namespace_size("joint_pos_feats"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "depencency_classifier_args": {
                "hidden_dim": 128,
                "n_rels_ud": vocab.get_namespace_size("deps_ud"),
                "n_rels_eud": vocab.get_namespace_size("deps_eud"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "misc_classifier_args": {
                "hidden_dim": 256,
                "n_classes": vocab.get_namespace_size("miscs"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "deepslot_classifier_args": {
                "hidden_dim": 512,
                "n_classes": vocab.get_namespace_size("deepslots"),
                "activation": "relu",
                "dropout": 0.1,
            },
            "semclass_classifier_args": {
                "hidden_dim": 512,
                "n_classes": vocab.get_namespace_size("semclasses"),
                "activation": "relu",
                "dropout": 0.1,
            }
        }
    }
    model = MorphoSyntaxSemanticsParser(**model_args)
    print(model)


def predict(conllu_path, optional_arg):
    print("Prediction mode activated.")
    print(f"Input data path: {conllu_path}")
    if optional_arg:
        print(f"Using optional argument: {optional_arg}")
    else:
        print("No optional arguments provided.")


def main():
    parser = argparse.ArgumentParser(description="A simple application with train and predict modes.")

    #parser.add_argument(
    #    "mode",
    #    choices=["train", "predict"],
    #    help="Mode of operation: 'train' or 'predict'."
    #)

    # Subparsers for mode-specific arguments
    subparsers = parser.add_subparsers(dest="subparser_name")

    # Train mode arguments
    train_parser = subparsers.add_parser("train", help="Arguments for training mode.")
    train_parser.add_argument(
        "train_conllu_path",
        type=str,
        help="Path to the training .conllu file."
    )
    train_parser.add_argument(
        "val_conllu_path",
        type=str,
        help="Path to the validation .conllu file."
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for dataloaders."
    )


    # Predict mode arguments
    predict_parser = subparsers.add_parser("predict", help="Arguments for prediction mode.")
    predict_parser.add_argument(
        "conllu_path",
        type=str,
        help="Path to the input .conllu file for prediction."
    )

    args = parser.parse_args()

    if args.subparser_name == "train":
        if not hasattr(args, 'train_conllu_path') or not hasattr(args, 'val_conllu_path'):
            print("Error: train mode requires 'train_conllu_path' and 'val_conllu_path'.")
            sys.exit(1)
        train(args.train_conllu_path, args.val_conllu_path, args.batch_size)
    elif args.subparser_name == "predict":
        if not hasattr(args, 'conllu_path'):
            print("Error: predict mode requires 'conllu_path'.")
            sys.exit(1)
        predict(args.conllu_path)
    else:
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    main()


