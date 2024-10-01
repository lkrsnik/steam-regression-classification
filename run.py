import argparse
import os
import time

import joblib
import logging
from scipy import stats
import numpy as np
from wandb.apis.importers.internals.util import Namespace

from bert import BertModel
import polars as pl

seed = 23

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logger")
logger.setLevel(logging.WARNING)


def create_models(classifier, fname, X_train, y_train, X_test, y_true):
    classifier.random_state = seed
    print('###############################################################')
    print('Starting calculation..')
    model = classifier.fit(X_train, y_train)
    _ = joblib.dump(model, os.path.join(args.output, fname), compress=9)
    y_test = model.predict(X_test)

    pearson = stats.pearsonr(y_test, y_true)
    spearman = stats.spearmanr(y_test, y_true)
    with open(os.path.join(args.output, fname) + '.result', 'w') as f:
        f.write(f'pearson={str(pearson)}|spearman={str(spearman)}')
    os.path.join(args.output, fname)
    print(f'{fname} - Pearson: {pearson} | Spearman: {spearman}')


def main(args: Namespace, other_args: dict):
    # reading data
    model = BertModel(args=args)

    model.train()
    model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract structures from a parsed corpus.')
    parser.add_argument('input',
                        help='Input file in parquet format.')
    parser.add_argument('--save_dir',
                        help='A location of a model.')
    parser.add_argument('--train', action='store_true',
                        help='Train a model.')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate a model.')
    parser.add_argument('--eval_test', action='store_true',
                        help='Evaluate a model on test set.')
    parser.add_argument('--predict', action='store_true',
                        help='Predict using input.')
    parser.add_argument('--type',
                        help='Write a type of model you would like to train.')
    parser.add_argument('--manual_seed', type=int, default=23,
                        help='Manual seed.')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training data.')
    parser.add_argument('--eval_size', type=int, default=50000,
                        help='Size of training data.')

    # BERT params
    parser.add_argument('--bert', default='roberta-large',
                        help='Path to BERT/bert name.')
    parser.add_argument('--bert_type', default='roberta',
                        help='Type of bert used.')
    # roberta-large - 32 OK
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of a batch.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs number.')
    args, other_args = parser.parse_known_args()

    np.random.seed(args.manual_seed)
    pl.set_random_seed(args.manual_seed)
    start = time.time()
    main(args, other_args)
    logging.info("TIME: {}".format(time.time() - start))
