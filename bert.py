from argparse import Namespace
import os
from typing import Iterable

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging

from model import Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

class BertModel(Model):
    args: Namespace
    model_args: ClassificationArgs = ClassificationArgs()
    best_model_dir: str | None = None
    model_dir: str | None = None
    model: ClassificationModel | None = None

    def _create_model_args(self) -> ClassificationArgs:
        model_args: ClassificationArgs = ClassificationArgs()
        model_args.num_train_epochs = self.args.epochs
        # model_args.regression = True
        model_args.manual_seed = self.args.manual_seed
        # model_args.overwrite_output_dir = True
        model_args.save_steps = -1
        model_args.train_batch_size = self.args.batch_size
        # model_args.early_stopping_metric = 'spearmanr'
        # model_args.early_stopping_metric_minimize = False
        model_args.best_model_dir = self.best_model_dir
        return model_args

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.best_model_dir = os.path.join(self.args.save_dir, 'best_model')
        self.model_dir = os.path.join(self.args.save_dir, 'model')
        # model = ClassificationModel(
        #     "roberta", "roberta-base"
        # )

        self.model = ClassificationModel(
            self.args.bert_type,
            self.args.bert,
            num_labels=2,
            args=self._create_model_args()
        )

    def load_model(self) -> ClassificationModel:
        # Load best model
        return ClassificationModel(
            self.args.bert_type,
            self.best_model_dir,
            num_labels=1,
            args=self.model_args
        )

    def predict(self, input_features: Iterable):
        predictions, raw_outputs = self.model.predict(self.df_eval['review_text'])
        with open(os.path.join(self.best_model_dir, 'predictions.tbl'), 'w') as f:
            for text, real_pred, program_pred in zip(self.df_eval['review_text'], self.df_eval['recommend'], predictions):
                f.write(f'{text}\t{real_pred}\t{program_pred}\n')

    def eval(self):
        result, model_outputs, wrong_predictions = self.model.eval_model(self.df_eval)
        with open(os.path.join(self.best_model_dir, 'test_results.txt'), 'w') as f:
            for key, val in result.items():
                f.write(f'{key} = {val}\n')

    def train(self):
        self.model.train_model(self.df_train.to_pandas(), output_dir=self.model_dir)
