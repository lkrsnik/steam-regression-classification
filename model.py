from abc import abstractmethod
from argparse import Namespace
from collections.abc import Iterable
from pathlib import Path

from data import read_parquet_train, read_parquet_dev, read_parquet_test
from pydantic import BaseModel

import polars as pl


class Model(BaseModel, arbitrary_types_allowed=True):
    args: Namespace
    df_train: pl.DataFrame | None = None
    df_eval: pl.DataFrame | None = None

    def model_post_init(self, __context):
        Path(self.args.save_dir).mkdir(exist_ok=True, parents=True)
        if self.args.train:
            self.df_train = read_parquet_train(self.args.input)
            if self.args.train_size is not None:
                self.df_train = self.df_train.sample(self.args.train_size, seed=self.args.manual_seed, shuffle=True)
            self.df_train = self.df_train.select(['review_text', 'recommended'])
            self.df_train = self.df_train.cast({'recommended': pl.Int8})

        if self.args.eval or self.args.train:
            self.df_eval = read_parquet_dev(self.args.input)
            if self.args.eval_size is not None:
                self.df_eval = self.df_eval.sample(self.args.eval_size, seed=self.args.manual_seed, shuffle=True)
            self.df_eval = self.df_eval.select(['review_text', 'recommended'])
            self.df_eval = self.df_eval.cast({'recommended': pl.Int8})

        if self.args.eval_test:
            self.df_eval = read_parquet_test(self.args.input)
            if self.args.eval_size is not None:
                self.df_eval = self.df_eval.sample(self.args.eval_size, seed=self.args.manual_seed, shuffle=True)
            self.df_eval = self.df_eval.select(['review_text', 'recommended'])
            self.df_eval = self.df_eval.cast({'recommended': pl.Int8})

        # TODO move this to jupyter/preprocessing


        # self.df_train['recommended'] = self.df_train.with_columns(
        #    pl.col('recommended').replace_strict({True: "R", False: "U"})
        # )
        # self.df_train['recommended'] = self.df_train.with_columns(
        #     pl.col('recommended').replace_strict({True: "R", False: "U"})
        # )
        #
        # self.df_eval['recommended'] = self.df_eval.with_columns(
        #     pl.col('recommended').replace_strict({True: "R", False: "U"})
        # )


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def predict(self, input_features: Iterable):
        pass