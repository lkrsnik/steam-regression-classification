import polars as pl

def write_parquet(df_dict: dict, filename: str):
    df_dict[("train",)].write_parquet(filename + '_train.parquet')
    df_dict[("test",)].write_parquet(filename + '_test.parquet')
    df_dict[("dev",)].write_parquet(filename + '_dev.parquet')

def read_parquet_train(filename: str) -> pl.DataFrame:
    return pl.read_parquet(filename + '_train.parquet')

def read_parquet_test(filename: str) -> pl.DataFrame:
    return pl.read_parquet(filename + '_test.parquet')

def read_parquet_dev(filename: str) -> pl.DataFrame:
    return pl.read_parquet(filename + '_dev.parquet')