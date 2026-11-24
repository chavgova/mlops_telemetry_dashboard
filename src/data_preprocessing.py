from datetime import date

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from configs.configs import PathsConfig, run_config


def get_nested_column_extended(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    For the column name given in the data, expands the list elements of the column into separate columns.

    WARNING: Drops the col_name

    Example:

    input:

        +--------+
        |   A    |
        +--------+
        | [1, 2] |
        | [3, 4] |
        +--------+

    Output:

        +-----+-----+
        | A_0 | A_1 |
        +-----+-----+
        |   1 |   2 |
        |   3 |   4 |
        +-----+-----+

    :param data: Data that has list as elements in col_name
    :param col_name: The column with list elements
    :return: Data returned with list elements as separate columns
    """
    expanded_col = pd.DataFrame(
        data[col_name].tolist(),
        columns=[col_name + "_" + str(i) for i in range(len(data[col_name][0]))],
    )
    data = data.drop(col_name, axis=1)
    return pd.concat([data, expanded_col], axis=1)


def _get_data_after(start_date: date) -> DataFrame:
    """
    Will read and return the data after (including) start_date.

    See also: data_processing.data_processing()

    :param start_date: The date data should be started to read from.
    :return: All data including and after the start_date
    """
    # I don't want all the cores to be eaten up by Spark, even if this is a minimal application.
    spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
    # Remember the lazy execution:
    # - The order will be decided by Spark.
    # - Spark will try to filter and select before reading. So you will end up reading only the required parts.
    return (
        spark.read.parquet(PathsConfig.raw_data_path)
        .select(["payment_type", "category", "amount", "fraud", "chck_date"])
        .filter(F.col("chck_date") >= start_date)
    )


def get_preprocessed_data(start_date: date) -> DataFrame:
    """
    Preprocessing step.

    Will read and pre-process the data starting from start_date. All the data from (including) start_date to later time
    will be preprocessed. The date will be got from the chck_date.

    Example:

        +-----+------------+---------------+
        | Row | chck_date  | will_process? |
        +-----+------------+---------------+
        |   1 | 2025-01-01 | No            |
        |   2 | 2025-01-01 | No            |
        |   3 | 2025-01-02 | Yes           |
        |   4 | 2025-01-02 | Yes           |
        |   5 | 2025-01-03 | Yes           |
        |   6 | 2024-12-30 | No            |
        |   7 | 2099-01-01 | Yes           |
        +-----+------------+---------------+

    :param start_date: The date the preprocessing should start with. This step will process all the data starting
    (including) this date, and follow up with the later dates.
    :return: Preprocessed data
    """
    data = _get_data_after(start_date)
    # Remember the lazy execution:
    # - This will actually (should) help with reading even fewer data.
    data = data.sample(
        run_config.sample_rate
    )  # Not making this a func. as it is a simple one-liner.
    # I don't have to deal with string encoding like .csv here, as Parquet encodes string properly.
    return data
