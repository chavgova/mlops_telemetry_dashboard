import pandas as pd
from pyspark.sql import SparkSession

from configs.configs import PathsConfig


if __name__ == "__main__":
    raw_csv = pd.read_csv("data/trx_data.csv")
    UTC_EPOCH = pd.Timestamp(year=1970, month=1, day=1)

    raw_csv.drop("Unnamed: 0", axis=1, inplace=True)
    for col in ["payment_type", "category", "customer", "gender", "merchant"]:
        raw_csv[col] = raw_csv[col].str.replace("'", "")

    raw_csv["chck_date"] = pd.to_datetime(
        raw_csv["chck_date"], errors="raise", format="%Y-%m-%d %H:%M:%S"
    )
    raw_csv["chck_date"] = raw_csv["chck_date"].fillna(UTC_EPOCH)
    raw_csv = raw_csv.astype({"amount": "float32", "fraud": "bool"})
    raw_csv["chck_date_day"] = raw_csv["chck_date"].dt.strftime("%Y-%m-%d")

    # I don't want all the cores to be eaten up by Spark, even if this is a minimal application.
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    spark_df = spark.createDataFrame(raw_csv)

    spark_df.write.option("maxRecordsPerFile", 100_000).parquet(
        PathsConfig.raw_data_path, mode="overwrite", partitionBy="chck_date_day"
    )
