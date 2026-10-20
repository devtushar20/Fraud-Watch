from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import (StructType, StructField, LongType, IntegerType, StringType, DoubleType)

RAW_SCHEMA = StructType([
    StructField("transaction_id", LongType()), StructField("customer_id", IntegerType()),
    StructField("merchant_id", IntegerType()), StructField("event_time", StringType()),
    StructField("amount", DoubleType()), StructField("merchant_category", StringType()),
    StructField("channel", StringType()), StructField("city", StringType()),
    StructField("latitude", DoubleType()), StructField("longitude", DoubleType()),
    StructField("distance_from_home_km", DoubleType()), StructField("is_foreign", IntegerType()),
    StructField("txn_count_1h", IntegerType()), StructField("is_fraud", IntegerType())])

def clean_transactions(raw_df):
    return (raw_df
        .withColumn("timestamp", to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss"))
        .filter(col("timestamp").isNotNull())
        .dropna(subset=["transaction_id", "customer_id", "merchant_id", "amount"])
        .filter((col("amount") > 0) & (col("amount") <= 500000))
        .filter(col("latitude").between(6.0, 38.0))
        .dropDuplicates(["transaction_id"]))
