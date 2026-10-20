"""Spark version of the batch pipeline (needs: pip install pyspark, Java 17).
Usage: python pyspark/run_spark_pipeline.py   (reads data/raw_transactions.csv)"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from spark_session import get_spark_session
from preprocessing import RAW_SCHEMA, clean_transactions
from feature_engineering import add_features_and_score
from hotspot_detection import merchant_hotspots
from analytics import city_summary

spark = get_spark_session()
raw = spark.read.csv("data/raw_transactions.csv", header=True, schema=RAW_SCHEMA)
clean = clean_transactions(raw)
proc = add_features_and_score(clean)
proc.write.mode("overwrite").partitionBy("city").parquet("out/spark_processed")
merchant_hotspots(proc).limit(10).show(truncate=False)
city_summary(proc).show()
spark.stop()
