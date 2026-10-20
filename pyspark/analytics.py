from pyspark.sql.functions import avg, count, round as spark_round

def city_summary(df):
    return df.groupBy("city").agg(count("*").alias("total_txns"),
            spark_round(avg("amount"), 2).alias("mean_amount"),
            spark_round(avg("is_fraud") * 100, 2).alias("fraud_pct"))

def hourly_profile(df):
    return df.groupBy("hour_of_day").agg(spark_round(avg("is_fraud") * 100, 3).alias("fraud_pct")).orderBy("hour_of_day")
