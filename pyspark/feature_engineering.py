from pyspark.sql import Window
from pyspark.sql.functions import (col, count, percentile_approx, least, greatest, lit, when,
                                   hour, dayofweek, round as spark_round)

def add_features_and_score(df):
    df = (df.withColumn("hour_of_day", hour("timestamp"))
            .withColumn("day_of_week", (dayofweek("timestamp") + 5) % 7)      # Monday = 0
            .withColumn("is_weekend", (col("day_of_week") >= 5).cast("int"))
            .withColumn("is_night", col("hour_of_day").between(0, 4).cast("int"))
            .withColumn("is_online", (col("channel") == "ONLINE").cast("int")))
    w_cust = (Window.partitionBy("customer_id").orderBy(col("timestamp").cast("long")).rangeBetween(-3600, -1))
    df = df.withColumn("txn_count_1h", count("*").over(w_cust))
    df = df.withColumn("amount_ratio", col("amount") / percentile_approx("amount", 0.5).over(Window.partitionBy("customer_id")))
    s_amt  = least(col("amount_ratio") / 6.0, lit(1.0)) * 100.0
    s_dist = least(col("distance_from_home_km") / 500.0, lit(1.0)) * 100.0
    s_vel  = least(col("txn_count_1h") / 5.0, lit(1.0)) * 100.0
    s_ctx  = least(col("is_night")*40.0 + col("is_online")*30.0 + col("is_foreign")*30.0, lit(100.0))
    raw_score = 0.35 * s_amt + 0.25 * s_dist + 0.25 * s_vel + 0.15 * s_ctx
    df = df.withColumn("risk_score", spark_round(greatest(lit(0.0), least(raw_score, lit(100.0))), 2))
    df = df.withColumn("risk_level", when(col("risk_score") < 25.0, "LOW")
            .when(col("risk_score") < 45.0, "MEDIUM").otherwise("HIGH"))
    return df
