from pyspark.sql.functions import col, count, avg, lit, least, round as spark_round

def merchant_hotspots(df, min_txns=100):
    g = (df.groupBy("merchant_id").agg(
            count("*").alias("txns"),
            (avg("is_fraud") * 100).alias("fraud_pct"),
            avg("amount").alias("mean_amount"),
            avg("risk_score").alias("avg_risk_score")))
    return (g.filter(col("txns") >= min_txns)
             .withColumn("merchant_severity_score", spark_round(
                 (col("fraud_pct") * 0.50) + (col("avg_risk_score") * 0.35) +
                 (least(col("mean_amount") / 20000.0, lit(1.0)) * 100 * 0.15), 2))
             .orderBy(col("merchant_severity_score").desc()))
