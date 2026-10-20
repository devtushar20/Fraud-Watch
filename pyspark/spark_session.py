from pyspark.sql import SparkSession

def get_spark_session(app_name="FraudDetectionBigData"):
    builder = (SparkSession.builder.appName(app_name).master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "true"))
    return builder.getOrCreate()
