CREATE DATABASE IF NOT EXISTS fraud_db LOCATION '/fraud/hive_warehouse';
USE fraud_db;
CREATE EXTERNAL TABLE IF NOT EXISTS processed_transactions (
    transaction_id BIGINT, customer_id INT, merchant_id INT, event_time TIMESTAMP,
    amount DOUBLE, merchant_category STRING, channel STRING, distance_from_home_km DOUBLE,
    is_foreign INT, txn_count_1h INT, hour_of_day INT, is_night INT, is_online INT,
    amount_ratio DOUBLE, risk_score DOUBLE, risk_level STRING, is_fraud INT)
PARTITIONED BY (city STRING)
STORED AS PARQUET LOCATION '/fraud/processed/events';
MSCK REPAIR TABLE processed_transactions;

SELECT city, COUNT(*) AS total_txns,
       ROUND(AVG(amount),2) AS mean_amount,
       ROUND(SUM(is_fraud)*100.0/COUNT(*),2) AS fraud_pct
FROM processed_transactions GROUP BY city
ORDER BY fraud_pct DESC;
