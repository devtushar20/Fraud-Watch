#!/bin/bash
hdfs dfs -mkdir -p /fraud/raw
hdfs dfs -mkdir -p /fraud/staging/spool
hdfs dfs -mkdir -p /fraud/cleaned
hdfs dfs -mkdir -p /fraud/processed/events
hdfs dfs -mkdir -p /fraud/processed/hotspots
hdfs dfs -mkdir -p /fraud/streaming/checkpoints
hdfs dfs -mkdir -p /fraud/models
hdfs dfs -chmod -R 775 /fraud
