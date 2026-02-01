# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.socialmedia;
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.socialmedia.bronze;
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.socialmedia.silver;
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.socialmedia.gold;
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

df_raw = spark.read.csv(
    "/Volumes/workspace/socialmedia/social_media_data/instagram_usage_lifestyle.csv",
    header=True,
    inferSchema=True
)

df_bronze = df_raw.withColumn(
    "ingestion_ts",
    F.current_timestamp()
)

df_bronze.write.format("delta") \
    .mode("overwrite") \
    .save("/Volumes/workspace/socialmedia/bronze/instagram_usage")

print("✅ Bronze ingestion completed")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS workspace.socialmedia.bronze_instagram_usage
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT * FROM delta.`/Volumes/workspace/socialmedia/bronze/instagram_usage`;
# MAGIC

# COMMAND ----------

df_bronze.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.bronze_instagram_usage")


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL workspace.socialmedia.bronze_instagram_usage;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.socialmedia.bronze_instagram_usage;
# MAGIC