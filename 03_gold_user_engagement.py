# Databricks notebook source
from pyspark.sql import functions as F

# Read Silver table
df_silver = spark.table("workspace.socialmedia.silver_instagram_usage")

# Create user-level engagement summary
df_gold_user = df_silver.groupBy("user_id", "country", "age", "gender") \
    .agg(
        F.avg("daily_active_minutes_instagram").alias("avg_daily_minutes"),
        F.avg("sessions_per_day").alias("avg_sessions_per_day"),
        F.avg("time_on_reels_per_day").alias("avg_reels_time"),
        F.avg("time_on_feed_per_day").alias("avg_feed_time"),
        F.avg("average_session_length_minutes").alias("avg_session_length"),
        F.avg("user_engagement_score").alias("engagement_score")
    )

# Write Gold table
df_gold_user.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.gold_user_engagement")

print("✅ Gold User Engagement table created")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.socialmedia.gold_user_engagement;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM workspace.socialmedia.gold_user_engagement
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.socialmedia.gold_country_engagement;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM workspace.socialmedia.gold_country_engagement
# MAGIC ORDER BY high_risk_percentage DESC;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   MIN(engagement_score) AS min_score,
# MAGIC   MAX(engagement_score) AS max_score,
# MAGIC   AVG(engagement_score) AS avg_score
# MAGIC FROM workspace.socialmedia.gold_user_engagement;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS workspace.socialmedia.gold_user_risk_segmentation;
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table("workspace.socialmedia.gold_user_engagement")

window_spec = Window.orderBy(F.col("engagement_score").desc())

df_gold_risk = df.withColumn(
    "risk_percentile",
    F.percent_rank().over(window_spec)
).withColumn(
    "addiction_risk_level",
    F.when(F.col("risk_percentile") <= 0.20, "High")
     .when(F.col("risk_percentile") <= 0.50, "Medium")
     .otherwise("Low")
)

df_gold_risk.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.gold_user_risk_segmentation")

print("✅ Gold risk segmentation table recreated successfully")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT addiction_risk_level, COUNT(*) AS users
# MAGIC FROM workspace.socialmedia.gold_user_risk_segmentation
# MAGIC GROUP BY addiction_risk_level;
# MAGIC

# COMMAND ----------

df_risk = spark.table("workspace.socialmedia.gold_user_risk_segmentation")


# COMMAND ----------

df_gold_user = spark.table("workspace.socialmedia.gold_user_engagement")

# Drop 'addiction_risk_level' if it exists to avoid duplicate column after join
df_gold_user = df_gold_user.drop("addiction_risk_level")

df_gold_user_enriched = df_gold_user.join(
    df_risk.select("user_id", "addiction_risk_level"),
    on="user_id",
    how="left"
)


# COMMAND ----------

df_gold_user_enriched.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.socialmedia.gold_user_engagement")

print("✅ gold_user_engagement enriched with risk level")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT addiction_risk_level, COUNT(*)
# MAGIC FROM workspace.socialmedia.gold_user_engagement
# MAGIC GROUP BY addiction_risk_level;
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

df_gold_user = spark.table("workspace.socialmedia.gold_user_engagement")

df_gold_country = df_gold_user.groupBy("country") \
    .agg(
        F.countDistinct("user_id").alias("total_users"),
        F.avg("avg_daily_minutes").alias("avg_daily_minutes"),
        F.avg("avg_sessions_per_day").alias("avg_sessions_per_day"),
        F.avg("engagement_score").alias("avg_engagement_score"),
        F.sum(
            F.when(F.col("addiction_risk_level") == "High", 1).otherwise(0)
        ).alias("high_risk_users")
    ) \
    .withColumn(
        "high_risk_percentage",
        F.round((F.col("high_risk_users") / F.col("total_users")) * 100, 2)
    )

df_gold_country.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.gold_country_engagement")

print("✅ Gold Country Engagement table updated correctly")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM workspace.socialmedia.gold_country_engagement
# MAGIC ORDER BY high_risk_percentage DESC;
# MAGIC