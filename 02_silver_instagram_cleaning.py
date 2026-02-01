# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import *

df_bronze = spark.table("workspace.socialmedia.bronze_instagram_usage")

df_bronze.printSchema()
df_bronze.count()


# COMMAND ----------

# Standardize Yes / No Columns

yes_no_cols = [
    "has_children",
    "smoking",
    "uses_premium_features",
    "two_factor_auth_enabled",
    "biometric_login_used"
]

df_silver = df_bronze

for col in yes_no_cols:
    df_silver = df_silver.withColumn(
        col,
        F.when(F.lower(F.trim(F.col(col))).isin("yes", "y", "true", "1"), F.lit(True))
         .when(F.lower(F.trim(F.col(col))).isin("no", "n", "false", "0"), F.lit(False))
         .otherwise(None)
    )

df_silver.select(yes_no_cols).show(5)


# COMMAND ----------

# Standardize Categorical Columns

cat_cols = [
    "gender",
    "income_level",
    "education_level",
    "employment_status",
    "relationship_status",
    "urban_rural"
]

for col in cat_cols:
    df_silver = df_silver.withColumn(
        col,
        F.initcap(F.trim(F.col(col)))
    )

df_silver.select(cat_cols).show(10)


# COMMAND ----------

# Standardize Numerical Columns

df_silver = (
    df_silver
    # Age
    .withColumn(
        "age",
        F.when((F.col("age") < 13) | (F.col("age") > 90), None)
         .otherwise(F.col("age"))
    )

    # BMI
    .withColumn(
        "body_mass_index",
        F.when((F.col("body_mass_index") < 10) | (F.col("body_mass_index") > 60), None)
         .otherwise(F.col("body_mass_index"))
    )

    # Sleep
    .withColumn(
        "sleep_hours_per_night",
        F.when((F.col("sleep_hours_per_night") < 0) | (F.col("sleep_hours_per_night") > 16), None)
         .otherwise(F.col("sleep_hours_per_night"))
    )

    # Exercise
    .withColumn(
        "exercise_hours_per_week",
        F.when((F.col("exercise_hours_per_week") < 0) | (F.col("exercise_hours_per_week") > 40), None)
         .otherwise(F.col("exercise_hours_per_week"))
    )

    # Time spent
    .withColumn(
        "daily_active_minutes_instagram",
        F.when(F.col("daily_active_minutes_instagram") > 1440, None)
         .otherwise(F.col("daily_active_minutes_instagram"))
    )
)


# COMMAND ----------

# Quick Validation
df_silver.select(
    F.min("age"), F.max("age"),
    F.min("sleep_hours_per_night"), F.max("sleep_hours_per_night"),
    F.min("body_mass_index"), F.max("body_mass_index")
).show()


# COMMAND ----------

df_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/Volumes/workspace/socialmedia/silver/instagram_usage")


# COMMAND ----------

df_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.socialmedia.silver_instagram_usage")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.socialmedia.silver_instagram_usage;
# MAGIC