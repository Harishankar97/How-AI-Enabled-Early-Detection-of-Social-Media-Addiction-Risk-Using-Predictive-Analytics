# Databricks notebook source
from pyspark.sql import functions as F

# Read Gold user table
df_gold = spark.table("workspace.socialmedia.gold_user_engagement")

# Create binary target variable
df_ml = df_gold.withColumn(
    "is_high_risk",
    F.when(F.col("addiction_risk_level") == "High", 1).otherwise(0)
)

# Select features for ML
feature_cols = [
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_reels_time",
    "avg_feed_time",
    "avg_session_length",
    "age"
]

df_ml_selected = df_ml.select(
    "is_high_risk",
    *feature_cols
)

display(df_ml_selected)


# COMMAND ----------

df_ml_selected.groupBy("is_high_risk").count()


# COMMAND ----------

from pyspark.sql import functions as F

df_gold = spark.table("workspace.socialmedia.gold_user_engagement")

df_gold.select(
    "user_id",
    "age",
    "gender",
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "engagement_score",
    "addiction_risk_level"
).limit(5).display()


# COMMAND ----------

df_ml = df_gold.withColumn(
    "label",
    F.when(F.col("addiction_risk_level") == "Low", 0)
     .when(F.col("addiction_risk_level") == "Medium", 1)
     .otherwise(2)
)

df_ml.select("addiction_risk_level", "label").groupBy("addiction_risk_level", "label").count().display()


# COMMAND ----------

df_ml = df_ml.withColumn(
    "gender_encoded",
    F.when(F.col("gender") == "Male", 1)
     .when(F.col("gender") == "Female", 0)
     .otherwise(-1)
)

df_ml.select("gender", "gender_encoded").groupBy("gender", "gender_encoded").count().display()


# COMMAND ----------

df_ml_final = df_ml.select(
    "user_id",
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "age",
    "gender_encoded",
    "label"
)

df_ml_final.printSchema()
df_ml_final.count()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark


# COMMAND ----------

feature_cols = [
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "age",
    "gender_encoded"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_features = assembler.transform(df_ml_final).select("user_id", "features", "label")

train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Shared/socialmedia_addiction_risk")


# COMMAND ----------

import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/socialmedia/gold/mlflow_tmp"


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

mlflow.set_experiment("/Shared/socialmedia_addiction_risk")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_df)

with mlflow.start_run(run_name="RandomForest_Addiction_Risk"):
    predictions = rf_model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)

    mlflow.log_param("num_trees", 50)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.spark.log_model(
        rf_model,
        artifact_path="rf_addiction_model",
        dfs_tmpdir="/Volumes/workspace/socialmedia/gold/mlflow_tmp"
    )

    print("✅ Model trained & logged successfully")
    print(f"🎯 Accuracy: {accuracy:.4f}")

# COMMAND ----------

from pyspark.sql import functions as F

# Read Gold user table
df_gold_user = spark.table("workspace.socialmedia.gold_user_engagement")

# Add new features
df_features = df_gold_user.withColumn(
    "reels_ratio", F.col("avg_reels_time") / (F.col("avg_daily_minutes") + 1e-6)  # avoid div by 0
).withColumn(
    "feed_ratio", F.col("avg_feed_time") / (F.col("avg_daily_minutes") + 1e-6)
).withColumn(
    "session_ratio", F.col("avg_session_length") / (F.col("avg_daily_minutes") + 1e-6)
).withColumn(
    "age_scaled", F.col("age") / 100.0  # scale age to 0-1
).withColumn(
    "gender_encoded", F.when(F.col("gender") == "Male", 1).otherwise(0)
).withColumn(
    "label", F.when(F.col("addiction_risk_level") == "High", 1).otherwise(0)
)

# Select only features and label for ML
df_ml_ready = df_features.select(
    "user_id",
    "country",
    "age",
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "reels_ratio",
    "feed_ratio",
    "session_ratio",
    "age_scaled",
    "gender_encoded",
    "label"
)

# Show sample
df_ml_ready.show(5)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Split into train and test
train_df, test_df = df_ml_ready.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train_df.count()}")
print(f"Test rows: {test_df.count()}")

# Features for ML
feature_cols = [
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "reels_ratio",
    "feed_ratio",
    "session_ratio",
    "age_scaled",
    "gender_encoded"
]

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ML model
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50)

# Pipeline
pipeline = Pipeline(stages=[assembler, gbt])

# Train model
model = pipeline.fit(train_df)

# Evaluate
predictions = model.transform(test_df)
accuracy = predictions.filter(col("label") == col("prediction")).count() / test_df.count()
print(f"🎯 Accuracy: {accuracy:.4f}")


# COMMAND ----------

from pyspark.sql.functions import col

# Predictions on train set
train_preds = model.transform(train_df)
train_accuracy = train_preds.filter(col("label") == col("prediction")).count() / train_df.count()

# Predictions on test set
test_preds = model.transform(test_df)
test_accuracy = test_preds.filter(col("label") == col("prediction")).count() / test_df.count()

print(f"🎯 Training Accuracy: {train_accuracy:.4f}")
print(f"🎯 Test Accuracy: {test_accuracy:.4f}")


# COMMAND ----------

import pandas as pd

# Feature names (same order as vector assembler)
feature_names = [
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "age",
    "gender_encoded"
]

# Extract feature importance
importances = rf_model.featureImportances.toArray()

df_feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

df_feature_importance


# COMMAND ----------

from pyspark.sql import Row

# Convert pandas to Spark DataFrame
spark_feature_importance = spark.createDataFrame(
    [Row(feature=row["feature"], importance=float(row["importance"]))
     for _, row in df_feature_importance.iterrows()]
)

# Write to Gold layer
spark_feature_importance.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.gold_feature_importance")

print("✅ Gold Feature Importance table created")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM workspace.socialmedia.gold_feature_importance
# MAGIC ORDER BY importance DESC;
# MAGIC

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()

models = client.search_registered_models()

for m in models:
    print(m.name)


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Define feature columns (update as needed)
feature_cols = [
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "reels_ratio",
    "feed_ratio",
    "session_ratio",
    "age_scaled",
    "gender_encoded"
]

# If df_ml_ready is available, use it; otherwise, update to your latest DataFrame
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_features = assembler.transform(df_ml_ready).select(
    "user_id",
    "country",
    "age",
    *feature_cols,
    "features",
    "label"
)

train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10, seed=42)
model = rf.fit(train_df)

# COMMAND ----------

import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")

# Prepare input_example and pred_example from df_features
input_example = df_features.limit(1).toPandas()
pred_example = model.transform(df_features).limit(1).toPandas()

# Convert DenseVector in 'features' column to list for JSON serialization
if "features" in input_example.columns:
    input_example["features"] = input_example["features"].apply(lambda x: x.toArray().tolist())
if "features" in pred_example.columns:
    pred_example["features"] = pred_example["features"].apply(lambda x: x.toArray().tolist())

signature = infer_signature(input_example, pred_example)

with mlflow.start_run():
    mlflow.spark.log_model(
        spark_model=model,
        artifact_path="model",
        registered_model_name="workspace.socialmedia.instagram_addiction_model",
        signature=signature,
        input_example=input_example
    )

print("✅ Model registered successfully in Unity Catalog")

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

for m in client.search_registered_models():
    print(m.name)


# COMMAND ----------

import mlflow

client = mlflow.MlflowClient()

# Get latest version number
model_name = "workspace.socialmedia.instagram_addiction_model"

versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max([int(v.version) for v in versions])

client.set_registered_model_alias(
    name=model_name,
    alias="prod",
    version=latest_version
)

print(f"✅ Alias 'prod' set for model version {latest_version}")


# COMMAND ----------

import mlflow.spark

model_uri = "models:/workspace.socialmedia.instagram_addiction_model@prod"

loaded_model = mlflow.spark.load_model(model_uri)

print("✅ Model loaded from Unity Catalog using prod alias")


# COMMAND ----------

df_predictions = loaded_model.transform(test_df)

df_predictions.select(
    "user_id",
    "prediction",
    "probability"
).show(5)


# COMMAND ----------

from pyspark.sql import functions as F

df_prediction_clean = df_predictions.select(
    "user_id",
    "country",
    "avg_daily_minutes",
    "avg_sessions_per_day",
    "avg_feed_time",
    "avg_reels_time",
    "avg_session_length",
    "age",
    "gender_encoded",
    F.col("prediction").alias("predicted_addiction_risk"),
    F.col("probability").alias("risk_probability")
)


# COMMAND ----------

df_prediction_final = df_prediction_clean.withColumn(
    "addiction_risk_label",
    F.when(F.col("predicted_addiction_risk") == 1, "High")
     .otherwise("Low")
)


# COMMAND ----------

df_prediction_final.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.socialmedia.gold_user_addiction_predictions")

print("✅ GOLD prediction table created")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT addiction_risk_label, COUNT(*) 
# MAGIC FROM workspace.socialmedia.gold_user_addiction_predictions
# MAGIC GROUP BY addiction_risk_label;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM workspace.socialmedia.gold_user_addiction_predictions
# MAGIC LIMIT 10;
# MAGIC