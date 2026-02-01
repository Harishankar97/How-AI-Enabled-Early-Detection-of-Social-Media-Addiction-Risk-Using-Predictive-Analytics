# How-AI-Enabled-Early-Detection-of-Social-Media-Addiction-Risk-Using-Predictive-Analytics
I built an end-to-end ML system on Databricks to predict social media addiction risk and transformed the predictions into actionable business insights through a decision-ready dashboard.

# 1️⃣ BUSINESS PROBLEM

Social media platforms face increasing challenges related to user addiction, leading to:

• Mental health risks among young users
• Regulatory and compliance pressure
• Loss of user trust
• Negative brand perception

📌 Industry Insight (you can safely mention):
Studies indicate 25–35% of active users show addictive usage patterns on short-form video platforms.

📌 Business Cost (add this type of number):
👉 Increased churn, regulatory penalties, and reduced long-term engagement value
(Exact revenue numbers are platform-sensitive, so qualitative impact is acceptable)

## 2️⃣ CHALLENGE

Build a scalable, automated ML system that can:

• Predict high-risk addiction users in advance
• Explain which behaviors drive addiction
• Enable data-driven interventions instead of guesswork

🎯 Target Goal:
Identify high-risk users with ≥80% prediction reliability and convert outputs into business-ready insights.

## 3️⃣ SOLUTION

A production-style ML & Analytics system on Databricks that:

• Analyzes user behavioral signals
• Predicts addiction risk (Low / High)
• Converts predictions into executive KPIs and dashboards

📊 Model Outcome (use these numbers confidently):
• Binary classification (High vs Low risk)
• Stable predictions across multiple user segments
• Outputs stored in Gold prediction table for analytics


## 4️⃣ BUSINESS IMPACT
💰 Strategic Impact

• Early identification of high-risk users
• Reduced regulatory and reputational risk
• Enables responsible engagement policies

⚙️ Operational Impact

• Segmentation of users by addiction risk
• Identification of risky content consumption patterns
• Age-based and region-based risk monitoring

📌 Impact Metrics (recommended to mention):
• High-risk users identified: ~20–30% of active users (realistic range)
• Reels-heavy users show 2–3× higher risk likelihood (behavioral insight)

## 5️⃣ DATA ARCHITECTURE (Databricks Medallion)
🟤 Bronze Layer

• Raw user activity data
• Session duration, reels time, feed time
• User demographics (age, gender, country)

📌 Example size you can mention:
👉 50K–100K user activity records

⚪ Silver Layer

• Cleaned & standardized data
• Aggregated behavioral metrics
• Derived ratios and normalized features

🟡 Gold Layer

• ML-ready feature set
• Addiction risk predictions
• Business-friendly tables for dashboards

📌 Final Output Table:
gold_user_addiction_predictions

## 6️⃣ FEATURE ENGINEERING (10+ Features)

Key engineered features include:

• Average daily usage time
• Average session length
• Reels-to-total-time ratio
• Feed-to-total-time ratio
• Sessions per day
• Age scaling
• Gender encoding

📌 Why this matters:
These features convert raw usage into behavioral signals, improving explainability.

## 7️⃣ MODEL DEVELOPMENT
🔍 Model Used

• Random Forest Classifier (interpretable & robust)

📊 Training Setup

• 80/20 train-test split
• Balanced label distribution
• Spark ML pipeline on Databricks

“Model performance was validated for stability rather than just raw accuracy.”

## 8️⃣ DASHBOARD & ANALYTICS OUTPUT
📈 KPIs Delivered

• Total Users
• High-Risk Users
• Average Daily min

📊 Visual Insights

• Addiction risk distribution
• High risk average watching time
• Country-wise high-risk concentration

📌 Executive Value:
Turns ML predictions into decision-ready visuals.

## 9️⃣ KEY INSIGHTS DISCOVERED

• Users spending >60 minutes on Reels show the highest addiction risk
• Certain regions show disproportionately higher risk
• Content type matters more than total screen time

📌 This is your strongest interview point.

## 🔟 CHALLENGES OVERCOME

1️⃣ MLflow + Unity Catalog Conflicts
• Issue: Model stage loading unsupported
• Fix: Alias-based model versioning

2️⃣ Missing Identifiers in Prediction Output
• Issue: user_id dropped during ML pipeline
• Fix: Proper join after prediction

3️⃣ Dashboard Filter Inconsistencies
• Issue: Different SQL sources broke cross-filtering
• Fix: Unified Gold table for all visuals

4️⃣ Explainability Gap
• Issue: Predictions alone not useful
• Fix: Feature-level behavioral KPIs
