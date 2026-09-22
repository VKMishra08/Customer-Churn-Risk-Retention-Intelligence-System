# Companion analysis notes. Run from project root.
from src.churn_engine import load_data, train, score, add_features

df, source = load_data()
model, metrics = train(df)
scored = score(model, df)
print(source)
print(df.shape)
print(metrics)
print(scored[['customerID','ChurnProbability','RiskLevel','RiskDriver','RetentionAction']].head())
