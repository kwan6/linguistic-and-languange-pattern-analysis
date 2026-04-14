from src.data_loader import load_and_combine_data
from src.model_ml import train_ml_models
from src.model_indobert import train_indobert

df = load_and_combine_data()

results = train_ml_models(df)

for model, res in results.items():
    print(f"\n{model}")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

print("Shape data:", df.shape)
print(df.head())
print(df.columns)
print(df["label"].value_counts())
print("\n=== IndoBERT ===")
indobert_result = train_indobert(df)

for k, v in indobert_result.items():
    print(f"{k}: {v}")