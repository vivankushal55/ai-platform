import pandas as pd
from datasets import load_dataset

# Downloads the Bitext customer-support dataset (~27k rows)
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = ds["train"].to_pandas()

# Save locally so we don't re-download
df.to_csv("data/tickets.csv", index=False)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nCategories:")
print(df["category"].value_counts())
print("\nSample row:")
print(df.iloc[0])