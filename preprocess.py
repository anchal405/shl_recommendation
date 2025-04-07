# STEP 1: Install required libraries
!pip install -q sentence-transformers

# STEP 2: Import libraries
import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

# STEP 3: Upload the CSV file
uploaded = files.upload()

# STEP 4: Load the DataFrame
filename = list(uploaded.keys())[0]

if filename.endswith(".csv"):
    df = pd.read_csv(filename)
else:
    print(f" The uploaded file '{filename}' is not a CSV. Please upload the correct file.")

# STEP 5: Confirm column names
print("Available columns in the CSV file:", list(df.columns))

# STEP 6: Parse embeddings (convert stringified list to real list)
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# STEP 7: Embed user query
query = input("Enter a job role or skill you're interested in: ")
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([query])[0]

# STEP 8: Calculate cosine similarities
df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])

# STEP 9: Show top 5 recommendations
top_matches = df.sort_values(by="similarity", ascending=False).head(5)
print("\nTop 5 recommended assessments:\n")
print(top_matches[["Assessment name", "similarity"]])
