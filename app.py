import pandas as pd
import ast
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("assessments_with_embeddings.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
all_embeddings = np.vstack(df["embedding"].to_numpy())

# Inference function
def recommend_assessments(user_input):
    input_embedding = model.encode([user_input])
    similarities = cosine_similarity(input_embedding, all_embeddings)[0]
    df_copy = df.copy()
    df_copy["similarity"] = similarities

    top_matches = df_copy.sort_values(by="similarity", ascending=False).head(10)

    return top_matches[[
        "Assessment name",
        "Remote Testing Support",
        "Adaptive/IRT",
        "Test type",
        "Duration(min)"
    ]]

# UI with Gradio Blocks
with gr.Blocks() as interface:
    gr.Markdown("## 🔍 SHL Assessment Recommender")
    gr.Markdown("Enter a job role or skill to get the top 10 relevant SHL assessments.")

    user_input = gr.Textbox(placeholder="e.g., Data Analyst, Communication Skills", label="Enter Job Role or Skill")
    output_table = gr.Dataframe(
        headers=[
            "Assessment name",
            "Remote Testing Support",
            "Adaptive/IRT",
            "Test type",
            "Duration(min)"
        ],
        label="Recommended Assessments"
    )

    user_input.change(fn=recommend_assessments, inputs=user_input, outputs=output_table)

interface.launch()
