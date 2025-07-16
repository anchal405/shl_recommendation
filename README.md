# SHL Assessment  Recommendation Engine

This project provides intelligent SHL assessment recommendations based on natural language queries such as job descriptions or skill requirements. It combines a semantic search backend powered by SentenceTransformers with an interactive Gradio-based frontend interface.

The system is deployed across two services:
- **Frontend**: Hosted on Hugging Face Spaces, offering a user-friendly interface for recruiters and hiring managers.
- **Backend API**: Hosted via Railway, supporting JSON-based recommendations programmatically.

---

## Key Features

-  Accepts free-text input (e.g., job role, skillset).
-  Uses semantic embedding and cosine similarity to find relevant SHL assessments.
-  Powered by `all-MiniLM-L6-v2` from the SentenceTransformers library.
-  API with `/health` for status check and `/recommend` for JSON responses.
-  Frontend displays results in a clickable, tabular interface using Gradio Blocks.

---

## Technologies Used

| Layer       | Stack                                          |
|-------------|------------------------------------------------|
| **Model**   | SentenceTransformers (`all-MiniLM-L6-v2`)       |
| **Frontend**| Gradio (Blocks UI)                             |
| **Backend** | Flask (REST API), Flask-CORS                   |
| **Similarity** | Cosine Similarity via `scikit-learn`        |
| **Deployment** | Hugging Face Spaces (UI), Railway (API)    |
| **Data**    | Precomputed SHL assessment embeddings (`.csv`) |

---

## File Overview

```text
 SHL-Recommendation-Engine
├── app.py                         # Gradio UI (Frontend)
├── back.py                        # Flask API (Backend)
├── assessments_with_embeddings-2.csv  # Assessment metadata and vector embeddings
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
```
## 🔗 Project Link

- Live Demo: [SHL Recommender on Hugging Face](https://huggingface.co/spaces/Lastinn/shl_recommendation_engine)
  

