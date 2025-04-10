import gradio as gr
import requests

BASE_URL = "https://flask-recommender-api-production.up.railway.app"

def fetch_recommendations(user_query):
    try:
        
        health_resp = requests.get(f"{BASE_URL}/health")
        if health_resp.status_code != 200:
            return "❌ Backend is not healthy or unavailable."

        
        headers = {"Content-Type": "application/json"}
        payload = {"query": user_query}

        response = requests.post(f"{BASE_URL}/recommend", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "recommended_assessments" not in data:
            return "No assessments found or unexpected response."

      
        rows = []
        for item in data["recommended_assessments"]:
            url_link = f'<a href="{item["url"]}" target="_blank">link</a>'
            test_types = ", ".join(item["test_type"])
            rows.append(f"""
                <tr>
                    <td>{item["description"]}</td>
                    <td>{url_link}</td>
                    <td>{item["adaptive_support"]}</td>
                    <td>{item["remote_support"]}</td>
                    <td>{item["duration"]} min</td>
                    <td>{test_types}</td>
                </tr>
            """)

        html_table = f"""
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Description</th>
                <th>URL</th>
                <th>Adaptive</th>
                <th>Remote</th>
                <th>Duration</th>
                <th>Test Type</th>
            </tr>
            {''.join(rows)}
        </table>
        """

        return html_table

    except requests.exceptions.RequestException as e:
        return f"❌ Error contacting backend: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("## 🔍 SHL Assessment Recommender (Frontend)")
    gr.Markdown("Type your job description or skillset and click *Enter* to get relevant assessments.")
    
    with gr.Row():
        user_input = gr.Textbox(placeholder="e.g. Looking for a software engineer", label="Job Query", lines=1)
        submit_btn = gr.Button("Enter")

    output_html = gr.HTML(label="Recommended Assessments")

    submit_btn.click(fn=fetch_recommendations, inputs=user_input, outputs=output_html)

if __name__ == "__main__":
    demo.launch()
