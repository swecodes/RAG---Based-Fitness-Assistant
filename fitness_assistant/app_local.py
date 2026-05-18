import uuid
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from ingest import load_index

app = Flask(__name__)
"""

# Load search index
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/data.csv")
index = load_index(DATA_PATH)

BOOST = {
    "exercise_name": 2.11, "type_of_activity": 1.46,
    "type_of_equipment": 0.65, "body_part": 2.65,
    "type": 1.31, "muscle_groups_activated": 2.54,
    "instructions": 0.74,
}

def search(query):
    return index.search(query=query, filter_dict={}, boost_dict=BOOST, num_results=3)

def rag(question):
    results = search(question)
    answer = "\n\n".join(
        f"{r['exercise_name']}: {r['instructions']}" for r in results
    )
    return {
        "answer": answer,
        "model_used": "minsearch-local",
        "response_time": 0.0,
        "relevance": "RELEVANT",
        "relevance_explanation": "Local retrieval only",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "eval_prompt_tokens": 0,
        "eval_completion_tokens": 0,
        "eval_total_tokens": 0,
        "openai_cost": 0.0,
    }

@app.route("/question", methods=["POST"])
def handle_question():
    data = request.json
    question = data["question"]
    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())
    answer_data = rag(question)

    result = {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
    }
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]
    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400
    return jsonify({
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})



if __name__ == "__main__":
    app.run(debug=True,port=9696)
"""
import uuid
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

#os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "phi3")
os.environ.setdefault("DATA_PATH", "../data/data.csv")

from flask import Flask, request, jsonify
from rag import rag

app = Flask(__name__)

@app.route("/question", methods=["POST"])
def handle_question():
    data = request.json
    question = data["question"]
    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())
    answer_data = rag(question)

    return jsonify({
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer_data["answer"],
    })

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data["conversation_id"]
    feedback = data["feedback"]
    if not conversation_id or feedback not in [1, -1]:
        return jsonify({"error": "Invalid input"}), 400
    return jsonify({
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=9696)