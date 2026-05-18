import os
import uuid
from flask import Flask, request, jsonify

from db import save_conversation, save_feedback, get_recent_conversations, get_feedback_stats
from rag import rag

app = Flask(__name__)


@app.route("/question", methods=["POST"])
def handle_question():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    conversation_id = str(uuid.uuid4())

    answer_data = rag(question)

    save_conversation(conversation_id, question, answer_data)

    return jsonify({
        "conversation_id": conversation_id,
        "question":        question,
        "answer":          answer_data["answer"],
    })


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.json
    conversation_id = data.get("conversation_id")
    feedback        = data.get("feedback")

    if not conversation_id or feedback not in (1, -1):
        return jsonify({"error": "Invalid input. Provide conversation_id and feedback (1 or -1)"}), 400

    save_feedback(conversation_id, feedback)

    return jsonify({
        "message": f"Feedback received for conversation {conversation_id}: {feedback}"
    })


@app.route("/recent", methods=["GET"])
def recent_conversations():
    limit     = request.args.get("limit", 5, type=int)
    relevance = request.args.get("relevance")
    rows = get_recent_conversations(limit=limit, relevance=relevance)
    return jsonify([dict(r) for r in rows])


@app.route("/stats", methods=["GET"])
def feedback_stats():
    stats = get_feedback_stats()
    return jsonify(dict(stats))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
