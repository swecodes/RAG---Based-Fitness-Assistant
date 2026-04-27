from flask import Flask, request, jsonify
from rag_test import rag   # or rag.py if you rename

app = Flask(__name__)

@app.route("/question", methods=["POST"])
def handle_question():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = rag(question)

    return jsonify({
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)