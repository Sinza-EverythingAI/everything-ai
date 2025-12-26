import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from orchestrator import orchestrate_question

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        mode = (data.get("mode") or "balanced").strip().lower()

        if not question:
            return jsonify({"error": True, "message": "No question provided."}), 400
        if mode not in {"balanced", "factual", "creative", "fast"}:
            mode = "balanced"

        result = orchestrate_question(question, mode)

        return jsonify({
            "error": False,
            "answer": result["final_answer"],
            "models": result["models"],
        }), 200

    except Exception as e:
        return jsonify({
            "error": True,
            "message": "Server error occurred while processing your question.",
            "details": str(e),
        }), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Everything AI backend is running."}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)