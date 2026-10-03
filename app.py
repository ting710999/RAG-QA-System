from flask import Flask, request, jsonify, render_template
from rag_retriever import ask
import os  # 加上这一行

app = Flask(__name__)

# ===================== 首页（解决 / 404） =====================
@app.route("/")
def home():
    return render_template("chat.html")


# ===================== API =====================
@app.route("/ask", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "")

        print("\nFlask收到问题:", question)

        answer = ask(question)

        print("返回答案:", answer[:100])

        return jsonify({
            "question": question,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    print("服务启动: http://0.0.0.0:" + str(port))
    app.run(host="0.0.0.0", port=port)