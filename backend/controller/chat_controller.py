# controller.py

from flask import Blueprint, request, jsonify

from service.chat_service import get_response, predict_intent

chatbot_api = Blueprint('chatbot_api', __name__)

@chatbot_api.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input.strip():
        return jsonify({"error": "Empty message"}), 400

    intent = predict_intent(user_input)
    if not intent:
        return jsonify({"intent": "unknown", "response": "Sorry, I didn't understand that."})

    response = get_response(intent)
    return jsonify({
        "intent": intent,
        "response": response
    })
