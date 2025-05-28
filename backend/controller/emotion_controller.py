from flask import Blueprint, request, jsonify
from service.emotion_service import predict_emotion

emotion_blueprint = Blueprint('emotion_blueprint', __name__)
@emotion_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Ensure image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']

        # ✅ Call the emotion prediction service
        emotion = predict_emotion(image_file)

        return jsonify({
            "emotion": emotion
        }), 200

    except Exception as e:
        print("❌ Error during image prediction:", str(e))
        return jsonify({"error": str(e)}), 500