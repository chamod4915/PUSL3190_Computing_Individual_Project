import os
import mimetypes

from flask import Flask
from flask_cors import CORS
from flask import send_file, abort

from controller.admin_controller import admin_bp
from controller.book_controller import book_api
from controller.chat_controller import chatbot_api
from controller.emotion_controller import emotion_blueprint
from controller.user_controller import user_api
from flask import send_from_directory

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit
# Serve uploaded files (images, PDFs)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.root_path, 'uploads', filename)

    if not os.path.exists(file_path):
        abort(404)

    # Guess the correct MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'application/octet-stream'  # fallback for unknown types

    return send_file(
        file_path,
        mimetype=mime_type,
        as_attachment=False,
        download_name=filename
    )

# Register blueprint
app.register_blueprint(book_api)
app.register_blueprint(user_api)
app.register_blueprint(chatbot_api)
app.register_blueprint(admin_bp)
app.register_blueprint(emotion_blueprint)

if __name__ == '__main__':
    app.run(debug=True)