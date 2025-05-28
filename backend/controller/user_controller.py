from flask import Blueprint, request, jsonify
from functools import wraps
import jwt
import os

from service.user_service import register_user, login_user, get_all_users

user_api = Blueprint('user_api', __name__)

# === Secret key (in real apps, keep in env variable or config) ===
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # replace 'your-secret-key'

# === Decorator to verify JWT token ===
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = token.replace("Bearer ", "")
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return f(decoded, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
    return decorated

# === Register Route ===
@user_api.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not all([name, email, username, password]):
        return jsonify({"error": "All fields are required"}), 400

    try:
        result = register_user(name, email, username, password)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Login Route ===
@user_api.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"error": "Username and password required"}), 400

    try:
        result = login_user(username, password)
        return jsonify({
            "message": "Login successful",
            "user": result["user"],
            "token": result["token"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 401

# === Get User Profile Route ===
@user_api.route('/user/profile', methods=['GET'])
@token_required
def get_user_profile(decoded_user):
    return jsonify({
        "name": decoded_user.get("name"),
        "username": decoded_user.get("username"),
        "email": decoded_user.get("email")
    }), 200

@user_api.route('/users', methods=['GET'])
def fetch_all_users():
    try:
        users = get_all_users()
        return jsonify({"users": users}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
