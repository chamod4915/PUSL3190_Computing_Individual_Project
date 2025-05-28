from flask import Blueprint, request, jsonify
from service.admin_service import register_admin, login_admin
import os

admin_bp = Blueprint('admin_bp', __name__)
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # Not used here directly, used in service

# === Admin Registration ===
@admin_bp.route('/admin/register', methods=['POST'])
def admin_register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    result, status = register_admin(username, password)
    return jsonify(result), status

# === Admin Login ===
@admin_bp.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    result, status = login_admin(username, password)
    return jsonify(result), status
