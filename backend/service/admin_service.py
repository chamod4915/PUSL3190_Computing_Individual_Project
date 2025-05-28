import bcrypt
import jwt
import datetime
from dbconnect import get_db_connection

SECRET_KEY = "your-secret-key"  # Replace with an environment variable in production


def register_admin(username, plain_password):
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM admins WHERE username = %s", (username,))
        if cursor.fetchone():
            return {"error": "Username already exists"}, 400

        cursor.execute(
            "INSERT INTO admins (username, password) VALUES (%s, %s)",
            (username, hashed_pw.decode('utf-8'))
        )
        conn.commit()
        return {"message": "Admin registered successfully"}, 201
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        cursor.close()
        conn.close()


def login_admin(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM admins WHERE username = %s", (username,))
        admin = cursor.fetchone()

        if not admin:
            return {"error": "Admin not found"}, 404

        if not bcrypt.checkpw(password.encode('utf-8'), admin['password'].encode('utf-8')):
            return {"error": "Incorrect password"}, 401

        payload = {
            "admin_id": admin['id'],
            "username": admin['username'],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }

        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        del admin['password']  # Don't return the password

        return {"admin": admin, "token": token}, 200
    except Exception as e:
        return {"error": str(e)}, 500
    finally:
        cursor.close()
        conn.close()
