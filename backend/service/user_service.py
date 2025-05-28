import bcrypt
from dbconnect import get_db_connection
import datetime
import jwt


SECRET_KEY = "your-secret-key"  # replace with env-secured value in production

def register_user(name, email, username, plain_password):
    # Hash the password
    hashed_pw = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

    # Save user to database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email, username, password) VALUES (%s, %s, %s, %s)",
        (name, email, username, hashed_pw.decode('utf-8'))
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "User registered successfully"}



def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        raise Exception("User not found")

    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        raise Exception("Incorrect password")

    # Create JWT token
    payload = {
        "user_id": user['id'],
        "username": user['username'],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    # Remove password before returning
    del user['password']
    return {"user": user, "token": token}

def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, email, username, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return users