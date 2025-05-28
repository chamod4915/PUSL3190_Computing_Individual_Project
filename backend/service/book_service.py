import os
import uuid
from dbconnect import get_db_connection

UPLOAD_FOLDER = 'uploads'
def save_book_data(name, author, summary, description, pdf_file, image_file):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    pdf_filename = f"{uuid.uuid4().hex}_{pdf_file.filename}"
    image_filename = f"{uuid.uuid4().hex}_{image_file.filename}"

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    pdf_file.save(pdf_path)
    image_file.save(image_path)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO books (name, author, summary, description, pdf_path, image_path)
        VALUES (%s, %s, %s, %s, %s, %s)
    ''', (name, author, summary, description, pdf_path, image_path))
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "Book uploaded successfully"}

def get_all_books():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM books ORDER BY created_at DESC")
    books = cursor.fetchall()
    cursor.close()
    conn.close()
    return books


def get_book_by_id(book_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM books WHERE id = %s", (book_id,))
    book = cursor.fetchone()
    cursor.close()
    conn.close()

    if not book:
        raise Exception("Book not found")

    return book

def update_book_by_id(book_id, name, author, summary, description, pdf_file=None, image_file=None):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # First fetch existing data
    cursor.execute("SELECT * FROM books WHERE id = %s", (book_id,))
    book = cursor.fetchone()

    if not book:
        cursor.close()
        conn.close()
        raise Exception("Book not found")

    pdf_path = book['pdf_path']
    image_path = book['image_path']

    # Replace PDF if a new one is uploaded
    if pdf_file:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        pdf_filename = f"{uuid.uuid4().hex}_{pdf_file.filename}"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        pdf_file.save(pdf_path)

    # Replace image if a new one is uploaded
    if image_file:
        if os.path.exists(image_path):
            os.remove(image_path)
        image_filename = f"{uuid.uuid4().hex}_{image_file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        image_file.save(image_path)

    cursor.execute('''
        UPDATE books
        SET name = %s, author = %s, summary = %s, description = %s,
            pdf_path = %s, image_path = %s
        WHERE id = %s
    ''', (name, author, summary, description, pdf_path, image_path, book_id))

    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "Book updated successfully"}


def delete_book_by_id(book_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Get file paths first
    cursor.execute("SELECT pdf_path, image_path FROM books WHERE id = %s", (book_id,))
    book = cursor.fetchone()

    if not book:
        cursor.close()
        conn.close()
        raise Exception("Book not found")

    # Delete files if they exist
    if os.path.exists(book['pdf_path']):
        os.remove(book['pdf_path'])
    if os.path.exists(book['image_path']):
        os.remove(book['image_path'])

    # Delete DB record
    cursor.execute("DELETE FROM books WHERE id = %s", (book_id,))
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": "Book deleted successfully"}
