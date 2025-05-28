from flask import Blueprint, request, jsonify

from service.book_service import save_book_data, get_all_books, get_book_by_id, update_book_by_id, delete_book_by_id

book_api = Blueprint('book_api', __name__)
@book_api.route('/upload-book', methods=['POST'])
def upload_book():
    try:
        name = request.form['name']
        author = request.form['author']
        summary = request.form['summary']
        description = request.form['description']  # NEW FIELD
        pdf_file = request.files['pdf']
        image_file = request.files['image']

        if not all([name, author, summary, description, pdf_file, image_file]):
            return jsonify({'error': 'All fields are required'}), 400

        result = save_book_data(name, author, summary, description, pdf_file, image_file)
        return jsonify(result), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@book_api.route('/books', methods=['GET'])
def books():
    try:
        books = get_all_books()
        return jsonify({"books": books}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@book_api.route('/book', methods=['GET'])
def book_by_id():
    book_id = request.args.get("id")

    if not book_id:
        return jsonify({"error": "Book ID is required"}), 400

    try:
        book = get_book_by_id(book_id)
        return jsonify({"book": book}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@book_api.route('/book/<book_id>', methods=['PUT'])
def update_book(book_id):
    try:
        name = request.form.get('name')
        author = request.form.get('author')
        summary = request.form.get('summary')
        description = request.form.get('description')
        pdf_file = request.files.get('pdf')      # optional
        image_file = request.files.get('image')  # optional

        if not all([name, author, summary, description]):
            return jsonify({'error': 'Missing required fields'}), 400

        result = update_book_by_id(
            book_id, name, author, summary, description, pdf_file, image_file
        )
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@book_api.route('/book/<book_id>', methods=['DELETE'])
def delete_book(book_id):
    try:
        result = delete_book_by_id(book_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
