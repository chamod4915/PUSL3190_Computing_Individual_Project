import React, { useEffect, useState } from 'react';
import axios from 'axios';
import NavBar from './navbar';
import { FiEdit2, FiSave, FiX, FiTrash2 } from 'react-icons/fi';

function ManageBooks() {
  const [books, setBooks] = useState([]);
  const [editBookId, setEditBookId] = useState(null);
  const [deleteBookId, setDeleteBookId] = useState(null);
  const [form, setForm] = useState({
    name: '',
    author: '',
    summary: '',
    description: '',
  });
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = async () => {
    try {
      const res = await axios.get('http://localhost:5000/books');
      setBooks(res.data.books);
    } catch {
      setMessage('❌ Failed to load books');
    }
  };

  const startEdit = (book) => {
    setEditBookId(book.id);
    setForm({
      name: book.name,
      author: book.author,
      summary: book.summary,
      description: book.description,
    });
  };

  const cancelEdit = () => {
    setEditBookId(null);
    setForm({ name: '', author: '', summary: '', description: '' });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleUpdate = async () => {
    try {
      const data = new FormData();
      for (let key in form) data.append(key, form[key]);

      await axios.put(`http://localhost:5000/book/${editBookId}`, data);
      setMessage('✅ Book updated successfully');
      setEditBookId(null);
      fetchBooks();
    } catch (err) {
      setMessage(err.response?.data?.error || '❌ Update failed');
    }
  };

  const confirmDelete = async () => {
    try {
      await axios.delete(`http://localhost:5000/book/${deleteBookId}`);
      setMessage('✅ Book deleted successfully');
      setDeleteBookId(null);
      fetchBooks();
    } catch (err) {
      setMessage(err.response?.data?.error || '❌ Delete failed');
    }
  };

  return (
    <>
      <NavBar />
      <div className="min-h-screen bg-black p-4 text-white font-sans">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-3xl text-yellow-400 font-bold mb-6 text-center">Manage Books</h1>
          {message && <p className="text-center mb-4 text-yellow-300">{message}</p>}

          {books.length === 0 ? (
            <p className="text-center text-gray-500">No books found.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {books.map((book) => (
                <div
                  key={book.id}
                  className="bg-zinc-900 p-5 rounded-xl border border-yellow-700 hover:shadow-xl transition-all relative"
                >
                  {editBookId === book.id ? (
                    <>
                      <input
                        type="text"
                        name="name"
                        value={form.name}
                        onChange={handleChange}
                        className="w-full p-2 mb-3 bg-zinc-800 border border-yellow-500 text-white rounded focus:ring-2 focus:ring-yellow-400"
                        placeholder="Book Name"
                      />
                      <input
                        type="text"
                        name="author"
                        value={form.author}
                        onChange={handleChange}
                        className="w-full p-2 mb-3 bg-zinc-800 border border-yellow-500 text-white rounded focus:ring-2 focus:ring-yellow-400"
                        placeholder="Author"
                      />
                      <textarea
                        name="summary"
                        value={form.summary}
                        onChange={handleChange}
                        className="w-full p-2 mb-3 bg-zinc-800 border border-yellow-500 text-white rounded focus:ring-2 focus:ring-yellow-400"
                        placeholder="Summary"
                      />
                      <textarea
                        name="description"
                        value={form.description}
                        onChange={handleChange}
                        className="w-full p-2 mb-4 bg-zinc-800 border border-yellow-500 text-white rounded focus:ring-2 focus:ring-yellow-400"
                        placeholder="Description"
                      />
                      <div className="flex justify-end gap-3">
                        <button
                          onClick={handleUpdate}
                          className="flex items-center gap-1 bg-green-500 hover:bg-green-600 text-black font-bold py-1 px-4 rounded"
                        >
                          <FiSave /> Save
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="flex items-center gap-1 bg-gray-600 hover:bg-gray-500 text-white font-bold py-1 px-4 rounded"
                        >
                          <FiX /> Cancel
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <h2 className="text-xl text-yellow-300 font-semibold mb-1">{book.name}</h2>
                      <p className="text-sm text-gray-300 mb-1">Author: {book.author}</p>
                      <p className="text-sm text-gray-400 mb-1">Summary: {book.summary}</p>
                      <p className="text-sm text-gray-400 mb-2">Description: {book.description}</p>
                      <div className="flex justify-between">
                        <button
                          onClick={() => startEdit(book)}
                          className="flex items-center gap-1 bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-1 px-4 rounded"
                        >
                          <FiEdit2 /> Edit
                        </button>
                        <button
                          onClick={() => setDeleteBookId(book.id)}
                          className="flex items-center gap-1 bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-4 rounded"
                        >
                          <FiTrash2 /> Delete
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      {deleteBookId && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-zinc-800 border border-yellow-600 p-6 rounded-xl max-w-sm w-full text-center shadow-xl">
            <h2 className="text-yellow-300 text-lg font-semibold mb-4">Confirm Deletion</h2>
            <p className="text-white mb-6">Are you sure you want to delete this book?</p>
            <div className="flex justify-center gap-4">
              <button
                onClick={confirmDelete}
                className="bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-md font-bold"
              >
                Yes, Delete
              </button>
              <button
                onClick={() => setDeleteBookId(null)}
                className="bg-gray-500 hover:bg-gray-600 text-white px-5 py-2 rounded-md font-bold"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default ManageBooks;
