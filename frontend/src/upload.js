import React, { useState } from 'react';
import axios from 'axios';
import NavBar from './navbar';

function Upload() {
  const [form, setForm] = useState({
    name: '',
    author: '',
    summary: '',
    description: '',
    pdf: null,
    image: null,
  });

  const [message, setMessage] = useState('');

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setForm({ ...form, [name]: files ? files[0] : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    for (let key in form) data.append(key, form[key]);

    try {
      const res = await axios.post('http://localhost:5000/upload-book', data);
      setMessage(res.data.message);
    } catch (err) {
      setMessage(err.response?.data?.error || 'Upload failed');
    }
  };

  return (
    <>
    <NavBar/>
    <div className="min-h-screen bg-black text-white flex justify-center items-center p-4 font-sans">
      <form
        onSubmit={handleSubmit}
        className="bg-zinc-900 p-8 rounded-xl shadow-lg w-full max-w-2xl border border-yellow-600"
      >
        <h1 className="text-3xl font-bold text-yellow-400 mb-6 text-center">Upload a Book</h1>

        <div className="space-y-4">
          <input
            type="text"
            name="name"
            placeholder="Book Name"
            onChange={handleChange}
            className="w-full p-3 bg-zinc-800 text-white border border-yellow-500 rounded-md focus:outline-none focus:ring-2 focus:ring-yellow-400"
            required
          />

          <input
            type="text"
            name="author"
            placeholder="Author"
            onChange={handleChange}
            className="w-full p-3 bg-zinc-800 text-white border border-yellow-500 rounded-md focus:outline-none focus:ring-2 focus:ring-yellow-400"
            required
          />

          <textarea
            name="summary"
            placeholder="Summary"
            onChange={handleChange}
            className="w-full p-3 bg-zinc-800 text-white border border-yellow-500 rounded-md h-24 focus:outline-none focus:ring-2 focus:ring-yellow-400"
            required
          />

          <textarea
            name="description"
            placeholder="Description"
            onChange={handleChange}
            className="w-full p-3 bg-zinc-800 text-white border border-yellow-500 rounded-md h-24 focus:outline-none focus:ring-2 focus:ring-yellow-400"
            required
          />

          <div>
            <label className="block text-sm font-medium text-yellow-300 mb-1">PDF File</label>
            <input
              type="file"
              name="pdf"
              accept=".pdf"
              onChange={handleChange}
              className="w-full bg-zinc-800 text-white border border-yellow-500 p-2 rounded-md"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-yellow-300 mb-1">Cover Image</label>
            <input
              type="file"
              name="image"
              accept="image/*"
              onChange={handleChange}
              className="w-full bg-zinc-800 text-white border border-yellow-500 p-2 rounded-md"
              required
            />
          </div>
        </div>

        <button
          type="submit"
          className="mt-6 w-full bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-2 rounded-md transition"
        >
          Upload Book
        </button>

        {message && (
          <p className="mt-4 text-center text-yellow-400 font-medium">{message}</p>
        )}
      </form>
    </div>
    </>
  );
}

export default Upload;
