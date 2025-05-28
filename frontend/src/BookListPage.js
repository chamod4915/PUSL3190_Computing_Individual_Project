import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import NavBar from "./navbar";

export default function BookListPage() {
  const [books, setBooks] = useState([]);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    axios
      .get("http://127.0.0.1:5000/books")
      .then((res) => setBooks(res.data.books))
      .catch((err) =>
        setError(err.response?.data?.error || "Failed to load books")
      );
  }, []);

  const handleBookClick = (id) => {
    navigate(`/books?id=${id}`);
  };

  return (
    <>
      <NavBar />

      <div className="min-h-screen bg-zinc-900 text-white px-4 py-10">
        <h1 className="text-3xl font-bold text-yellow-400 mb-8 text-center">
          📚 Explore Books
        </h1>

        {error && (
          <p className="text-center text-red-400 mb-6">{error}</p>
        )}

        <div className="grid gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 max-w-7xl mx-auto">
          {books.map((book) => (
            <div
              key={book.id}
              onClick={() => handleBookClick(book.id)}
              className="cursor-pointer border border-yellow-600 bg-zinc-800 rounded-lg overflow-hidden 
                         hover:shadow-2xl hover:scale-[1.02] transition-all duration-300 ease-in-out 
                         hover:border-yellow-400"
            >
              <img
                src={`http://127.0.0.1:5000/${book.image_path}`}
                alt={book.name}
                className="w-full h-52 object-cover"
              />
              <div className="p-4">
                <h2 className="text-xl font-semibold text-yellow-300 mb-1">{book.name}</h2>
                <p className="text-sm text-gray-300 mb-1">by {book.author}</p>
                <p className="text-sm text-gray-400 line-clamp-2">{book.summary}</p>
              </div>
            </div>
          ))}
        </div>

        {books.length === 0 && !error && (
          <p className="text-center text-gray-400 mt-10">No books available.</p>
        )}
      </div>
    </>
  );
}
