import React, { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import axios from "axios";
import NavBar from "./navbar";

export default function ViewBookPage() {
  const [searchParams] = useSearchParams();
  const [book, setBook] = useState(null);
  const [error, setError] = useState("");
  const bookId = searchParams.get("id");

  const synthRef = useRef(window.speechSynthesis);

  useEffect(() => {
    if (bookId) {
      axios
        .get(`http://127.0.0.1:5000/book?id=${bookId}`)
        .then((res) => setBook(res.data.book))
        .catch((err) =>
          setError(err.response?.data?.error || "Failed to load book")
        );
    }
  }, [bookId]);

  const handleSpeakSummary = () => {
    if (book?.summary) {
      const utterance = new SpeechSynthesisUtterance(book.summary);
      utterance.lang = "en-US";
      synthRef.current.cancel(); // stop any existing speech
      synthRef.current.speak(utterance);
    }
  };

  return (
    <>
      <NavBar />

      <div className="min-h-screen bg-zinc-900 text-white px-4 py-10">
        {error && <p className="text-center text-red-400 text-lg">{error}</p>}

        {!book && !error && (
          <p className="text-center text-yellow-300 text-lg">Loading...</p>
        )}

        {book && (
          <div className="max-w-6xl mx-auto">
            {/* Book Title */}
            <h1 className="text-4xl font-extrabold text-yellow-400 mb-2 text-center">
              {book.name}
            </h1>

            <p className="text-center text-gray-400 text-lg mb-6">
              📘 By {book.author}
            </p>

            {/* Summary & Read Button */}
            <div className="bg-zinc-800 border border-yellow-700 rounded p-6 mb-8 shadow-md">
              <h2 className="text-2xl font-bold text-yellow-300 mb-2">
                Book Summary
              </h2>
              <p className="text-gray-300 leading-relaxed">{book.summary}</p>
              <button
                onClick={handleSpeakSummary}
                className="mt-4 px-4 py-2 bg-yellow-500 text-black rounded hover:bg-yellow-600 transition"
              >
                🔊 Read Summary
              </button>
            </div>

            {/* PDF Viewer */}
            <div className="border border-yellow-600 rounded-lg overflow-hidden shadow-lg">
              <iframe
                src={`http://127.0.0.1:5000/${book.pdf_path.split("/").pop()}`}
                title="PDF Viewer"
                width="100%"
                height="800px"
                className="bg-white"
              ></iframe>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
