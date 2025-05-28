import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import axios from "axios";
import NavBar from "./navbar";

export default function BookDetailsPage() {
  const [searchParams] = useSearchParams();
  const [book, setBook] = useState(null);
  const [error, setError] = useState("");

  const bookId = searchParams.get("id");

  useEffect(() => {
    if (bookId) {
      axios
        .get(`http://127.0.0.1:5000/book?id=${bookId}`)
        .then((res) => setBook(res.data.book))
        .catch((err) =>
          setError(err.response?.data?.error || "Failed to load book details")
        );
    }
  }, [bookId]);



  return (
    <>
      <NavBar />

      <div className="min-h-screen bg-zinc-900 text-white px-4 py-12">
        {error && (
          <p className="text-center text-red-400 text-lg">{error}</p>
        )}

        {!book && !error && (
          <p className="text-center text-yellow-300 text-lg">
            Loading book details...
          </p>
        )}

        {book && (
          <div className="max-w-5xl mx-auto">
            <div className="flex flex-col md:flex-row gap-10">
              <img
                src={`http://127.0.0.1:5000/${book.image_path.split("/").pop()}`}
                alt={book.name}
                className="w-full md:w-1/2 h-80 object-cover rounded-lg border border-yellow-600 shadow-lg"
              />

              <div className="flex-1">
                <h1 className="text-3xl font-bold text-yellow-400 mb-3">
                  {book.name}
                </h1>
                <h2 className="text-xl text-yellow-300 mb-2 italic">
                  by {book.author}
                </h2>
                <p className="text-gray-300 mb-4">{book.summary}</p>
                <p className="text-gray-400">{book.description}</p>

<a
  href={`/view?id=${book.id}`}
  className="mt-6 inline-block bg-yellow-500 hover:bg-yellow-600 text-black font-semibold px-6 py-2 rounded shadow transition"
>
Read
</a>


              </div>
            </div>
          </div>
        )}
      </div>

      {/* Call-to-Action Section */}
      <section className="py-20 bg-yellow-500 text-black px-6 md:px-12 text-center">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Want to Explore More Books?
          </h2>
          <p className="mb-8">
            Head back to the library and dive into new stories and knowledge.
          </p>
          <a
            href="/library"
            className="bg-black text-yellow-400 font-semibold px-8 py-4 rounded-lg shadow hover:bg-gray-800 transition"
          >
            Browse Library
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black text-yellow-200 py-12 px-6 md:px-12">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h4 className="font-bold mb-3">About LibroSphere</h4>
            <p className="text-sm text-gray-400">
              A platform to upload, read, and share books securely and beautifully.
            </p>
          </div>
          <div>
            <h4 className="font-bold mb-3">Quick Links</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="/privacy" className="hover:underline">Privacy Policy</a></li>
              <li><a href="/terms" className="hover:underline">Terms of Service</a></li>
              <li><a href="/support" className="hover:underline">Help Center</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-3">Contact</h4>
            <p className="text-sm text-gray-400">Email: support@librosphere.com</p>
            <p className="text-sm text-gray-400">Phone: +94 77 123 4567</p>
          </div>
        </div>
        <p className="text-center text-xs text-gray-500 mt-8">
          &copy; 2025 LibroSphere. All rights reserved.
        </p>
      </footer>
    </>
  );
}
