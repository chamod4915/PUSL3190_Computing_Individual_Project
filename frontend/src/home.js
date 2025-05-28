import React from "react";
import { Link } from "react-router-dom";
import {
  FiBookOpen,
  FiUpload,
  FiUserCheck,
  FiSearch,
  FiHeart,
  FiHelpCircle,
} from "react-icons/fi";
import NavBar from "./navbar";

export default function HomePage() {
  return (
    <>
    <NavBar/>
      <div className="flex flex-col min-h-screen bg-black text-white">
        {/* Hero Section */}
        <section className="relative bg-gradient-to-br from-yellow-900 via-black to-black py-32 px-6 md:px-12">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-extrabold mb-6 text-yellow-400">
              Welcome to LibroSphere
            </h1>
            <p className="text-lg md:text-xl mb-8 text-gray-200">
              Dive into the dreamland of books – discover, read, and share your favorite titles.
            </p>
            <div className="space-x-4">
              <Link
                to="/upload"
                className="bg-yellow-400 text-black font-semibold px-6 py-3 rounded-lg shadow hover:bg-yellow-300 transition"
              >
                Upload a Book
              </Link>
              <Link
                to="/library"
                className="border border-yellow-400 text-yellow-300 font-semibold px-6 py-3 rounded-lg hover:bg-yellow-400 hover:text-black transition"
              >
                Browse Library
              </Link>
            </div>
          </div>

          {/* Floating Circles */}
          <div className="absolute top-10 left-10 w-24 h-24 bg-yellow-200 opacity-10 rounded-full animate-ping" />
          <div className="absolute bottom-10 right-10 w-32 h-32 bg-yellow-300 opacity-10 rounded-full animate-ping" />
        </section>

        {/* Features */}
        <section className="py-20 px-6 md:px-12 bg-zinc-900">
          <div className="max-w-5xl mx-auto text-center mb-12">
            <h2 className="text-3xl font-bold mb-4 text-yellow-300">Core Features</h2>
            <p className="text-gray-400">
              Discover the benefits of our digital reading platform.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto text-gray-300">
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiBookOpen className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">Vast Digital Library</h3>
              <p>
                Access a growing collection of fiction, research, textbooks, and more.
              </p>
            </div>
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiUpload className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">Upload & Share</h3>
              <p>
                Contribute to the community by uploading and reviewing books.
              </p>
            </div>
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiSearch className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">Smart Search</h3>
              <p>
                Search books by name, author, or summary using advanced filters.
              </p>
            </div>
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiUserCheck className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">User Accounts</h3>
              <p>
                Save favorites, manage uploads, and track reading history.
              </p>
            </div>
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiHeart className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">Favorites & Likes</h3>
              <p>
                Show appreciation for books and keep a list of personal favorites.
              </p>
            </div>
            <div className="p-6 border border-zinc-700 rounded-lg hover:shadow-xl transition">
              <FiHelpCircle className="text-yellow-400 text-4xl mb-4" />
              <h3 className="text-xl font-semibold mb-2">24/7 Support</h3>
              <p>
                Reach out anytime through our support team or community forums.
              </p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="py-20 bg-yellow-500 text-black px-6 md:px-12 text-center">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Join the Reading Revolution
            </h2>
            <p className="mb-8">
              Be a part of a thriving community. Register now and start exploring!
            </p>
            <Link
              to="/register"
              className="bg-black text-yellow-400 font-semibold px-8 py-4 rounded-lg shadow hover:bg-gray-800 transition"
            >
              Create an Account
            </Link>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-black text-yellow-200 py-12 px-6 md:px-12 mt-auto">
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h4 className="font-bold mb-3">About LibroSphere</h4>
              <p className="text-sm text-gray-400">
                A platform to upload, read, and share books securely and beautifully.
              </p>
            </div>
            <div>
              <h4 className="font-bold mb-3">Links</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link to="/privacy" className="hover:underline">Privacy Policy</Link></li>
                <li><Link to="/terms" className="hover:underline">Terms of Service</Link></li>
                <li><Link to="/support" className="hover:underline">Help Center</Link></li>
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
      </div>
    </>
  );
}
