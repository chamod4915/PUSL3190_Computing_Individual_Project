import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { FiUser, FiLock, FiCheckCircle } from "react-icons/fi";
import NavBar from "./navbar";

export default function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

   try {
  const res = await axios.post("http://127.0.0.1:5000/admin/login", {
    username,
    password,
  });

  // Ensure the returned user object has a role, e.g., "admin"
  const userData = {
    ...res.data.user,
    role: "admin", // 🔑 Add this if your backend doesn't provide it
  };

  localStorage.setItem("user", JSON.stringify(userData));
  localStorage.setItem("token", res.data.token);

  setMessage("Login successful ✅");
  setTimeout(() => navigate("/dashboard"), 1000);
} catch (error) {
  setMessage(error.response?.data?.error || "Login failed");
}

  };

  return (
    <>
      <NavBar />
      <div className="min-h-screen flex bg-zinc-900 text-white">
        {/* Left Side Panel */}
        <div className="hidden md:flex w-1/2 bg-gradient-to-br from-yellow-700 to-yellow-900 items-center justify-center p-10">
          <div className="text-black text-center">
            <h1 className="text-4xl font-bold mb-4">Admin Panel</h1>
            <p className="text-lg">Restricted access for administrators only.</p>
          </div>
        </div>

        {/* Right Side Form */}
        <div className="flex-1 flex items-center justify-center px-4">
          <form
            onSubmit={handleLogin}
            className="bg-zinc-800 border border-yellow-600 p-8 rounded-lg shadow-lg w-full max-w-md"
          >
            <h2 className="text-2xl font-bold mb-6 text-center text-yellow-400">
              Admin Login
            </h2>

            {message && (
              <div
                className={`mb-4 flex items-center justify-center gap-2 text-sm text-center ${
                  message.includes("success") ? "text-green-400" : "text-red-400"
                }`}
              >
                {message.includes("success") && (
                  <FiCheckCircle className="text-green-400 text-lg" />
                )}
                <span>{message}</span>
              </div>
            )}

            <div className="mb-4">
              <label className="block mb-1 text-yellow-300">Username</label>
              <div className="flex items-center border border-yellow-500 bg-zinc-900 rounded px-3 py-2">
                <FiUser className="text-yellow-400 mr-2" />
                <input
                  type="text"
                  className="w-full bg-transparent outline-none text-white"
                  placeholder="admin123"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="mb-6">
              <label className="block mb-1 text-yellow-300">Password</label>
              <div className="flex items-center border border-yellow-500 bg-zinc-900 rounded px-3 py-2">
                <FiLock className="text-yellow-400 mr-2" />
                <input
                  type="password"
                  className="w-full bg-transparent outline-none text-white"
                  placeholder="********"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-yellow-500 text-black py-2 rounded font-semibold hover:bg-yellow-600 transition"
            >
              Log In
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
