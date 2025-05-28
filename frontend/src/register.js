import { useState } from "react";
import axios from "axios";
import { FiUser, FiMail, FiLock, FiCheckCircle } from "react-icons/fi";
import NavBar from "./navbar";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.post("http://127.0.0.1:5000/register", {
        name,
        username,
        email,
        password,
      });

      setMessage(res.data.message);
    } catch (error) {
      setMessage(error.response?.data?.error || "Registration failed");
    }
  };

  return (
    <>
      <NavBar />

      <div className="min-h-screen flex bg-zinc-900 text-white">
        {/* Left Panel */}
        <div className="hidden md:flex flex-col justify-center items-center w-1/2 bg-gradient-to-br from-yellow-600 to-yellow-700 p-10 text-black">
          <h1 className="text-4xl font-extrabold mb-4">Join LibroSphere</h1>
          <p className="text-lg text-center max-w-sm">
            Explore, upload, and enjoy the world of digital reading.
          </p>
        </div>

        {/* Right Panel - Form */}
        <div className="flex-1 flex items-center justify-center px-4">
          <form
            onSubmit={handleRegister}
            className="bg-zinc-800 p-8 rounded-lg shadow-lg w-full max-w-md border border-yellow-600"
          >
            <h2 className="text-2xl font-bold mb-6 text-center text-yellow-400">
              Register
            </h2>

       {message && (
  <div
    className={`mb-4 flex items-center justify-center gap-2 text-sm text-center ${
      message.includes("success") ? "text-green-400" : "text-red-400"
    }`}
  >
    {message.includes("success") && <FiCheckCircle className="text-green-400 text-lg" />}
    <span>{message}</span>
  </div>
)}


            {/* Name */}
            <div className="mb-4">
              <label className="block mb-1 text-yellow-300">Full Name</label>
              <div className="flex items-center border border-yellow-500 bg-zinc-900 rounded px-3 py-2">
                <FiUser className="text-yellow-400 mr-2" />
                <input
                  type="text"
                  className="w-full bg-transparent outline-none text-white"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Username */}
            <div className="mb-4">
              <label className="block mb-1 text-yellow-300">Username</label>
              <div className="flex items-center border border-yellow-500 bg-zinc-900 rounded px-3 py-2">
                <FiUser className="text-yellow-400 mr-2" />
                <input
                  type="text"
                  className="w-full bg-transparent outline-none text-white"
                  placeholder="john123"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div className="mb-4">
              <label className="block mb-1 text-yellow-300">Email</label>
              <div className="flex items-center border border-yellow-500 bg-zinc-900 rounded px-3 py-2">
                <FiMail className="text-yellow-400 mr-2" />
                <input
                  type="email"
                  className="w-full bg-transparent outline-none text-white"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Password */}
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
              Register
            </button>

            <p className="mt-6 text-sm text-center text-gray-300">
              Already have an account?{" "}
              <a href="/login" className="text-yellow-400 hover:underline">
                Sign in
              </a>
            </p>
          </form>
        </div>
      </div>
    </>
  );
}
