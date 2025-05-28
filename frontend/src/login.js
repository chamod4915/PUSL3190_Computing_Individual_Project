import { useState } from "react";
import axios from "axios";
import { FiUser, FiLock, FiCheckCircle } from "react-icons/fi";
import NavBar from "./navbar";
import { useNavigate } from "react-router-dom"; 

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
const navigate = useNavigate(); 

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage("");

    try {
      const res = await axios.post("http://127.0.0.1:5000/login", {
        username,
        password,
      });

      localStorage.setItem("user", JSON.stringify(res.data.user));
      localStorage.setItem("token", res.data.token);
      setMessage("Login successful");
       navigate("/"); 
    } catch (error) {
      setMessage(error.response?.data?.error || "Login failed");
    }
  };

  return (
    <>
      <NavBar />

      <div className="min-h-screen flex bg-zinc-900 text-white">
        {/* Left Panel */}
        <div className="hidden md:flex flex-col justify-center items-center w-1/2 bg-gradient-to-br from-yellow-600 to-yellow-700 p-10 text-black">
          <h1 className="text-4xl font-extrabold mb-4">Welcome Back</h1>
          <p className="text-lg text-center max-w-sm">
            Log in to access your digital reading dashboard.
          </p>
        </div>

        {/* Right Panel - Login Form */}
        <div className="flex-1 flex items-center justify-center px-4">
          <form
            onSubmit={handleLogin}
            className="bg-zinc-800 p-8 rounded-lg shadow-lg w-full max-w-md border border-yellow-600"
          >
            <h2 className="text-2xl font-bold mb-6 text-center text-yellow-400">
              Login
            </h2>

            {message && (
              <div
                className={`mb-4 flex items-center justify-center gap-2 text-sm ${
                  message.includes("success")
                    ? "text-green-400"
                    : "text-red-400"
                }`}
              >
                {message.includes("success") && (
                  <FiCheckCircle className="text-green-400 text-lg" />
                )}
                <span>{message}</span>
              </div>
            )}

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
              Login
            </button>

            <p className="mt-6 text-sm text-center text-gray-300">
              Don’t have an account?{" "}
              <a href="/register" className="text-yellow-400 hover:underline">
                Register now
              </a>
            </p>
          </form>
        </div>
      </div>
    </>
  );
}
