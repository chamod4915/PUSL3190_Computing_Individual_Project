import React, { useEffect, useState } from "react";
import axios from "axios";
import NavBar from "./navbar";
import { FiUser, FiMail, FiHash } from "react-icons/fi";

export default function ProfilePage() {
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("token");

    if (token) {
      axios
        .get("http://127.0.0.1:5000/user/profile", {
          headers: { Authorization: `Bearer ${token}` },
        })
        .then((res) => setUser(res.data))
        .catch((err) =>
          setError(err.response?.data?.error || "Failed to load profile")
        );
    } else {
      setError("Token missing. Please log in again.");
    }
  }, []);

  return (
    <>
      <NavBar />
      <div className="min-h-screen bg-zinc-900 text-white px-4 py-12 flex items-center justify-center">
        <div className="bg-zinc-800 border border-yellow-600 rounded-lg shadow-xl w-full max-w-md p-8">
          <h2 className="text-3xl font-bold text-yellow-400 mb-6 text-center">
            👤 Your Profile
          </h2>

          {error && (
            <p className="text-red-400 text-sm text-center mb-4">{error}</p>
          )}

          {user && (
            <div className="space-y-5">
              <div className="flex items-center gap-3 border-b border-yellow-700 pb-3">
                <FiUser className="text-yellow-400 text-xl" />
                <div>
                  <p className="text-sm text-gray-400">Name</p>
                  <p className="font-medium text-white">{user.name}</p>
                </div>
              </div>

              <div className="flex items-center gap-3 border-b border-yellow-700 pb-3">
                <FiHash className="text-yellow-400 text-xl" />
                <div>
                  <p className="text-sm text-gray-400">Username</p>
                  <p className="font-medium text-white">{user.username}</p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <FiMail className="text-yellow-400 text-xl" />
                <div>
                  <p className="text-sm text-gray-400">Email</p>
                  <p className="font-medium text-white">{user.email}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
