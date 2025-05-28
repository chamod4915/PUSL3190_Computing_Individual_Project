import React, { useEffect, useState } from 'react';
import axios from 'axios';
import NavBar from './navbar';

function ManageUsers() {
  const [users, setUsers] = useState([]);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);

const token = localStorage.getItem("auth_token");

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const res = await axios.get('http://localhost:5000/users', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setUsers(res.data.users);
    } catch (err) {
      setMessage(err.response?.data?.error || '❌ Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <NavBar />
      <div className="min-h-screen bg-black p-6 text-white font-sans">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-yellow-400 text-center mb-6">Manage Users</h1>
          {message && <p className="text-center text-yellow-300 mb-4">{message}</p>}

          {loading ? (
            <p className="text-center text-gray-400">Loading users...</p>
          ) : users.length === 0 ? (
            <p className="text-center text-gray-400">No users found.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-yellow-600 text-black">
                    <th className="p-3 border border-yellow-700">ID</th>
                    <th className="p-3 border border-yellow-700">Name</th>
                    <th className="p-3 border border-yellow-700">Username</th>
                    <th className="p-3 border border-yellow-700">Email</th>
                    <th className="p-3 border border-yellow-700">Role</th>
                    <th className="p-3 border border-yellow-700">Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id} className="bg-zinc-900 hover:bg-zinc-800 transition">
                      <td className="p-3 border border-yellow-700">{user.id}</td>
                      <td className="p-3 border border-yellow-700">{user.name}</td>
                      <td className="p-3 border border-yellow-700">{user.username}</td>
                      <td className="p-3 border border-yellow-700">{user.email}</td>
                      <td className="p-3 border border-yellow-700 capitalize">{user.role}</td>
                      <td className="p-3 border border-yellow-700">{new Date(user.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default ManageUsers;
