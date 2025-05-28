import React, { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import {
  FiMenu, FiX, FiUser, FiLogOut, FiHome, FiUpload,
  FiBookOpen, FiMessageCircle, FiUsers, FiGrid, FiSettings
} from "react-icons/fi";

export default function NavBar() {
  const [user, setUser] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const raw = localStorage.getItem("user");
    if (raw && raw !== "undefined") {
      try {
        setUser(JSON.parse(raw));
      } catch (err) {
        console.warn("Invalid JSON in localStorage ‘user’:", err);
        localStorage.removeItem("user");
        setUser(null);
      }
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    setUser(null);
    navigate("/");
  };

  useEffect(() => {
    setMenuOpen(false);
  }, [location]);

  const linkClasses = (path) =>
    `flex items-center px-3 py-2 rounded-md transition-colors ${
      location.pathname === path
        ? "bg-yellow-500 text-black"
        : "text-yellow-300 hover:bg-yellow-600 hover:text-white"
    }`;

  const isAdmin = user?.role === "admin";
  const isNormal = user && !isAdmin;

  return (
    <nav className="sticky top-0 bg-zinc-900 shadow-lg z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          <Link to="/" className="flex items-center text-yellow-400 text-2xl font-bold">
            <FiBookOpen className="mr-2" /> LibroSphere
          </Link>

          {/* Desktop */}
          <div className="hidden md:flex md:items-center md:space-x-4">
            <Link to="/" className={linkClasses("/")}>
              <FiHome className="mr-1" /> Home
            </Link>

            {!user && (
              <>
                <Link to="/login" className={linkClasses("/login")}>
                  <FiUser className="mr-1" /> Login
                </Link>
                <Link to="/register" className={linkClasses("/register")}>
                  <FiUser className="mr-1" /> Register
                </Link>
              </>
            )}

            {isNormal && (
              <>
                <Link to="/chat" className={linkClasses("/chat")}>
                  <FiMessageCircle className="mr-1" /> Chat
                </Link>
                    <Link to="/recommand" className={linkClasses("/recommand")}>
                  <FiMessageCircle className="mr-1" /> Recommended Books
                </Link>
                <Link to="/profile" className={linkClasses("/profile")}>
                  <FiSettings className="mr-1" /> Profile
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center px-3 py-2 text-red-500 hover:bg-red-600 hover:text-white rounded-md transition-colors"
                >
                  <FiLogOut className="mr-1" /> Logout
                </button>
              </>
            )}

            {isAdmin && (
              <>
                
                <Link to="/upload" className={linkClasses("/upload")}>
                  <FiUpload className="mr-1" /> Upload
                </Link>
                <Link to="/manage-books" className={linkClasses("/manage-books")}>
                  <FiBookOpen className="mr-1" /> Manage Books
                </Link>
                <Link to="/manage-users" className={linkClasses("/manage-users")}>
                  <FiUsers className="mr-1" /> Manage Users
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center px-3 py-2 text-red-500 hover:bg-red-600 hover:text-white rounded-md transition-colors"
                >
                  <FiLogOut className="mr-1" /> Logout
                </button>
              </>
            )}
          </div>

          {/* Mobile Toggle */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="p-2 rounded-md text-yellow-300 hover:bg-yellow-600 hover:text-white"
            >
              {menuOpen ? <FiX size={24} /> : <FiMenu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="md:hidden bg-zinc-900 border-t border-yellow-700">
          <div className="px-2 pt-2 pb-3 space-y-1">
            <Link to="/" className={linkClasses("/")}>
              <FiHome className="mr-1" /> Home
            </Link>

            {!user && (
              <>
                <Link to="/login" className={linkClasses("/login")}>
                  <FiUser className="mr-1" /> Login
                </Link>
                <Link to="/register" className={linkClasses("/register")}>
                  <FiUser className="mr-1" /> Register
                </Link>
              </>
            )}

            {isNormal && (
              <>
                <Link to="/chat" className={linkClasses("/chat")}>
                  <FiMessageCircle className="mr-1" /> Chat
                </Link>
                 <Link to="/recommand" className={linkClasses("/recommand")}>
                  <FiMessageCircle className="mr-1" /> Recommended Books
                </Link>
                <Link to="/profile" className={linkClasses("/profile")}>
                  <FiSettings className="mr-1" /> Profile
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center px-3 py-2 text-red-500 hover:bg-red-600 hover:text-white rounded-md w-full text-left"
                >
                  <FiLogOut className="mr-1" /> Logout
                </button>
              </>
            )}

            {isAdmin && (
              <>
            
                <Link to="/upload" className={linkClasses("/upload")}>
                  <FiUpload className="mr-1" /> Upload
                </Link>
                <Link to="/manage-books" className={linkClasses("/manage-books")}>
                  <FiBookOpen className="mr-1" /> Manage Books
                </Link>
                <Link to="/manage-users" className={linkClasses("/manage-users")}>
                  <FiUsers className="mr-1" /> Manage Users
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center px-3 py-2 text-red-500 hover:bg-red-600 hover:text-white rounded-md w-full text-left"
                >
                  <FiLogOut className="mr-1" /> Logout
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
