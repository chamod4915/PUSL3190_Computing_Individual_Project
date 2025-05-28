import { BrowserRouter, Routes, Route } from "react-router-dom";
import Upload from "./upload";
import Home from "./home";
import RegisterPage from "./register";
import LoginPage from "./login";
import BookListPage from "./BookListPage";
import BookDetailsPage from "./BookDetailsPage";
import ViewBookPage from "./ViewBookPage";
import ChatBot from "./ChatBot";
import ProfilePage from "./profile";
import AdminLogin from "./AdminLogin";
import ManageBooks from "./ManageBooks";
import ManageUsers from "./ManageUsers";
import EmotionBasedRecommendations from "./EmotionBasedRecommendations";


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/login" element={<LoginPage />} />
<Route path="/library" element={<BookListPage />} />
<Route path="/books" element={<BookDetailsPage />} />
<Route path="/view" element={<ViewBookPage />} />
<Route path="/chat" element={<ChatBot />} />

<Route path="/profile" element={<ProfilePage />} />
<Route path="/admin/login" element={<AdminLogin />} />
<Route path="/manage-books" element={<ManageBooks />} />
<Route path="/manage-users" element={<ManageUsers />} />
<Route path="/recommand" element={<EmotionBasedRecommendations />} />


      </Routes>
    </BrowserRouter>
  );
}

export default App;
