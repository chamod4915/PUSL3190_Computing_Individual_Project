import React, { useRef, useState } from "react";
import axios from "axios";
import Webcam from "react-webcam";
import { useNavigate } from "react-router-dom";
import NavBar from "./navbar";

export default function EmotionBasedRecommendations() {
  const webcamRef = useRef(null);
  const [imageData, setImageData] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [books, setBooks] = useState([]);
  const [filteredBooks, setFilteredBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
const navigate = useNavigate();

  const captureImage = () => {
    const screenshot = webcamRef.current.getScreenshot();
    if (screenshot) {
      setImageData(screenshot);
      setEmotion("");
      setFilteredBooks([]);
      setMessage("");
    }
  };

  const handleEmotionCheck = async () => {
    if (!imageData) {
      setMessage("📸 Please capture an image first.");
      return;
    }

    setLoading(true);
    setMessage("🔍 Predicting emotion...");
    try {
      // Convert base64 to file
      const blob = await (await fetch(imageData)).blob();
      const formData = new FormData();
      formData.append("image", new File([blob], "webcam.jpg", { type: "image/jpeg" }));

      const emotionRes = await axios.post("http://localhost:5000/predict", formData);
      const detectedEmotion = emotionRes.data.emotion;
      setEmotion(detectedEmotion);
      setMessage(`✅ Emotion detected: ${detectedEmotion}`);

      const booksRes = await axios.get("http://localhost:5000/books");
      const recommended = booksRes.data.books.filter((book) =>
        book.description.toLowerCase().includes(detectedEmotion.toLowerCase())
      );
      setBooks(booksRes.data.books);
      setFilteredBooks(recommended);
    } catch (err) {
      setMessage("❌ Failed to predict emotion or fetch books.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
   <>
   <NavBar/>
    <div className="min-h-screen bg-black text-white p-6 font-sans">
      <h1 className="text-3xl font-bold text-yellow-400 mb-4">Emotion-Based Book Recommender</h1>

      <div className="flex flex-col md:flex-row gap-6 items-center mb-6">
        <div className="border border-yellow-500 rounded-lg overflow-hidden">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={320}
            height={240}
            className="rounded"
          />
        </div>

        <div className="flex flex-col gap-3 w-full md:w-auto">
          <button
            onClick={captureImage}
            className="bg-yellow-400 text-black font-semibold px-6 py-2 rounded hover:bg-yellow-300 transition"
          >
            📸 Capture Image
          </button>
          <button
            onClick={handleEmotionCheck}
            className="bg-green-500 text-white font-semibold px-6 py-2 rounded hover:bg-green-600 transition"
            disabled={loading}
          >
            {loading ? "Processing..." : "🎯 Get Book Recommendations"}
          </button>
        </div>

        {imageData && (
          <div className="border border-gray-600 rounded overflow-hidden mt-4 md:mt-0">
            <img src={imageData} alt="Captured" className="w-40 h-32 object-cover rounded" />
          </div>
        )}
      </div>

      {message && <p className="text-yellow-300 mb-6">{message}</p>}
{filteredBooks.length > 0 && (
  <>
    <h2 className="text-xl font-semibold text-yellow-300 mb-4">
      Recommended Books for: <span className="capitalize">{emotion}</span>
    </h2>
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
      {filteredBooks.map((book) => (
        <div
          key={book.id}
          className="bg-zinc-800 p-4 rounded-lg shadow border border-yellow-600 flex flex-col"
        >
          <img
            src={`http://localhost:5000/${book.image_path.replace("\\", "/")}`}
            alt={book.name}
            className="w-full h-48 object-cover rounded mb-3"
          />
          <h3 className="text-lg font-bold text-yellow-400">{book.name}</h3>
          <p className="text-sm text-gray-300">by {book.author}</p>
          <p className="text-sm text-gray-400 mt-2 flex-1">{book.description}</p>
          <button
            onClick={() => navigate(`/books?id=${book.id}`)}
            className="mt-4 bg-yellow-400 text-black font-semibold py-2 rounded hover:bg-yellow-300 transition"
          >
            View Details
          </button>
        </div>
      ))}
    </div>
  </>
)}

      {!loading && emotion && filteredBooks.length === 0 && (
        <p className="text-gray-400 mt-4">No books match the emotion "{emotion}".</p>
      )}
    </div>
   
   </>
  );
}
