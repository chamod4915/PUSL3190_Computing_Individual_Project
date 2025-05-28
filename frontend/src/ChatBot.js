import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FiSend, FiUser, FiCpu } from "react-icons/fi";
import NavBar from "./navbar";

export default function ChatBot() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      sender: "user",
      text: input,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages((prev) => [...prev, userMessage]);
    setTyping(true);

    try {
      const res = await axios.post("http://127.0.0.1:5000/chat", { message: input });
      const botMessage = {
        sender: "bot",
        text: res.data.response,
        intent: res.data.intent,
        timestamp: new Date().toLocaleTimeString()
      };
      setTimeout(() => {
        setMessages((prev) => [...prev, botMessage]);
        setTyping(false);
      }, 1000); // simulate typing delay
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Server error. Please try again later.",
          intent: "error",
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
      setTyping(false);
    }

    setInput("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  return (
   <>
   <NavBar/>
    <div className="min-h-screen bg-zinc-900 text-white flex flex-col items-center p-4">
      <div className="w-full max-w-2xl bg-zinc-800 rounded-lg shadow-lg flex flex-col h-[90vh] overflow-hidden border border-yellow-500">
        <div className="bg-yellow-500 text-black font-bold text-xl text-center py-4 shadow">
          🧠 Mental Health Chatbot
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-zinc-900">
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className="flex items-end space-x-2 max-w-[80%]">
                {msg.sender === "bot" && <div className="text-yellow-400"><FiCpu /></div>}
                <div className={`rounded-xl px-4 py-2 shadow transition-all duration-200
                  ${msg.sender === "user" 
                    ? "bg-yellow-400 text-black rounded-br-none hover:shadow-yellow-500" 
                    : "bg-zinc-700 text-yellow-300 rounded-bl-none hover:shadow-zinc-600"}`}>
                  <div>{msg.text}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    {msg.timestamp} {msg.intent && msg.sender === "bot" && <span className="italic">({msg.intent})</span>}
                  </div>
                </div>
                {msg.sender === "user" && <div className="text-yellow-400"><FiUser /></div>}
              </div>
            </motion.div>
          ))}

          {typing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-yellow-300 italic text-sm"
            >
              Bot is typing...
            </motion.div>
          )}

          <div ref={bottomRef}></div>
        </div>

        <div className="flex border-t border-zinc-700 bg-zinc-800 p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 bg-zinc-700 text-white placeholder-gray-400 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400 transition"
          />
          <button
            onClick={sendMessage}
            className="ml-3 bg-yellow-500 hover:bg-yellow-400 text-black px-4 py-2 rounded-lg font-semibold shadow-lg transition"
          >
            <FiSend />
          </button>
        </div>
      </div>
    </div>
   </>
  );
}
