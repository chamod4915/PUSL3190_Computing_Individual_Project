# 📘 LibroSphere: AI-Enhanced E-Library System

LibroSphere is an intelligent, accessible, and AI-driven digital library platform that transforms the traditional e-reading experience. It offers personalized book recommendations based on user mood, text-to-speech for auditory access, AI-generated summaries, and chatbot assistants that simulate literary characters — all through a modern full-stack web application.

---

## 🌟 Features

- 📖 **Mood Based Book Recommendations** – Get suggestions based on user mood and behavior.
- 📝 **AI Book Summarization** – Quickly grasp book content through concise NLP-generated summaries.
- 🗣️ **Text-to-Speech (TTS)** – Audio narration of summaries with multilingual support (English, Sinhala, Tamil).
- 🤖 **AI Chatbot Characters** – Interact with simulated book characters or ask questions.
- 🔍 **Semantic Smart Search** – Search books using keywords, context, or emotional cues.
- 🧑‍🦯 **Accessible Design** – Complies with WCAG 2.1 for visually impaired and neurodivergent users.
- 🔐 **Secure Role-Based Access** – Admin, Librarian, Reader, and Guest roles with JWT authentication.

---

## 🛠️ Required Software

Ensure the following software is installed on your system:

1. **[XAMPP](https://www.apachefriends.org/index.html)** – Apache & MySQL server.
2. **[Python 3.12](https://www.python.org/downloads/release/python-3120/)** – For running the AI backend.
3. **[Node.js](https://nodejs.org/)** – For React frontend.
4. **[PyCharm IDE](https://www.jetbrains.com/pycharm/)** – Backend development.
5. **[Visual Studio Code](https://code.visualstudio.com/)** – Frontend development.

---

## 📂 Project Structure
LibroSphere/
├── backend/ # Python backend with AI services
│ └── app.py
│ └── summarizer/
│ └── chatbot/
│ └── requirements.txt
│
├── frontend/ # React frontend
│ └── src/
│ └── public/
│ └── package.json
│
├── database/ # SQL file for phpMyAdmin import
│ └── librosphere_db.sql


---

## ▶️ How to Start the App (Step-by-Step Guide)

### 1. Start Apache & MySQL Using XAMPP
- Open **XAMPP Control Panel**.
- Start **Apache** and **MySQL**.

### 2. Create Database in phpMyAdmin
- Go to `http://localhost/phpmyadmin`.
- Create a new database (e.g., `librosphere_db`).
- Import the `librosphere_db.sql` file into the database.

### 3. Run the Backend in PyCharm
- Open the `backend` folder in **PyCharm**.
- Install dependencies:
  ```bash
  pip install -r requirements.txt

### 4. Run the Frontend in VS Code
- Open the frontend folder in Visual Studio Code.
- Open a new terminal and run:
- cd frontend
- npm install
- npm start

## 🔧 Tech Stack

## Frontend
- React.js
- Tailwind CSS
- React Router
- Axios

## Backend
- Node.js with Express.js
- MySQL
- JWT (Authentication)

## AI & NLP
- Python 3.12
- Hugging Face Transformers (BART, GPT)
- OpenAI GPT-3.5-turbo (for chatbot)
- Google Cloud Text-to-Speech API
- TensorFlow, PyTorch

### 📈 Future Enhancements
- 📱 Develop a mobile app (React Native / Flutter)
- 🌍 Add full multilingual UI with i18n support
- 📶 Enable offline book access and TTS playback
- 🎮 Gamification (book challenges, reading streaks)
- 🧠 AI analytics dashboard for admins

### 👨‍💻 Author
- H R C T Kumara
- BSc (Hons) Software Engineering – Plymouth University
- Student ID: 10898536
- Supervisor: Ms. Pavithra Subhashini

## ![LinkedIn Icon](https://example.com/animated-linkedin.gif) Connect with Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/chamod-thilina-6a8563249/details/projects/)

