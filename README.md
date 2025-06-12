# 🦙 Alfred – Karan Poddar's Professional Assistant

Alfred is an intelligent chatbot built with Streamlit and powered by LLMs to answer questions about Karan Poddar based on his resume. Whether you're a recruiter, colleague, or collaborator, Alfred will respond with precise and context-aware information derived only from the resume data.


## 🚀 Features

- 🔍 Answers resume-based questions accurately
- 📄 Rejects personal or inappropriate questions
- 💬 Chat interface using Streamlit
- 🧠 Powered by a Groq API-backed prompt engine
- 🎯 Focused, non-speculative responses
- 🧹 Option to clear chat history


## 📂 Project Structure

📁 alfred-chatbot/
│
├── resume_groq_api.py # Contains the prompt formatting and LLM call logic
├── app.py # Main Streamlit frontend
├── README.md # You're reading it!


## 💡 How It Works

### `resume_groq_api.py`

This file defines the prompt logic using a structured system prompt for the LLM. It ensures that:

- Only resume context is used for answers
- Responses follow a consistent, professional format
- Inappropriate questions are politely declined

This is the Streamlit app that powers the interactive chat UI. Key features include:

Sidebar with links to Karan’s Resume and GitHub

Chat history preservation using st.session_state

Real-time streaming of LLM-generated responses

"Clear Chat History" button to reset the session

🛠️ Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/alfred-chatbot.git
cd alfred-chatbot
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Create .env or config for API Keys
Store your Groq or other model API keys as needed.

Run the App

bash
Copy
Edit
streamlit run app.py

✍️ Example Questions You Can Ask

"What technical skills does Karan have?"

"Describe a project Karan led involving machine learning."

"Has he demonstrated leadership in previous roles?"

"What cloud platforms has he worked with?"

🛡️ Limitations

Alfred cannot answer personal or unrelated questions.

Responses are limited to the data provided in the resume context.

It doesn't browse the internet or external sources.