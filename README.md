# Conversational RAG Chatbot

1. `python -m venv venv && source venv/bin/activate`  (Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your free key from https://console.groq.com
4. (Optional) Put your own `.txt` / `.md` files in `data/` to answer from them.
   If `data/` is empty, the bot answers using the chat history only.
5. `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Open http://localhost:8000  (phone: http://<your-pc-ip>:8000 on same Wi-Fi)

After changing files in `data/`, delete the `chroma_db/` folder so it re-indexes.

Pages:
- http://localhost:8000/          -> general chatbot
- http://localhost:8000/map.html  -> Pimpri-Chinchwad map + AI guide (answers from data/pimpri_chinchwad.md and chat history; no fine-tuning)

Files:
- static/index.html -> chatbot frontend
- static/map.html   -> Pimpri-Chinchwad map frontend
- rag_backend.py    -> LangChain / Groq / Chroma logic
- main.py           -> FastAPI server
