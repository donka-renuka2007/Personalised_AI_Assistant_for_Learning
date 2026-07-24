import json
import os
from dotenv import load_dotenv
import requests
from typing import TypedDict, List, Dict, Any
from flask import Flask, render_template, request, session, redirect, url_for
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langgraph.graph import StateGraph, START, END
import markdown2
import pypdf
import math
import re
from collections import Counter
# ==========================================
# 0. INITIALIZATION & CONFIG
# ==========================================
load_dotenv()
app = Flask(__name__)           
app.secret_key = "renu10karthika26roshan53"

Mistral_api_key = os.environ.get("MISTRAL_KEY")
client = MistralClient(api_key=Mistral_api_key)

chat_history: List[ChatMessage] = []

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    input: str
    output: str
    quiz_data: List[Dict[str, Any]]

# ==========================================
# 2. TOOLS
# ==========================================
def book_recommend_tool():
    google_books_api_key = os.environ.get("GOOGLE_BOOKS_KEY")
    
    formatted_history = [
        {"role": msg.role, "content": msg.content} for msg in chat_history
    ]
    
    # Infer search topic
    result = client.chat(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": f"Based on context, extract a single concise subject keyword (1-2 words, e.g., 'Artificial Intelligence', 'Python'): {formatted_history}"
            }
        ]
    )
    topic = result.choices[0].message.content.strip().replace('"', '')
    
    # Target exact title/subject matching
    query_url = f"https://www.googleapis.com/books/v1/volumes?q={topic}+subject:computers&maxResults=5&key={google_books_api_key}"
    response = requests.get(query_url)
    
    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        
        cards_html = []
        for item in items:
            info = item.get("volumeInfo", {})
            title = info.get("title", "Unknown Title")
            authors = ", ".join(info.get("authors", ["Unknown Author"]))
            desc = info.get("description", "No description available.")[:130] + "..."
            link = info.get("infoLink", "#")
            
            image_links = info.get("imageLinks", {})
            thumbnail = image_links.get("thumbnail", "https://via.placeholder.com/128x192?text=No+Cover")
            thumbnail = thumbnail.replace("http://", "https://")

            card = f"""
            <div style="display: flex; gap: 15px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 12px; margin-bottom: 12px; align-items: start;">
                <img src="{thumbnail}" alt="Cover" style="width: 70px; height: 100px; object-fit: cover; border-radius: 4px; flex-shrink: 0;">
                <div style="flex-grow: 1;">
                    <h4 style="margin: 0 0 4px 0; font-size: 16px;">
                        <a href="{link}" target="_blank" style="text-decoration: none; color: #00f0ff;">{title}</a>
                    </h4>
                    <p style="margin: 0 0 6px 0; font-size: 13px; color: #cbd5e1;"><strong>By:</strong> {authors}</p>
                    <p style="margin: 0 0 8px 0; font-size: 12px; color: #94a3b8; line-height: 1.4;">{desc}</p>
                    <a href="{link}" target="_blank" style="display: inline-block; padding: 4px 10px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; font-size: 12px;">View Book</a>
                </div>
            </div>
            """
            cards_html.append(card)
            
        if cards_html:
            return f"<h3 style='margin-bottom: 15px; color: #00f0ff;'>Recommended Books for '{topic}':</h3>" + "".join(cards_html)
            
    return f"Sorry, I couldn't find relevant book recommendations for topic: {topic}"


def dictionary_tool(query: str):
    words = [w.strip(",.?!").lower() for w in query.split(" ") if w.strip()]
    stop_words = {"what", "is", "the", "define", "meaning", "of", "and", "a", "an"}
    definitions_html = []
    
    for word in words:
        if len(word) <= 2 or word in stop_words:
            continue
            
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        
        if response.status_code == 200:
            word_data = response.json()[0]
            w = word_data.get('word', word)
            phonetic = word_data.get('phonetic', '')
            meanings = word_data.get('meanings', [])
            
            mean_items = []
            for m in meanings[:2]:
                part = m.get('partOfSpeech', '')
                defs = [d['definition'] for d in m.get('definitions', [])[:1]]
                if defs:
                    mean_items.append(
                        f"<div style='margin-top: 6px;'>"
                        f"<span style='background: rgba(0,240,255,0.15); color: #00f0ff; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;'>{part}</span> "
                        f"<span style='font-size: 13px; color: #e2e8f0;'>{defs[0]}</span>"
                        f"</div>"
                    )
                
            if mean_items:
                card = f"""
                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); border-left: 4px solid #00f0ff; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: baseline; gap: 8px;">
                        <h4 style="margin: 0; color: #ffffff; font-size: 18px; text-transform: capitalize;">{w}</h4>
                        <span style="color: #94a3b8; font-size: 12px; font-style: italic;">{phonetic}</span>
                    </div>
                    {"".join(mean_items)}
                </div>
                """
                definitions_html.append(card)
            
    return "".join(definitions_html) if definitions_html else "<div style='color: #94a3b8; font-style: italic;'>No dictionary definitions found for your query.</div>"


def quiz_generate(query: str):
    context_data = client.chat(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": f"Based on the user query generate learning content. User query: {query}"
            }
        ]
    )
    context = context_data.choices[0].message.content.strip()

    quiz_data = client.chat(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": f"""
Generate 10 multiple choice quiz questions based on the context.

Return ONLY valid JSON in this format:
[
  {{
    "question": "Question text",
    "options": ["option 1", "option 2", "option 3", "option 4"],
    "answer": "correct option"
  }}
]

Context:
{context}
"""
            }
        ]
    )

    quiz_raw = quiz_data.choices[0].message.content.strip()
    quiz_clean = quiz_raw.replace("```json", "").replace("```", "").strip()
    start = quiz_clean.find("[")
    end = quiz_clean.rfind("]") + 1

    if start != -1 and end != -1:
        quiz_clean = quiz_clean[start:end]
    
    try:
        return json.loads(quiz_clean)
    except Exception:
        return []


def chat(query: str):
    formatted_messages = [
        {"role": msg.role, "content": msg.content} for msg in chat_history
    ]
    
    response = client.chat(
        model="mistral-small-latest",
        messages=formatted_messages
    )
    
    raw_markdown = response.choices[0].message.content
    
    # Render Markdown into rich HTML with tables, code blocks, and strike-throughs
    formatted_html = markdown2.markdown(
        raw_markdown, 
        extras=["fenced-code-blocks", "tables", "strike", "task_list", "code-friendly"]
    )
    
    return formatted_html

# ==========================================
# 3. ROUTER & NODES
# ==========================================
def router(state: AgentState) -> str:
    query = state["input"]
    prompt = f"""
    Classify the user intent into EXACTLY one category:
    - 'book' (if asking for book recommendations or reading materials)
    - 'dictionary' (if asking for word definitions or meanings)
    - 'quiz' (if asking to create/take a test, quiz, or multiple choice practice)
    - 'chat' (if general question, greeting, casual talk, or general advice)

    User Query: "{query}"

    Return ONLY the category name.
    """
    response = client.chat(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    category = response.choices[0].message.content.strip().lower()
    if category not in ["book", "dictionary", "quiz", "chat"]:
        category = "chat"
    return category


def book_recommend_tool_node(state: AgentState):
    output_text = book_recommend_tool()
    return {"output": output_text}


def dictionary_tool_node(state: AgentState):
    output_text = dictionary_tool(state["input"])
    return {"output": output_text}


def quiz_generate_node(state: AgentState):
    user_query = state["input"]
    
    # 1. Clean the user query to use as a topic (e.g., "quiz on ML" -> "ML")
    clean_topic = user_query.lower()
    for filler in ["generate quiz on", "quiz on", "quiz about", "test me on", "create a quiz on", "quiz"]:
        clean_topic = clean_topic.replace(filler, "")
    clean_topic = clean_topic.strip().title() or "General Knowledge"
    
    # 2. Save the new topic to Flask session
    session["quiz_topic"] = clean_topic
    
    # 3. Generate quiz
    quiz = quiz_generate(user_query)
    return {"quiz_data": quiz, "output": "QUIZ_GENERATED"}


def chat_node(state: AgentState):
    output_text = chat(state["input"])
    return {"output": output_text}

# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================
graph_builder = StateGraph(AgentState)

# Add Nodes
graph_builder.add_node("router_node", lambda state: state)
graph_builder.add_node("book_recommend_tool", book_recommend_tool_node)
graph_builder.add_node("dictionary_tool", dictionary_tool_node)
graph_builder.add_node("quiz_generate", quiz_generate_node)
graph_builder.add_node("chat", chat_node)

# Flow Setup
graph_builder.add_edge(START, "router_node")

graph_builder.add_conditional_edges(
    "router_node",
    router,
    {
        "book": "book_recommend_tool",
        "dictionary": "dictionary_tool",
        "quiz": "quiz_generate",
        "chat": "chat"
    }
)

graph_builder.add_edge("book_recommend_tool", END)
graph_builder.add_edge("dictionary_tool", END)
graph_builder.add_edge("quiz_generate", END)
graph_builder.add_edge("chat", END)

# Compile Graph
compiled_agent = graph_builder.compile()

def run_agent(query: str):
    """Execution wrapper function for compiled graph."""
    state_input = {"input": query, "output": "", "quiz_data": []}
    return compiled_agent.invoke(state_input)

# ==========================================
# 5. FLASK ROUTES
# ==========================================
@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')
@app.route('/MindCraft', methods=["GET", "POST"])
def chat_bot():
    if request.method == "POST":
        query = request.form.get("query", "")
        
        # Append user message to history
        chat_history.append(ChatMessage(role="user", content=query))
        
        # Invoke LangGraph Agent
        result = run_agent(query)
        
        # Check if output requires redirection to Quiz page
        if result.get("output") == "QUIZ_GENERATED":
            quiz = result.get("quiz_data", [])
            session["quiz_answers"] = [q["answer"] for q in quiz if "answer" in q]
            return render_template("quiz.html", quiz=quiz)
            
        output = result.get("output", "")
        
        # Append assistant response to history
        chat_history.append(ChatMessage(role="assistant", content=output))
        
        return render_template("agent_chatbot.html", query=query, output=output, chat_history=chat_history)
        
    return render_template("agent_chatbot.html", chat_history=chat_history)


@app.route("/submit-quiz", methods=["POST"])
def submit_quiz():
    try:
        correct_answers = session.get("quiz_answers", [])
        topic = session.get("quiz_topic", "General Knowledge")
        
        if not correct_answers:
            bot_msg = "⚠️ No quiz data was found for this submission. Please generate a new quiz."
            return render_template("score.html", score=0, total=0, feedback=bot_msg)

        user_score = 0
        total = len(correct_answers)

        for i, correct_ans in enumerate(correct_answers):
            selected = request.form.get(f"q_{i}")
            if selected == correct_ans:
                user_score += 1

        # Formatted result string for chatbot & score page
        bot_msg = f"📊 **Quiz Completed for {topic}!**\nYou got **{user_score}/{total}**!"

        # Save to Chat History so chatbot remembers it
        if 'chat_history' in globals():
            chat_history.append(ChatMessage(role="user", content=f"Submitted Quiz on {topic}"))
            chat_history.append(ChatMessage(role="assistant", content=bot_msg))

        # Save to Session Progress History
        if "progress_history" not in session:
            session["progress_history"] = []
            
        progress_history = list(session["progress_history"])
        progress_history.append({
            "topic": topic,
            "score": user_score,
            "total": total
        })
        session["progress_history"] = progress_history

        return render_template("score.html", score=user_score, total=total, feedback=bot_msg)

    except Exception as e:
        print(f"Error in submit_quiz: {e}")
        return f"An error occurred while submitting the quiz: {e}", 500
# 4. Book Library Page
@app.route('/library', methods=["GET"])
def book_library():
    raw_query = request.args.get('query', '')
    google_books_api_key = os.environ.get("GOOGLE_BOOKS_KEY")
    books = []

    if raw_query:
        # Strip common natural language filler words for better Google Books API results
        clean_query = raw_query.lower()
        for filler in ["suggest books on", "recommend books for", "books about", "books on", "suggest", "find"]:
            clean_query = clean_query.replace(filler, "")
        clean_query = clean_query.strip() or raw_query

        query_url = f"https://www.googleapis.com/books/v1/volumes?q={clean_query}&maxResults=12&key={google_books_api_key}"
        response = requests.get(query_url)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", []) or []
            
            for item in items:
                info = item.get("volumeInfo", {})
                image_links = info.get("imageLinks", {}) or {}
                thumbnail = image_links.get("thumbnail", "https://via.placeholder.com/128x192?text=No+Cover")
                thumbnail = thumbnail.replace("http://", "https://")

                books.append({
                    "title": info.get("title", "Unknown Title"),
                    "authors": ", ".join(info.get("authors", ["Unknown Author"])),
                    "description": (info.get("description") or "No description available.")[:110] + "...",
                    "link": info.get("infoLink", "#"),
                    "thumbnail": thumbnail
                })

    return render_template('book_library.html', books=books)


import math
import re
from collections import Counter
import pypdf

# ==========================================
# LIGHTWEIGHT PURE-PYTHON RAG LOGIC
# ==========================================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract raw text from an uploaded PDF stream."""
    reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text


def tokenize(text: str):
    """Simple word tokenizer for TF-IDF."""
    return re.findall(r'\w+', text.lower())


def rag_retrieve_chunks(text: str, query: str = "key concepts core facts definition", top_k: int = 5) -> str:
    """Chunks PDF text and retrieves top sections using pure-Python TF-IDF cosine similarity."""
    # Split PDF into paragraphs/chunks
    raw_chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    
    if not raw_chunks:
        # Fallback split if double newlines are missing
        raw_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
    if not raw_chunks:
        return text[:3000]

    # Tokenize corpus & query
    corpus = [tokenize(c) for c in raw_chunks]
    query_tokens = tokenize(query)
    
    if not query_tokens:
        return "\n\n".join(raw_chunks[:top_k])

    # Compute Document Frequencies
    num_docs = len(corpus)
    df = Counter()
    for doc in corpus:
        for word in set(doc):
            df[word] += 1

    # Score chunks against query terms using TF-IDF
    scores = []
    for idx, doc in enumerate(corpus):
        if not doc:
            scores.append((0, idx))
            continue
            
        doc_tf = Counter(doc)
        score = 0.0
        for token in query_tokens:
            if token in doc_tf:
                tf = doc_tf[token] / len(doc)
                idf = math.log((num_docs + 1) / (df[token] + 1)) + 1
                score += tf * idf
                
        scores.append((score, idx))

    # Sort chunks by relevance score
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Extract top k chunks
    top_indices = [idx for _, idx in scores[:top_k]]
    retrieved_chunks = [raw_chunks[idx] for idx in top_indices]
    
    return "\n\n".join(retrieved_chunks)


def generate_rag_quiz(extracted_text: str):
    """Retrieves relevant context from PDF and generates quiz."""
    context = rag_retrieve_chunks(extracted_text)
    
    # Pass context to your existing quiz_generate function
    return quiz_generate(f"Generate 10 multiple choice questions based on this text:\n\n{context}")
@app.route('/upload-quiz', methods=["GET", "POST"])
def upload_pdf_quiz():
    if request.method == "POST":
        if 'pdf_file' not in request.files:
            return redirect(request.url)
            
        file = request.files['pdf_file']
        if file.filename == '':
            return redirect(request.url)
            
        # Inside upload_pdf_quiz route in app.py:
        if file and file.filename.endswith('.pdf'):
            # Extract clean name without .pdf extension
            clean_topic = file.filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
            session["quiz_topic"] = f"PDF: {clean_topic}"  # 👈 Saves topic to session
            
            pdf_text = extract_text_from_pdf(file)
            quiz = generate_rag_quiz(pdf_text)
            session["quiz_answers"] = [q["answer"] for q in quiz if "answer" in q]
            return render_template("quiz.html", quiz=quiz)

    return render_template('upload_quiz.html')
# ==========================================
# PROGRESS PAGE ROUTE
# ==========================================
@app.route('/progress', methods=["GET"])
def show_progress():
    history = session.get("progress_history", [])
    return render_template('progress.html', history=history)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)