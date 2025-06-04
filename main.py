from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from dotenv import load_dotenv
import os

# Optional: Load .env variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Load and wrap the document ---
try:
    with open("your_data.txt", "r", encoding="utf-8") as file:
        content = file.read()
    documents = [Document(page_content=content)]
except Exception as e:
    raise RuntimeError(f"Error reading your_data.txt: {e}")

# --- Build the vector store with HuggingFace embeddings ---
embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma.from_documents(documents, embedding)

# --- Initialize the free LLM model ---
model_name = "google/flan-t5-base"  # T5 model for Q&A
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Change this line - use the correct model class for T5
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Configure the pipeline with better parameters
pipe = pipeline(
    "text2text-generation",  # This is the correct task for T5 models
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,  # Lower temperature for more focused answers
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Build the QA retrieval chain ---
retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Define endpoint to ask questions ---
@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    query = body.get("question", "").strip()
    
    if not query:
        return {"error": "No question provided"}
    
    try:
        # Check for common FAQ questions first
        faq_keywords = {
            "attendance": "Q: Will we get attendance if we don't attend the event?\nA: No. Attendance is only granted to those who are physically present, as per institutional guidelines.",
            "registration fee": "Q: What is the registration fee?\nA: ₹300 for teams (up to 4 members), ₹100 for individuals, ₹100 for audience entry.",
            "rewards": "Q: What are the rewards for winners?\nA: A ₹25,000 prize pool, trophies, certificates, mentorship, and potential incubation support.",
            "participate": "Q: Who can participate?\nA: Any student from any college or university. Open to B.Tech, MBA, and other streams."
        }
        
        # Check if query contains any FAQ keywords
        for keyword, answer in faq_keywords.items():
            if keyword.lower() in query.lower():
                # Extract just the answer part
                answer_text = answer.split("A: ")[1] if "A: " in answer else answer
                return {"answer": answer_text}
        
        # If not a direct FAQ match, use the regular QA chain
        prompt = f"""Answer the following question about the INNOVATE-X event based on the provided context.
        Be concise and accurate. Pay special attention to FAQ sections in the data.
        If the answer cannot be found in the context, say "I don't have information about that in my database."
        
        Question: {query}
        
        Context: """
        
        result = qa_chain({"query": prompt})
        answer = result["result"]
        
        # Clean up the response
        if "Based on the following information" in answer:
            answer = answer.replace("Based on the following information", "").strip()
        if "Answer:" in answer:
            answer = answer.replace("Answer:", "").strip()
        
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# Add a new endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot is running"}

# Add this new endpoint
@app.get("/faq")
async def get_faq():
    return {
        "faqs": [
            {"question": "What is INNOVATE-X?", "answer": "A 4-day entrepreneurship and innovation marathon to simulate real-world startup building for students."},
            {"question": "Who can participate?", "answer": "Any student from any college or university. Open to B.Tech, MBA, and other streams."},
            {"question": "What is the registration fee?", "answer": "₹300 for teams (up to 4 members), ₹100 for individuals, ₹100 for audience entry."},
            {"question": "When and where is the event?", "answer": "May 9, 10, 11, and 13, 2025 at LNCT Campus and Aryabhat Auditorium, Bhopal."},
            {"question": "What are the rewards?", "answer": "A ₹25,000 prize pool, trophies, certificates, mentorship, and potential incubation support."}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
