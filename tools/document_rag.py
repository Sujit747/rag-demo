import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 300
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load sentence embedding model
model = SentenceTransformer(MODEL_NAME)

# Global state
doc_chunks, doc_embeddings = [], []

class DocumentRAGTool:
    def __init__(self):
        self.name = "document_rag"
        self.description = "Answer questions based on an uploaded document"
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def handle_file_upload(self, file):
        """Processes the uploaded PDF and caches its embeddings."""
        global doc_chunks, doc_embeddings
        if not file:
            return False, "⚠️ Please upload a file."
        try:
            text = self.extract_pdf_text(file)
            doc_chunks = self.split_text(text)
            doc_embeddings = model.encode(doc_chunks, convert_to_numpy=True)
            return True, f"✅ Processed {len(doc_chunks)} chunks."
        except Exception as e:
            return False, f"❌ Failed to process file: {e}"

    def extract_pdf_text(self, file_obj):
        """Extracts and joins text from all pages of a PDF."""
        reader = PdfReader(file_obj)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def split_text(self, text, size=CHUNK_SIZE):
        """Splits text into fixed-size word chunks."""
        words = text.split()
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    def get_top_chunks(self, query, k=3):
        """Finds top-k relevant chunks using cosine similarity."""
        global doc_chunks, doc_embeddings
        if not doc_chunks or not doc_embeddings.any():
            return None
        try:
            query_emb = model.encode([query], convert_to_numpy=True)
            if query_emb.size == 0 or doc_embeddings.size == 0:
                return None
            sims = cosine_similarity(query_emb, doc_embeddings)[0]
            indices = np.argsort(sims)[::-1][:k]
            return "\n\n".join([doc_chunks[i] for i in indices if i < len(doc_chunks)])
        except Exception as e:
            logger.error(f"Error in get_top_chunks: {e}")
            return None

    def call_openai_ai(self, context, question):
        """Calls OpenAI API for chat completion."""
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set in environment.")
            return "❌ Error: API key not configured."
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer only using the provided context. If the context does not contain relevant information to answer the question, respond with 'I cannot answer this from the document.'"},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=512
            )
            content = response.choices[0].message.content
            return content if content != "I cannot answer this from the document." else "unanswerable"
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"❌ Error: {str(e)}"

    def _run(self, question: str) -> str:
        if not doc_chunks:
            return "⚠️ No document uploaded. Please upload a PDF first."
        context = self.get_top_chunks(question)
        if context is None:
            return "unanswerable"
        answer = self.call_openai_ai(context, question)
        return answer

    async def _arun(self, question: str) -> str:
        return self._run(question)