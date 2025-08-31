from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.memory import ChatMemoryBuffer  # Added missing import
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

class LlamaIndexMemoryManagerFAISS:
    def __init__(self, persist_dir: str = "./memory_storage", openai_api_key: str = None, dimension: int = 1536):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.dimension = dimension
        
        self.faiss_index_path = self.persist_dir / "faiss_index.bin"
        self.metadata_path = self.persist_dir / "metadata.pkl"
        
        self._initialize_faiss_index()
        
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )
        
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        print(f"✅ FAISS Memory Manager initialized with dimension {self.dimension}")
    
    def _initialize_faiss_index(self):
        try:
            if self.faiss_index_path.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))
                print(f"📂 Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                print("🆕 Created new FAISS index")
                
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                print(f"📂 Loaded {len(self.metadata_store)} metadata entries")
            else:
                self.metadata_store = {}
                print("🆕 Created new metadata store")
                
        except Exception as e:
            print(f"⚠️ Error loading FAISS index, creating new one: {str(e)}")
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = {}
    
    def _save_faiss_index(self):
        try:
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            print(f"💾 Saved FAISS index with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {str(e)}")
    
    def add_interaction(self, user_id: str, user_message: str, assistant_response: str):
        try:
            self.cleanup_old_interactions(user_id)
            interaction_text = f"User: {user_message}\nAssistant: {assistant_response}"
            embedding = self.embed_model.get_text_embedding(interaction_text)
            embedding_array = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            self.faiss_index.add(embedding_array)
            doc_id = f"interaction_{user_id}_{datetime.now().isoformat()}_{self.faiss_index.ntotal-1}"
            self.metadata_store[self.faiss_index.ntotal-1] = {
                "doc_id": doc_id,
                "user_id": user_id,
                "text": interaction_text,
                "timestamp": datetime.now().isoformat(),
                "interaction_type": "conversation"
            }
            self._save_faiss_index()
            print(f"➕ Added interaction to FAISS memory: {len(interaction_text)} chars")
        except Exception as e:
            print(f"❌ Error adding interaction to FAISS memory: {str(e)}")
    
    def add_financial_data(self, user_id: str, data_type: str, data: str):
        try:
            embedding = self.embed_model.get_text_embedding(data)
            embedding_array = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            self.faiss_index.add(embedding_array)
            doc_id = f"data_{user_id}_{data_type}_{datetime.now().isoformat()}_{self.faiss_index.ntotal-1}"
            self.metadata_store[self.faiss_index.ntotal-1] = {
                "doc_id": doc_id,
                "user_id": user_id,
                "text": data,
                "data_type": data_type,
                "timestamp": datetime.now().isoformat()
            }
            self._save_faiss_index()
            print(f"➕ Added financial data to FAISS memory: {data_type}")
        except Exception as e:
            print(f"❌ Error adding financial data to FAISS memory: {str(e)}")
    
    def query_memory(self, user_id: str, query: str, top_k: int = 3) -> str:
        try:
            if self.faiss_index.ntotal == 0:
                return "No previous interactions found."
            query_embedding = self.embed_model.get_text_embedding(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            similarities, indices = self.faiss_index.search(query_array, min(top_k * 2, self.faiss_index.ntotal))
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx in self.metadata_store:
                    metadata = self.metadata_store[idx]
                    if metadata.get("user_id") == user_id and similarity > 0.1:
                        results.append({
                            "text": metadata["text"],
                            "similarity": float(similarity),
                            "timestamp": metadata["timestamp"]
                        })
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
            if not results:
                return "No relevant previous interactions found."
            response_parts = []
            for i, result in enumerate(results, 1):
                response_parts.append(f"{i}. {result['text'][:200]}... (Similarity: {result['similarity']:.3f})")
            return "Previous relevant interactions:\n" + "\n\n".join(response_parts)
        except Exception as e:
            print(f"❌ Error querying FAISS memory: {str(e)}")
            return "Error retrieving previous interactions."
    
    def get_user_context(self, user_id: str) -> str:
        try:
            return self.query_memory(user_id, "recent interactions and preferences", top_k=5)
        except Exception as e:
            print(f"❌ Error getting user context: {str(e)}")
            return "No previous context available."

    def cleanup_old_interactions(self, user_id: str, days_threshold: int = 30):
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
            indices_to_remove = []
            for idx, metadata in self.metadata_store.items():
                if (metadata.get("user_id") == user_id and 
                    metadata.get("timestamp", "") < cutoff_date):
                    indices_to_remove.append(idx)
            for idx in indices_to_remove:
                if idx in self.metadata_store:
                    del self.metadata_store[idx]
            if indices_to_remove:
                print(f"🧹 Marked {len(indices_to_remove)} old interactions for cleanup")
                self._save_faiss_index()
        except Exception as e:
            print(f"⚠️ Cleanup error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.faiss_index.ntotal,
            "total_metadata": len(self.metadata_store),
            "dimension": self.dimension,
            "index_type": type(self.faiss_index).__name__
        }