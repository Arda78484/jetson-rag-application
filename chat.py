import os
from glob import glob
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
import json
import requests
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    text: str
    distance: float

class RAGChat:
    """A class implementing RAG (Retrieval-Augmented Generation) based chat functionality."""
    
    def __init__(self, llm_model: str, embedding_model: str, nim_api: str, system_prompt: str):
        """Initialize the RAG chat system with necessary clients and configurations."""
        self.collection_name = "rag_chat"
        self._initialize_milvus_client()
        self._initialize_embedding_client(nim_api, embedding_model)
        self._initialize_llm_client(llm_model, nim_api)
        self.system_prompt = system_prompt

    def _initialize_milvus_client(self) -> None:
        """Initialize the Milvus vector database client."""
        self.milvus_client = MilvusClient(uri="./milvus_demo.db")

    def _initialize_embedding_client(self, nim_api: str, embedding_model: str) -> None:
        """Initialize the embedding generation client."""
        self.embedding_client = OpenAI(
            api_key=nim_api,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.embedding_model = embedding_model

    def _initialize_llm_client(self, llm_model: str, nim_api: str) -> None:
        """Initialize the language model client based on model type."""
        if llm_model == "local":
            self.llm_client = OpenAI(
                api_key="*",
                base_url="http://0.0.0.0:9000/v1"
            )
        else:
            self.llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nim_api
            )
        self.llm_model = llm_model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text."""
        response = self.embedding_client.embeddings.create(
            input=text,
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    def _search_similar_documents(self, embedding: List[float], limit: int = 150) -> List[RetrievedDocument]:
        """Search for similar documents in the vector database."""
        search_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
        return [
            RetrievedDocument(res["entity"]["text"], res["distance"]) 
            for res in search_results[0]
        ]

    def _rerank_documents(self, question: str, documents: List[RetrievedDocument]) -> Optional[str]:
        """Rerank retrieved documents using NVIDIA's reranking API."""
        headers = {
            "Authorization": f"Bearer {self.embedding_client.api_key}",  # Using the NIM API key from initialization
            "Accept": "application/json",
        }
        payload = {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": question},
            "passages": [{"text": doc.text} for doc in documents],
        }

        try:
            with requests.Session() as session:
                response = session.post(
                    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                if "rankings" not in data:
                    raise ValueError("Unexpected response format: 'rankings' not found")
                
                return documents[data["rankings"][0]["index"]].text
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents[0].text

    def answer_question(self, question: str, temperature, max_tokens, top_p) -> str:
        """Generate an answer for the given question using RAG."""
        # Generate embedding for the question
        question_embedding = self.generate_embedding(question)
        
        # Retrieve similar documents
        retrieved_documents = self._search_similar_documents(question_embedding)
        
        # Rerank and get the best context
        context = self._rerank_documents(question, retrieved_documents)
        
        # Generate the answer using the LLM
        user_prompt = self._create_prompt(context, question)
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,  # Fixed: Added missing model parameter
            messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        
        return response.choices[0].message.content

    @staticmethod
    def _create_prompt(context: str, question: str) -> str:
        """Create a formatted prompt for the LLM."""
        return f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """