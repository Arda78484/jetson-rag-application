import os
from typing import List, Dict, Any
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
import json

class RAGCreate:
    """A class to create and manage RAG (Retrieval-Augmented Generation) embeddings."""
    
    def __init__(self, filepath: list, model_name:str, nim_api: str):
        """
        Initialize the RAG creation system.
        
        Args:
            filepath: Path to the input file
            nim_api: NVIDIA API key
            collection_name: Name of the Milvus collection
        """
        self.collection_name = "rag_chat"
        self.text_lines: List[str] = []
        self.embedding_dim: int = 0
        self.model_name = model_name

        self._initialize_clients(nim_api)
        self.read_text(filepath)
        self._generate_test_embedding()

    def _initialize_clients(self, nim_api: str) -> None:
        """Initialize Milvus and OpenAI clients."""
        self.milvus_client = MilvusClient(uri="./milvus_demo.db")
        self.openai_client = OpenAI(
            api_key=nim_api,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        
        self._reset_collection_if_exists()

    def _reset_collection_if_exists(self) -> None:
        """Drop existing collection if it exists."""
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

    def read_text(self, filepath_list) -> None:
        """
        Read and process text from input file.
        
        Args:
            filepath: Path to the input file
        """
        try:
            for filepath in filepath_list:
                with open(filepath, 'r') as file:
                    lines = [line.strip() for line in file.readlines() if line.strip()]
                    # Remove header if present
                    if lines and lines[0].startswith("TIME"):
                        lines = lines[1:]
                    self.text_lines = lines
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def emb_text(self, text: str) -> List[float]:
        """
        Create embedding for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.model_name,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"}
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    def _generate_test_embedding(self) -> None:
        """Generate test embedding to determine embedding dimension."""
        test_embedding = self.emb_text("Rag Chatbot")
        self.embedding_dim = len(test_embedding)
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Test embedding sample: {test_embedding[:10]}")

    def create_collection(self) -> None:
        """Create and populate Milvus collection with embeddings."""
        if not self.text_lines:
            raise ValueError("No text data loaded. Call read_text first.")

        self._create_milvus_collection()
        self._insert_embeddings()

    def _create_milvus_collection(self) -> None:
        """Create new Milvus collection with appropriate parameters."""
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="IP",  # Inner Product distance metric
            consistency_level="Strong"
        )

    def _insert_embeddings(self) -> None:
        """Generate and insert embeddings into Milvus collection."""
        data = []
        for i, line in enumerate(tqdm(self.text_lines, desc="Creating embeddings")):
            data.append({
                "id": i,
                "vector": self.emb_text(line),
                "text": line
            })

        self.milvus_client.insert(
            collection_name=self.collection_name,
            data=data
        )
