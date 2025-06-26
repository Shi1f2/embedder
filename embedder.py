import weaviate
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Optional


class BertEmbedder:
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 weaviate_url: Optional[str] = "https://osvcpzz6sjaaloedhjbha.c0.europe-west3.gcp.weaviate.cloud",
                 weaviate_api_key: Optional[str] = "N2plUjYzemxTVHhXY1poNl9SZzA2bisxdjZQYTlPTTJvUEI2L0dlVkFEN1d1bHJwOTR1bmFNZTBtRHFNPV92MjAw"):
        """
        Initialize BERT embedder with Weaviate connection
        
        Args:
            model_name: Sentence transformer model name
            weaviate_url: Weaviate Cloud cluster URL
            weaviate_api_key: Weaviate Cloud API key
        """
        # Initialize BERT model
        self.model = SentenceTransformer(model_name)
        
        # Connect to Weaviate Cloud or local
        if weaviate_url and weaviate_api_key:
            # Connect to Weaviate Cloud
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
            )
        else:
            print("clound is not working")
        
        # Create schema if it doesn't exist
        self._create_schema()
    
    def _create_schema(self):
        """Create Weaviate schema for storing text embeddings"""
        from weaviate.classes.config import Configure, Property, DataType
        
        # Check if collection already exists
        if not self.client.collections.exists("TextEmbedding"):
            self.client.collections.create(
                name="TextEmbedding",
                description="Text documents with BERT embeddings",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT, description="The original text content"),
                    Property(name="text_id", data_type=DataType.TEXT, description="Unique identifier for the text")
                ]
            )
            print("Created TextEmbedding collection in Weaviate")
        else:
            print("TextEmbedding collection already exists")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate BERT embedding for input text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def store_embedding(self, text: str, text_id: Optional[str] = None) -> str:
        """
        Store text and its embedding in Weaviate
        
        Args:
            text: Input text to embed and store
            text_id: Optional custom ID, if not provided, random UUID will be generated
            
        Returns:
            The ID of the stored object
        """
        # Generate embedding
        embedding = self.embed_text(text)
        
        # Generate ID if not provided
        if text_id is None:
            text_id = str(uuid.uuid4())
        
        # Prepare data object
        data_object = {
            "text": text,
            "text_id": text_id
        }
        
        # Store in Weaviate with embedding
        collection = self.client.collections.get("TextEmbedding")
        collection.data.insert(
            properties=data_object,
            vector=embedding
        )
        
        print(f"Stored text with ID: {text_id}")
        return text_id
    
    def batch_store_embeddings(self, texts: List[str]) -> List[str]:
        """
        Store multiple texts and their embeddings in batch
        
        Args:
            texts: List of texts to embed and store
            
        Returns:
            List of IDs for stored objects
        """
        # Generate embeddings for all texts
        embeddings = self.model.encode(texts)
        
        stored_ids = []
        
        # Use batch operations for efficiency
        from weaviate.classes.data import DataObject
        collection = self.client.collections.get("TextEmbedding")
        objects_to_insert = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            text_id = str(uuid.uuid4())
            
            data_object = DataObject(
                properties={
                    "text": text,
                    "text_id": text_id
                },
                vector=embedding.tolist()
            )
            
            objects_to_insert.append(data_object)
            stored_ids.append(text_id)
        
        collection.data.insert_many(objects_to_insert)
        
        print(f"Stored {len(texts)} texts in batch")
        return stored_ids
    
    def close(self):
        """Close the Weaviate connection"""
        self.client.close()


# Example usage
if __name__ == "__main__":
    embedder = None
    try:
        # Initialize embedder
        embedder = BertEmbedder()
        
        # Example texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology.",
            "Weaviate is a vector database for storing embeddings.",
            "BERT creates contextual word embeddings."
        ]
        
        # Store single text
        single_id = embedder.store_embedding("This is a test sentence.")
        
        # Store multiple texts in batch
        batch_ids = embedder.batch_store_embeddings(sample_texts)
        
        print(f"Single text stored with ID: {single_id}")
        print(f"Batch texts stored with IDs: {batch_ids}")
        
    finally:
        if embedder:
            embedder.close()
            print("✅ Connection closed properly")
