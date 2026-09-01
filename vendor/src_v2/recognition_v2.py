# recognition_v2.py
# Face recognition using ChromaDB vector DB (v2 - with Quantum Haar Wavelet)
# Uses 8-qubit Hierarchical QCNN with Haar Wavelet feature extraction
# Stores embeddings in Vector DB_v2/ (separate from v1)

import os
import numpy as np
import joblib
import chromadb
from chromadb.config import Settings
from .qcnn_recognition_v2 import QCNNEmbeddingExtractor


class FaceRecognizer:
    def __init__(self, db_dir=None, collection_name='face_embeddings_v2',
                 threshold=0.75, metric='cosine', seed: int = 42,
                 embedding_mode='optimized'):
        """
        Initialize FaceRecognizer v2 with ChromaDB backend.

        This is the v2 version using Quantum Haar Wavelet for feature extraction.
        Uses a separate database (Vector DB_v2/) from v1.

        Args:
            db_dir: Directory for ChromaDB persistence (defaults to Vector DB_v2/)
            collection_name: Name of the collection
            threshold: Similarity threshold for recognition (0.0 to 1.0)
            metric: Distance metric - 'cosine' or 'l2' (Euclidean)
            seed: Random seed for reproducible QCNN embeddings
            embedding_mode: Always 'optimized' in v2 (512-dim, grayscale)
        """
        # Use absolute path to Vector DB_v2/ (separate from v1's Vector DB/)
        if db_dir is None:
            db_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'Vector DB_v2'
            )
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.threshold = threshold
        self.metric = metric.lower()
        self.seed = seed
        self.embedding_mode = 'optimized'  # v2 always uses optimized mode
        self.pickle_path = os.path.join('models', 'recognition_db_v2.pkl')

        # Initialize 8-qubit Hierarchical QCNN with Haar Wavelet feature extraction
        self.embedding_extractor = QCNNEmbeddingExtractor(seed=seed, use_optimized=True)

        self._init_chromadb()
        self._load_pickle_backup()

    def _init_chromadb(self):
        os.makedirs(self.db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_dir, settings=Settings(allow_reset=True))
        existing_collections = [c.name for c in self.client.list_collections()]

        # Determine HNSW space based on metric
        hnsw_space = "cosine" if self.metric == "cosine" else "l2"

        # v2 only uses optimized collection (512-dim with Haar Wavelet)
        if self.collection_name not in existing_collections:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": hnsw_space, "embedding_dim": "512", "version": "2.0"}
            )
            print(f"Created v2 ChromaDB collection (512-dim, Haar Wavelet) with {hnsw_space} metric")
        else:
            self.collection = self.client.get_collection(self.collection_name)

        print(f"v2 Active collection: {self.collection.name} ({self.embedding_extractor.embedding_dim}-dim)")

    def _load_pickle_backup(self):
        # For backward compatibility: load old pickle DB if exists
        self.embeddings = {}
        if os.path.exists(self.pickle_path):
            try:
                self.embeddings = joblib.load(self.pickle_path)
            except Exception as e:
                print(f"Warning: Could not load pickle backup: {e}")
                self.embeddings = {}

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding using 8-qubit Hierarchical QCNN with Haar Wavelet.

        Args:
            image: RGB or BGR image (any size, will be resized to 64x64)

        Returns:
            L2-normalized 512-dim embedding
        """
        return self.embedding_extractor.extract_embedding(image)

    def recognize(self, query_embedding, n_results=3):
        """
        Recognize a face from its embedding using ChromaDB query matching.

        Args:
            query_embedding: List or numpy array of floats (512-dim)
            n_results: Number of top matches to consider (default: 3)

        Returns:
            Tuple of (name, confidence_score)
        """
        # Convert to list of native Python floats
        if isinstance(query_embedding, np.ndarray):
            query_list = query_embedding.astype(float).tolist()
        else:
            query_list = [float(x) for x in query_embedding]

        # Check if database has any embeddings
        if self.collection.count() == 0:
            return ("Unknown", 0.0)

        # Query ChromaDB for top N matches
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=min(n_results, self.collection.count()),
            include=['distances', 'metadatas', 'documents']
        )

        # Check for valid results
        if not results['ids'] or not results['ids'][0]:
            return ("Unknown", 0.0)

        if not results['distances'] or not results['distances'][0]:
            return ("Unknown", 0.0)

        # Convert distance to similarity based on metric
        distance = results['distances'][0][0]
        score = self._distance_to_similarity(distance)

        meta = results['metadatas'][0][0] if results['metadatas'] and results['metadatas'][0] else {}
        name = meta.get('name', 'Unknown')

        # Log top matches for debugging
        print(f"  Top {len(results['ids'][0])} matches:")
        for i, (dist, meta_item) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
            sim = self._distance_to_similarity(dist)
            print(f"    {i+1}. {meta_item.get('name', 'Unknown')} - similarity: {sim:.4f} (distance: {dist:.4f})")

        if score >= self.threshold:
            return (name, float(score))
        else:
            return ("Unknown", float(score))

    def _distance_to_similarity(self, distance):
        """
        Convert distance to similarity score based on the metric.

        ChromaDB cosine distance = 1 - cosine_similarity
        So: cosine_similarity = 1 - distance (range: distance [0,2] -> similarity [1,-1])

        For L2 distance: similarity = 1 / (1 + distance) (range: [0, inf] -> [1, 0])

        Args:
            distance: Raw distance from ChromaDB

        Returns:
            Similarity score (for cosine: higher = more similar)
        """
        if self.metric == "cosine":
            similarity = 1.0 - distance
            return similarity
        else:
            similarity = 1.0 / (1.0 + distance)
            return similarity

    def recognize_with_voting(self, query_embedding, n_results=5):
        """
        Recognize using majority voting from top N matches.
        More robust when multiple embeddings per person exist.

        Args:
            query_embedding: List or numpy array of floats
            n_results: Number of top matches to consider

        Returns:
            Tuple of (name, confidence_score, vote_count)
        """
        # Convert to list of native Python floats
        if isinstance(query_embedding, np.ndarray):
            query_list = query_embedding.astype(float).tolist()
        else:
            query_list = [float(x) for x in query_embedding]

        # Check if database has any embeddings
        if self.collection.count() == 0:
            return ("Unknown", 0.0, 0)

        # Query ChromaDB for top N matches
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=min(n_results, self.collection.count()),
            include=['distances', 'metadatas']
        )

        if not results['ids'] or not results['ids'][0]:
            return ("Unknown", 0.0, 0)

        # Count votes for each person (weighted by similarity)
        votes = {}
        for dist, meta in zip(results['distances'][0], results['metadatas'][0]):
            sim = self._distance_to_similarity(dist)
            name = meta.get('name', 'Unknown')

            if name not in votes:
                votes[name] = {'count': 0, 'total_sim': 0.0, 'max_sim': 0.0}

            votes[name]['count'] += 1
            votes[name]['total_sim'] += sim
            votes[name]['max_sim'] = max(votes[name]['max_sim'], sim)

        # Find best match (highest vote count, then highest avg similarity)
        best_name = "Unknown"
        best_score = 0.0
        best_count = 0

        for name, data in votes.items():
            avg_sim = data['total_sim'] / data['count']
            if data['count'] > best_count or (data['count'] == best_count and avg_sim > best_score):
                best_name = name
                best_score = data['max_sim']  # Use max similarity as confidence
                best_count = data['count']

        print(f"  Voting results: {votes}")

        if best_score >= self.threshold:
            return (best_name, float(best_score), best_count)
        else:
            return ("Unknown", float(best_score), best_count)

    def add_person(self, name, embeddings, phone=None, age=None, user_id=None):
        """
        Add a person with multiple face embeddings.

        Args:
            name: Person's name
            embeddings: List of numpy arrays or lists (each 512-dim)
            phone: Optional phone number
            age: Optional age
            user_id: Optional user_id for linking to PQC encrypted metadata
        """
        # Convert embeddings to list of lists of native Python floats
        embedding_lists = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                embedding_lists.append(emb.astype(float).tolist())
            else:
                embedding_lists.append([float(x) for x in emb])

        # Generate user_id if not provided (for linking to PQC metadata)
        if user_id is None:
            user_id = name.lower().replace(' ', '_')

        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"{name}_{existing_count + i}" for i in range(len(embedding_lists))]

        # Metadata for each embedding (includes user_id for PQC metadata lookup)
        metadatas = [{
            "name": name,
            "user_id": user_id,
            "phone": phone or "",
            "age": str(age) if age else "",
            "version": "2.0"
        }] * len(embedding_lists)

        # Add to ChromaDB
        self.collection.add(
            embeddings=embedding_lists,
            ids=ids,
            metadatas=metadatas,
            documents=[name] * len(embedding_lists)
        )

        # Also update pickle backup
        if name in self.embeddings:
            self.embeddings[name].extend([np.array(e) for e in embeddings])
        else:
            self.embeddings[name] = [np.array(e) for e in embeddings]

        print(f"Added {len(embedding_lists)} v2 embeddings for '{name}' (user_id: {user_id}) to database")

    def save_database(self):
        """Save pickle backup of embeddings"""
        # ChromaDB is persistent, but keep pickle backup
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        joblib.dump(self.embeddings, self.pickle_path)

    def get_stats(self):
        """Get database statistics"""
        from collections import Counter

        count = self.collection.count()
        all_names = []

        if count > 0:
            data = self.collection.get(include=["metadatas"])
            all_names = [m.get("name", "Unknown") for m in data["metadatas"]]

        name_counts = Counter(all_names)

        return {
            'total_people': len(name_counts),
            'total_embeddings': count,
            'people': list(name_counts.keys()),
            'embeddings_per_person': dict(name_counts),
            'version': '2.0',
            'feature_extraction': 'Quantum Haar Wavelet'
        }

    def get_person_count(self):
        """Get the number of unique persons in the database"""
        stats = self.get_stats()
        return stats['total_people']

    def delete_person(self, name):
        """Delete all embeddings for a person"""
        total_deleted = 0

        if self.collection.count() > 0:
            data = self.collection.get(include=["metadatas"])
            ids_to_delete = []
            for i, meta in enumerate(data["metadatas"]):
                if meta.get("name") == name:
                    ids_to_delete.append(data["ids"][i])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                total_deleted = len(ids_to_delete)

        # Remove from pickle backup
        if name in self.embeddings:
            del self.embeddings[name]

        return total_deleted
