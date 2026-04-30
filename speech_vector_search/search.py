import numpy as np

from speech_vector_search import util_math
from speech_vector_search.metadata import VectorMetadatas
import faiss


class VectorIndex:
    '''vector search interface.
    vectors                 speech vectors 
    metadata                aligned metadata 
    '''

    def __init__(self, vectors, metadatas, backend="faiss"):
        self.vectors = util_math.l2_normalize_rows(vectors)
        self.metadatas = metadatas
        self.backend_name = backend
        self._validate()
        self._set_backend()
        self.max_k = self.vectors.shape[0]
        self.embedding_dim = self.vectors.shape[1]

    def _set_backend(self):
        if self.backend_name == "brute_force":
            self.backend = BruteForceIndex(self.vectors)
        elif self.backend_name == "faiss": 
            self.backend = FaissIndex(self.vectors)
        else:
            m = f"unsupported backend: {self.backend_name}" 
            m += " (supported: 'brute_force', 'faiss')"
            raise ValueError(m)

    def queries(self, query_vectors, top_k=5):
        '''query by external vectors.
        query_vectors          query matrix
        top_k                  number of neighbours to return
        '''
        top_k = self._check_top_k(top_k)
        query_vectors = np.asarray(query_vectors, dtype=float)
        check_multiple_vectors(query_vectors)
        query_vectors = self._check_query_vector(query_vectors)
        scores, indices = self.backend.search(query_vectors, top_k)
        results = []
        for i in range(len(query_vectors)):
            results.append(self._result(scores[i], indices[i]))
        return results

    def query(self, vector, top_k=5):
        '''query by external vector.
        vector                 query vector
        top_k                  number of neighbours to return
        '''
        top_k = self._check_top_k(top_k)
        vector = np.asarray(vector, dtype=float)
        check_single_vector(vector)
        vector = vector.reshape(1, -1)
        vector = self._check_query_vector(vector)
        scores, indices = self.backend.search_single(vector, top_k)
        return self._result(scores, indices)

    def query_by_index(self, index, top_k=5):
        '''query by prototype index.
        index                  prototype index
        top_k                  number of neighbours to return
        '''
        if index < 0:
            m = f'index must be non-negative but is {index}'
            raise ValueError(m)
        if index >= self.vectors.shape[0]:
            m = f'index {index} out of bounds {self.vectors.shape[0]}'
            raise ValueError(m)
        return self.query(self.vectors[index], top_k=top_k)

    def _result(self, scores, indices):
        '''attach metadata to search output.
        scores                 similarity scores
        indices                prototype indices
        '''
        bad = []
        for index in indices:
            if index < 0 or index >= len(self.metadatas): bad.append(index)
        if bad:
            m = f'indices out of bounds: {bad} (max {len(self.metadatas)-1})'
            raise ValueError(m)
        metadatas = [self.metadatas[int(index)] for index in indices]
        d = {"scores": scores, "indices": indices, "metadatas": metadatas}
        return d
    
    def _validate(self):
        if not isinstance(self.vectors, np.ndarray):
            raise ValueError('vectors must be a numpy array')
        if len(self.vectors.shape) != 2:
            raise ValueError('vectors must be a 2D array')
        if not isinstance(self.metadatas, VectorMetadatas):
            raise ValueError('metadatas must be a VectorMetadatas instance')
        if self.vectors.shape[0] != len(self.metadatas):
            raise ValueError('number of vectors must match number of metadatas')

    def _check_top_k(self, top_k):
        if top_k > self.max_k: 
            top_k = self.max_k
            print(f'Warning: using {self.max_k} as top_k (all vectors)')
        return top_k

    def _check_query_vector(self, vector):
        if vector.shape[1] != self.embedding_dim:
            m = f'query vector dimension {vector.shape[1]} '
            m += f'does not match index {self.embedding_dim}'
            raise ValueError(m)
        return vector

class BruteForceIndex:
    '''brute-force cosine search.
    vectors                  normalized vectors
    '''

    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, query_vectors, top_k):
        '''search nearest neighbours.
        query_vectors          query matrix
        top_k                  number of neighbours to return
        '''
        query_vectors = util_math.l2_normalize_rows(query_vectors)
        scores = np.dot(query_vectors, self.vectors.T)
        order = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores, order

    def search_single(self, query_vector, top_k):
        '''search single query vector.
        query_vector           query vector
        top_k                  number of neighbours to return
        '''
        scores, indices = self.search(query_vector, top_k)
        return scores[0], indices[0]


class FaissIndex:
    '''faiss inner-product search.
    vectors                  normalized vectors
    '''
    def __init__(self, vectors):
        vectors = vectors.astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_vectors, top_k):
        '''search nearest neighbours.
        query_vectors          query matrix
        top_k                  number of neighbours to return
        '''
        query_vectors = util_math.l2_normalize_rows(query_vectors)
        query_vectors = query_vectors.astype("float32")
        scores, indices = self.index.search(query_vectors, top_k)
        return scores, indices

    def search_single(self, query_vector, top_k):
        '''search single query vector.
        query_vector           query vector
        top_k                  number of neighbours to return
        '''
        scores, indices = self.search(query_vector, top_k)
        return scores[0], indices[0]


def build_index(vectors, metadata, backend="faiss"):
    '''create prototype index.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    backend                  search backend name
    '''
    return VectorIndex(vectors, metadata, backend=backend)

def check_single_vector(vector):
    if not isinstance(vector, np.ndarray):
        raise ValueError('vector must be a numpy array')
    if vector.ndim != 1:
        raise ValueError(f'vector must be 1D {vector.shape}')

def check_multiple_vectors(vectors):
    if not isinstance(vectors, np.ndarray):
        raise ValueError('vectors must be a numpy array')
    if vectors.ndim != 2:
        raise ValueError(f'vectors must be 2D {vectors.shape}')
