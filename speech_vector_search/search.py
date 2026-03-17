import numpy as np

from speech_vector_search.normalize import l2_normalize, l2_normalize_rows

try:
    import faiss
except ImportError:
    faiss = None


class BruteForceIndex:
    '''brute-force cosine search.
    vectors                  normalized vectors
    '''

    def __init__(self, vectors):
        self.vectors = l2_normalize_rows(vectors)

    def search(self, query_vectors, top_k):
        '''search nearest neighbours.
        query_vectors          query matrix
        top_k                  number of neighbours to return
        '''
        query_vectors = l2_normalize_rows(query_vectors)
        scores = np.dot(query_vectors, self.vectors.T)
        order = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores, order


class FaissIndex:
    '''faiss inner-product search.
    vectors                  normalized vectors
    '''

    def __init__(self, vectors):
        if faiss is None:
            raise ImportError("faiss is not installed")
        vectors = l2_normalize_rows(vectors).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_vectors, top_k):
        '''search nearest neighbours.
        query_vectors          query matrix
        top_k                  number of neighbours to return
        '''
        query_vectors = l2_normalize_rows(query_vectors).astype("float32")
        scores, indices = self.index.search(query_vectors, top_k)
        return scores, indices


class PrototypeIndex:
    '''unified prototype search interface.
    vectors                  prototype matrix
    '''

    def __init__(self, vectors, metadata, backend="auto"):
        self.vectors = l2_normalize_rows(vectors)
        self.metadata = list(metadata)
        self.backend_name = self._choose_backend(backend)
        if self.backend_name == "faiss":
            self.backend = FaissIndex(self.vectors)
        else:
            self.backend = BruteForceIndex(self.vectors)

    def _choose_backend(self, backend):
        '''select backend name.
        backend                requested backend
        '''
        if backend == "auto":
            return "faiss" if faiss is not None else "brute_force"
        if backend == "faiss":
            if faiss is None:
                raise ImportError("faiss backend requested but not installed")
            return "faiss"
        if backend == "brute_force":
            return "brute_force"
        raise ValueError("backend must be 'auto', 'faiss', or 'brute_force'")

    def query(self, vector, top_k=5):
        '''query by external vector.
        vector                 query vector
        top_k                  number of neighbours to return
        '''
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        scores, indices = self.backend.search(vector, top_k)
        return self._format_result(scores[0], indices[0])

    def query_by_index(self, index, top_k=5):
        '''query by prototype index.
        index                  prototype index
        top_k                  number of neighbours to return
        '''
        return self.query(self.vectors[index], top_k=top_k)

    def _format_result(self, scores, indices):
        '''attach metadata to search output.
        scores                 similarity scores
        indices                prototype indices
        '''
        rows = [self.metadata[int(index)] for index in indices]
        return {
            "scores": np.asarray(scores),
            "indices": np.asarray(indices),
            "metadata": rows,
        }


def build_index(vectors, metadata, backend="auto"):
    '''create prototype index.
    vectors                  prototype matrix
    metadata                 prototype metadata rows
    backend                  search backend name
    '''
    return PrototypeIndex(vectors, metadata, backend=backend)
