import os

# On macOS, faiss and torch bundle separate OpenMP runtimes. Loading both in
# one test process aborts unless duplicates are allowed; single-threaded
# OpenMP avoids crashes from the two runtimes. Test arrays need no threads.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
