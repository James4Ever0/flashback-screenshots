from rank_bm25 import BM25Okapi

# Example corpus
corpus = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document?",
]

# Tokenize corpus
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "first document"
tokenized_query = query.lower().split()

# Get scores
scores = bm25.get_scores(tokenized_query)

# Get top 2 results
top_results = bm25.get_top_n(tokenized_query, corpus, n=2)

print("Top results:")
for result in top_results:
    print(result)