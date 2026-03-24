# pip install bm25s
import bm25s
# Corpus of documents
corpus = [
   "a cat is a feline and likes to purr",
   "a dog is the human's best friend and loves to play",
   "a bird is a beautiful animal that can fly",
   "a fish is a creature that lives in water and swims",
]
# Tokenize corpus
corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
# Create BM25 retriever and index corpus
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
# Query
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query)
# Retrieve top 2 results
results, scores = retriever.retrieve(query_tokens, k=2, corpus=corpus)
for rank, (doc, score) in enumerate(zip(results[0], scores[0]), start=1):
   print(f"Rank {rank} (score: {score:.2f}): {doc}")