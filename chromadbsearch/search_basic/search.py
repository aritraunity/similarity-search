import chromadb

from chromadb.utils import embedding_functions

# Step 1: Initialize and Embedding using Sentence Transformers
# During research, look for models which server the embedding purpose
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

# Step 2: Initialize the Chroma DB client 
client = chromadb.Client()
# Step 3: Define a collection name
collection_name = "my_grocery_collection"

# Data Inputs
texts = [
    'fresh red apples',
    'organic bananas',
    'ripe mangoes',
    'whole wheat bread',
    'farm-fresh eggs',
    'natural yogurt',
    'frozen vegetables',
    'grass-fed beef',
    'free-range chicken',
    'fresh salmon fillet',
    'aromatic coffee beans',
    'pure honey',
    'golden apple',
    'red fruit'
]

def main ():

    # Step 4 : Creating the Collection in DB Store of Chroma DB
    # Remeber Chroma DB stores embeddings
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "A Collection for storing Grocery Data"
        },
        # This is where it becomes configurable, HNSW: hierarchially navigable small worlds algorithm
        # Todo: Clarify about the collection configurations
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": ef
        }
    )
    # Unique ID for each text
    ids = [f"food_{id + 1}" for id, text in enumerate(texts)]
    print (f"Collection created {collection.name}")
    # Step 5 Adding into collection
    # Adds docs, updates metadata for each, adds the ids
    collection.add(
        documents=texts,
        metadatas=[{"source": "grocery_store","category": "food"} for _ in texts],
        ids = ids
    )
    # Step 6: Getting all Items
    all_items = collection.get()
    print("Collection Contents")
    print (f"Number of Documents: {len(all_items['documents'])}")
    
    #Step 7: Performing Similarity Search
    # At this point we have the collections with metadatas and ids
    query_term = "apple"
    results = collection.query(
        query_texts=[query_term],
        # top k
        n_results= 3
    )
    print(f"Query Results {query_term}")
    print(results)
main()