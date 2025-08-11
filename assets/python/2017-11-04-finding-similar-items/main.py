import os

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "demo_collection"
# DB_PATH = "path/to/db"
DISTANCE = models.Distance.COSINE
DOCS_PATH = "../../../_posts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# running qdrant in local mode suitable for experiments
client = QdrantClient(":memory:")

payload = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".md"):
        with open(os.path.join(DOCS_PATH, file), "r") as f:
            content = f.read()
            payload.append(
                {
                    "document": content,
                    "source": file,
                }
            )

docs = [models.Document(text=data["document"], model=MODEL_NAME) for data in payload]
ids = [42, 2]

client.create_collection(
    COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(MODEL_NAME), distance=DISTANCE
    ),
)

client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=docs,
    ids=ids,
    payload=payload,
)

search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=models.Document(text="partial correlations", model=MODEL_NAME),
).points
print(search_result)
