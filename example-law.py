
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from scipy.spatial import distance
import time

# Start measuring time for loading model
start_time = time.time()

# Load the model
print("Loading model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Print model loading time
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time} seconds")

# Start measuring time for embedding the messages
start_time = time.time()

# File path where the JSON file is located.
# Replace this with your JSON file path. 
file_path = 'articles.json'

# Open the JSON file and load the data.
with open(file_path, 'r', encoding='utf-8') as file:
    messages = json.load(file)

message_embeddings = embed(messages)

# Print message embedding time
message_embedding_time = time.time() - start_time
print(f"Messages embedded in {message_embedding_time} seconds")

# Function to calculate cosine similarity
def rank_messages(query):
    query_embedding = embed([query])
    similarity_scores = 1 - distance.cdist(query_embedding, message_embeddings, "cosine")[0]
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_messages = [(messages[index], similarity_scores[index]) for index in ranked_indices]
    return ranked_messages

# Starts the loop to get queries
while True:
    query = input("Enter the query (or press enter to exit): ")

    # breaks loop if enter is pressed
    if query == '':
        break

    # start measuring query execution time
    start_query_time = time.time()

    newline = '\n'

    # Ranks messages based on query
    ranked_messages = rank_messages(query)
    for message, score in ranked_messages[:10]:
        print(f'{score}: {message.partition(newline)[0]}')

    # Print query execution time
    query_time = time.time() - start_query_time
    print(f"Query executed in {query_time} seconds")