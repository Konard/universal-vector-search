import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from scipy.spatial import distance
import time

# Start measuring time
start_time = time.time()

# Load the model
print("Loading model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Define your predefined list of messages
messages = ["Hello, my name is AI.", "Привет, меня зовут AI."]
message_embeddings = embed(messages)

# Print startup time
startup_time = time.time() - start_time
print(f"Model loaded in {startup_time} seconds")

# Function to calculate cosine similarity
def rank_messages(query):
    query_embedding = embed([query])
    similarity_scores = 1 - distance.cdist(query_embedding, message_embeddings, "cosine")
    ranked_indices = np.argsort(similarity_scores[0])[::-1]
    ranked_messages = [messages[index] for index in ranked_indices]
    return ranked_messages

# Starts the loop to get queries
while True:
    query = input("Enter the query (or press enter to exit): ")

    # breaks loop if enter is pressed
    if query == '':
        break

    # Start measuring query execution time
    start_query_time = time.time()

    # Ranks messages based on query
    ranked_messages = rank_messages(query)
    print('Ranked Messages:', ranked_messages)

    # Print query execution time
    query_time = time.time() - start_query_time
    print(f"Query executed in {query_time} seconds")