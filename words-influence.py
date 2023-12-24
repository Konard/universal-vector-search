import math
from collections import defaultdict

def make_doc_words_count_dict(doc_words):
  """ 
  Make a dictionary that contains count of each word in a document.
  doc_words - the list of words of the text document.
  """

  doc_word_count_dict = defaultdict(int)
  for word in doc_words:
    doc_word_count_dict[word] += 1
  return doc_word_count_dict

def make_word_docs_count_dict(docs_words):
  """ 
  Make a dictionary that contains count of docs for each word in these docs.
  docs_words - a list of documents (each document represented as a list of words).
  """

  word_docs_count_dict = defaultdict(int)
  for doc in docs_words:
    for word in set(doc):
      word_docs_count_dict[word] += 1
  return word_docs_count_dict

def compute_tf(doc_word_count_dict, doc_words):
  """ 
  Compute term frequency.
  doc_word_count_dict - a dictionary with words and their counts.
  doc_words - the document text.
  """

  tf_dict = {}
  total_words = len(doc_words)
  for word, count in doc_word_count_dict.items():
    tf_dict[word] = count / float(total_words)
  return tf_dict

def compute_idf(docs_words):
  """
  Compute inverse document frequency.
  docs_words - a list of documents (each document represented as a list of words).
  """

  idf_dict = {}
  total_docs = len(docs_words)
  
  word_docs_count_dict = make_word_docs_count_dict(docs_words)

  # Compute IDF for each word
  for word, count in word_docs_count_dict.items():
    idf_dict[word] = math.log(total_docs / float(count))
    
  return idf_dict

def compute_tfidf(tf, idf):
  """
  Compute TF*IDF.
  tf - term frequency of words in a document.
  idf - inverse document frequency of words.
  """
  
  tfidf = {}
  for word, tf_val in tf.items():
    tfidf[word] = tf_val * idf[word]
  return tfidf

docs = ['the cat sat on my bed', 'there bird sat on my bed', 'the bird sat on my window']

docs_words = [doc.split(' ') for doc in docs]

idf = compute_idf(docs_words)
print(idf)

for i, doc_words in enumerate(docs_words):
  doc_word_count_dict = make_doc_words_count_dict(doc_words)
  tf = compute_tf(doc_word_count_dict, doc_words)
  tfidf = compute_tfidf(tf, idf)
  print(f"TF-IDF of Doc {i+1}: {tfidf}")
