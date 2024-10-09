# Importações
import nltk
import pandas as pd
import re
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')  # Não baixa se já estiver atualizado!
from nltk.corpus import stopwords

# Pré-processamento
doc_mestre = ""
