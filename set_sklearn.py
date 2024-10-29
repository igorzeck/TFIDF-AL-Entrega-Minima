# Importações
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')  # Não baixa se já estiver atualizado!
from nltk.corpus import stopwords
from meta_funcs import selecionar_dataset
import imp_setup as imps
import meta_funcs as meta
from tfidf_sklearn import *

#
# Variáveis globais
#
meta.sw = set()
# Segmentar funções de TF e de TF-IDF, além de BOW em geral

#
# Funções
#
def rodar_sklearn(index: int, raw_query=""):
    print("\n\n- Modo Sci-kit e NLTK - \n\n")
    df_top_x = pd.DataFrame()
    if index == -1:
        index = selecionar_dataset(index)
        if index < 0:
            return df_top_x

    df_mestre = pd.read_csv(imps.df_metadatasets.loc[index, "Caminho"], sep=imps.df_metadatasets.loc[index, "Separador"], low_memory=False)

    # Pega a query do usuário
    while not raw_query:
        raw_query = input("Query: ")
    
    if not raw_query:
        print("Input inválido")
        return df_top_x
    #
    # TF-IDF
    #
    lista_campos = imps.df_metadatasets.loc[index, "Exibir"].split(" ")
    df_sim = exe_tfidf(df_mestre,
                        imps.df_metadatasets.loc[index, "Campo"],
                        imps.df_metadatasets.loc[index, "Idioma"],
                        raw_query)
    # TODO: Arrumar para caso a query esteja vazia!
    df_mestre["Similaridade"] = df_sim[1:].reset_index(drop=True)

    #
    # TOP 10
    #
    df_top_x = df_mestre.copy()
    df_top_x.sort_values("Similaridade", ascending=False, inplace=True)
    df_top_x = df_top_x.head(10)
    df_top_x = df_top_x[df_top_x["Similaridade"] != 0.0]
    df_top_x = df_top_x[lista_campos]

    # Exibição
    if not df_top_x.empty:
        print("\n-- Top 10 --\n\n", df_top_x)
    else:
        print(f"String '{raw_query}' não encontrado!")
    
    return df_top_x
