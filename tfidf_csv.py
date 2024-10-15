# Importações
import nltk
import pandas as pd
import re
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')  # Não baixa se já estiver atualizado!
from nltk.corpus import stopwords
import imp_setup as imps

#
# Variáveis globais
#
sw = set()


#
# Funções
#
def limpeza_str(texto: str):
    global sw
    texto = texto.lower()
    temp_texto = []
    # Retira todo e qualquer caractere especial (incluindo UNICODE)
    pals = re.sub(r'[^\w\s]|_', ' ', texto, flags=re.UNICODE).split()
    for pal in pals:
        if pal not in sw:
            if pal.isalnum():
                temp_texto.append(pal)
            else:
                temp_texto.append(" ")
    return ' '.join(temp_texto)

def rodar(index: int):
    global sw
    sw = set(stopwords.words(imps.df_metadatasets.loc[index, "Idioma"]))

    col_texto = imps.df_metadatasets.loc[index, "Campo"]
    df_mestre = pd.read_csv(imps.df_metadatasets.loc[index, "Caminho"], sep=imps.df_metadatasets.loc[index, "Separador"], low_memory=False)

    df_tinindo = df_mestre.copy()
    df_tinindo[col_texto] = df_tinindo[col_texto].apply(limpeza_str)

    # Pega a query do usuário
    raw_query = input("Query: ")
    if not raw_query:
        # query = "O gato comeu o rato"
        raw_query = "O gato bebeu o leite"

    # Limpa a query e coloca ela no DataFrame
    query = limpeza_str(raw_query)
    lista_tinindo = df_tinindo[col_texto].to_list()

    #
    # TF
    #

    #
    # Vetorização
    #
    print("\n\n------ TF (com BOW simples) ------\n\n")
    cont_vectz = CountVectorizer()  # Sem limite para max_features
    
    # Query vai ao final da lista para não bangunçar os índices
    pal_f = cont_vectz.fit_transform(lista_tinindo + [query])

    df_pal_f = pd.DataFrame(pal_f.toarray(), columns=cont_vectz.get_feature_names_out())
    df_pal_f.index = pd.Index(df_pal_f.index.to_list()[:-1] + ["q"])


    # Pega o term frequency
    arr_pal_tf = pal_f.toarray()
    df_pal_tf = pd.DataFrame(arr_pal_tf / arr_pal_tf.shape[1], columns=cont_vectz.get_feature_names_out())
    df_pal_tf.index = pd.Index(df_pal_tf.index.to_list()[:-1] + ["q"])

    print("\n -- TF -- \n\n", df_pal_tf.head(10))

    #
    # Semelhança de cossenos
    #
    lista_cos_sim = cosine_similarity(pal_f, pal_f[-1])[:-1]

    # Computa pontuação de semelhantes
    # Se coloca em um Series para saber a ordem em que se encontravam antes de ordenados
    sim_scores = pd.Series(lista_cos_sim.ravel()).sort_values(ascending=False)

    print("\n -- BOW -- \n\n", df_pal_f.head())

    #
    # TOP X
    #
    top_sim_scores = sim_scores[0:min(len(sim_scores), 10)]
    df_top_x = df_mestre.loc[top_sim_scores.index]

    # Adição da coluna de similaridade
    df_top_x.insert(0, "Similaridade", top_sim_scores.to_list(), True)
    df_top_x = df_top_x.loc[df_top_x.index, [col_texto, "Similaridade"]]

    # Exibição
    if df_top_x["Similaridade"].iloc[0] != 0.0:
        print("\n-- Top 10 --\n\n", df_top_x.head(10))
    else:
        print(f"String '{raw_query}' não encontrado!")

    #
    # TF-IDF
    #

    #
    # Vetorização
    #
    print("\n\n------ TF-IDF ------")
    print('*O SciKit utiliza, no seu log, base natural (e) e não base 10\n\n')
    tfidfv = TfidfVectorizer()

    # O Tfidfv da Sci-kit usa IDF(t)=log[N/(1+N)] para a fórmula (evitando assim divisão por zero)
    pal_tfidf = tfidfv.fit_transform(lista_tinindo + [query])

    # Termos (palavras tokens) para servirem de colunas
    termos = tfidfv.get_feature_names_out()

    df_pal_tfidf = pd.DataFrame(pal_tfidf.toarray(), columns=termos)

    print(f"\n-- IDF -- \n\n{pd.Series(tfidfv.idf_, index=termos)}")

    print("\n -- BOW -- \n\n", df_pal_tfidf)

    #
    # Semelança de cossenos
    #
    lista_cos_sim = cosine_similarity(pal_tfidf, pal_tfidf[-1])[:-1]

    # Computa pontuação de semelhantes
    # Se coloca em um Series para saber a ordem em que se encontravam antes de ordenados
    sim_scores = pd.Series(lista_cos_sim.ravel()).sort_values(ascending=False)

    #
    # TOP X
    #
    top_sim_scores = sim_scores[0:min(len(sim_scores), 10)]
    # Dropa valores == 0.0
    top_sim_scores = top_sim_scores[top_sim_scores != 0.0]
    df_top_x = df_mestre.loc[top_sim_scores.index]

    # Adição da coluna de similaridade
    df_top_x.insert(0, "Similaridade", top_sim_scores.to_list(), True)
    df_top_x = df_top_x.loc[df_top_x.index]

    # Exibição
    if not df_top_x.empty:
        print("\n-- Top 10 --\n\n", df_top_x.head(10))
    else:
        print(f"String '{raw_query}' não encontrado!")
