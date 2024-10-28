# Importações
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
from nltk.corpus import stopwords


#
# Variáveis
#
sw = set()

#
# Funções
#
def limpar_str(texto: str):
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


def exe_tfidf(df: pd.DataFrame, col:str, lang='english', query = None):
    global sw
    sw = stopwords.words(lang)
    
    if query:
        query_limpa = limpar_str(query)
    else:
        query_limpa = query

    l_limpa = df[col].apply(limpar_str).tolist()

    return _tfidf_nltk(l_limpa, query_limpa)


def _tfidf_nltk(l_limpa, query_limpa = None, idf_suav = True):
    """
    Gera matriz TFIDF com lista limpa e query, se houver
    """
    tfidfv = TfidfVectorizer(smooth_idf=idf_suav)
    if query_limpa:
        l_query = [query_limpa] + l_limpa
        tfidf_arr = tfidfv.fit_transform(l_query).toarray()
        cos_sim = cosine_similarity(tfidf_arr, [tfidf_arr[0]])
        df_sim = pd.DataFrame(cos_sim, l_query, [query_limpa])
    else:
        tfidf_arr = tfidfv.fit_transform(l_limpa).toarray()
        cos_sim = cosine_similarity(tfidf_arr)
        df_sim = pd.DataFrame(cos_sim, l_limpa, l_limpa)
    return df_sim


def _fallback_(query = None):
    #path_arq = "datasets/descricao_sistema_harmonizado_ncm.csv"
    #campo_busca = "NO_NCM_POR"
    separador = ","
    path_arq = "datasets/teste_objects_description_pt.csv"
    campo_busca = "description"
    try:
        df = pd.read_csv(path_arq, sep=separador)
    except FileExistsError:
        print(f"Arquivo {path_arq} não encontrado!")
        return False

    if not query:
        query = input("\nQuery: ")

    sim_df = exe_tfidf(df, campo_busca, "portuguese", query)
    tfidf_df = df.copy()
    tfidf_df.insert(0, "Similaridade", sim_df[1:].reset_index(drop=True))

    # Coloca coluna no final do DataFrame
    col_busca = tfidf_df.pop(campo_busca)
    tfidf_df.join(col_busca)

    tfidf_df.sort_values("Similaridade", ascending=False, inplace=True)
    tfidf_df = tfidf_df.head(10)
    tfidf_df = tfidf_df[tfidf_df["Similaridade"] != 0.0]

    return tfidf_df


if __name__== "__main__":
    print(_fallback_())
