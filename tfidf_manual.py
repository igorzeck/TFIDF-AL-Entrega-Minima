# 
# -- Importações --
#
import re
import math
if __name__ == "__main__":
    import pandas as pd
    import nltk
    nltk.download('stopwords')  # Não baixa se já estiver atualizado!
    from nltk.corpus import stopwords
else:
    import imp_setup as imps
    import meta_funcs as meta
    from imp_setup import pd, stopwords
    from meta_funcs import selecionar_dataset

#
# -- Variáveis --
#
sw = set()

#
# -- Funções --
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


#
# Funções de auxílio
#
def ciclar_v(v, n=-1):
    """
    Percorre ciclicamente o vetor (lista) v em n passos
    Se n não for fornecido, percorre quantas vezes for chamado
    """
    x = 0
    for _ in range(n):
        yield v[x]
        x = (x + 1) if (x + 1) < len(v) else 0


def aplicar_op(v1, op, v2):
    """
    Aplica operação op nos elementos de v1 a partir dos de v2 e retorna resultado
    em um vetor do mesmo tamanho de v1
    """
    vr = []
    genv2 = ciclar_v(v2, len(v1))
    for i in range(len(v1)):
        vr.append(op(v1[i], next(genv2)))
    return vr


# Generalização de multiplicação para elementos inteiros e/ou float
_multx_ = lambda n1, n2: n1 * n2


#
# Funções de álgebra
#
def prod_escalar(v1, v2) -> float:
    """
    Retorna o produto escalar de dois vetores.
    Implicitio que eles tem a mesma dimensão
    """
    return sum(aplicar_op(v1, _multx_, v2))


def comp_sim_cos(list_v, vect) -> list:
    """
    Computa similaridade de cosseno entre uma coleção de vetores e um vetor
    Retorna lista de cossenos
    """
    return [(prod_escalar(v_el, vect)/(math.sqrt(prod_escalar(v_el, v_el))*math.sqrt(prod_escalar(vect, vect)))) for v_el in list_v]


#
# Funções de TF-IDF
#
def dimensionar(lista_limpa) -> tuple:
    """
    Faz, a partir de uma coleção limpa e utilizando-se Sets,
    uma tupla com as dimensões
    """
    dimen_set = set()
    for texto in lista_limpa:
        for pal in texto.split(" "):
            dimen_set.add(pal)
    return tuple(dimen_set)


def arr_bowrizar(lista_limpa, lista_dimen):
    """
    Gera, a partir de um array (coleção), lista Bag of Words (BOW)
    Elementos são Term Frequency (TF) Raw (contangem absoluta por documento)
    """
    arr_dimen = []
    for doc in lista_limpa:
        vect = []
        doc_list = doc.split(" ")
        for token in lista_dimen:
            vect.append(doc_list.count(token))
        arr_dimen.append(vect)
    return arr_dimen


def term_frequency(l_vect, rel=True) -> list[list]:
    """
    rel: Define se ocorrerá divisão de cada elemento da coleção por algum valor, relativando
    """
    if rel:
        dimen = len(l_vect[0])
        n_vect = [aplicar_op(vect, _multx_, [1/dimen]) for vect in l_vect]
    else:
        n_vect = l_vect
    return n_vect


def inverse_doc_f(l_vect, suav=True, modo_garcia=False):
    """
    Coleção de documentos vetorizados -> lista com idfs por termo
    """
    l_idfs = []
    n_docs = len(l_vect)
    n_dimen = len(l_vect[0])  # Pega dimensão do primeiro vetor
    base = 10 if modo_garcia else math.e
    suav = float(suav)
    for i_termo in range(n_dimen):
        # Conta ocorrência transdocumental
        idf_t = math.log(
                (n_docs + suav)/([doc[i_termo] != 0 for doc in l_vect].count(True) + suav),
                base
            ) + float(not modo_garcia)
        l_idfs.append(idf_t)

    return l_idfs


def tfidf(*args, suav_idf=True, modo_garcia=False, rel_tf=False):
    """
    Aceita lista com strings limpas e lista com dimensões nesta ordem
    ou TF, IDF nesta ordem
    """
    if isinstance(args[0][0], str):  # Por agora essa é a solução
        docs_limpos = args[0]
        if len(args) >= 2:
            lista_dimen = args[1]
        else:
            lista_dimen = dimensionar(docs_limpos)
        arr_bow_ = arr_bowrizar(docs_limpos, lista_dimen)
        tf_ = term_frequency(arr_bow_, rel_tf)
        idf_ = inverse_doc_f(arr_bow_, suav_idf, modo_garcia)
    else:
        tf_ = args[0]
        idf_ = args[1]
    l_tfidf_ = []
    for v in tf_:
        v_tfidif = aplicar_op(v, _multx_, idf_)
        l_tfidf_.append(v_tfidif)

    return l_tfidf_


def on_start(lang: str):
    global sw
    sw = set(stopwords.words(lang))


def _fallback_():
    on_start('portuguese')
    path_arq = "datasets/descricao_sistema_harmonizado_ncm.csv"
    campo_busca = "NO_NCM_POR"
    separador = ";"
    try:
        df = pd.read_csv(path_arq, sep=separador)
        df = df[:len(df):10]  # Usa apenas 10% do arquivo
    except FileExistsError:
        print(f"Arquivo {path_arq} não encontrado!")
        return False

    query = input("\nQuery: ")

    query = limpar_str(query)
    df_limpo = df[campo_busca].apply(limpar_str)
    lista_limpa = [query] + df_limpo.to_list()
    lista_tfidf = tfidf(lista_limpa)

    lista_cos = comp_sim_cos(lista_tfidf, lista_tfidf[0])

    tfidf_df = df.copy()
    tfidf_df.insert(0, "Similaridade", lista_cos[1:])

    # Coloca coluna no final do DataFrame
    col_busca = tfidf_df.pop(campo_busca)
    tfidf_df = tfidf_df.join(col_busca)


    tfidf_df.sort_values("Similaridade", ascending=False, inplace=True)
    tfidf_df = tfidf_df.head(10)
    tfidf_df = tfidf_df[tfidf_df["Similaridade"] != 0.0]

    return tfidf_df

if __name__ == "__main__":
    tfidf_df = _fallback_()
    print("\n-- Top 10 -- \n")
    print(tfidf_df)
    count = 1
    for _, row in tfidf_df.iterrows():
        print(f"\n -- #{count} --\n")
        print("Similaridade:", row["Similaridade"],
            "\nDescrição:", row["NO_NCM_POR"])
        count += 1