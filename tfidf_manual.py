import imp_setup as imps
import meta_funcs as meta
from imp_setup import pd, stopwords
from meta_funcs import selecionar_dataset, limpar_str
import math


#
# -- Funções --
#
# TODO: Corrigir formatação docstring

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
def prod_escalar(v1, v2) -> int:
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


def arr_bowrizar(lista_limpa, lista_dimen=[]):
    """
    Gera, a partir de um array (coleção), lista Bag of Words (BOW)
    Elementos são Term Frequency (TF) Raw (contangem absoluta por documento)\n
    <h2>Parâmetros</h2>
    **lista_limpa**:Lista com elementos de onde será formada a Bag of Words\n
    **lista_dimen**:Lista com dimensões (tokens), se não for passada será criada
    """
    arr_dimen = []
    if lista_dimen:
        lista_dimen = dimensionar(lista_limpa)
    for doc in lista_limpa:
        vect = []
        for token in lista_dimen:
            vect.append(doc.count(token))
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


def tfidf(*args, suav_idf=False, modo_garcia=False, rel_tf=True):
    """
    Aceita lista com strings limpas, lista com dimensões nesta ordem
    ou TF, IDF nesta ordem
        <h2>Parâmetros</h2>
            \n>
            **modo_garcia** (*bool*): Se True fórmula exata da aula.
            \n>
            **rel_tf** (*bool*): Se False, tf é frequência absoluta e se True é relativo a dimensão (frequência relativa)
    """
    # TODO: Arrumar isso
    if isinstance(args[0][0], str):  # Por agora essa é a solução
        docs_limpos = args[0]
        lista_dimen = args[1]
        arr_bow_ = arr_bowrizar(docs_limpos, lista_dimen)
        tf_ = term_frequency(arr_bow_, rel_tf)
        idf_ = inverse_doc_f(arr_bow_, suav_idf, modo_garcia)
    else:
        tf_ = args[0]
        idf_ = args[1]
    l_tfidf_ = []
    for v in tf_:
        l_tfidf_.append(aplicar_op(v, _multx_, idf_))
    return l_tfidf_


# 
# Ciclo principal
# 
def rodar_manual(index: int = -1):
    print("\n\n- Modo manual - \n\n")
    # Seleção dos datasets
    if index == -1:
        index = selecionar_dataset()
        if index < 0:
            return False
    
    # Limpeza
    meta.sw = set(stopwords.words(imps.df_metadatasets.loc[index, "Idioma"]))
    col_texto = imps.df_metadatasets.loc[index, "Campo"]
    df_mestre = pd.read_csv(imps.df_metadatasets.loc[index, "Caminho"], sep=imps.df_metadatasets.loc[index, "Separador"], low_memory=False)

    df_tinindo = df_mestre.copy()
    df_tinindo[col_texto] = df_tinindo[col_texto].apply(limpar_str)

    # Query do usuário
    raw_query = input("Query: ")

    # Limpeza a query e adição em um DataFrame
    query = limpar_str(raw_query)
    lista_tinindo = [query] + df_tinindo[col_texto].to_list()
    print("Limpeza feito")

    #
    # BOW
    #
    lista_dimen = dimensionar(lista_tinindo)
    print("Tokens encontrados")
    bow = arr_bowrizar(lista_tinindo, lista_dimen)
    print("BOW feito")

    #
    # TF
    #
    lista_tf = term_frequency(bow, rel=False)
    print("TF feito")

    #
    # IDF
    #
    lista_idf = inverse_doc_f(bow, False, False)
    print("IDF feito")

    #
    # TF-IDF
    #
    lista_tfidf = tfidf(lista_tf, lista_idf)
    print("TF-IDF feito")

    #
    # Similaridade de cossenos
    #
    lista_cos = comp_sim_cos(lista_tfidf, lista_tfidf[0])
    print("Similaridade do cosseno feito")


    #
    # TOP 10 Resultados
    #
    df_top_x = pd.DataFrame({"Similaridade":lista_cos[1:],
                             col_texto:lista_tinindo[1:]})
    df_top_x.sort_values("Similaridade", ascending=False, inplace=True)
    df_top_x = df_top_x[0:min(len(df_top_x), 10)]

    df_top_x = df_top_x[df_top_x["Similaridade"] != 0.0]

    # Exibição
    if not df_top_x.empty:
        print("\n-- Top 10 --\n\n", df_top_x)
    else:
        print(f"String '{raw_query}' não encontrado!")
    return True
    