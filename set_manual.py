import re
import imp_setup as imps
import meta_funcs as meta
from imp_setup import pd, stopwords
from meta_funcs import selecionar_dataset
import math
from tfidf_manual import *

# 
# Ciclo principal
# 
def rodar_manual(index: int, query=""):
    print("\n\n- Modo manual - \n\n")
    df_top_x = pd.DataFrame()
    # Seleção dos datasets
    if index == -1:
        index = selecionar_dataset()
        if index < 0:
            return df_top_x

    # Setup
    on_start(imps.df_metadatasets.loc[index, "Idioma"])
    col_texto = imps.df_metadatasets.loc[index, "Campo"]
    df_mestre = pd.read_csv(imps.df_metadatasets.loc[index, "Caminho"], sep=imps.df_metadatasets.loc[index, "Separador"], low_memory=False)
    if len(df_mestre) > 1000:
        # Abre apenas 1000 entries (igualmente espaçadas)
        print("Dataset tem mais de 1000 linahs!\n"
              "Abrindo apenas 1000 linhas (ao longo do documento)\n")
        n_row = len(df_mestre)
        df_mestre = df_mestre[:n_row:int(n_row/1000)]

    # Limpeza
    df_tinindo = df_mestre.copy()
    df_tinindo[col_texto] = df_tinindo[col_texto].apply(limpar_str)

    # Query do usuário
    while not query:
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
    # TOP 10 Resultados com campos para exibir
    #
    lista_campos = imps.df_metadatasets.loc[index, "Exibir"].split(" ")
    df_mestre["Similaridade"] = lista_cos[1:]
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
    