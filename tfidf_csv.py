# Importações
import glob
import nltk
import pandas as pd
import csv
import re
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')  # Não baixa se já estiver atualizado!
from nltk.corpus import stopwords

#
# Setup
#
# Lista com todos os registros dos datasets
metadatasets = []

def detectar_separador(arquivo_csv):
    with open(arquivo_csv, 'r', encoding='utf-8') as f:
        try:
            sniffer = csv.Sniffer()
            amostra = f.read(1024)
            return sniffer.sniff(amostra).delimiter
        except csv.Error:
            print("Delimitador não encontrado!")
            return input("Insira o delimitador do arquivo: ")


class MetaDataset():
    def __init__(self, nome, caminho):
        campo = None
        delimitador = None

        # Constrói o meta_dataset
        delimitador = detectar_separador(caminho)

        self.nome = nome
        self.caminho = caminho
        self.campo = campo
        self.sep = delimitador

    # Para exibição com 'print'
    def __str__(self):
        texto = f" - Registro do DataSet - \ncNome: {self.nome}\n Caminho: {self.caminho} \n Campo: {self.campo} \n Delimitador: {self.sep} \n - Fim do Registro -"
        return texto


# doc_mestre_path = "descricao_sistema_harmonizado_ncm.csv"
# separador = ';'
# doc_mestre_path = "teste_csv_exemplo.csv"
doc_mestre_path = "teste_csv_exemplo 1.csv"
sw = set(stopwords.words('portuguese'))

# 
# Funções de seleção
# 
def pegar_input(op_texto = ''):
    return input(f'{op_texto}: ')


def registrar_dataset():
    print("Escolha qual DataSet deseja registrar: ")
    arqs = glob.glob("datasets/*.csv")
    max_choice = len(arqs)
    op = 0
    col_texto = ''
    # 
    # Olha arquivos do diretório 'dataset'
    #

    # Se não houver arquivos adequados
    if not arqs:
        print("Sem datasets em formato .csv no diretório datasets!")
        return False

    for i, arq in enumerate(arqs):
        print(f"{i + 1:<4} {arq.replace("datasets/", "")}")

    while (op == 0) or (op > max_choice):
        op = input("Escolha o número do dataset a ser usado: ")
        if op.isnumeric():
            op = int(op)
        else:
            op = 0
    
    doc_mestre_path = arqs[op - 1]
    print(f"arquivo '{doc_mestre_path}' selecionado!")

    meta_dt = MetaDataset('DataSet de Teste', doc_mestre_path)

    df_mestre = pd.read_csv(doc_mestre_path, sep=meta_dt.sep)

    print(df_mestre)
    print(" - Campos - ")
    for i, campo in enumerate(df_mestre.columns.to_list()):
        print(f"{i:<4} {campo}")

    while (col_texto != None) and (col_texto not in df_mestre.columns):
        col_texto = input("Campo: ")

    meta_dt.campo = col_texto
    metadatasets.append(meta_dt)
    print("Registro completo!")
    print(metadatasets[-1])

    return True


# 
# Menu de seleção
# 
# TODO: Transformar em dicionário as listas de opções
ops = [("Registrar DataSet", registrar_dataset), ("Sair", lambda: False)]
ops_l = [x[0].lower() for x in ops]

while True:
    print("Escolha uma das opções: ")
    for i, op in enumerate(ops):
        print(f"{i:<2} - {op[0]}")
    inp = pegar_input()
    if inp.isnumeric():
        inp = int(inp)
    else:
        # Por agora não aceita strings
        continue
    if not ops[inp][1]():
        # break
        break
    

# #
# # Preparo (olha arquivos no diretório)
# #
# doc_choice = 0
# arqs = glob.glob("datasets/*.csv")
# max_choice = len(arqs)

# # Se não houver arquivos adequados
# if not arqs:
#     print("Sem datasets em formato .csv no diretório datasets!")
#     exit()

# # 
# # Olha arquivos do diretório 'dataset'
# #
# for i, arq in enumerate(arqs):
#     print(f"{i + 1:<4} {arq}")

# while (doc_choice == 0) or (doc_choice > max_choice):
#     doc_choice = input("Escolha o número do dataset a ser usado: ")
#     if doc_choice.isnumeric():
#         doc_choice = int(doc_choice)
#     else:
#         doc_choice = 0

# doc_mestre_path = arqs[doc_choice - 1]
# print(f"arquivo '{doc_mestre_path}' selecionado!")

# meta_dt = MetaDataset('DataSet de Teste', doc_mestre_path)

# df_mestre = pd.read_csv(doc_mestre_path, sep=meta_dt.sep)

# print(" - Campos - ")
# for i, campo in enumerate(df_mestre.columns.to_list()):
#     print(f"{i:<4} {campo}")

# while (col_texto != None) and (col_texto not in df_mestre.columns):
#     col_texto = input("Campo: ")

# meta_dt.campo = col_texto
#
# Pré-processamento
#

col_texto = metadatasets[0].campo
df_mestre = pd.read_csv(metadatasets[0].caminho, sep=metadatasets[0].sep, low_memory=False)

def limpeza_str(texto: str):
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
    print(f"Termo '{raw_query}' não encontrado!")

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
df_top_x = df_mestre.loc[top_sim_scores.index]

# Adição da coluna de similaridade
df_top_x.insert(0, "Similaridade", top_sim_scores.to_list(), True)
df_top_x = df_top_x.loc[df_top_x.index, [col_texto, "Similaridade"]]

# Exibição
if df_top_x["Similaridade"].iloc[0] != 0.0:
    print("\n-- Top 10 --\n\n", df_top_x.head(10))
else:
    print(f"Termo '{raw_query}' não encontrado!")
