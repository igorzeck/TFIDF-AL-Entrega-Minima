# 
# Funções de registro
# 
import csv
import re
import imp_setup as imps
from imp_setup import pd, stopwords, glob
from tfidf_sklearn import exe_tfidf

#
# Variáveis globais
#
sw = set()

#
# Funções
#
# Detecção de separadores
def detectar_separador(arquivo_csv):
    with open(arquivo_csv, 'r', encoding='utf-8') as f:
        try:
            sniffer = csv.Sniffer()
            amostra = f.read(8192)
            delimitador = sniffer.sniff(amostra).delimiter
            print(f"Delimitador do arquivo identificado: {delimitador}")
            return delimitador
        except csv.Error:
            print("Delimitador não encontrado!")
            return input("Insira o delimitador do arquivo: ")


# Escolha do arquivo .csv
def _reg_caminho(dict_inp: dict):
    arqs_csv = glob.glob("datasets/*.csv")
    arqs_tsv = glob.glob("datasets/*.tsv")
    arqs = arqs_csv + arqs_tsv
    max_choice = len(arqs)
    op = 0

    # 
    # Olha arquivos do diretório 'dataset'
    #

    # Se não houver arquivos adequados
    if not arqs:
        print("Sem datasets em formato .csv ou .tsv no diretório datasets!")
        return False
    
    for i, arq in enumerate(arqs):
        print(f"{i + 1:<4} {arq.replace("datasets/", "")}")

    while (op == 0) or (op > max_choice):
        op = input("Escolha o número do dataset a ser usado: ")
        if op.isnumeric():
            op = int(op)
        else:
            op = 0

    print(f"Arquivo '{arqs[op - 1]}' selecionado!")
    return arqs[op - 1]

def _reg_sep(dict_inp: dict):
    return detectar_separador(dict_inp["Caminho"])

# Escolha da coluna do dataset para se usar de coleção de documentos
def _reg_campo(dict_inp: dict):
    df_mestre = pd.read_csv(dict_inp["Caminho"], sep=dict_inp["Separador"])

    # print(df_mestre)
    print("Escolha o campo no qual fará a busca")
    col_n = 0
    for i, campo in enumerate(df_mestre.columns.to_list()):
        print(f"{i:<4} {campo}")
        col_n += 1
    
    col_texto = None
    while not col_texto:
        inp = input("Campo: ")
        if not inp.isnumeric():
            if not df_mestre.columns[df_mestre.columns == inp].empty:
                col_texto = df_mestre.columns[df_mestre.columns == inp][0]
        else:
            inp = int(inp)
            if (inp >= 0 and inp < col_n):
                col_texto = df_mestre.columns[inp]
    print(f"Coluna '{col_texto}' selecionada!")
    return col_texto


def _reg_idioma(dict_inp: dict):
    # Detecção de idioma
    df_linha = pd.read_csv(dict_inp["Caminho"], sep=dict_inp["Separador"])
    df_linha = df_linha.head(5)
    langs = stopwords.fileids()
    f_total = []
    for lang in langs:
        f_lang = 0
        sw_atual = stopwords.words(lang)
        for row_list in df_linha[dict_inp["Campo"]]:
            for word in row_list.split(" "):
                if word in sw_atual:
                    f_lang += 1
        f_total.append(f_lang)
    langs_sort = sorted(list(zip(langs, f_total)), key=lambda tupla: tupla[1], reverse=True)
    langs_sort = [tupla[0] for tupla in langs_sort]
    op_clean = langs_sort[0] if langs_sort[1] != 'english' else langs_sort[1]
    op = input(f"Idioma detectado: '{op_clean}'. Selecionar este? (S/N)\n:")
    if not op.lower().startswith("s"):
        op = None
        print("Não")
    while not op:
        op = input("Insíra o nome do idioma (em inglês) dos documentos: ")
        op_clean = op.strip().lower()
        if op_clean not in stopwords.fileids():
            print("Idioma inválido!")
            op = None
        else:
            print(f"Idioma '{op}' selecionado!")
    return op_clean


def _reg_exibir(dict_inp: dict):
    df_mestre = pd.read_csv(dict_inp["Caminho"], sep=dict_inp["Separador"])
    campo_exibir = []
    op = "Similaridade"
    while op:
        campo_exibir.append(op)
        op = selecionar_op(
            df_mestre.to_dict(),
            "Selecione os campos para exibir como resultado (um de cada vez)\n"\
            "Enter para terminar",
            retorno="Chave"
        )
    if len(campo_exibir) == 1:
        campo_exibir.append(dict_inp["Campo"])
    return " ".join(campo_exibir)


def _reg_imagem(dict_inp: dict):
    df_mestre = pd.read_csv(dict_inp["Caminho"], sep=dict_inp["Separador"])
    return selecionar_op(
        df_mestre.to_dict(),
        "Selecione o campo contendo o caminho das imagems"\
        " (Enter para caso não haja nenhum)",
        retorno="Chave"
    )


def _reg_matriz(dict_inp: dict):
    """
    Criar matriz de recomendação (similaridade de cosseno entre todos elementos)
    """
    to_reg = selecionar_op({"Sim": True, "Não": False},
                  "Criar similaridade de cosseno para todos elementos do dataset?"\
                    "\nIsso vai possibilitar recomendação entre itens, mas pode demorar.")
    if to_reg:
        df_mestre = pd.read_csv(dict_inp["Caminho"], low_memory=False)
        df_sim = exe_tfidf(df_mestre, dict_inp["Campo"], dict_inp["Idioma"])
        df_sim.to_csv(f"save_state/{dict_inp["Nome"]}")
        return True
    else:
        return False
        

def _reg_nome(dict_inp: dict):
    while True:
        inp_nome = input("Nome do registro: ")
        if "Nome" in dict_inp.keys():
            if inp_nome in dict_inp["Nome"]:
                print("Nome já registrado!")
                inp_nome = None
        else:
            return inp_nome


def _reg_descricao(_):
    return input("Breve descrição do dataset\n: ")


def get_reg_ops():
    dict_op = {
        "Caminho": _reg_caminho,
        "Separador": _reg_sep,
        "Campo": _reg_campo,
        "Idioma": _reg_idioma,
        "Exibir": _reg_exibir,
        "Imagem": _reg_imagem,
        "Nome": _reg_nome,
        "Descricao": _reg_descricao,
        "Recomendar": _reg_matriz,
    }
    return dict_op


def mudar_modo():
    """
    Muda modo de "Manual" para "Sci-kit" ou vice-versa
    """
    modo_atual = imps.default_params["Modo"][0]
    imps.default_params["Modo"] = "Sci-kit" if modo_atual == "Manual" else "Manual"
    return True


def selecionar_op(ops: dict, preamble: str, query: str = ": ", retorno: str = "Valor"):
    """
    Exibe e seleciona as opções de um dicionário, variável 'retorno' define o que retornar
    Se retorna 'Valor', 'Chave' ou 'Retorno' (para o retorno de uma função no dicionário)
    """
    print(preamble)
    # Exibição
    for i_op, op in enumerate(ops):
        print(f"{i_op:<2} - {op}")
    
    while True:
        inp = input(query).lower()
        for i_op, op in enumerate(ops):
            if (op.lower() == inp) or (inp == str(i_op)):
                if retorno == "Retorno":
                    return ops[op]() # Retorna resultado
                elif retorno == "Valor":
                    return ops[op]
                elif retorno == "Chave":
                    return op  # Retorna chave
        if inp == "":
            break
        print("Opção inválida!")
    return False


def registrar_dataset():
    dict_modo = {"Direto": True,
                 "Completo": False}
    dict_op = get_reg_ops()

    modo_direto = selecionar_op(
                  dict_modo,
                  "Modo rápido (Direto) ou completo de registro?"
                  )

    linha_dict = {}
    for k in dict_op.keys():
        if modo_direto:
            if k == "Nome":
                linha_dict[k] = linha_dict["Caminho"]
                continue
            if k == "Exibir":
                linha_dict[k] = "Similaridade " + linha_dict["Campo"]
                continue
            if k == "Imagem":
                linha_dict[k] = None
                continue
            if k == "Recomendar":
                linha_dict[k] = False
                continue
            if k == "Descricao":
                linha_dict[k] = ""
                continue
        ret_val = dict_op[k](linha_dict)
        if ret_val:
            linha_dict[k] = ret_val
    
    # Adição da linha contendo os metadatasets
    df_criado = pd.DataFrame([linha_dict])
    imps.df_metadatasets = pd.concat([imps.df_metadatasets, df_criado], ignore_index=True)
    print(imps.df_metadatasets.iloc[-1])
    imps.save_mdt(imps.df_metadatasets)
    print("Registro completo!")
    return df_criado


def exibir_datasets():
    """
    Exibe os datasets registrados no DataFrame de registro de datasets principal
    """
    print(f"\n- Datasets registrados -\n\n{imps.df_metadatasets}\n")
    return True


def selecionar_dataset(inp: int = -1):
    exibir_datasets()
    # Verifica se hpa algum dataset a ser selecionado
    if imps.df_metadatasets.empty:
        print("Não há datasets para serem selecionados")
        return -1
    while inp == -1:
        inp = input("Selecione o ID do dataset: ")
        if inp.isnumeric():
            inp = int(inp)
            if inp >= 0 and inp < imps.df_metadatasets.shape[0]:
                break
        
        # Caso não saia da função
        print("Valor inválido")
        inp = -1
    return inp


def recomendar(_):
    pass