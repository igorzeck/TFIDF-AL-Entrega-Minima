# 
# Funções de registro
# 
import csv
import imp_setup as imps
from imp_setup import pd, stopwords, glob

# lista_col = ["Nome", "Caminho", "Separador", "Campo", "Idioma"]

# Detecção de separadores
def detectar_separador(arquivo_csv):
    with open(arquivo_csv, 'r', encoding='utf-8') as f:
        try:
            sniffer = csv.Sniffer()
            amostra = f.read(4096)
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
    print(" - Campos - ")
    for i, campo in enumerate(df_mestre.columns.to_list()):
        print(f"{i:<4} {campo}")
    
    col_texto = None
    while (col_texto == None) or (col_texto not in df_mestre.columns):
        col_texto = input("Campo: ")
    
    print(f"Coluna '{col_texto}' selecionada!")
    return col_texto


def _reg_idioma(_):
    op = None
    while True:
        op = input("Insira o nome do idioma (em inglês) dos documentos: ")
        op_clean = op.strip().lower()
        if op_clean not in stopwords.fileids():
            print("Idioma inválido!")
        else:
            print(f"Idioma {op} selecionado!")
            break
    return op_clean



def registrar_dataset():
    dict_op = {
        "Nome": lambda _: input("Nome do registro: "),
        "Caminho": _reg_caminho,
        "Separador": _reg_sep,
        "Campo": _reg_campo,
        "Idioma": _reg_idioma
    }
    linha_dict = {}
    for k in dict_op.keys():
        ret_val = dict_op[k](linha_dict)
        if ret_val:
            linha_dict[k] = ret_val
        else:
            return False
    
    # Adição da linha contendo os metadatasets
    imps.df_metadatasets = pd.concat([imps.df_metadatasets, pd.DataFrame([linha_dict])], ignore_index=True)
    print(imps.df_metadatasets.iloc[-1])
    imps.save_mdt(imps.df_metadatasets)
    print("Registro completo!")
    return True


def exibir_datasets():
    """
    Exibe os datasets registrados no DataFrame de registro de datasets principal
    """
    print(imps.df_metadatasets)
    return True
