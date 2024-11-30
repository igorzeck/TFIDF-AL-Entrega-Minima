import pandas as pd
import imp_setup as imps
import meta_funcs as meta, glob
from set_sklearn import rodar_sklearn
from set_manual import rodar_manual
from fasthtml.common import *


#
# Globais
#
df_novo = pd.DataFrame()
app = FastHTML()
cache_dict = {}
index = 0
modo_atual = "manual"

# TODO: Página com resultado dos registros
# TODO: Página confirmando criação de dataset
# TODO: Utilizar o campo de descrição para poder fazer buscas nos metadatasets
@app.get("/")
def home(query: str = ""):
    df = pd.DataFrame()
    if index != -1:
        df_nome = imps.df_metadatasets.iloc[index]["Nome"]
        df_desc = imps.df_metadatasets.iloc[index]["Descricao"]

    if query:
        if modo_atual == "Scikit":
            df = rodar_sklearn(index, query)
        else:
            df = rodar_manual(index, query)
    elif index == -1:
        df = imps.df_metadatasets.copy()
        df_nome = "Datasets"
        df_desc = "Datasets registrados."

    tab = Table(eval(html2ft(df.to_html())))

    return Title("Pesquisa"), Main(
                Style(imps.styles),
                Div(
                    Form(
                        Button(f"Modo para: {"Manual" if modo_atual == "Scikit" else "Scikit"}"),
                        action="/toggle_tfidf", method="get"
                    )
                ),
                H1("Buscar", cls="h1"),
                P("Dataset ", B(f'{df_nome}', style="color: #4CAF50"), " selecionado!"),
                P("Descrição: ", df_desc),
                Div(
                    Form(
                        Button("Registrar dataset"),
                        action="/registrar", method="get",
                        cls="h-item-form"
                    ),
                    Form(
                        Button("Exibir datasets"),
                        action="/exibir_dataset", method="post",
                        cls="h-item-form"
                    ),
                    Form(
                        Button("Selecionar datasets"),
                        action="/selecionar_dataset", method="get",
                        cls="h-item-form"
                    ),
                    cls="horizontal-container"
                ),
                Div(
                    Form(
                        Input(type="text", name="data"),
                        Button("Pesquisar"),
                        action="/", method="post",
                        cls="search-container"
                    ),
                ),
                Div(
                    H2("Resultados"),
                    Table(tab),
                ))


#
# Funções com decorators
#
@app.post("/")
def set_query(data:str):
    return home(data)


#
# Registro de Datasets
#
@app.get("/registrar")
def registrar_dataset():
    title_doc = Title("Registrar Dataset")
    inp_list = []
    sep_ = None
    cam_ = None
    finalizavel = False

    # Caminho (pra caso tenha sido registrado)
    if "col_Caminho" in cache_dict:
        cam_ = cache_dict["col_Caminho"]

    if "col_Separador" in cache_dict:
        sep_ = cache_dict["col_Separador"]
        if "dt_Caminho" not in cache_dict:
            cache_dict["dt_Caminho"] = pd.read_csv(cam_, sep=sep_)
    elif "col_Caminho" in cache_dict:
        sep_ = meta.detectar_separador(cache_dict["col_Caminho"])

    # Dataset
    div_dataset = Div(
        P("Dataset"),
            Select(
                *[Option(dt, value=dt, selected=(True if dt == cam_ else False)) for dt in glob.glob("datasets/*")],
                name="col_Caminho"
        ),
    )
    inp_list.append(div_dataset)

    # Separador
    if "col_Caminho" in cache_dict:
        sep_ 
        div_sep = Div(
            P("Separador"),
            P(f"Separador detetado: '{sep_}'"),
            Input(name="col_Separador", value=sep_),
            )
        inp_list.append(div_sep)
    
    if "col_Separador" in cache_dict:
        # Campo de comparação
        div_campo_idioma = Div(
            P("Campo"),
            Select(
                *[Option(campo, name=f"col_Campo_{campo}", value=campo) for campo in cache_dict["dt_Caminho"].columns],
                name="col_Campo"
            ),
            P("Idioma"),
            Select(
                *[Option(lang, name=f"col_Idioma_{lang}", value=lang) for lang in meta.stopwords.fileids()],
                name="col_Idioma"
            ),
            cls="search-container"
        )
        inp_list.append(div_campo_idioma)

        # Colunas
        div_exibir = Div(
            P("Exibir"),
            *[Input(campo, name=f"col_Exibir_{campo}", value=campo, type="checkbox") for campo in cache_dict["dt_Caminho"].columns],
            cls="search-container"
        )
        inp_list.append(div_exibir)

        # Campo da Imagem
        div_imagem = Div(
            P("Campo da Imagem"),
            Select(
                Option(None, value="Nan"),
                *[Option(campo, name=f"col_Imagem_{campo}", value=campo) for campo in cache_dict["dt_Caminho"].columns],
                name="col_Imagem"
            ),
            cls="search-container"
        )
        inp_list.append(div_imagem)

        # Campo do Nome
        div_final = Div(
            P("Nome"),
            Input(name="col_Nome"),
            P("Descrição"),
            Input(name="col_Descricao"),
            P("Matriz de recomendações?"),
            Select(
                Option("Não", value="False"),
                Option("Sim", value="True")
            )
        )
        inp_list.append(div_final)
        finalizavel = True


    main_doc = Main(
        Style(imps.styles),
        H1("Registrar datasets"),
        Div(
            Form(
                *inp_list,
                Button(("Registrar" if finalizavel else "Próximo"), type="submit"),
                action="/registrar_info_dataset", method="post",
                cls="search-container",
            ),
            Form(
                Button("Limpar registro", type="submit"),
                action="limpar_registro", method="post",
                cls="search-container"
            )
        )
    )

    return title_doc, main_doc


#
# Posts
#
@app.post("/registrar_info_dataset")
def registrar_info_dataset(campos: dict):
    global cache_dict
    # Acúmulo de informações temporárias
    for campo in campos:
        cache_dict[campo] = campos[campo]
    if "col_Nome" in cache_dict:
        dt_temp_reg = pd.DataFrame(columns=imps.df_metadatasets.columns)
        dt_temp_reg.loc[0, "Exibir"] = "Similaridade"
        
        # Retirada das informações do campo
        for campo in campos:
            campo_info = campo.split("_")
            campo_val = campo_info[1]
            if campo_val in dt_temp_reg.columns: 
                if campo_val != "Exibir":
                    dt_temp_reg[campo_val] = campos[campo]
                else:
                    dt_temp_reg.loc[0, "Exibir"] += " " + cache_dict[campo]
        
        # Registro do Dataset
        imps.df_metadatasets = pd.concat([imps.df_metadatasets, dt_temp_reg], ignore_index=True)
        imps.save_mdt(imps.df_metadatasets)
        cache_dict = dict()
        print("Registro completo!")
        return home()
    else:
        return registrar_dataset()
        

@app.post("/limpar_registro")
@app.post("/limpar_cahce")
def limpar_registro():
    global cache_dict
    cache_dict = dict()
    return registrar_dataset()


@app.get("/toggle_tfidf")
def toggle_mode():
    global modo_atual
    modo_atual = "Scikit" if modo_atual == "Manual" else "Manual"
    print("Modo atual: ", modo_atual)
    return home()
#
# Seleção e visualização de datasets
#
@app.get("/selecionar_dataset")
def selecionar_dataset():
    title_doc = Title("Selecionar Dataset")
    btn_list = [(Button(imps.df_metadatasets.iloc[i_]["Nome"], name="data", type="submit", value=i_) if i_ != index else None) for i_ in range(len(imps.df_metadatasets))]
    main_doc = Main(
        Style(imps.styles),
        H1("Datasets"),
        P("Dataset atual:"),
        P(imps.df_metadatasets.iloc[index]["Nome"]),
        Div(
            Form(
                *btn_list,
                action="/change_index", method="post",
                cls="search-container",
            )
        )
    )
    return title_doc, main_doc


@app.post("/change_index")
def set_dataset(data:int):
    global index
    index = data
    return home()


@app.post("/exibir_dataset")
def exibir_datasets():
    global index
    index = -1
    return home()


serve()
