from fasthtml.common import *
import pandas as pd
import imp_setup as imps
import meta_funcs as meta, glob
from set_sklearn import rodar_sklearn
from set_manual import rodar_manual


#
# Globais
#
df_novo = pd.DataFrame()
app = FastHTML()
index = 0

# TODO: Página com resultado dos registros
@app.get("/")
def home(query: str = ""):
    df = pd.DataFrame()
    if index != -1:
        df_nome = imps.df_metadatasets.iloc[index]["Nome"]

    if query:
        df = rodar_sklearn(index, query)
    elif index == -1:
        df = imps.df_metadatasets.copy()
        df_nome = "Datasets"

    tab = Table(eval(html2ft(df.to_html())))

    return Title("Pesquisa"), Main(
                Style(imps.styles),
                H1("Buscar", cls="h1"),
                P("Dataset ", B(f'{df_nome}', style="color: #4CAF50"), " selecionado!"),
                Div(
                    Form(
                        Button("Registrar dataset"),
                        action="/registrar_dataset_1", method="get",
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
@app.get("/registrar_dataset_1")
def registrar_dataset_1(passo: int = 1):
    title_doc = Title("Selecionar Dataset")
    ops = [op for op in meta.get_reg_ops()][1:3]
    inp_list = [
        Div(
            P("Dataset"),
            Select(
                *[Option(dt, value=dt) for dt in glob.glob("datasets/*")],
                name="caminho"
            )
        ),]\
        +\
        [
        Div(
            P(op),
            Input(type="text", name=f"{op}".lower().replace(" ", "_"), placeholder=f"{op}"),
            )
        for op in ops
        ]

    main_doc = Main(
        Style(imps.styles),
        H1("Registrar datasets"),
        Div(
            Form(
                *inp_list,
                Button("Confirmar", type="submit"),
                action="/registrar_1", method="post",
                cls="search-container",
            )
        )
    )

    return title_doc, main_doc


@app.get("/registrar_dataset_2")
def registrar_dataset_2():
    title_doc = Title("Selecionar Dataset")
    ops = [op for op in meta.get_reg_ops()][4:6]
    inp_list = [
        Div(
            P("Idioma"),
            Select(
                *[Option(lang, value=lang) for lang in meta.stopwords.fileids()],
                name="idioma"
            )
        )
    ]\
    +\
    [
        Div(
            P(op),
            Input(type="text", name=f"{op}".lower().replace(" ", "_"), placeholder=f"{op}"),
            )
        for op in ops
        ]

    main_doc = Main(
        Style(imps.styles),
        H1("Registrar datasets"),
        Div(
            Form(
                *inp_list,
                Button("Confirmar", type="submit"),
                action="/registrar_2", method="post",
                cls="search-container",
            )
        )
    )

    return title_doc, main_doc


@app.get("/registrar_dataset_3")
def registrar_dataset_3():
    title_doc = Title("Selecionar Dataset")
    ops = [op for op in meta.get_reg_ops()][6:]
    inp_list = [
        Div(
            P(op),
            Input(type="text", name=f"{op}".lower().replace(" ", "_"), placeholder=f"{op}"),
            )
        for op in ops
        ]

    main_doc = Main(
        Style(imps.styles),
        H1("Registrar datasets"),
        Div(
            Form(
                *inp_list,
                Button("Confirmar", type="submit"),
                action="/registrar_3", method="post",
                cls="search-container",
            )
        )
    )

    return title_doc, main_doc


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


@app.post("/registrar_1")
def registrar_1(caminho: str, separador: str, campo: str):
    global df_novo
    if (caminho == "") or (separador == "") or (campo == ""):
        print("Inválido!")
        return home()
    else:
        df_novo = pd.DataFrame({"Caminho": [caminho],
                            "Separador": [separador],
                            "Campo":[campo]},
                            columns=imps.df_metadatasets.columns)
        imps.df_cache = df_novo
        return registrar_dataset_2()


@app.post("/registrar_2")
def registrar_2(idioma: str, exibir: str, imagem: str):
    global df_novo
    if idioma == "":
        print("Inválido!")
        imps.df_metadatasets = imps.df_metadatasets[:-1]
        return home()
    else:
        # Verifica se campos anteriores obrigatório estão presentres
        if "Caminho" not in imps.df_cache:
            print("Perda de informação!")
            return registrar_dataset_1()
        imps.df_cache.loc[0, "Idioma"] = idioma
        imps.df_cache.loc[0, "Exibir"] = exibir
        imps.df_cache.loc[0, "Imagem"] = imagem
        print(imps.df_cache)
        return registrar_dataset_3()


@app.post("/registrar_3")
def registrar_3(nome: str, descricao: str, recomendar: str):
    global index
    if nome == "":
        print("Inválido!")
    else:
        imps.df_cache.loc[0, "Nome"] = nome
        imps.df_cache.loc[0, "Descricao"] = descricao
        imps.df_cache.loc[0, "Recomendar"] = recomendar
        imps.df_metadatasets = pd.concat([imps.df_metadatasets, imps.df_cache])
        imps.df_cache = pd.DataFrame()
        # Salva registro
        imps.save_mdt(imps.df_metadatasets)
        # Seleciona registro
        index = imps.df_metadatasets.index[-1]
        print("Registro criado!")
    return home()


@app.post("/exibir_dataset")
def exibir_datasets():
    global index
    index = -1
    return home()


serve()
