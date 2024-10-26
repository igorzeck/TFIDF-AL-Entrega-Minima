import imp_setup as imps
import meta_funcs
import tfidf_sklearn
import tfidf_manual

# 
# Menu de seleção
# 
# TODO: Nome do registro no final ou depois de selecionr arquivo
# TODO: Escolher campos para mostrar, além da similaridade (default é "Campo" do texto)
# TODO: Registro rápido ou completo
# TODO: Fallback para rodar arquivos tfidf standalone
def executar_busca(index: int = -1) -> bool:
    return tfidf_manual.rodar_manual(index) if imps.default_params["Modo"] == "Manual" else tfidf_sklearn.rodar_nltk(index)

ops = [("Registrar DataSet", meta_funcs.registrar_dataset),
        ("Rodar", executar_busca),
        ("Datasets registrados", meta_funcs.exibir_datasets),
        (f"Mudar modo TFIDF (Manual <-> Sci-kit)", meta_funcs.mudar_modo),
        ("Sair", lambda: False)]

ops_l = [x[0].lower() for x in ops]

while True:
    print()
    # Verifica se há datasets, senão seleciona o padrão
    # Seleciona configurações padrões por agora
    selecao = True
    if (not imps.default_params["IdDataSetDefault"].isin(imps.df_metadatasets.index).any())\
          or selecao:
        print("Modo atual:", imps.default_params["Modo"][0])
        print("Escolha uma das opções: ")
    # TODO: Melhorar isso
    # TODO: Criar menu de seleção de DataSets
        for i, op in enumerate(ops):
            print(f"{i:<2} - {op[0]}")
        inp = input(": ")
        if inp.isnumeric():
            inp = int(inp)
        else:
            # Por agora não aceita strings
            continue
        if not ops[min(inp, len(ops) - 1)][1]():
            # break
            break
    else:
        # TODO: TIrar redundância
        if imps.df_metadatasets.loc[imps.default_params["IdDataSetDefault"], "Nome"].empty:
            print("O Dataset está vazio!")
        else:
            print(f"Dataset {imps.df_metadatasets.loc[imps.default_params["IdDataSetDefault"], "Nome"][0]} selecionado!")
        if not tfidf_sklearn.rodar_nltk(imps.default_params["IdDataSetDefault"][0]):
            break
