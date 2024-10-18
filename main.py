import imp_setup as imps
import meta_funcs
import tfidf_csv

# 
# Menu de seleção
# 
# TODO: Nome do registro no final ou depois de selecionr arquivo
# TODO: Consertar launch.json para sempre rodar o main.py
# TODO: Mostrar só campo e similaridade? Adicionar para que possa mostrar tabela acima?
ops = [("Registrar DataSet", meta_funcs.registrar_dataset),
        ("Rodar", tfidf_csv.rodar_nltk),
        ("Datasets registrados", meta_funcs.exibir_datasets),
        ("Sair", lambda: False)]

ops_l = [x[0].lower() for x in ops]

while True:
    # Verifica se há datasets, senão seleciona o padrão
    # Seleciona configurações padrões por agora
    selecao = True
    if (not imps.default_params["IdDataSetDefault"].isin(imps.df_metadatasets.index).any())\
          or selecao:
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
        if not ops[inp][1]():
            # break
            break
    else:
        # TODO: TIrar redundância
        if imps.df_metadatasets.loc[imps.default_params["IdDataSetDefault"], "Nome"].empty:
            print("O Dataset está vazio!")
        else:
            print(f"Dataset {imps.df_metadatasets.loc[imps.default_params["IdDataSetDefault"], "Nome"][0]} selecionado!")
        if not tfidf_csv.rodar_nltk(imps.default_params["IdDataSetDefault"][0]):
            break
