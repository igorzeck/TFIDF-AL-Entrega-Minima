import imp_setup as imps
import meta_funcs
import tfidf_csv

# 
# Menu de seleção
# 
# TODO: Transformar em dicionário as listas de opções
# TODO: Nome do registro no final ou depois de selecionr arquivo
ops = [("Registrar DataSet", meta_funcs.registrar_dataset),
        ("Rodar", lambda: tfidf_csv.rodar(int(input("Índice do dataset: ")))),
        ("Datasets registrados", meta_funcs.exibir_datasets),
        ("Sair", lambda: False)]

ops_l = [x[0].lower() for x in ops]

while True:
    print("Escolha uma das opções: ")
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

