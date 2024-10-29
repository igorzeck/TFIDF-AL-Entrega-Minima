import imp_setup as imps
import meta_funcs
from meta_funcs import selecionar_op, pd
import set_sklearn
import set_manual


def executar_busca(index: int = -1) -> bool:
    """
    Executa busca utilizando modo Manual ou do Scikit
    """
    if imps.default_params["Modo"][0] == "Manual":
        return set_manual.rodar_manual(index) 
    else:
        return set_sklearn.rodar_sklearn(index)


def main():
    ops_dict = {
                "Registrar DataSet": meta_funcs.registrar_dataset,
                "Rodar": executar_busca,
                "Datasets registrados": meta_funcs.exibir_datasets,
                "Mudar modo TFIDF (Manual <-> Sci-kit)": meta_funcs.mudar_modo,
                "Sair": lambda: False
            }
    while True:
        print("")
        if (imps.default_params["IdDataSetDefault"].isin(imps.df_metadatasets.index).any()):
            print("Modo atual:", imps.default_params["Modo"][0])
            op_ret = selecionar_op(ops_dict,
                          "Escolha uma das opções: ",
                          retorno="Retorno")
            if isinstance(op_ret, pd.DataFrame):
                if op_ret.empty:
                    break
            else:
                if op_ret:
                    break
        else:
            print("O Dataset padrão está vazio!")
            ops_dict["Registrar DataSet"]()


if __name__ == "__main__":
    main()