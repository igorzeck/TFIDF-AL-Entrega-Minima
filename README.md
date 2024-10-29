#
# README
#

O programa permite o registro e a busca simplificada de qualquer dataset
Utiliza _similaridade de cosseno_ em cima de _Term Frequency-Inverse Documento Frequency_ (TFIDF) ou apenas _Term Frequency_ (TF).

## Datasets utilizados
### Descrição do Sistema Harmonizado
**Arquivo**: _descricao_sistema_harmonizado_ncm.csv_

link [aqui](https://repositorio.seade.gov.br/dataset/comercio-exterior/resource/ffc3925a-a27b-4707-88fd-430d43cce512)

### Descrição do Sistema Harmonizado
**Arquivo**: _Books_simplificado.csv_

link [aqui](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Arqiuivo _Books_simplificado.csv_ criado pegando 50000 (25%) do arquivo _Books.csv_ linhas igualmente espaçados ao longo do arquivo acima.

## Setup
1. Verificar se há **descricao_sistema_harmonizado_ncm.csv** na pasta **datasets** (é o dataset default)
2. Verificar se há registros de datasets em **save/save_principal**


## Passo a passo:
1. Utilizar um interpretador python com nltk, sklearn, pandas e python-fasthtml, senão `pip install requirements.txt`
2. Executar no VS Code ou via cmd por meio de `python main.py`, onde `pyhton`é o interpretador com os módulos acima instalados
3. Caso a interface não funcione, rodar main.py
4. Caso main.py apresentar problemas pode se rodar **tfidf_manual.py** e **tfidf_sklearn.py** diretamente.
