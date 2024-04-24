# Projeto-Analise-de-emocoes (Mineração de Dados)


## Integrantes:

| Nome                               | RA          |
|------------------------------------|-------------|
| Fernanda Felix da Silva            | 11201921613 |
| Fernando Schroder                  |             |
| Winicius Pontes                    | 11201810196 |

## Descrição do Projeto:

 O Projeto visa aplicar conhecimentos adquiridos na disciplina de mineração de dados (UFABC), desta forma, caracteriza-se no desenvolvimento de um modelo de Machine Learning capaz de classificar as emoções humanas em imagens. Para isto, fora utilizado um dataset obtido no Kaggle, ao qual é composto por um grande número de imagens rotuladas com diferentes emoções humanas. 
 O dataset é dividido em conjuntos de treinamento e teste, uma para treinamento do modelo e outra para validação dos resultados, respectivamente.

## Referencias e Tecnologias Utilizadas:

* Dataset Utilizado: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
* Python como a linguagem de programação;
* TensorFlow para construir e treinar o modelo;
* OpenCV para processamento das imagens;
* Algumas bibliotecas auxiliares, como NumPy e Pandas, para manipulação de dados.


## Detalhamento do Código:
- Importação das bibliotecas.
- Divisão dos dados entre teste, treino e validação.
- Construção da arquitetura da rede neural.
- Metrificação da acurácia do nosso modelo. 
- Tunning do hiperparametros, caso necessário.
- Export da rede neural para que não haja a necessidade de ficar treinando o modelo sempre que for utilizar
- Export da estrutura em json para que outras pessoas possam se inspirar na arquitetura

## Como executar o Código:
