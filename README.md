# Projeto: Analise de Emoções (Mineração de Dados)


## Integrantes:

| Nome                               | RA          |
|------------------------------------|-------------|
| Fernanda Felix da Silva            | 11201921613 |
| Fernando Schroder                  | 11201921885 |
| Winicius Pontes                    | 11201810196 |

## Descrição do Projeto:

 O Projeto visa aplicar conhecimentos adquiridos na disciplina de mineração de dados (UFABC), desta forma, caracteriza-se no desenvolvimento de um modelo de Machine Learning capaz de classificar as emoções humanas em imagens. Para isto, fora utilizado um dataset obtido no Kaggle, ao qual é composto por um grande número de imagens rotuladas com diferentes emoções humanas. 
 O dataset é dividido em conjuntos de treinamento e teste, uma para treinamento do modelo e outra para validação dos resultados, respectivamente.
 Como resultado final tem-se um modelo de classificação de emoções em tempo real, utilizando a câmera da webcam, além de um modelo que classifica emoções a partir de um vídeo pré estabelecido.

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
- OBS: todos os arquivos de código possuem comentários sobre cada função

## Como executar o Código:

- Instalar o VS Code com as seguintes extensões: Python, Tensorflow 2.0 Snippets e Tensorflow Snippets.
- Instalar bibliotecas presentes no arquivo requirements.txt (pip install -r requirements.txt)
- Executar o programa ResultadoTempoReal.py caso queira executar o código ao qual abre a câmera da webcam para fazer a análise de emoções ao vivo.
- Executar o programa ResultadoVideo.py caso queira executar o código ao qual abre um vídeo pré-definido (já presente o arquivo filme_cena.mp4) para fazer a análise de emoções.
