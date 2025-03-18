
<img src="../assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=30% height=30%>

# AI Project Document - Módulo 5 - FIAP

## Nome do Grupo
Deniz Feital Armanhe - individual

#### Nomes dos integrantes do grupo
Deniz Feital Armanhe


## Sumário

[1. Introdução](#c1)

[2. Visão Geral do Projeto](#c2)

[3. Desenvolvimento do Projeto](#c3)

[4. Resultados e Avaliações](#c4)

[5. Conclusões e Trabalhos Futuros](#c5)

[6. Referências](#c6)

[Anexos](#c7)

<br>

# <a name="c1"></a>1. Introdução

## 1.1. Escopo do Projeto

### 1.1.1. Capítulo 1 - FarmTech na era da cloud computing

*Nesta nova fase do projeto, Fase 5, vamos trabalhar com machine learning e Computação em Nuvem.

### 1.1.2. Descrição da Solução Desenvolvida

*A solução foi desenvolvida utilizando atividades apresentadas nos módulos anteriores. Neste caso, criamos um programa em Python para demonstrar uma análise exploratória dos dados, definir tendências para os rendimentos das plantações e demonstramos cenários de modelos preditivos.*

# <a name="c2"></a>2. Visão Geral do Projeto

## 2.1. Objetivos do Projeto

*Demonstrar as novas funcionalidades deste módulo com a adição das mesmas ao ambiente previamente criado, tornando-o assim mais efetivo. O objetivo é continuar com o aprendizado, e melhor a qualidade das entregas iniciadas em módulos anteriores*

## 2.2. Público-Alvo

*Bem, isto é um trabalho da FIAP, mas obviamente a idea é que o aprendizado aqui seja aplicado nas empresas do ramo agrícola. Obviamente o conhecimento aqui é agnóstico, servirá para qualquer área de atuação, o que é excelente*

## 2.3. Metodologia

*A metodologia foi utilizar toda a nova documentação *

# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

  Python
  
  Visual Studio Code
  
  
  ChatGPT
  
  Além das bibliotecas que contans no arquivo requirements.txt


## 3.2. Modelagem e Algoritmos

Utilizeamos as seguintes bibliotecas no código Python.
pandas
sklearn
xgboost 
matplotlib.pyplot 
seaborn
numpy


## 3.3. Treinamento e Teste

Utilizamos o Python e modelos de regressão linear para estas atividades.

# <a name="c4"></a>4. Resultados e Avaliações

## 4.1. Análise dos Resultados
 
Utilizamos um arquivo com pouco mais de 150 registros, onde tivemos informações sobre 4 culturas além de informações climáticas e a produção das culturas. Percebemos uma clusterização bem definidas das culturas. Em termos de outliers elas estavam bem equilibradas, o que se mostrou bem desequilibrada foi a produção em si, com valores muito diferentes.

Estatísticas descritivas para cada coluna:
       Precipitacao    Humidade  Humidade_relativa  Temperatura       Colheita
count    156.000000  156.000000         156.000000    156.00000     156.000000
mean    2486.498974   18.203077          84.737692     26.18359   56153.096154
std      289.457914    0.293923           0.996226      0.26105   70421.958897
min     1934.620000   17.540000          82.110000     25.56000    5249.000000
25%     2302.990000   18.030000          84.120000     26.02000    8327.750000
50%     2424.550000   18.270000          84.850000     26.13000   18871.000000
75%     2718.080000   18.400000          85.510000     26.30000   67518.750000
max     3085.790000   18.700000          86.100000     26.81000  203399.000000

Abaixo os gráficos gerados:


<img width="494" alt="image" src="https://github.com/user-attachments/assets/717a5f53-810e-40e5-a283-92a1ee92bb2d" />

A clusterização mostra um equilíbrio bem definido entre as quatro culturas.


<img width="460" alt="image" src="https://github.com/user-attachments/assets/87ddb5d9-4d7b-4e24-b228-6a7dd4180cce" />


a produção (colheita) mostra uma diferença fenomenal entre as informações apresentadas, ou seja, praticamente nenhum padrão com relação ao produto ou clima.


Modelos preditivos.

Analisamos os dados através de 5 modelos distintos e os resultados foram bem interessantes:


Regressão Linear:

![image](https://github.com/user-attachments/assets/09b60b5d-8cc9-43cf-a55c-19308112e3d3)


O resultado não foi muito bom.


Árvore de decisão:

![image](https://github.com/user-attachments/assets/88226a1c-d27b-4d39-9cad-bb52040f0103)


Percebemos que foi um dos melhores modelos, onde a linha preditiva se assimilou muito com o resultado original.

SVR

![image](https://github.com/user-attachments/assets/84680813-6f91-4238-aef6-50c8c3864317)


Também não apresentou um bom resultado

KNN

![image](https://github.com/user-attachments/assets/4d0ec6e4-9838-43ff-9b3f-f58d84adc6ac)

A utilização deste modelo também deixou a desejar.

XGBoost Regressor

![image](https://github.com/user-attachments/assets/db1b7e34-f9e5-42f5-aebd-9120c1260207)

Este modelo apresentou um resultado satisfatório.

No geral, em ordem abaixo a classificação dos modelos:

Classificação dos Modelos com base no RMSE (do melhor para o pior):
1. Decision Tree: RMSE = 5450.09987293811
2. XGBoost: RMSE = 6293.051723925364
3. KNN: RMSE = 35675.868050599
4. Linear Regression: RMSE = 65364.56901634608
5. SVM: RMSE = 71312.75791075568

Classificação dos Modelos com base no R² (do melhor para o pior):
1. Decision Tree: R² = 0.9923424063976768
2. XGBoost: R² = 0.9897904396057129
3. KNN: R² = 0.6718801604527653
4. Linear Regression: R² = -0.10145864648572833
5. SVM: R² = -0.31104578138210703

O R² (R-quadrado) é uma métrica que indica a proporção da variabilidade dos dados que é explicada pelo modelo. Ele varia de 0 a 1, sendo que valores mais próximos de 1 indicam que o modelo explica bem os dados, enquanto valores próximos de 0 indicam que o modelo tem pouco poder explicativo.

Já o RMSE (Root Mean Squared Error) é a raiz quadrada da média dos erros quadráticos. Ele mede o quão distante as previsões estão dos valores reais, com unidades iguais às da variável alvo. Quanto menor o RMSE, melhor o modelo, pois significa que as previsões estão mais próximas dos valores reais.

