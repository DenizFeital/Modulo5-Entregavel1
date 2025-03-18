
<img src="../assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=30% height=30%>

# AI Project Document - Módulo 5 - FIAP (Dois entregáveis neste Readme)

## Nome do Grupo
Deniz Feital Armanhe - individual

#### Nomes dos integrantes do grupo
Deniz Feital Armanhe


## Sumário - Entregável 1

[1. Introdução](#c1)

[2. Visão Geral do Projeto](#c2)

[3. Desenvolvimento do Projeto](#c3)

[4. Resultados e Avaliações](#c4)


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

A metodologia foi utilizar o conhecimento adquirido principalmente nos módulos 4 e 5, mas sem dúvida dos módulos anteriores também.

# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

  Python
  
  Visual Studio Code
    
  ChatGPT

  Google Colab

  Jupiter
  
  Além das bibliotecas que constam no arquivo requirements.txt


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

Aqui mostramos algumas informações básicas do dataset, como quantidade de colunas e tipos de dados:

![image](https://github.com/user-attachments/assets/4cbbc9b2-67af-4343-829e-c2fc6c475a40)

![image](https://github.com/user-attachments/assets/6edc6078-39df-441a-a39f-c6924c413d98)

Mostramos também que o dataset está bem estruturado, sem nenhuma coluna com valores nulos.

![image](https://github.com/user-attachments/assets/ec357e90-578c-41ff-a29f-d922e2a24dde)

 
Utilizamos um arquivo com pouco mais de 150 registros, onde tivemos informações sobre 4 culturas (Cocoa Beans, Oil Palm fruit, Rice paddy e Rubber natural) além de informações climáticas e a produção das mesmas. Percebemos uma clusterização bem definidas das culturas. Em termos de outliers elas estavam bem equilibradas, o que se mostrou bem desequilibrada foi o resultado da produção em si, com valores muito diferentes.

Estatísticas descritivas para cada componente:

![image](https://github.com/user-attachments/assets/f9c6b300-f020-45b2-9a54-f157ce50f99e)



## Abaixo os gráficos gerados:

<img width="494" alt="image" src="https://github.com/user-attachments/assets/717a5f53-810e-40e5-a283-92a1ee92bb2d" />

A clusterização mostra um equilíbrio bem definido entre as quatro culturas, percebemos que existe um agrupamento natural entre elas ((Cocoa Beans, Oil Palm fruit, Rice paddy e Rubber natural).

<img width="460" alt="image" src="https://github.com/user-attachments/assets/87ddb5d9-4d7b-4e24-b228-6a7dd4180cce" />

A produção (colheita) mostra uma diferença fenomenal entre as informações apresentadas, ou seja, praticamente nenhum padrão com relação ao produto ou clima teve resultado direto na produção, foi como se não houvesse relação entre eles.

**Modelos preditivos.**

Analisamos os dados através de 5 modelos distintos e os resultados foram bem interessantes:


## Regressão Linear:

![image](https://github.com/user-attachments/assets/2f0fd671-2918-4cc5-ac14-6f62f80b8c81)


A utilização deste modelo não trouxe resultados satisfatórios, percebemos uma diferença considerável entre previsão e resultado real.


## Árvore de decisão:

![image](https://github.com/user-attachments/assets/763fd70e-f6b4-46b4-95b9-c187b02b8143)


A árvore de decisão apresentou um dos melhores resultados, onde a linha preditiva se assimilou muito com o resultado original.

## SVR

![image](https://github.com/user-attachments/assets/f7e6de65-e48e-4a0b-aa97-ab1920358c4f)


Também não apresentou um bom resultado, notamos uma discrepância muito grande entre o previsto e real.

## KNN

![image](https://github.com/user-attachments/assets/c83d3428-1cc5-45e4-a48d-05fc35171a39)

A utilização deste modelo também deixou a desejar, bem diferente da previsão.

## XGBoost Regressor

![image](https://github.com/user-attachments/assets/b1cfb5b0-b17f-47f9-bc29-916ad802de07)

Este modelo apresentou um resultado satisfatório, Muito similar ao da árvore de decisão.

## **Resumo geral:**

Classificação dos Modelos com base no RMSE (do melhor para o pior):


O RMSE (Root Mean Squared Error) é a raiz quadrada da média dos erros quadráticos. Ele mede o quão distante as previsões estão dos valores reais, com unidades iguais às da variável alvo. Quanto menor o RMSE, melhor o modelo, pois significa que as previsões estão mais próximas dos valores reais.

1. Decision Tree: RMSE = 5450.09987293811
2. XGBoost: RMSE = 6293.051723925364
3. KNN: RMSE = 35675.868050599
4. Linear Regression: RMSE = 65364.56901634608
5. SVM: RMSE = 71312.75791075568

Classificação dos Modelos com base no R² (do melhor para o pior):

O R² (R-quadrado) é uma métrica que indica a proporção da variabilidade dos dados que é explicada pelo modelo. Ele varia de 0 a 1, sendo que valores mais próximos de 1 indicam que o modelo explica bem os dados, enquanto valores próximos de 0 indicam que o modelo tem pouco poder explicativo.

1. Decision Tree: R² = 0.9923424063976768
2. XGBoost: R² = 0.9897904396057129
3. KNN: R² = 0.6718801604527653
4. Linear Regression: R² = -0.10145864648572833
5. SVM: R² = -0.31104578138210703


## Sumário - Entregável 2

Análise de opções de aquisição de serviços na AWS.

## **Requisitos:**

Usar  a calculadora da AWS, realizar uma estimativa de custos (On-Demand – 100%) para usar uma máquina Linux simples, comparando os valores cotados para a região de São Paulo (BR) e para a região da Virgínia do Norte (EUA). A máquina será utilizada para hospedar uma API que receberá dados dos sensores que coletam as variáveis da Entrega 1 e onde rodará a Machine Learning. Qual a solução mais barata com as seguintes configurações?

•	2 CPUs.

•	1 GIB de memória.

•	Até 5 Gigabit de rede.

•	50 GB de armazenamento (HD).

## **Como calcular:**

I site da AWS fornece uma calculadora que nos ajuda muito neste quesito:

Primeiro precisamos acessar o link: https://calculator.aws/#/

![image](https://github.com/user-attachments/assets/fea78009-5e29-48ab-93d2-1d35bf478f59)

Feito isso, selecionamos "Create estimate", que nos levará a uma tela onde escolhemos algumas opções como o tipo de serviço e a região.

![image](https://github.com/user-attachments/assets/95e9838a-d56d-4b95-8b15-4553502865ba)

Com estas seleções efetuadas, podemos agora selecionar o tipo de hardware que configuraremos para o servidor Linux, informando memória, quantidade de CPUs entre outros parâmetros:

![image](https://github.com/user-attachments/assets/568583bb-2140-427d-ac41-a9be5e6ddfea)

Feito isso, selecionamos "Save and add service" para que possamos selecionar outra região e efetuar a comparação. Quando tiver selecionado todas as regiões, basta selecionar "Save and View Summary".







## **Comparativo:**

![image](https://github.com/user-attachments/assets/00e6b891-84b6-4dc8-a094-8a0b761c6f2e)

Em termos financeiros claramente a melhor opção seria hospedar a máquina nos Estados Unidos. Porém temos que levar em consideração dois pontos importantes do enunciado:

Suponha também que você precisa acessar rapidamente os dados dos sensores:

Há restrições legais para armazenamento no exterior.

Qual opção você escolheria? Justifique.

Com estas duas considerações a resposta deixa de ser simpoles.

Vamos comentar pelo segundo ponto, que é decisivo, se há restrições legais para armazenar dados no exterior, isso elimina qualquer possibilidade de usarmos o datacenter nos Estados Unidos. Quando a "acessar rapidamente" isso pode ser subjetivo, pois o que é rapidamente? Milisegundos, segundos, minutos? Em se tratando de IoT eu entendo que estamos falando em acesso muito rápido, pois podemos estar falando em tomadas de decisões, como acionamento de um motor, um alarme e isso não poderia esperar. Desta forma, estas duas condições nos remete a solução de hospedar o servidor Linux no Brasil.
