import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("crop_yield.csv", encoding="utf-8")  # Pode testar 'latin1' ou 'ISO-8859-1' se houver erro de encoding

#Exibindo informacoes basicas do dataset
print("")
print ("Exibindo informacoes basicas do dataset")
print("")


# Exibir as primeiras linhas
print(df.head())

# Informações gerais sobre o dataset
print(df.info())

# Estatísticas descritivas das colunas numéricas
print(df.describe())

# Exibir nomes das colunas
print(df.columns)

#identificando valores nulos
print("")
print ("Identificado valores nulos no dataset")
print("")
print(df.isnull().sum())  # Contar valores nulos por coluna
