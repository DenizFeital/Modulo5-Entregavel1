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


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Função para limpar a tela
def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

limpar_tela()

# Carregar os dados
df = pd.read_csv('crop_yield.csv')

# Selecionar as colunas numéricas relevantes para a clusterização
colunas = [
    'Precipitation (mm day-1)', 
    'Specific Humidity at 2 Meters (g/kg)', 
    'Relative Humidity at 2 Meters (%)', 
    'Temperature at 2 Meters (C)', 
    'Yield'
]
df_selected = df[colunas].dropna()

# Normalizar os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Aplicar K-Means para clusterização (definindo 3 clusters, por exemplo)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_selected['Cluster'] = kmeans.fit_predict(df_scaled)

# Identificar outliers com Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_selected['Outlier'] = iso_forest.fit_predict(df_scaled)
df_selected['Outlier'] = df_selected['Outlier'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')

# Adicionar a coluna Crop para referência
df_selected['Crop'] = df['Crop']

# Aplicar PCA para reduzir a dimensionalidade para 2 componentes
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_selected['PCA1'] = principal_components[:, 0]
df_selected['PCA2'] = principal_components[:, 1]

# Visualizar os clusters e outliers com os componentes do PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_selected,
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    style='Outlier',
    palette='viridis',
    s=100
)
plt.title('Visualização dos Clusters com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Exibir os outliers identificados (alguns exemplos)
outliers = df_selected[df_selected['Outlier'] == 'Outlier']
print("Outliers identificados:")
print(outliers[['Crop', 'Yield', 'Temperature at 2 Meters (C)', 'PCA1', 'PCA2']].head(10))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Carregar o arquivo CSV
data = pd.read_csv("crop_yield.csv")

# Visualizar as primeiras linhas do dataset
print(data.head())

# A variável dependente (target) é o "Yield"
X = data[['Precipitation (mm day-1)', 'Specific Humidity at 2 Meters (g/kg)', 
          'Relative Humidity at 2 Meters (%)', 'Temperature at 2 Meters (C)']]
y = data['Yield']

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados (escala de 0 a 1 para melhorar o desempenho de alguns algoritmos)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Construir e treinar os modelos
# Regressão Linear
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_lr = linear_regressor.predict(X_test_scaled)

# Avaliar o modelo
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Regressão Linear - RMSE: {rmse_lr}, R²: {r2_lr}")


# AdaBoost
adaboost = AdaBoostRegressor(n_estimators=50, random_state=42)
adaboost.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_ab = adaboost.predict(X_test_scaled)

# Avaliar o modelo
mse_ab = mean_squared_error(y_test, y_pred_ab)
rmse_ab = np.sqrt(mse_ab)
r2_ab = r2_score(y_test, y_pred_ab)

print(f"AdaBoost - RMSE: {rmse_ab}, R²: {r2_ab}")


# Árvore de Decisão
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_dt = decision_tree.predict(X_test_scaled)

# Avaliar o modelo
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Árvore de Decisão - RMSE: {rmse_dt}, R²: {r2_dt}")

# Floresta Aleatória
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_rf = random_forest.predict(X_test_scaled)

# Avaliar o modelo
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Floresta Aleatória - RMSE: {rmse_rf}, R²: {r2_rf}")

# K-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_knn = knn.predict(X_test_scaled)

# Avaliar o modelo
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"K-Nearest Neighbors - RMSE: {rmse_knn}, R²: {r2_knn}")


# Visualização dos resultados
models = ['Regressão Linear', 'AdaBoost', 'Árvore de Decisão', 'Floresta Aleatória', 'KNN']
rmse_values = [rmse_lr, rmse_ab, rmse_dt, rmse_rf, rmse_knn]

plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Modelos')
plt.ylabel('RMSE')
plt.title('Comparação de Modelos - RMSE')
plt.show()
