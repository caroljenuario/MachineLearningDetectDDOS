from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

# Carregando os dados
#caminho_arquivo = 'DadosFiltrados\mesclados\merged_file_final3.csv'
caminho_arquivo = 'dados_concatenados.csv'
dados = pd.read_csv(caminho_arquivo)

# Convertendo as labels em números usando LabelEncoder
le = LabelEncoder()
dados['label'] = le.fit_transform(dados['label'])

# Separando features e labels
X = dados.drop(columns=['label'])
y = dados['label']

# Distribuição das classes antes do balanceamento
rotulos_originais = le.inverse_transform(y.value_counts().index)
contagens_originais = y.value_counts().tolist()
print("Distribuicao das classes antes do balanceamento:")
for rotulo, contagem in zip(rotulos_originais, contagens_originais):
    print(f"{rotulo}: {contagem}")

# Dividindo os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicando o balanceamento usando RandomOverSampler da biblioteca Imbalanced-Learn
sampler = RandomOverSampler(random_state=42)
X_treino_resampled, y_treino_resampled = sampler.fit_resample(X_treino, y_treino)

# Avaliando a importância das features
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_treino_resampled, y_treino_resampled)
importancias_features = modelo.feature_importances_

# Criando um DataFrame com as importâncias das features
importancias_df = pd.DataFrame({'Feature': X.columns, 'Importancia': importancias_features})

# Exibindo as importâncias das features
print("\nImportancia das Features:")
print(importancias_df)

# Exibindo o gráfico de correlação de significância
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('Correlação de Significância Antes de Remover Colunas')
#plt.show()

# Excluindo as features com importância zero
importancias_df = importancias_df[importancias_df['Importancia'] != 0]

# Salvando as colunas relevantes em um novo arquivo CSV
colunas_relevantes = importancias_df['Feature'].tolist()
dados_novos = dados[colunas_relevantes + ['label']]
dados_novos.to_csv('dados_novos.csv', index=False)

# Distribuição das classes após o balanceamento
rotulos_balanceados = le.inverse_transform(pd.Series(y_treino_resampled).value_counts().index)
contagens_balanceadas = pd.Series(y_treino_resampled).value_counts().tolist()
print("\nDistribuicao das classes apos o balanceamento:")
for rotulo, contagem in zip(rotulos_balanceados, contagens_balanceadas):
    print(f"{rotulo}: {contagem}")

# Treinando um modelo de Floresta Aleatória após a remoção das features irrelevantes
modelo.fit(X_treino_resampled, y_treino_resampled)

# Fazendo previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# Calculando as métricas de avaliação do modelo
acuracia = accuracy_score(y_teste, previsoes)
precisao = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')

# Imprimindo as métricas
print(f"\nAcuracia: {acuracia:.9f}")
print(f"Precisao: {precisao:.9f}")
print(f"Recall: {recall:.9f}")
print(f"F1 Pontos: {f1:.9f}")

# Exibindo o relatório de classificação com números decimais formatados
print("\nRelatorio de Classificacao:")
report = classification_report(y_teste, previsoes, target_names=rotulos_originais, digits=9)
print(report)


# Criando um gráfico de matriz de correlação usando seaborn
#matriz_correlacao = dados.corr()
#plt.figure(figsize=(12, 8))
#sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
#plt.title('Matriz de Correlaçao')
#plt.show()
