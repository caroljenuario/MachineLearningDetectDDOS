import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from google.colab import files
import pandas as pd

uploaded = files.upload()
file_name = list(uploaded.keys())[0]


dados = pd.read_csv(file_name)

# Filtrando as linhas com os campos específicos e então salvar um novo arquivo
campos_especificos = ['BenignTraffic', 'Mirai-greip_flood', 'Mirai-udpplain', 'Mirai-greeth_flood']
dados_filtrados = dados[dados['label'].isin(campos_especificos)]

print(dados_filtrados.head())

dados_filtrados.to_csv('dados_filtrados19.csv', index=False)

# Mensagem de confirmação
print('Linhas filtradas foram salvas em um novo arquivo CSV.')