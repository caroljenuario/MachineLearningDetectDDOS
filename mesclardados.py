import pandas as pd
import os

# Caminho para o diretório onde os arquivos CSV estão localizados
diretorio = 'DadosFiltrados'

# Lista para armazenar os DataFrames
dataframes = []

# Carregar o primeiro arquivo CSV (com cabeçalho)
arquivo_principal = 'mesclados/merged_file_final3.csv'
df_principal = pd.read_csv(os.path.join(diretorio, arquivo_principal))
dataframes.append(df_principal)

# Intervalo de arquivos CSV para mesclar (de 20 a 50)
inicio = 20
fim = 168

# Carregar e concatenar os arquivos CSV adicionais (sem cabeçalho)
for i in range(inicio, fim + 1):
    arquivo_adicional = f'dados_filtrados{i}.csv'
    caminho_arquivo = os.path.join(diretorio, arquivo_adicional)
    
    if os.path.exists(caminho_arquivo):
        df_adicional = pd.read_csv(caminho_arquivo, skiprows=1, header=None)
        df_adicional.columns = df_principal.columns
        dataframes.append(df_adicional)
    else:
        print(f"Arquivo {arquivo_adicional} não encontrado.")

# Concatenar todos os DataFrames
df_concatenado = pd.concat(dataframes, ignore_index=True)

# Exibir o resultado concatenado
print(df_concatenado)

# Opcional: Salvar o resultado concatenado em um novo arquivo CSV
df_concatenado.to_csv('dados_concatenados.csv', index=False)

