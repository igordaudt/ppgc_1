import pandas as pd

# Nome do arquivo original e do arquivo final que será gerado
arquivo_entrada = 'trabalho2/dalagnol/materiais_clean_ordenado.tsv'
arquivo_saida = 'trabalho2/dalagnol/materiais_clean_ordenado.tsv'

# Carrega o arquivo TSV
df = pd.read_csv(arquivo_entrada, sep='\t')

# Ordena as linhas primeiro por 'ID_CAT' e depois por 'ID_SUB'
df_ordenado = df.sort_values(by=['ID_CAT', 'ID_Sub'])

# Salva o resultado em um novo arquivo TSV
df_ordenado.to_csv(arquivo_saida, sep='\t', index=False)

print(f"Arquivo ordenado com sucesso! Salvo em: {arquivo_saida}")