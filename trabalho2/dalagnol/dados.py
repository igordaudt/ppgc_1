import pandas as pd

# 1. Carregar as duas listas .tsv (substitua pelo nome dos seus arquivos)
# O parâmetro sep='\t' indica que as colunas são separadas por "Tab"
df1 = pd.read_csv('trabalho2/dalagnol/materiais_clean_ordenado.tsv', sep='\t')
df2 = pd.read_csv('trabalho2/dalagnol/subcats_atualizado.tsv', sep='\t')

# Limpar possíveis espaços vazios ou aspas que vêm do .tsv nos nomes das colunas
df1.columns = df1.columns.str.strip().str.replace('"', '')
df2.columns = df2.columns.str.strip().str.replace('"', '')

# =====================================================================
# OBJETIVO 1: O que da Lista 2 está na Lista 1 e a quantidade
# =====================================================================
contagem_lista1 = df1.groupby(['ID_CAT', 'ID_Sub']).size().reset_index(name='Qtd_Produtos')
lista2_sanitizada = df2.merge(contagem_lista1, on=['ID_CAT', 'ID_Sub'], how='left')
lista2_sanitizada['Qtd_Produtos'] = lista2_sanitizada['Qtd_Produtos'].fillna(0).astype(int)
lista2_sanitizada = lista2_sanitizada.sort_values(by='Qtd_Produtos', ascending=False)

# =====================================================================
# OBJETIVO 2: O que da Lista 1 está desatualizado (Não está na Lista 2)
# =====================================================================
checagem_lista1 = df1.merge(df2[['ID_CAT', 'ID_Sub']], on=['ID_CAT', 'ID_Sub'], how='left', indicator=True)
desatualizados_lista1 = checagem_lista1[checagem_lista1['_merge'] == 'left_only'].drop(columns=['_merge'])

# =====================================================================
# EXPORTAR OS RESULTADOS EM .TSV
# =====================================================================
# Exportando no mesmo formato original (.tsv)
lista2_sanitizada.to_csv('trabalho2/dalagnol/resultado_lista2_com_quantidades.tsv', sep='\t', index=False)
desatualizados_lista1.to_csv('trabalho2/dalagnol/resultado_lista1_produtos_desatualizados.tsv', sep='\t', index=False)

print("Sanitização concluída com sucesso!")