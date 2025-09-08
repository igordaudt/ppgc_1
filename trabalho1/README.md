# TRABALHO DE APRENDIZADO DE MAQUINA

# instruções de uso
1. "limpar arquivo":
- usando clean_file.py, limpar o arquivo original "materiais.csv"
- foi removida a ultima coluna que não será usada.
- detecta o separador atual (conta ocorrências de >, \t, ;, ,) e salva como TAB (TSV).
- preserva caracteres especiais (tenta utf-8-sig, cai para utf-8, depois latin-1).
- gerado o arquivo de saída materiais_clean em .tsv (É padrão em datasets técnicos/científicos porque garante que o separador nunca vai colidir com caracteres comuns de texto.). .csv foi criado também em paralelo, para facilitar abertura e leitura em softwares comuns (libreoffice, msoffice, etc)
- 

