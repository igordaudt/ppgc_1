# ppgc_1
Atividade Prática 1: Questionário sobre KNN

## Referências
*https://www.youtube.com/watch?v=hDKCxebp88A
Machine Learning with Python and Scikit-Learn – Full Course (18 horas)


1. Crie e ative um venv e instale libs:
- VS Code → Terminal → Novo Terminal (PowerShell)
```
py -m venv .venv
* em caso de erro, liberar a execução de scripts: "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install scikit-learn pandas matplotlib
```

3. Instalar as bibliotecas necessárias
- criar arquivo requirements.txt e inserir o texto:
```
scikit-learn
pandas
matplotlib
```
2. Criar arquivo de código python
- knn_vinhos.py (ver o código no arquivo)

3. versão criada com Streamlit (opcional para visualização dinâmica)
- ver, baixar e instalar aqui: https://streamlit.io/
- criados os arquivos versão streamlit para visualização.
 - Obs: Foram criados apenas 2 para modelo
 - app_knn_2f_norm.py
 - app_knn_11f_norm.py

5. Criada uma versão mais simples do código com interação do usuário:
- KNN_aula1.py (plota grafico simples no final)
