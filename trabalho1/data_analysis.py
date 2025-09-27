# analise_materiais.py
import pandas as pd
import sys
import unicodedata
import re
from pathlib import Path

def norm_text(s: str) -> str:
    if pd.isna(s):
        return ""
    # minúsculas
    s = str(s).lower().strip()
    # normaliza acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # remove pontuação comum e múltiplos espaços
    s = re.sub(r"[^\w\s]", " ", s)       # mantém letras/números/_ e espaço
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(path="./trabalho1/cenario/materiais_clean.tsv"):
    path = Path(path)
    if not path.exists():
        print(f"[ERRO] Arquivo não encontrado: {path.resolve()}")
        sys.exit(1)

    # Lê TSV
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"ID_CAT": "Int64", "Cat_Name": "string",
               "ID_Sub": "Int64", "Sub_Name": "string",
               "ID_Prod": "Int64", "Prod_Desc": "string"},
        on_bad_lines="skip",
        encoding="utf-8"
    )

    # --- (1) Quantas amostras por subcategoria (desc) ---
    # Usamos par (ID_Sub, Sub_Name) para evitar nomes repetidos de subcategorias diferentes
    counts_sub = (
        df.groupby(["ID_Sub", "Sub_Name"], dropna=False)
          .size()
          .sort_values(ascending=False)
          .rename("qtde_amostras")
    )

    print("\n=== (1) Amostras por subcategoria (ordenado decrescente) ===")
    # Mostra tudo; se preferir resumir, use .head(20)
    for (id_sub, sub_name), n in counts_sub.items():
        print(f"[ID_Sub={id_sub}] {sub_name}: {n}")

    # --- (2) Quantas subcategorias no total ---
    total_subcats = counts_sub.index.nunique()
    print(f"\n=== (2) Subcategorias no total ===")
    print(total_subcats)

    # --- (3) Quantos dados no total (linhas) ---
    total_rows = len(df)
    print(f"\n=== (3) Dados no total ===")
    print(total_rows)

    # --- (4) Quantas amostras repetidas (e quais suas subcategorias) ---
    # Definição: amostras repetidas por Prod_Desc NORMALIZADO (casefold/acentos/espaços).
    df["Prod_Desc_norm"] = df["Prod_Desc"].map(norm_text)

    # Opcional: remove strings vazias após normalização
    df_nonempty = df[df["Prod_Desc_norm"].str.len() > 0].copy()

    dup = (
        df_nonempty.groupby("Prod_Desc_norm", dropna=False)
        .agg(
            ocorrencias=("Prod_Desc_norm", "size"),
            exemplos=("Prod_Desc", lambda s: sorted(set(filter(pd.notna, s)))[:3]),
            subcats=("Sub_Name", lambda s: sorted(set(filter(pd.notna, s))))
        )
        .reset_index()
    )

    dup = dup[dup["ocorrencias"] > 1].sort_values("ocorrencias", ascending=False)

    # Contagens úteis
    grupos_duplicados = len(dup)                 # quantos descritores têm mais de 1 ocorrência
    linhas_repetidas = int((dup["ocorrencias"] - 1).sum())  # “excessos” sobre a primeira ocorrência

    print("\n=== (4) Amostras repetidas ===")
    print(f"Grupos de descrições repetidas: {grupos_duplicados}")
    print(f"Linhas repetidas (ocorrências excedentes): {linhas_repetidas}")

    # Listagem das repetidas com subcategorias
    # (mostra as top 50 para não poluir muito; ajuste conforme necessário)
    print("\nTop repetições (até 50 linhas):")
    for i, row in dup.head(50).iterrows():
        exemplos = "; ".join(row["exemplos"])
        subcats = "; ".join(row["subcats"])
        print(f"- '{row['Prod_Desc_norm']}' -> ocorrencias={row['ocorrencias']} | subcategorias=[{subcats}] | exemplos=[{exemplos}]")

    # Também salva um CSV detalhado (útil para auditoria)
    out_csv = path.with_name("amostras_repetidas_por_desc.csv")
    dup.to_csv(out_csv, index=False)
    print(f"\n[OK] Relatório detalhado de duplicatas salvo em: {out_csv.resolve()}")

if __name__ == "__main__":
    # Passe o caminho do TSV como argumento opcional:
    #   python analise_materiais.py ./trabalho1/cenario/materiais_clean.tsv
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
