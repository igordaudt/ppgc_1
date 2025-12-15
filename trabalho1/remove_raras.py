# filter_rare_classes.py
import pandas as pd
import sys
from pathlib import Path

def main(path="./trabalho1/cenario/materiais_clean.tsv", min_count=3):
    p = Path(path)
    if not p.exists():
        print(f"[ERRO] Arquivo não encontrado: {p.resolve()}")
        sys.exit(1)

    # Lê TSV
    df = pd.read_csv(
        p,
        sep="\t",
        dtype=str,              # mantém tudo como string
        on_bad_lines="skip",
        encoding="utf-8"
    ).fillna("")

    # Verificações básicas
    for col in ("ID_Sub", "Sub_Name"):
        if col not in df.columns:
            print(f"[ERRO] Coluna obrigatória ausente: '{col}'. Colunas: {list(df.columns)}")
            sys.exit(1)

    total_rows = len(df)

    # Contagem por classe (ID_Sub, Sub_Name)
    group_cols = ["ID_Sub", "Sub_Name"]
    counts = (
        df.groupby(group_cols, dropna=False)
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )

    # Identifica raras
    rares = counts[counts["count"] < int(min_count)].copy()
    n_rare_classes = len(rares)
    rare_rows = (
        df.merge(rares[group_cols], on=group_cols, how="inner")
    )
    n_rare_rows = len(rare_rows)

    # Filtra dataset
    df_filtered = (
        df.merge(rares[group_cols].assign(_drop=1), on=group_cols, how="left")
          .loc[lambda d: d["_drop"].isna()]
          .drop(columns=["_drop"])
    )
    kept_rows = len(df_filtered)

    # Salvar resultados
    out_tsv = p.with_name("materiais_clean_min5.tsv")
    df_filtered.to_csv(out_tsv, sep="\t", index=False, encoding="utf-8")

    counts.to_csv(p.with_name("class_counts_before.csv"), index=False, encoding="utf-8")
    rares.to_csv(p.with_name("rare_classes_removed.csv"), index=False, encoding="utf-8")

    print("=== Remoção de classes raras (por subcategoria) ===")
    print(f"Arquivo de entrada: {p.resolve()}")
    print(f"Min. amostras por classe: {min_count}")
    print(f"Linhas totais (antes): {total_rows}")
    print(f"Classes totais (antes): {len(counts)}")
    print(f"Classes removidas (<{min_count}): {n_rare_classes}")
    print(f"Linhas removidas (pertencentes a classes raras): {n_rare_rows}")
    print(f"Linhas finais (depois): {kept_rows}")
    print("\nArquivos gerados:")
    print(f"- Dataset filtrado: {out_tsv.resolve()}")
    print(f"- Contagens (antes): {p.with_name('class_counts_before.csv').resolve()}")
    print(f"- Classes raras removidas: {p.with_name('rare_classes_removed.csv').resolve()}")

if __name__ == "__main__":
    # Uso:
    #   python filter_rare_classes.py                          # usa materiais_clean.tsv e min_count=5
    #   python filter_rare_classes.py caminho/arquivo.tsv 5    # caminho e limiar customizados
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1], 5)
    else:
        main(sys.argv[1], int(sys.argv[2]))
