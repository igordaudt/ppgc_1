# dedup_normalizado.py
import pandas as pd
import sys, re, unicodedata
from pathlib import Path

REPLACEMENTS = {
    "×": "x", "÷": "/", "Ø": "o", "ø": "o", "º": "o", "ª": "a",
    "–": "-", "—": "-", "‐": "-", "−": "-",
}

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    for a, b in REPLACEMENTS.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    return df

def main(path="./trabalho1/cenario/materiais_clean.tsv"):
    p = Path(path)
    if not p.exists():
        print(f"[ERRO] Arquivo não encontrado: {p.resolve()}")
        sys.exit(1)

    df = pd.read_csv(p, sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip")
    df = strip_all_strings(df)

    # Normalização do campo de descrição
    df["Prod_Desc_norm"] = df["Prod_Desc"].map(normalize_text)

    total = len(df)
    exatos = df.duplicated(keep=False).sum()
    por_desc = df.duplicated(subset=["Prod_Desc_norm"], keep=False).sum()
    por_sub_desc = df.duplicated(subset=["ID_Sub","Prod_Desc_norm"], keep=False).sum()

    print("=== Diagnóstico ===")
    print(f"Linhas totais: {total}")
    print(f"Duplicados EXATOS (todas as colunas iguais): {exatos}")
    print(f"Duplicados por Prod_Desc_norm: {por_desc}")
    print(f"Duplicados por (ID_Sub, Prod_Desc_norm): {por_sub_desc}")

    # Ordenação para escolher quem manter
    try:
        df["_ID_Prod_num"] = pd.to_numeric(df["ID_Prod"], errors="coerce")
        df = df.sort_values(["ID_Sub", "Prod_Desc_norm", "_ID_Prod_num"], na_position="last")
    except Exception:
        pass

    # A) Dedup por subcategoria + descrição normalizada
    df_a = df.drop_duplicates(subset=["ID_Sub", "Prod_Desc_norm"], keep="first").copy()
    removed_a = total - len(df_a)

    # B) Dedup global por descrição normalizada
    df_b = df.drop_duplicates(subset=["Prod_Desc_norm"], keep="first").copy()
    removed_b = total - len(df_b)

    # --- Remover colunas auxiliares ANTES de salvar ---
    helper_cols = {"Prod_Desc_norm", "_ID_Prod_num"}
    out_cols_a = [c for c in df_a.columns if c not in helper_cols]
    out_cols_b = [c for c in df_b.columns if c not in helper_cols]

    df_a_out = df_a[out_cols_a].copy()
    df_b_out = df_b[out_cols_b].copy()

    # Salvar
    out_a = p.with_name("materiais_clean_unique_by_sub_desc.tsv")
    out_b = p.with_name("materiais_clean_unique_by_desc.tsv")
    df_a_out.to_csv(out_a, sep="\t", index=False, encoding="utf-8")
    df_b_out.to_csv(out_b, sep="\t", index=False, encoding="utf-8")

    print("\n=== Resultados ===")
    print(f"[A] Unique por (ID_Sub, Prod_Desc_norm): {len(df_a_out)} linhas (removidas {removed_a})")
    print(f"[B] Unique por (Prod_Desc_norm):        {len(df_b_out)} linhas (removidas {removed_b})")
    print("\nArquivos gerados (sem colunas auxiliares):")
    print(f"- {out_a.resolve()}")
    print(f"- {out_b.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
