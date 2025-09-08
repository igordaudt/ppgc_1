# clean_materiais.py
from pathlib import Path
import pandas as pd
import re

INPUT = Path("./trabalho1/materiais.csv")
OUTPUT_TSV = Path("./trabalho1/materiais_clean.tsv")
OUTPUT_CSV_TAB = Path("./trabalho1/materiais_clean.csv")  # CSV com separador TAB (para quem prefere .csv)

EXPECTED_COLS = ["ID_CAT", "Cat_Name", "ID_Sub", "Sub_Name", "ID_Prod", "Prod_Desc", "Prod_Det"]

def try_read_csv(path: Path, sep, header, encoding):
    return pd.read_csv(
        path,
        sep=sep,
        engine="python",        # necessário para regex no sep
        header=header,          # 0 ou None
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        quotechar='"',
        escapechar='\\',
        on_bad_lines="error",   # se quiser pular linhas ruins: "skip"
    )

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        c = str(c).strip()
        c = c.strip('"').strip("'")
        c = re.sub(r"\s+", " ", c)  # colapsa espaços múltiplos
        return c
    df.columns = [norm(c) for c in df.columns]
    return df

def drop_prod_det(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    if "prod_det" in lower_map:
        return df.drop(columns=[lower_map["prod_det"]])
    return df

def ensure_prod_desc(df: pd.DataFrame) -> pd.DataFrame:
    if "Prod_Desc" in df.columns:
        return df
    # tenta localizar variações
    for c in df.columns:
        key = c.lower().replace(" ", "").replace("-", "_")
        if key in ("prod_desc", "proddesc", "prod__desc"):
            return df.rename(columns={c: "Prod_Desc"})
    raise KeyError(f"Coluna 'Prod_Desc' não encontrada. Colunas vistas: {list(df.columns)}")

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT.resolve()}")

    # Vamos checar rapidamente presença de possíveis separadores no arquivo bruto
    raw = INPUT.read_text(encoding="utf-8", errors="ignore")
    sample = "\n".join(raw.splitlines()[:200])

    count_gt = sample.count(">")
    count_tab = sample.count("\t")
    count_semic = sample.count(";")
    count_comma = sample.count(",")

    # Estratégia de separadores (mais tolerante primeiro)
    seps = []
    # Se encontramos '>' e/ou TAB, priorizar regex que aceita AMBOS
    if count_gt > 0 and count_tab > 0:
        seps.append(r">\t|[\t>]")
    elif count_gt > 0:
        # aceita > ou TAB (p/ cabeçalho que às vezes vem em TAB)
        seps.append(r">|[\t]")
        seps.append(">")
    elif count_tab > 0:
        seps.append(r"[\t]|>")
        seps.append("\t")
    else:
        # fallback comuns
        if count_semic > 0:
            seps.append(";")
        if count_comma > 0:
            seps.append(",")
        # por último, tentar regex bem abrangente
        seps.append(r"[>\t;,]")

    # Estratégia de cabeçalho
    headers = [0, None]  # 0 = primeira linha é cabeçalho; None = sem cabeçalho
    encodings = ["utf-8-sig", "utf-8", "latin-1"]

    last_err = None
    df = None
    chosen = None
    for enc in encodings:
        for sep in seps:
            for header in headers:
                try:
                    temp = try_read_csv(INPUT, sep=sep, header=header, encoding=enc)
                    # se não tinha cabeçalho, dar nomes provisórios
                    if header is None:
                        # se o arquivo tem 7 colunas, use EXPECTED_COLS; senão, nomes genéricos
                        if temp.shape[1] == len(EXPECTED_COLS):
                            temp.columns = EXPECTED_COLS
                        else:
                            temp.columns = [f"col_{i+1}" for i in range(temp.shape[1])]
                    temp = normalize_headers(temp)

                    # remover colunas totalmente vazias (se existirem)
                    temp = temp.dropna(axis=1, how="all")

                    # garantir Prod_Desc
                    temp = ensure_prod_desc(temp)

                    # pronto: conseguimos ler
                    df = temp.copy()
                    chosen = (enc, sep, header)
                    raise StopIteration
                except StopIteration:
                    break
                except Exception as e:
                    last_err = e
            if df is not None:
                break
        if df is not None:
            break

    if df is None:
        # relatório de erro útil
        raise RuntimeError(
            "Falha ao ler o arquivo com as tentativas. "
            f"Último erro: {repr(last_err)}"
        )

    # Remover Prod_Det
    df = drop_prod_det(df)

    # Strip em strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    # Salvar em TAB (TSV)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUT_CSV_TAB, sep="\t", index=False, encoding="utf-8-sig")

    enc, sep, header = chosen
    print("=== LIMPEZA CONCLUÍDA ===")
    print(f"Encoding usado: {enc}")
    print(f"Separador usado: {sep!r} (engine='python')")
    print(f"Cabeçalho: {'primeira linha' if header==0 else 'sem cabeçalho'}")
    print(f"Linhas: {len(df):,} | Colunas: {len(df.columns)}")
    print("Gerados:")
    print(f" - {OUTPUT_TSV}")
    print(f" - {OUTPUT_CSV_TAB}")

if __name__ == "__main__":
    main()