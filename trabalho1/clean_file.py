# clean_materiais.py
from pathlib import Path
import pandas as pd
import re

INPUT = Path("./trabalho1/cenario/materiais.csv")
OUTPUT = Path("./trabalho1/cenario/materiais_clean.tsv")
OUTPUT_CSV_TAB = Path("./trabalho1/cenario/materiais_clean.csv")  # CSV com separador TAB (para quem prefere .csv)

EXPECTED_COLS = ["ID_CAT", "Cat_Name", "ID_Sub", "Sub_Name", "ID_Prod", "Prod_Desc"]

# ---------- Utilidades ----------
def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    # fallback: ignora erros para não travar
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize_header_fields(fields):
    out = []
    for f in fields:
        f = (f or "").strip().strip('"').strip("'")
        f = re.sub(r"\s+", " ", f)
        out.append(f)
    return out

def canon_map(name: str) -> str:
    key = (name or "").lower().replace(" ", "").replace("-", "_")
    if key in ("id_cat", "idcat"): return "ID_CAT"
    if key in ("cat_name", "catname"): return "Cat_Name"
    if key in ("id_sub", "idsub"): return "ID_Sub"
    if key in ("sub_name", "subname"): return "Sub_Name"
    if key in ("id_prod", "idprod"): return "ID_Prod"
    if key in ("prod_desc", "proddesc"): return "Prod_Desc"
    if key in ("prod_det", "proddet"): return "Prod_Det"
    return name or ""

def coerce_to_6_fields(fields):
    # garante exatamente 6 colunas
    fields = list(fields)
    if len(fields) < 6:
        fields += [""] * (6 - len(fields))
    elif len(fields) > 6:
        head, tail = fields[:5], fields[5:]
        merged_last = "\t".join(tail)  # preserva excedente na última coluna
        fields = head + [merged_last]
    return fields

# ---------- Parser manual (estado de aspas) ----------
def parse_records(raw: str, delim: str = ">", quote: str = '"'):
    """
    Retorna lista de linhas, onde cada linha é lista de campos.
    Regras:
      - delim só separa fora de aspas
      - fim de linha só fecha registro fora de aspas
      - "" dentro de aspas vira aspas literal
    """
    rows = []
    field = []
    record = []
    in_quotes = False
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        if ch == quote:
            # aspas duplicadas dentro de campo -> aspas literal
            if in_quotes and i + 1 < n and raw[i+1] == quote:
                field.append(quote)
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue

        if ch == delim and not in_quotes:
            record.append("".join(field))
            field = []
            i += 1
            continue

        # tratar quebras de linha (CRLF/CR/LF) somente fora de aspas
        if (ch == "\n" or ch == "\r") and not in_quotes:
            # consome \r\n como uma quebra
            if ch == "\r" and i + 1 < n and raw[i+1] == "\n":
                i += 1
            record.append("".join(field))
            rows.append(record)
            field = []
            record = []
            i += 1
            continue

        field.append(ch)
        i += 1

    # flush final (última linha pode não terminar com \n)
    record.append("".join(field))
    rows.append(record)
    return rows

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT.resolve()}")

    raw = read_text_any(INPUT)
    rows = parse_records(raw, delim=">", quote='"')
    if not rows:
        raise RuntimeError("Arquivo vazio.")

    # detectar cabeçalho
    header = normalize_header_fields(rows[0])
    header_mapped = [canon_map(c) for c in header]
    header_present = any(h.lower() in ("prod_desc", "sub_name", "id_sub") for h in header_mapped)

    data_rows = rows[1:] if header_present else rows

    # normalizar cada linha para 6 colunas
    fixed = []
    for r in data_rows:
        # strip de espaços externos e aspas redundantes nas células
        r = [ (c or "").strip() for c in r ]
        fixed.append(coerce_to_6_fields(r))

    # definir nomes de colunas
    if header_present:
        cols = [canon_map(c) for c in header]
        # se o header não tiver 6, ajusta
        cols = coerce_to_6_fields(cols)
        cols = [canon_map(c) for c in cols]
    else:
        cols = EXPECTED_COLS[:]

    df = pd.DataFrame(fixed, columns=cols)

    # remover Prod_Det se existir
    if "Prod_Det" in df.columns:
        df = df.drop(columns=["Prod_Det"])

    # garantir e ordenar colunas canônicas
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[EXPECTED_COLS]

    # limpeza leve
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(r'^\s*"\s*|\s*"\s*$', "", regex=True)  # remove aspas externas
            )

    # filtra linhas sem texto e sem rótulo mínimo
    has_text = df["Prod_Desc"].astype(str).str.len() > 0
    has_label = df["Sub_Name"].astype(str).str.len() > 0
    df = df[has_text & has_label].reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, sep="\t", index=False, encoding="utf-8-sig")

    print("=== LIMPEZA OK ===")
    print(f"Linhas: {len(df):,} | Colunas: {len(df.columns)} (6 fixas)")
    print(f"Saída: {OUTPUT}")

if __name__ == "__main__":
    main()