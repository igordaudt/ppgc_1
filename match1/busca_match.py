import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# Funções auxiliares
# =============================

def remover_acentos(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_numbers_units(s: str) -> str:
    s = s.lower()
    s = s.replace("kv", " kv")
    s = s.replace("v ", " v ")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = s.replace("mm²", "mm2")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def aplicar_sinonimos(s: str) -> str:
    s = re.sub(r"\bhepr\b", "epr", s)
    s = re.sub(r"\bnao[- ]?halogenado\b|\blivre halogenio\b", "lszh", s)
    return s

def preprocess(s: str) -> str:
    return aplicar_sinonimos(normalize_numbers_units(remover_acentos(s)))

# =============================
# Classe do comparador
# =============================

class ComparadorProdutos:
    def __init__(self, df_produtos, col_id="ID_Prod", col_desc="Prod_Desc"):
        self.df = df_produtos[[col_id, col_desc]].copy()
        self.col_id = col_id
        self.col_desc = col_desc
        self.df["_norm"] = self.df[self.col_desc].astype(str).apply(preprocess)
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(self.df["_norm"].tolist())

    def buscar(self, consulta: str, top_k=10):
        qnorm = preprocess(consulta)
        qvec = self.vectorizer.transform([qnorm])
        sims = cosine_similarity(qvec, self.tfidf).flatten()
        idx = sims.argsort()[::-1][:top_k]
        return [{
            "id_prod": int(self.df[self.col_id].iloc[i]),
            "descricao": self.df[self.col_desc].iloc[i],
            "score": float(sims[i])
        } for i in idx]

# =============================
# Main
# =============================

if __name__ == "__main__":
    # Exemplo: carregando manualmente alguns produtos
    data = {
    "ID_Prod": [
        192, 194, 197, 257, 302, 319, 320, 321, 322, 371,
        372, 405, 406, 409, 431, 432, 1182, 1183, 1184, 1185,
        1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195,
        1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205,
        1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214        
    ],
    "Prod_Desc": [
        "Cabo cobre nú 50mm²",
        "CABO CONDUTOR 2,5mm² PVC BT 750V",
        "CABO CONDUTOR 16mm² EPR BT 1kV",
        "CABO DE COBRE FLEXÍVEL ISOLADO, 240 mm², ANTI-CHAMAS 0,6/1,0 kV",
        "CABO CONDUTOR 4mm² PVC BT 750V",
        "Cabo baixa tensão 50mm²  061kV  Isolação em EPR90º  Livre halogênio  Instalação tipo B1",
        "Cabo baixa tensão 70mm²  061kV  Isolação em EPR90º  Livre halogênio  Instalação tipo B1",
        "CABO DE COBRE FLEXÍVEL ISOLADO, 120 mm² , ANTI-CHAMAS 0,6/1,0 kV",
        "Cabo baixa tensão 240mm²  061kV  Isolação em EPR90º  Livre halogênio  Instalação tipo B1",
        "CABO CONDUTOR 6mm² PVC BT 750V",
        "CABO CONDUTOR 50mm² PVC BT 750V cor VERMELHA",
        "Otimizador 4510025232 Ts4ao Mlpe 1500v Mc4 Cabo 12m",
        "Cabo Solar 0.6-1KV 6mm²",
        "Cabo Solar Nexans ALDO Solar 47064 ENERGYFLEX Afitox 061KV 1500V",
        "Cabo PP 4 x 25mm",
        "Cabo Flexivel 1 x 15mm",
        "CABO DE COBRE NU 10 MM2 MEIO-DURO",
        "CABO DE COBRE NU 120 MM2 MEIO-DURO",
        "CABO DE COBRE NU 150 MM2 MEIO-DURO",
        "CABO DE COBRE NU 16 MM2 MEIO-DURO",
        "CABO DE COBRE NU 185 MM2 MEIO-DURO",
        "CABO DE COBRE NU 25 MM2 MEIO-DURO",
        "CABO DE COBRE NU 300 MM2 MEIO-DURO",
        "CABO DE COBRE NU 35 MM2 MEIO-DURO",
        "CABO DE COBRE NU 50 MM2 MEIO-DURO",
        "CABO DE COBRE NU 500 MM2 MEIO-DURO",
        "CABO DE COBRE NU 70 MM2 MEIO-DURO",
        "CABO DE COBRE NU 95 MM2 MEIO-DURO",
        "CABO DE COBRE RIGIDO, CLASSE 2, ISOLACAO EM PVC, ANTI-CHAMA BWF-B, 1 CONDUTOR, 450/750 V, DIAMETRO 120 MM2",
        "CABO DE COBRE UNIPOLAR 10 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 16 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 16 MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 25 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 25MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 35 MM2, BLINDADO, ISOLACAO 12/20 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 35 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 35 MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 50 MM2, BLINDADO, ISOLACAO 12/20 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 50 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 50 MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 70 MM2, BLINDADO, ISOLACAO 12/20 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 70 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 70 MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 95 MM2, BLINDADO, ISOLACAO 12/20 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 95 MM2, BLINDADO, ISOLACAO 3,6/6 KV EPR, COBERTURA EM PVC",
        "CABO DE COBRE UNIPOLAR 95 MM2, BLINDADO, ISOLACAO 6/10 KV EPR, COBERTURA EM PVC",
        "CABO FLEXIVEL COBRE HEPR 90G1KV 1X120MM PRETO ATOX",
        "CABO FLEXIVEL COBRE HEPR 90G1KV 1X120MM PRETO ATOX - CORFIO",
        "CABO FLEXIVEL COBRE HEPR 90G1KV 1X120MM PRETO ATOX - SIL SIL 0"
    ]
    }


    df = pd.DataFrame(data)

    # Instanciar modelo
    modelo = ComparadorProdutos(df)

    print("=== Comparador de Produtos ===")
    while True:
        demanda = input("\nDigite a demanda (ou ENTER para sair): ").strip()
        if not demanda:
            break

        resultados = modelo.buscar(demanda, top_k=10)
        if resultados:
            print("\nTop 10 resultados:")
            for r in resultados:
                print(f"[{r['id_prod']}] {r['descricao']}  (score={r['score']:.3f})")
        else:
            print("Nenhum produto encontrado.")
