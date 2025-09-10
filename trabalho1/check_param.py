import joblib

# carregar vectorizer
vectorizer = joblib.load("./trabalho1/tfidf_vectorizer.pkl")
print("Configurações do TF-IDF:", vectorizer.get_params())
print("Tamanho do vocabulário:", len(vectorizer.vocabulary_))
print("Algumas palavras:", list(vectorizer.vocabulary_.keys())[:20])

# carregar modelo
model_pack = joblib.load("./trabalho1/nb_tuned_model.pkl")
model = model_pack["model"]

print("Parâmetros do modelo:", model.get_params())
print("Classes aprendidas:", model.classes_[:10])  # só as 10 primeiras
print("Alpha:", model.alpha)
