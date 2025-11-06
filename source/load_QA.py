import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer

import pandas as pd

csv_path = "data/Questions-Export-2025-October-27-1237 (1)(Questions-Export-2025-October-2).csv"

# Essaie d’abord ISO-8859-1
df = pd.read_csv(csv_path, sep=';', encoding='ISO-8859-1')

print(df.head())

# --- 3. Vérification des colonnes disponibles ---
print("Colonnes détectées :", df.columns.tolist())

# --- 4. Sélection des colonnes nécessaires ---
# (adapte si les noms exacts diffèrent, ex: 'Question', 'Réponse', etc.)
df = df[['Title', 'Content', 'Écoles', 'Langues']].rename(columns={
    'Title': 'question',
    'Content': 'answer',
    'Écoles': 'ecole',
    'Langues': 'langue'
})
# --- Conversion sécurisée en chaînes ---
for col in ['question', 'answer', 'ecole', 'langue']:
    df[col] = df[col].astype(str)

# --- 5. Chargement du modèle d'embedding ---
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# --- 6. Génération des embeddings séparés ---
df["question_embedding"] = df["question"].fillna("").apply(lambda x: model.encode(x))
df["answer_embedding"]  = df["answer"].fillna("").apply(lambda x: model.encode(x))

# --- 7. Connexion à la base LanceDB ---
db = lancedb.connect("lancedb_questions")
table = db.create_table("qa_table", data=df.to_dict(orient="records"), mode="overwrite")

print(f"✅ Base créée : {len(df)} lignes insérées")
print("Champs vectoriels : question_embedding, answer_embedding")

# --- 8. Exemple de recherche ---
query = "Comment fonctionne l'apprentissage automatique ?"
query_vec = model.encode(query)
results = table.search(query_vec, vector_column_name="question_embedding").limit(3).to_pandas()

print("\n🔍 Résultats similaires :")
print(results[['question', 'answer', 'ecole', 'langue']])