import lancedb
from sentence_transformers import SentenceTransformer

# --- Charger la base LanceDB et le modèle ---
db = lancedb.connect("lancedb_questions")
table = db.open_table("qa_table")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Boucle interactive ---
while True:
    query = input("\n❓ Quelle est votre question ? (ou 'quit' pour sortir)\n> ")
    if query.lower() in ["quit", "exit"]:
        print("👋 Fin du programme.")
        break

    # Encoder la question et faire la recherche
    query_vec = model.encode(query)
    results = table.search(query_vec, vector_column_name="answer_embedding").limit(3).to_pandas()

    # Affichage des 3 résultats les plus pertinents
    print("\n🔍 Résultats les plus pertinents :\n")
    for i, row in results.iterrows():
        print(f"🧠 Question similaire {i+1}: {row['question']}")
        print(f"💬 Réponse : {row['answer']}")
        print(f"🏫 École : {row['ecole']} | 🌍 Langue : {row['langue']}")
        print("-" * 80)