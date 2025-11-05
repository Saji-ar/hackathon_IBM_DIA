import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# --- CONFIG (remplacez par vos infos) ---
DB_USER = "user"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "qa_db"

CSV_PATH = "data/Questions-Export-2025-October-27-1237 (1)(Questions-Export-2025-October-2).csv"

# --- 1. Charger CSV (même logique que [load_QA.py]) ---
df = pd.read_csv(CSV_PATH, sep=';', encoding='ISO-8859-1')
df = df[['Title', 'Content', 'Écoles', 'Langues']].rename(columns={
    'Title': 'question',
    'Content': 'answer',
    'Écoles': 'ecole',
    'Langues': 'langue'
})
for col in ['question', 'answer', 'ecole', 'langue']:
    df[col] = df[col].astype(str)

# --- 2. Charger modèle d'embeddings (même modèle que dans load_QA.py) ---
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# --- 3. Générer embeddings (convertis en lists pour JSON storage) ---
df['question_embedding'] = df['question'].apply(lambda x: model.encode(x).tolist())
df['answer_embedding']  = df['answer'].apply(lambda x: model.encode(x).tolist())

# --- 4. Connexion à MariaDB via SQLAlchemy ---
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
                       pool_pre_ping=True)

create_table_sql = """
CREATE TABLE IF NOT EXISTS qa_table (
  id INT AUTO_INCREMENT PRIMARY KEY,
  question TEXT,
  answer TEXT,
  ecole VARCHAR(255),
  langue VARCHAR(100),
  question_embedding JSON,
  answer_embedding JSON
) CHARACTER SET = utf8mb4;
"""
with engine.begin() as conn:
    conn.execute(text(create_table_sql))

# --- 5. Insertion (ex: boucle simple) ---
insert_sql = text("""
INSERT INTO qa_table (question, answer, ecole, langue, question_embedding, answer_embedding)
VALUES (:q, :a, :ec, :lg, :qemb, :aemb)
""")

with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(insert_sql, {
            "q": row['question'],
            "a": row['answer'],
            "ec": row['ecole'],
            "lg": row['langue'],
            "qemb": json.dumps(row['question_embedding']),
            "aemb": json.dumps(row['answer_embedding'])
        })

print("✅ Données insérées dans MariaDB")

# --- 6. Recherche : encodage de la requête puis récupération + similarité ---
def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

query = "Comment fonctionne l'apprentissage automatique ?"
query_vec = model.encode(query).tolist()

# Récupérer embeddings (attention volumétrie : paginer si gros dataset)
with engine.begin() as conn:
    rows = conn.execute(text("SELECT id, question, answer, ecole, langue, question_embedding FROM qa_table")).fetchall()

scores = []
for r in rows:
    qemb = json.loads(r['question_embedding']) if isinstance(r['question_embedding'], (str, bytes)) else r['question_embedding']
    sim = cosine_sim(query_vec, qemb)
    scores.append((sim, r['question'], r['answer'], r['ecole'], r['langue']))

top = sorted(scores, key=lambda x: x[0], reverse=True)[:5]
print("\nTop résultats (MariaDB + calcul Python):")
for s, q, a, ec, lg in top:
    print(f"{s:.4f} — {q} — {ec} — {lg}")