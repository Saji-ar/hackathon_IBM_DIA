import json
import os
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from source.search_question import search_question  # ⚠️ Ton fichier précédent
from sentence_transformers import SentenceTransformer
from langdetect import detect, detect_langs
from dotenv import load_dotenv


# Charger .env local si présent (optionnel)
load_dotenv()

# --- IBM Watsonx credentials (lues depuis les variables d'environnement) ---
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "eu-de")  # valeur par défaut si souhaitée

if not API_KEY or not PROJECT_ID:
    raise RuntimeError(
        "Missing IBM Watsonx credentials. Set IBM_WATSONX_API_KEY and IBM_WATSONX_PROJECT_ID in your environment."
    )


# --- Initialize credentials and model ---
creds = Credentials(
    url=f"https://{REGION}.ml.cloud.ibm.com",
    api_key=API_KEY
)

llm_model = ModelInference(
    model_id="mistralai/mistral-medium-2505",
    credentials=creds,
    project_id=PROJECT_ID
)


def school_assistant(question: str, school: str, chat_history=None):
    """
    Answer a student's question using retrieved Q&A + conversation context.
    chat_history: list[ {role: 'user'|'assistant', content: str, timestamp: str } ]
    """
    # Step 0 - Language from current question
    language = detect(question)
    if language == "fr":
        language_label = "Français"
        lang_rule = "Réponds STRICTEMENT en français. Ne mélange pas les langues."
    else:
        language_label = "English"
        lang_rule = "Answer STRICTLY in English. Do not mix languages."

    # Step 1 - Retrieve relevant Q&A
    retrieved = search_question(question, school, language_label)
    if retrieved is None or retrieved.empty:
        if language_label == "Français":
            print(f"❌ Aucun contexte trouvé pour '{school}', redirection vers un formulaire.")
            return "Je suis désolé, je n'ai pas pu trouver la réponse à votre question. S'il vous plaît utilisez le formulaire suivant: https://forms.office.com/"
        else:
            print(f"❌ No context found for '{school}', redirecting to form.")
            return "I'm sorry, I couldn't find relevant information. Please use the contact form: https://forms.office.com/"

    # Step 2 - Build retrieval context
    retrieval_context = ""
    for _, row in retrieved.iterrows():
        retrieval_context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

    # Step 3 - Build conversation context (short)
    conv_context = ""
    last_assistant_answer = ""
    if chat_history:
        # Keep last 8 messages
        for msg in chat_history[-8:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if language_label == "Français":
                if role == "user":
                    conv_context += f"Étudiant: {content}\n"
                elif role == "assistant":
                    conv_context += f"Assistant: {content}\n"
            else:
                if role == "user":
                    conv_context += f"Student: {content}\n"
                elif role == "assistant":
                    conv_context += f"Assistant: {content}\n"
        # Dernière réponse de l'assistant (utile pour les relances du type "et après ?")
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_answer = msg["content"]
                break

    # Step 4 - Prompt (favorise la continuité de la conversation et évite le FR/EN mix)
    if language_label == "Français":
        prompt = (
            f"TU dois faire ta réponse au format HTML."
            f"Tu es un assistant administratif pour l'école '{school.upper()}'. "
            f"{lang_rule} Sois concis (1–3 phrases). "
            f"Règles de cohérence:\n"
            f"1) Utilise le contexte conversationnel pour les relances elliptiques (ex: 'et il se passe quoi après ?'). "
            f"2) Si une information a déjà été donnée plus haut dans la conversation, ne la contredis pas. "
            f"3) Utilise le contexte Q&R pour vérifier les faits. En cas de conflit, signale l'incertitude et reste prudent. "
            f"4) Si ni la conversation ni le Q&R ne contiennent l'information, indique-le et propose le formulaire de contact.\n\n"
            f"--- Contexte Q&R ---\n{retrieval_context}\n"
            f"--- Contexte conversationnel (récent) ---\n{conv_context if conv_context else '(aucun)'}\n"
            f"--- Dernière réponse de l'assistant ---\n{(last_assistant_answer or '(aucune)')}\n"
            f"--- Nouvelle question ---\n{question}\n\n"
            f"Réponse:"
        )
    else:
        prompt = (
            f"you must answer in the HTML format."
            f"You are an administrative assistant for the school '{school.upper()}'. "
            f"{lang_rule} Be concise (1–3 sentences). "
            f"Consistency rules:\n"
            f"1) Use conversational context to resolve follow-ups (e.g., 'and then what?'). "
            f"2) Do not contradict information you previously provided. "
            f"3) Use the Q&A context to verify facts. If conflicting, acknowledge uncertainty and stay cautious. "
            f"4) If neither conversation nor Q&A has the info, say so and offer the contact form.\n\n"
            f"--- Q&A Context ---\n{retrieval_context}\n"
            f"--- Conversational Context (recent) ---\n{conv_context if conv_context else '(none)'}\n"
            f"--- Last assistant answer ---\n{(last_assistant_answer or '(none)')}\n"
            f"--- New Question ---\n{question}\n\n"
            f"Answer:"
        )

    params = {
        "max_new_tokens": 180,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        # "stop_sequences": ["\n\nQ:", "\n\nÉtudiant:", "\n\nStudent:"],  # optionnel
    }

    response = llm_model.generate(prompt=prompt, params=params)
    answer = response["results"][0]["generated_text"].strip()

    if language_label == "Français":
        print(f"\n🤖 Réponse de l'assistant:\n{answer}\n")
    else:
        print(f"\n🤖 Assistant response:\n{answer}\n")

    with open("assistant_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

    print("✅ Full JSON response saved to assistant_response.json")
    return answer


# --- Example usage ---
if __name__ == "__main__":
    q = "A combien d'absence ai-je droit?"
    s = "esilv"
    print(school_assistant(q, s, chat_history=[]))