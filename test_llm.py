import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

key = os.getenv("GROQ_API_KEY")
print(f"🔑 Clé trouvée : {key[:10]}..." if key else "❌ AUCUNE CLÉ DÉTECTÉE DANS .ENV")

if key:
    client = Groq(api_key=key)
    
    # Modèles de CHAT identifiés sur votre compte Groq
    chat_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "allam-2-7b"
    ]
    
    for model_id in chat_models:
        try:
            print(f"🚀 Essai du modèle de chat : {model_id}...")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Tu es DiaBot-RDC, un assistant médical intelligent et chaleureux."},
                    {"role": "user", "content": "Mbote ! Comment tu t'appelles et que penses-tu du pondu avec du fufu pour un diabétique ?"}
                ],
                max_tokens=300
            )
            
            print("\n🎉 SUCCÈS TOTAL ! RÉPONSE DE L'IA :")
            print("=" * 60)
            print(response.choices[0].message.content)
            print("=" * 60)
            break
        except Exception as e:
            print(f"⚠️ {model_id} indisponible : {e}")