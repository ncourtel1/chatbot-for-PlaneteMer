import os
import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration de l'API Gemini
genai.configure(api_key="AIzaSyB7VTLxWHoAQKFzlfj8goI3UbS0yCkCB8c")

# Configuration du modèle Gemini
generation_config = {
    "temperature": 0.05,
    "top_p": 0.96,
    "top_k": 30,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialisation des embeddings et de la base de données Chroma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./db-planet-mer", embedding_function=embedding_model)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 35,
        "lambda_mult": 0.7,
    }
)
print(" \n")

# Template du prompt pour Gemini
template = """Context: {context}

Question: {question}
Vous êtes un expert en biologie marine et en pêche. Vous devez répondre de manière précise sur les reglementations en vigueur, 
En vous appuyant sur la base documentaire. N'hesite pas a resumer l'information.
Si tu ne trouves pas la reponse dans le contexte, dis que tu ne sais pas.
Sois clair et concis.
"""

# Fonction pour formater les documents et extraire les sources
# Fonction pour formater les documents et extraire les chunks et leurs sources
def format_docs(docs):
    chunks_and_sources = []
    for doc in docs:
        content = doc.page_content  # Le chunk de texte
        source = doc.metadata.get("source", "Source inconnue")  # La source associée
        chunks_and_sources.append({"content": content, "source": source})
    return chunks_and_sources

# Fonction pour générer la réponse avec les sources
def generate_response_with_sources(retriever, question):
    try:
        # Récupération des documents pertinents
        docs = retriever.get_relevant_documents(question)
        chunks_and_sources = format_docs(docs)
    except Exception as e:
        return f"Erreur lors de la récupération des documents : {e}", []

    # Construire le contexte pour le prompt
    context = "\n\n".join([chunk["content"] for chunk in chunks_and_sources])
    sources = [chunk["source"] for chunk in chunks_and_sources]

    try:
        # Créer une session de chat et envoyer la requête
        chat_session = model.start_chat(history=[])
        full_prompt = f"Veuillez répondre en français.\n\n{template.format(context=context, question=question)}"
        response = chat_session.send_message(full_prompt)
    except Exception as e:
        return f"Erreur lors de la génération de la réponse : {e}", []

    return response.text, chunks_and_sources  # Retourne la réponse et les chunks avec leurs sources

# Configuration de la page Streamlit
st.set_page_config(page_title="Planète Mer ChatBot", page_icon="🐠")

# Ajout de styles personnalisés
st.markdown("""
    <style>
        .user-message {
            background-color: #e3f2fd;
            color: #0d47a1;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-end;
        }
        .assistant-message {
            background-color: #f1f8e9;
            color: #33691e;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Initialisation des messages si pas déjà fait
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣", [])}
    ]

# Sidebar avec historique
with st.sidebar:
    st.title("ChatBot Planète Mer")
    
    # Bouton pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.messages = [
            {"role": "assistant", "content": ("Bonjour! Je suis là pour répondre à vos questions sur la pêche et la vie marine. Comment puis-je vous aider? 🎣", [])}
        ]
    
    # Affichage de l'historique
    st.markdown("### Historique des conversations")
    for message in st.session_state.messages[1:]:
        if message["role"] == "user":
            st.markdown(f"**😎 Vous:** {message['content'][:100]}...")
        else:
            response, _ = message["content"]
            st.markdown(f"**🤖 Assistant:** {response[:100]}...")

# Zone principale de chat
st.title("Bonjour !")

# Affichage des messages

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    elif message["role"] == "assistant":
        response, chunks_and_sources = message["content"]
        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)


# Zone de saisie utilisateur
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Réflexion en cours..."):
        response, sources = generate_response_with_sources(retriever, prompt)
        st.session_state.messages.append({"role": "assistant", "content": (response, sources)})
        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f'- ***{source["source"].replace("/Users/nattan/Documents/cs/zone01/ai/chatbot-for-PlaneteMer/data/", "")}*** :\n "{source["content"]}"')
                    

