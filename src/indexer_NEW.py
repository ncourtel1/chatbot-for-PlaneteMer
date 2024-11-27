from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document  # Import nécessaire pour créer des objets Document

import os
import PyPDF2

# Chemin vers le dossier contenant les fichiers PDF
folder_path = "/Users/nattan/Documents/cs/zone01/ai/chatbot-for-PlaneteMer/data"

# Liste pour stocker les documents chargés
documents = []

# Parcourir tous les fichiers du répertoire
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Vérifier si c'est un fichier PDF
    if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
        # Ouvrir le fichier PDF avec PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extraire le texte de chaque page du PDF
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()  # Extraire le texte de la page
            
            # Créer un objet Document avec le texte extrait
            document = Document(page_content=text, metadata={"source": file_path})
            documents.append(document)  # Ajouter l'objet Document à la liste
        
# Afficher le nombre de documents chargés
print(f"Nombre de documents PDF chargés : {len(documents)}")

# Étape 2 : Diviser les documents en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    is_separator_regex=False
)
texts = text_splitter.split_documents(documents)
print(f"Nombre de chunks générés : {len(texts)}")

# Étape 3 : Initialiser le modèle Hugging Face pour les embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Étape 4 : Créer et persister la base de données Chroma
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory="./db-planet-mer"  # Chemin où persister les embeddings
)
print("Vectorstore créé et persisté.")

# Étape 5 : Charger la base de données pour vérification
loaded_vectorstore = Chroma(
    persist_directory="./db-planet-mer",
    embedding_function=embedding_model
)
print(f"Nombre de documents dans le vectorstore chargé : {len(loaded_vectorstore.get())}")
