from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document 
import csv, os, tqdm
from tqdm import tqdm

# Load the MathBERT model as an embedding function
embeddings = HuggingFaceEmbeddings(model_name="tbs17/MathBERT")
DATA_PATH="."
CHROMA_PATH = "chroma1"
# Create a new ChromaDB instance with MathBERT embeddings
chroma_db = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH  # Specify the directory for the database
)
examples = []
csv_file_path = os.path.join(DATA_PATH, "dataset.csv")  
def load_examples(csv_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, desc="Loading examples", unit="example"):
                examples.append({
                    "problem": row["problem"],
                    "reasoning": row["reasoning"]
                })
        print(f"Loaded {len(examples)} examples from CSV.")
        return examples

examples=load_examples(csv_file_path)
# Example: Adding documents to the database
documents = [
    Document(
            page_content=f"Problem: {ex['problem']}\nReasoning: {ex['reasoning']}",
            metadata={"problem": ex['problem']}  
        ) for ex in examples
]

# Add documents to the database with embeddings
chroma_db.add_documents(documents)

# # Persist the database
# chroma_db.persist()  