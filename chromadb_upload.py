import os
import csv
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document  
from tqdm import tqdm
from api_module import api_key

DATA_PATH="."
CHROMA_PATH = "chroma2"  

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

textbook_examples=load_examples(csv_file_path)


def store_examples_in_chroma_db():
    save_to_chroma(textbook_examples)

def save_to_chroma(examples: list):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

   
    embeddings = OpenAIEmbeddings(api_key=api_key)

   
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    documents = [
        Document(
            page_content=f"Problem: {ex['problem']}\nReasoning: {ex['reasoning']}",
            metadata={"problem": ex['problem']}  
        ) for ex in examples
    ]


    db.add_documents(documents)

    
    db.persist()
    print(f"Saved {len(documents)} examples to {CHROMA_PATH}.")

store_examples_in_chroma_db()





