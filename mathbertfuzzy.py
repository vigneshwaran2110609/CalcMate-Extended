import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document  
from langchain_openai import OpenAIEmbeddings
import csv
import re
from api_module import api_key
import warnings
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sympy import symbols, solve, Eq
import re,torch
from test_case_module import test_cases
from fuzzy_score import calculate_fuzzy_score
from transformers import AutoTokenizer,AutoModel,AutoConfig

tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
def custom_padding(batch):
    max_len = 512
    padded_batch = []
    for sequence in batch:
        padding = torch.full((max_len - len(sequence),), tokenizer.pad_token_id)
        padded_sequence = torch.cat((sequence, padding))
        padded_batch.append(padded_sequence)
    return padded_batch



warnings.simplefilter("ignore", DeprecationWarning)

tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
model = AutoModelForCausalLM.from_pretrained("tbs17/MathBERT")
model_config = AutoConfig.from_pretrained("tbs17/MathBERT")
print(model_config.max_position_embeddings) 
mathbert_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=350,
    truncation=True
)


# Create an LLM instance for MathBERT
mathbert_llm = HuggingFacePipeline(pipeline=mathbert_pipeline)

CHROMA_PATH = "chroma1"

def load_examples_from_csv(csv_file_path):
    textbook_examples = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            textbook_examples.append({
                "problem": row["problem"],
                "reasoning": row["reasoning"]
            })
    return textbook_examples

textbook_examples = load_examples_from_csv("dataset.csv")

def extract_final_equation(output):
    match = re.search(r"Final equation:\s*(.*)", output, re.IGNORECASE)
    if match:
        equations = match.group(1).strip()
        return equations
    return None

def find_most_similar_examples(input_text, examples, n=3):
    embeddings = HuggingFaceEmbeddings(model_name="tbs17/MathBERT")

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    results = db.similarity_search(input_text, k=n)

    similar_examples = []
    for doc in results:
        d = {}
        clean_content = doc.page_content.replace("\n", "")
        problem = clean_content.split("Reasoning:")[0]
        reasoning = clean_content.split("Reasoning:")[1]
        d['problem'] = problem.replace("Problem:", "").strip()
        d['reasoning'] = reasoning.strip()
        similar_examples.append(d)
   
    return similar_examples

example_template = """
Q. {problem}
A. {reasoning}
"""

prefix = """
You are a Math tutor and your role is to convert a word problem into a Maths equation. Convert it only to a math equation and do not give the answer.
"""

suffix = """
Now, convert the following word problem into a math equation:
Q. {wordproblem}
"""

weighted_fuzzy_score = 0
for test_case in test_cases:
    similar_examples = find_most_similar_examples(test_case["input"], textbook_examples)

    few_shot_prompt = FewShotPromptTemplate(
        examples=similar_examples,
        example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template), 
        prefix=prefix,
        suffix=suffix,
        input_variables=["wordproblem"],
    )

    # Tokenize only the input text without padding or truncation
    inputs = tokenizer(test_case["input"], return_tensors='pt', max_length=512, truncation=True)

    # Ensure to get the decoded string for the LLMChain
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    llm_chain = LLMChain(llm=mathbert_llm, prompt=few_shot_prompt)

    # Run the LLM chain with the string input
    output = llm_chain.run({"wordproblem": input_text})  # Pass as a dictionary for the LLMChain

    final_equation = extract_final_equation(output)
    print(final_equation)

    if final_equation:
        fuzzy_score = calculate_fuzzy_score(test_case["expected_equation"], final_equation)
        weighted_fuzzy_score += fuzzy_score * test_case["weight"]
        print("Test Case Input:", test_case["input"])
        print("Generated Final Equation:", final_equation)
        print("Expected Equation:", test_case["expected_equation"])
        print("Fuzzy Score:", fuzzy_score)
    else:
        print("No final equation found in the output.")
    
print("Weighted Fuzzy Score:", weighted_fuzzy_score / len(test_cases))
