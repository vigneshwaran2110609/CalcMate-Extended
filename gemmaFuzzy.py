import os
import csv
import re
from langchain_ollama import OllamaLLM
 # Import the LLaMA wrapper from langchain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from test_case_module import test_cases  # Your test cases module
from fuzzy_score import calculate_fuzzy_score  # Your fuzzy score calculation 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from api_module import api_key

CHROMA_PATH = "chroma"
llama_model_name = "mistral:latest"  # Use the correct model name
llm = OllamaLLM(model=llama_model_name)
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
    if "Final Equations:" in output:
        equations = output.split('Final Equations:')[-1].strip()
        steps = output.split('Final Equations:')[0].strip()
    elif "Final Equation:" in output:
        equations = output.split('Final Equation:')[-1].strip()
        steps = output.split('Final Equation:')[0].strip()
    else:
        equations = None
        steps = None
    return equations, steps

def find_most_similar_examples(input_text, examples, n=3):
    embeddings = OpenAIEmbeddings(api_key=api_key)

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    results = db.similarity_search(input_text, k=3)

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
You are a Math tutor, and your role is to convert a word problem into a Math equation. Convert it only to a math equation and do not give the answer.
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

    llm_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

    output = llm_chain.run(test_case["input"])

    final_equation = extract_final_equation(output)

    if final_equation:
        fuzzy_score = calculate_fuzzy_score(test_case["expected_equation"], final_equation)
        weighted_fuzzy_score += fuzzy_score * test_case["weight"]
        print("Test Case Input:", test_case["input"])
        print("Generated Final Equation:", final_equation)
        print("Expected Equation:", test_case["expected_equation"])
        print("Fuzzy Score:", fuzzy_score)
    else:
        print("No final equation found in the output.")
    
print(weighted_fuzzy_score / sum(test_case["weight"] for test_case in test_cases))  # Divide by the number of test cases
