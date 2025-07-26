import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document  
import csv
import re
from api_module import api_key
import warnings
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sympy import symbols, solve, Eq
import re
from test_case_module import test_cases
from equivalence_score import calculate_equivalence_score

warnings.simplefilter("ignore", DeprecationWarning)

openai_llm = OpenAI(api_key=api_key, temperature=0.6)

CHROMA_PATH = "chroma2"

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

def find_most_similar_examples(input_text,n=3):
    embeddings = OpenAIEmbeddings()

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

prefix = """ You are a Math tutor and your role is to convert a word problem into a Maths equation. 
             Convert it only to a math equation, in the form of numbered steps.
             Identification of variables must be present in the steps.
             Explanation of the formation of equation in the steps.
             Do not solve the equations and do not give the final answer.
         """

suffix = """ Now, convert the following word problem into a math equation: Q. {wordproblem}.
             Mention the Final Equations using the Keyword Final Equation or Final Equations. 
             Seperate the equations using commas."""

weighted_eq_score=0
for test_case in test_cases:
    similar_examples = find_most_similar_examples(test_case["input"])

    few_shot_prompt = FewShotPromptTemplate(
        examples=similar_examples,
        example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template), 
        prefix=prefix,
        suffix=suffix,
        input_variables=["wordproblem"],
    )

    llm_chain = LLMChain(llm=openai_llm, prompt=few_shot_prompt)

    output = llm_chain.run(test_case["input"])

    final_equation, steps = extract_final_equation(output)

    if final_equation:
        print(test_case["expected_equation"],final_equation)
        eq_score = calculate_equivalence_score(test_case["expected_equation"], final_equation)
        weighted_eq_score+=eq_score*test_case["weight"]
        print("Test Case Input:", test_case["input"])
        print("Generated Final Equation:", final_equation)
        print("Expected Equation:", test_case["expected_equation"])
        print("Equivalence Score:", eq_score)
    else:
        print("No final equation found in the output.")
    
print(weighted_eq_score/sum(tc["weight"] for tc in test_cases))
