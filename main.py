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

warnings.simplefilter("ignore", DeprecationWarning)

openai_llm = OpenAI(api_key=api_key, temperature=0.6)

CHROMA_PATH = "chroma"

# Function to load examples from a CSV file
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

# Load examples from CSV
textbook_examples = load_examples_from_csv("dataset.csv")

# Function to find most similar examples from the Chroma database
def find_most_similar_examples(input_text, examples, n=3):
    embeddings = OpenAIEmbeddings()
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)

    print(len(db))

    results = db.similarity_search(input_text, k=n)

    print(results)

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


# Sample input text for finding similar examples
input_text1 = "A store sells three types of fruits: apples, bananas, and cherries. The total number of fruits sold is 200. The number of apples is twice the number of bananas, and the number of bananas is 30 more than the number of cherries. Write a system of linear equations to represent this information and find the number of each type of fruit sold."
similar_examples = find_most_similar_examples(input_text1, textbook_examples)

print("HI")

print(similar_examples)


# Template for few-shot examples
example_template = """
Q. {problem}
A. {reasoning}
"""

prefix = """ You are a Math tutor and your role is to convert a word problem into a Maths equation. 
                    Convert it only to a math equation, in the form of numbered steps.
                    Identification of variables must be present in the steps.
                    Explanation of the formation of equation in the steps.
                    Do not give the final answer.
                    """
suffix = """Now, convert the following word problem into a math equation with clear, numbered steps:
1. Identify variables.
2. Set up equations based on the conditions provided.
3. Provide reasoning for each equation formation.But do not solve the equation
4. Conclude by stating the final equations with the label 'Final Equations: '.

Q. {wordproblem}"""


# Create FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=similar_examples,
    example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template), 
    prefix=prefix,
    suffix=suffix,
    input_variables=["wordproblem"],
)


# Create LLMChain with the OpenAI LLM
llm_chain = LLMChain(llm=openai_llm, prompt=few_shot_prompt)

# Run the LLMChain to get the output
output = llm_chain.run(input_text1)


# Initialize a dictionary to store variable solutions
variable_solution = {}

# Function to solve equations using sympy
def solve_equation(equations, distinct_symbols):
    symbol_objects = symbols(distinct_symbols)
    eqs = []

    for equation in equations:
        equation = equation.strip() 
        equation = equation.rstrip('.')

        if '=' in equation:
            parts = equation.split('=')
            if len(parts) == 2: 
                lhs_str, rhs_str = parts
                lhs_str = lhs_str.strip()
                rhs_str = rhs_str.strip()

                # Ensure multiplication is explicit
                lhs_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', lhs_str)
                rhs_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', rhs_str)

                lhs_str = re.sub(r'(\d)\s*\(', r'\1*(', lhs_str)
                rhs_str = re.sub(r'(\d)\s*\(', r'\1*(', rhs_str)

                lhs_str = re.sub(r'(\d)(,)', r'\1', lhs_str)
                rhs_str = re.sub(r'(\d)(,)', r'\1', rhs_str)

                try:
                    eval_context = dict(zip(distinct_symbols, symbol_objects))
                    lhs = eval(lhs_str.replace(' ', ''), {}, eval_context)
                    rhs = eval(rhs_str.replace(' ', ''), {}, eval_context)
                    eqs.append(Eq(lhs, rhs))
                except Exception as e:
                    print(f"Error evaluating equation: '{equation}' - {str(e)}")
            else:
                print(f"Warning: Equation '{equation}' has more than one '=' sign.")
        else:
            print(f"Warning: Equation '{equation}' does not contain an '=' sign.")

    if not eqs:
        return "No valid equations to solve.", {}

    # Solve the equations
    solution = solve(eqs, symbol_objects)
    solution_values = {}  # Initialize the solution values dictionary

    # Prepare the solution text
    solution_text = "After solving these set of equations, we get the following solutions:"
    if isinstance(solution, dict):
        for var in solution:
            if re.match(r"^\d+\.\d*0+$",str(solution[var])):  # if the result has trailing zeroes, cast to int
                solution[var] = int(solution[var])
        for var in solution:
            solution_text += f"\nValue of {var} = {solution[var]}"
            solution_values[str(var)] = solution[var]
    elif isinstance(solution, list):
        for sol in solution:
            if isinstance(sol, tuple):
                for var, val in zip(symbol_objects, sol):
                    if re.match(r"^\d+\.\d*0+$", str(val)):
                        val = int(val)
                    solution_text += f"\nValue of {var} = {val}"
                    solution_values[str(var)] = val
            else:
                if re.match(r"^\d+\.\d*0+$", str(sol)):
                    sol = int(sol)
                solution_text += f"\nValue = {sol}"
    else:
        solution_text += "\nNo valid solution found."

    return solution_text, solution_values

# Extract equations and steps from output
if "Final Equations:" in output:
    equations = output.split('Final Equations:')[-1].strip() 
    steps = output.split('Final Equations:')[0].strip()
elif "Final Equation:" in output:
    equations = output.split('Final Equation:')[-1].strip()
    steps = output.split('Final Equation:')[0]

# Extract variables from steps
try:
    matches = re.findall(r"(?:Let )?(\w+) be the (.+?)(?:,| and|\.|$)|(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", steps)

    flag = False
    for match in matches:
        if match[0]:
            flag = True

    if not flag:
        raise Exception("nothing in matches")
except:
    matches = re.findall(r"(?:Let\s+)?(\w+)\s*=\s*([^.,]+?)(?:,| and|\.|$)", steps)

variable_labels = {}
for match in matches:
    if match[0] not in variable_labels and re.match(r'^[a-zA-Z_]\w*$', match[0]):
        var = match[0] or match[2]  
        desc = match[1] or match[3]
        variable_labels[var] = desc.strip()

# Clean and prepare valid equations
equations = re.split(r',\s*|\s+and\s+', equations)
equations = [eq.strip() for eq in equations if '=' in eq]  

valid_equations = []
for eq in equations:
    if '=' in eq:
        valid_equations.append(eq)

# Find distinct symbols used in the equations
symbols_found = re.findall(r'[a-zA-Z]', " ".join(valid_equations))
distinct_symbols = tuple(set(symbols_found))

# Solve equations if valid equations are found
if valid_equations:
    solution_text, solution_values = solve_equation(valid_equations, distinct_symbols)
else:
    solution_text = "No valid equations found."

# Format the final result
final_result = "Therefore,\n"
for var in variable_labels:
    final_result += f"The value of {variable_labels[var]} is {solution_values.get(var, 'unknown')}.\n"

# Combine all results into the final solution
final_solution = steps + "\n" + solution_text + "\n" + final_result

# Print the final solution
print(final_solution)
