import streamlit as st
import os
import replicate
import os
import pdfplumber
import re
import spacy
from api_module import api_key
from dataset import textbook_examples
from langchain_openai import OpenAIEmbeddings
import warnings
from langchain.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sympy import symbols, solve, Eq
import streamlit as st 
import csv


warnings.simplefilter("ignore", DeprecationWarning)
openai_llm = OpenAI(api_key=api_key, temperature=0.6)
nlp = spacy.load("en_core_web_sm")
CHROMA_PATH="chroma"

def extract_equations_with_spacy(text):
    doc = nlp(text)
    equations = []
    equation_pattern = re.compile(r'([+-]?(\d+)?[A-Za-z]+|\d+)? *([-+]) *([+-]?(\d+)?[A-Za-z]+|\d+) *= *([+-]?(\d+)?[A-Za-z]+|\d+)')

    for sent in doc.sents:
        matches = equation_pattern.findall(sent.text)
        for match in matches:
            left_side = f"{match[0]} {match[2]} {match[3]}"
            right_side = match[5]
            equation = f"{left_side} = {right_side}"
            equations.append(equation.strip())

    return equations

def extract_variables(equations):
    variable_set = set()
    for equation in equations:
        found_vars = re.findall(r'[a-zA-Z]', equation)
        variable_set.update(found_vars)
    return tuple(sorted(variable_set)) 

def determine_final_equations(equations, variables):
    final_equations = []
    for equation in reversed(equations):
        if any(var in equation for var in variables):
            final_equations.append(equation)
        if len(final_equations) == len(variables):
            break
    return list(reversed(final_equations))

def find_most_similar_examples(input_text, examples, n=3):
        embeddings=OpenAIEmbeddings(api_key=api_key)
        db=Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        results=db.similarity_search(input_text,k=3)
        similar_examples=[]
        for doc in results:
            d={}
            clean_content = doc.page_content.replace("\n", "")
            problem=clean_content.split("Reasoning:")[0]
            reasoning=clean_content.split("Reasoning:")[1]
            d['problem'] = problem.replace("Problem:", "").strip()
            d['reasoning'] = reasoning.strip()
            similar_examples.append(d)
        return similar_examples

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
    solution_text = "After solving these set of equations, we get the following solutions: " + "\n"
    if isinstance(solution, dict):
        for var in solution:
            if re.match(r"^\d+\.\d*0+$",str(solution[var])):
                solution[var]=int(solution[var])
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

def standardize_output(llm_output):
    llm_output = llm_output.strip()
    print("Printing LLM OUTPUT IN SO")
    print(llm_output)
    if llm_output[0:3] == "A. ":
        llm_output = llm_output[3:]
    # if llm_output[0:2] == "1.":
    #     return llm_output
    # if re.match(r'^A\. \d+\. .*$', llm_output, re.MULTILINE):
    #     return llm_output
    parts = re.split(r'\n\s*\n|\d\.\s', llm_output)
    parts = [part.strip() for part in parts if part]
    standardized_output = []
    step = 1
    notin1 = [str(i) for i in range(1,len(parts)*3)]
    notin2 = [str(i)+" " for i in range(1,len(parts)*2)]
    notin1 += ["A","A "]
    print(parts)
    print("partyyyy ")
    for part in parts:
        if not bool(re.search(r'[a-zA-Z]', part)):
            continue
        standardized_output.append(f'{step}. {part}')
        step += 1          
    standardized_output = '\n'.join(standardized_output)
    return standardized_output

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

# App title
st.set_page_config(page_title="🧮💬 CalcMate")

# Replicate Credentials
with st.sidebar:
    st.title('🧮💬 CalcMate')
    st.write('Math Problem solver')

try:
    st.write("")  # Optional empty line for spacing
    st.markdown("<h3 style='font-size: 18px;'>🤖 Enter your math problem:</h3>", unsafe_allow_html=True)
    input_text = st.text_area("", height=200)

    # Add vertical space between the text area and the button
    st.write("")  # This adds an empty line
    st.write("")  # This adds another empty line (you can add more as needed)

    if st.button("Solve"):
        example_template = """ Q. {problem} A. {reasoning} """
        prefix = """ You are a Math tutor and your role is to convert a word problem into a Maths equation. 
                    Convert it only to a math equation, in the form of numbered steps.
                    Identification of variables must be present in the steps.
                    Do not give the final answer.
                    Just give the reasoning for the formation of the equation but do not solve the equations.
                    """
        suffix = """Now, convert the following word problem into a math equation with clear, numbered steps:
                    1. Identify variables.
                    2. Set up equations based on the conditions provided.
                    3. Conclude by stating the final equations with the label 'Final Equations: '.

                    Q. {wordproblem}"""
        few_shot_prompt = FewShotPromptTemplate(
            examples=[],
            example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template),
            prefix=prefix,
            suffix=suffix,
            input_variables=["wordproblem"],
        )
        llm_chain = LLMChain(llm=openai_llm, prompt=few_shot_prompt)
        csv_file_path = "dataset.csv"
        textbook_examples = load_examples_from_csv(csv_file_path)
        similar_examples = find_most_similar_examples(input_text, textbook_examples)

        print(similar_examples)
        #print(similar_examples,"helloooo oo o o o o o o o o o oo  oooooooooooooooooo o oo o o o")
        few_shot_prompt.examples = similar_examples
        llm_chain.prompt = few_shot_prompt
        output = llm_chain.run(input_text)

        if "Final Equations:" in output:
                equations = output.split('Final Equations:')[-1].strip() 
                steps=output.split('Final Equations:')[0].strip()
        elif "Final Equation:" in output:
                equations = output.split('Final Equation:')[-1].strip() 
                steps = output.split('Final Equation:')[0]
        elif "Final Equation" in output:
                equations = output.split("Final Equation")[-1].strip()
                steps = output.split('Final Equation')[0]
        elif "Final Equations" in output:
                equations = output.split("Final Equations")[-1].strip()
                steps = output.split('Final Equations')[0]

        try:
            matches = re.findall(r"(?:Let )?(\w+) be the (.+?)(?:,| and|\.|$)|(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", steps)
            flag=False
            for match in matches:
                if match[0]:
                    flag=True
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
        equations = re.split(r'\s*,\s*|\s+and\s+', equations)

        equations = [eq.strip().rstrip('.') for eq in equations if eq.strip()]

        equations = [eq.strip() for eq in equations if '=' in eq]
        valid_equations = []
        for eq in equations:
            if '=' in eq:
                if eq[0:4] == "and " or eq[0:3] == "And ":
                    eq = eq[4:]
                valid_equations.append(eq)

        symbols_found = re.findall(r'[a-zA-Z]', " ".join(valid_equations))
        distinct_symbols = tuple(set(symbols_found))
        print(valid_equations)
        if valid_equations:
            solution_text,solution_values = solve_equation(valid_equations, distinct_symbols)
        else:
            solution_text = "No valid equations found."
        try:
            matches = re.findall(r"(?:Let )?(\w+) be the (.+?)(?:,| and|\.|$)|(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", steps)
            flag = False
            for match in matches:
                if match[0]:
                    flag = True
            if not flag:
                raise Exception("nothing in matches")
        except:
            matches = re.findall(r"(?:Let\s+)?(\w+)\s*=\s*([^.,]+?)(?:,| and|\.|$)",steps)

        variable_labels = {}
        for match in matches:
            if match[0] not in variable_labels and re.match(r'^[a-zA-Z_]\w*$', match[0]):
                var = match[0] or match[2]  
                desc = match[1] or match[3]
                variable_labels[var] = desc.strip()

        final_result = "Therefore,\n"
        for var in variable_labels:
            final_result += f"The value of {variable_labels[var]} is {solution_values.get(var, 'unknown')}.\n"

        # final_solution=steps+"\n"+solution_text
        st.write("Here is the solution for the Problem: ")
        steps = standardize_output(steps)
        st.write(steps)
        st.write()
        st.write("Therefore, after solving the above equations: ")
        print(solution_values,variable_labels)
        solution_text = "<br>".join(
    [f"The value of {var} ({variable_labels[var]}) is {solution_values.get(var, 'unknown')}." for var in variable_labels]
)
        print(solution_text)
        st.markdown(solution_text, unsafe_allow_html=True)
except:
    st.write("Error in solving the problem, please try again later.")
