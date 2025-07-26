from flask import Flask, send_from_directory, request, jsonify, session,send_file
from flask_cors import CORS, cross_origin
from db_connection import conn
import os
import replicate
import pdfplumber
import math
import re
import re as regex_module
from sympy import symbols, Eq, solve, Poly, expand, I, re as sympy_re, im as sympy_im
import spacy
from api_module import api_key
from dataset import textbook_examples
from langchain_openai import OpenAIEmbeddings
import warnings
from langchain.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sympy import symbols, Eq, solve, Poly, expand, I, re as sympy_re, im as sympy_im
import csv
import redis
from langchain_ollama import OllamaLLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import textwrap
import joblib
import datetime
import json
import math
import re as regex_module

app = Flask(__name__, static_folder='angular-project/dist/angular-project/browser')
flagForFollowUp = False
memory = ConversationBufferMemory(input_key="wordproblem", memory_key="chat_history")
CORS(app, supports_credentials=True)
app.secret_key = "virat"
warnings.simplefilter("ignore", DeprecationWarning)
openai_llm = OpenAI(api_key=api_key, temperature=0.6)
llama_model_name = "llama3.1:latest"
llm_ollama = OllamaLLM(model=llama_model_name)
nlp = spacy.load("en_core_web_sm")
CHROMA_PATH = "chroma"
textbook_examples = None
model = joblib.load("random_forest_followup.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
@app.route('/')
def serve_angular():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/login', methods=["POST"])
def login():
    try:
        login_data = request.get_json()
        username = login_data.get('username')
        password = login_data.get('password')
        if not username or not password:
            return jsonify({"Message": "Username and password are required"}), 400
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM Users WHERE username=%s AND password=%s", (username, password))
        row = cur.fetchone()
        if row:
            user_id = row[0]
            f = open("temp_file","w+")
            f.write(str(user_id))
            f.close()
            cur.execute("SELECT MAX(chat_id) FROM problem_solutions WHERE user_id=%s", (user_id,))
            chat_id = cur.fetchone()[0]
            if not chat_id:
                chat_id = 1
            return jsonify({"Message": "Login successful","chat_id":chat_id}), 200
        else:
            return jsonify({"Message": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Error during login: {e}")
        return jsonify({"Message": "Error during login"}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        signup_data = request.get_json()
        username = signup_data.get('username')
        password = signup_data.get('password')
        email = signup_data.get('email')
        if not username or not password or not email:
            return jsonify({"Message": "Username, password, and email are required"}), 400
        cur = conn.cursor()
        cur.execute('SELECT user_id FROM Users WHERE username=%s', (username,))
        existing_user = cur.fetchone()
        if existing_user:
            return jsonify({"Message": "Username already exists"}), 409
        cur.execute('INSERT INTO Users(username, password, email) VALUES (%s, %s, %s)', (username, password, email))
        conn.commit()
        cur.execute("SELECT user_id FROM Users WHERE username=%s AND password=%s", (username, password))
        f = open("temp_file","w+")
        row = cur.fetchone()
        user_id = row[0]
        f.write(str(user_id))
        f.close()
        return jsonify({"Message": "Signup successful","chat_id":1}), 200
    except Exception as e:
        print(f"Error during signup: {e}")
        return jsonify({"Message": "Error during signup"}), 500


def standardize_output(llm_output):
    print(llm_output)
    llm_output = llm_output.strip()
    if llm_output[0:3] == "A. ":
        llm_output = llm_output[3:]
    parts = re.split(r'\n\s*\n|\d\.\s', llm_output)
    parts = [part.strip() for part in parts if part]
    standardized_output = []
    step = 1
    notin1 = [str(i) for i in range(1,len(parts)*3)]
    notin2 = [str(i)+" " for i in range(1,len(parts)*2)]
    notin1 += ["A","A "]
    for part in parts:
        if not bool(re.search(r'[a-zA-Z]', part)):
            continue
        standardized_output.append(f'{step}. {part}')
        step += 1          
    standardized_output = "<br>".join([f"Step{line.strip()}" for line in standardized_output])
    return standardized_output

def solve_equation(equations, distinct_symbols):
    """
    Solve a system of equations with support for complex solutions.
    
    Args:
        equations: List of equation strings
        distinct_symbols: List of symbol strings used in the equations
        
    Returns:
        Tuple of (solution text, solution values dictionary)
    """
    print("&&&&&&&&&&&&&&&&&")
    print(equations, distinct_symbols)
    symbol_objects = symbols(distinct_symbols)
    eqs = []
    
    # Process and parse each equation
    for equation in equations:
        equation = equation.strip().rstrip('.')
        
        if '=' in equation:
            parts = equation.split('=')
            if len(parts) == 2:
                lhs_str, rhs_str = parts
                lhs_str = lhs_str.strip()
                rhs_str = rhs_str.strip()
                
                # Preprocess the equation parts
                for i, part in enumerate([lhs_str, rhs_str]):
                    part = regex_module.sub(r'(\d)([a-zA-Z])', r'\1*\2', part)
                    part = regex_module.sub(r'([a-zA-Z])(\d)', r'\1**\2', part)  
                    part = regex_module.sub(r'([a-zA-Z])\(', r'\1*(', part)     
                    part = regex_module.sub(r'(\d)\s*\(', r'\1*(', part)         
                    part = regex_module.sub(r'\)\s*\(', r')*(', part)            
                    part = regex_module.sub(r'\)\s*([a-zA-Z])', r')*\1', part)   
                    part = regex_module.sub(r'(\d)(,)', r'\1', part)             
                    
                    # Handle adjacent symbols
                    for sym1 in distinct_symbols:
                        for sym2 in distinct_symbols:
                            if sym1 != sym2:
                                pattern = f'({sym1})({sym2})'
                                replacement = f'\\1*\\2'
                                part = regex_module.sub(pattern, replacement, part)
                    
                    if i == 0:
                        lhs_str = part
                    else:
                        rhs_str = part
                
                try:
                    # Set up evaluation context
                    eval_context = dict(zip(distinct_symbols, symbol_objects))
                    
                    # Add math functions
                    import math as math_module
                    for name in dir(math_module):
                        if not name.startswith('_'):
                            eval_context[name] = getattr(math_module, name)
                    
                    lhs = eval(lhs_str.replace(' ', ''), {"builtins": {}}, eval_context)
                    rhs = eval(rhs_str.replace(' ', ''), {"builtins": {}}, eval_context)
                    
                    # Add equation to our list
                    eqs.append(Eq(lhs, rhs))
                    
                except Exception as e:
                    print(f"Error evaluating equation: '{equation}' - {str(e)}")
            else:
                print(f"Warning: Equation '{equation}' has more than one '=' sign.")
        else:
            print(f"Warning: Equation '{equation}' does not contain an '=' sign.")
    
    # Single variable quadratic equation special handling
    if len(eqs) == 1 and len(distinct_symbols) == 1:
        var = symbol_objects[0]
        expr = eqs[0].lhs - eqs[0].rhs
        expr = expand(expr)
        
        try:
            poly = Poly(expr, var)
            
            if poly.degree() == 2:
                coeffs = poly.all_coeffs()
                if len(coeffs) == 3:  
                    a, b, c = coeffs
                    
                    # Convert to numerical values
                    a_val = float(a)
                    b_val = float(b)
                    c_val = float(c)
                    
                    discriminant = b_val**2 - 4*a_val*c_val
                    
                    solution_values = {}
                    solution_text = "After solving this quadratic equation, we get the following solutions: \n"
                    
                    solution_text += f"\nStandard form: {a_val}{var}² + {b_val}{var} + {c_val} = 0"
                    
                    if discriminant < 0:
                        # Complex solutions
                        sqrt_disc = math.sqrt(abs(discriminant))
                        
                        x1_real = -b_val / (2*a_val)
                        x1_imag = sqrt_disc / (2*a_val)
                        
                        if abs(x1_real - round(x1_real)) < 1e-10:
                            x1_real = int(round(x1_real))
                        if abs(x1_imag - round(x1_imag)) < 1e-10:
                            x1_imag = int(round(x1_imag))
                            
                        solution_text += f"\nValues of {var} = {x1_real} + {x1_imag}i or {x1_real} - {x1_imag}i"
                        solution_values[str(var)] = f"{x1_real} + {x1_imag}i or {x1_real} - {x1_imag}i"
                    elif discriminant == 0:
                        x = -b_val / (2*a_val)
                        if abs(x - round(x)) < 1e-10:
                            x = int(round(x))
                        solution_text += f"\nValue of {var} = {x} (repeated root)"
                        solution_values[str(var)] = x
                    else:
                        sqrt_disc = math.sqrt(discriminant)
                        
                        x1 = (-b_val + sqrt_disc) / (2*a_val)
                        if abs(x1 - round(x1)) < 1e-10:  
                            x1 = int(round(x1))
                            
                        x2 = (-b_val - sqrt_disc) / (2*a_val)
                        if abs(x2 - round(x2)) < 1e-10:
                            x2 = int(round(x2))
                        
                        solution_text += f"\nValues of {var} = {x1} or {x2}"
                        solution_values[str(var)] = f"{x1} or {x2}"
                    
                    return solution_text, solution_values
        except Exception as e:
            print(f"Error in quadratic analysis: {str(e)}")
           
    if not eqs:
        return "No valid equations to solve.", {}
    
    # Solve the system of equations
    all_solutions = solve(eqs, symbol_objects, dict=True)
    
    # Process solutions
    solution_text = "After solving this system of equations, we get the following solutions: \n"
    solution_values = {}
    
    # Define helper functions inside main function
    def is_positive_value(val):
        """Check if a value is positive, handling complex numbers."""
        if hasattr(val, 'is_real') and not val.is_real:
            return False  # Complex values are not positive
        
        try:
            # For expressions like 360/y, we need to check if numerically positive for sample values
            if hasattr(val, 'free_symbols') and val.free_symbols:
                # Try with simple positive values
                from sympy import Symbol
                sample_val = 5
                test_vals = {}
                for sym in val.free_symbols:
                    test_vals[sym] = sample_val
                numeric_val = val.subs(test_vals)
                if hasattr(numeric_val, 'evalf'):
                    float_val = float(numeric_val.evalf())
                else:
                    float_val = float(numeric_val)
            else:
                # Convert to float only for comparison
                if hasattr(val, 'evalf'):
                    float_val = float(val.evalf())
                else:
                    float_val = float(val)
            return float_val > 0
        except (TypeError, ValueError):
            return False
    
    def format_value(val):
        """Format a value for display, handling complex numbers, fractions, and expressions."""
        # Check if val contains other variables
        if hasattr(val, 'free_symbols') and val.free_symbols:
            # It's an expression containing other variables
            return str(val)
        
        if hasattr(val, 'is_real') and not val.is_real:
            # Handle complex numbers
            real_part = sympy_re(val)
            imag_part = sympy_im(val)
            
            # Convert to float for rounding checks
            try:
                real_float = float(real_part.evalf())
                imag_float = float(imag_part.evalf())
                
                # Round if close to integers
                if abs(real_float - round(real_float)) < 1e-10:
                    real_part = int(round(real_float))
                if abs(imag_float - round(imag_float)) < 1e-10:
                    imag_part = int(round(imag_float))
                    
                if imag_float == 0:  # No imaginary part
                    return str(real_part)
                elif real_float == 0:  # Only imaginary part
                    return f"{imag_part}i"
                elif imag_float > 0:  # Both parts, positive imaginary
                    return f"{real_part} + {imag_part}i"
                else:  # Both parts, negative imaginary
                    return f"{real_part} - {abs(imag_part)}i"
            except Exception:
                # Fall back to string representation
                return str(val)
        
        # Handle real numbers and fractions
        try:
            # Try to convert to a float to check for near-integers
            if hasattr(val, 'evalf'):
                float_val = float(val.evalf())
            else:
                float_val = float(val)
                
            if abs(float_val - round(float_val)) < 1e-10:
                return str(int(round(float_val)))
            return str(float_val)
        except (TypeError, ValueError):
            # Could be a fraction or expression - return the string representation
            return str(val)
    
    # Special handling for simple systems like x - y = 4, x * y = 192
    if len(equations) == 2 and len(distinct_symbols) == 2 and isinstance(all_solutions, list):
        # Try to compute numerical values for specific cases
        try:
            # Check if we have equations of the form x - y = A, x * y = B
            eq1_str = equations[0].replace(" ", "")
            eq2_str = equations[1].replace(" ", "")
            
            diff_product_pattern = any([
                (f"{distinct_symbols[0]}-{distinct_symbols[1]}=" in eq1_str and f"{distinct_symbols[0]}*{distinct_symbols[1]}=" in eq2_str),
                (f"{distinct_symbols[1]}-{distinct_symbols[0]}=" in eq1_str and f"{distinct_symbols[0]}*{distinct_symbols[1]}=" in eq2_str),
                (f"{distinct_symbols[0]}-{distinct_symbols[1]}=" in eq2_str and f"{distinct_symbols[0]}*{distinct_symbols[1]}=" in eq1_str),
                (f"{distinct_symbols[1]}-{distinct_symbols[0]}=" in eq2_str and f"{distinct_symbols[0]}*{distinct_symbols[1]}=" in eq1_str)
            ])
            
            if diff_product_pattern and len(all_solutions) == 2:
                # We likely have a quadratic system with two solutions
                # Let's compute them explicitly
                numeric_solutions = []
                
                for sol in all_solutions:
                    if all(isinstance(sol[sym], (int, float)) or (hasattr(sol[sym], 'is_Number') and sol[sym].is_Number) for sym in symbol_objects):
                        numeric_sols = {}
                        for sym in symbol_objects:
                            if hasattr(sol[sym], 'evalf'):
                                val = float(sol[sym].evalf())
                            else:
                                val = float(sol[sym])
                            
                            if abs(val - round(val)) < 1e-10:
                                numeric_sols[str(sym)] = int(round(val))
                            else:
                                numeric_sols[str(sym)] = val
                        
                        numeric_solutions.append(numeric_sols)
                
                if numeric_solutions:
                    solution_text = "After solving this system of equations, we get the following solutions: \n"
                    
                    # Sort solutions by first variable value
                    numeric_solutions.sort(key=lambda x: x[str(symbol_objects[0])])
                    
                    for i, sol in enumerate(numeric_solutions):
                        solution_text += f"\nSolution {i+1}:\n"
                        for var, val in sol.items():
                            solution_text += f"Value of {var} = {val}\n"
                    
                    # Use the first solution as the primary one
                    solution_values = numeric_solutions[0]
                    return solution_text, solution_values
        except Exception as e:
            print(f"Error in special case handling: {str(e)}")
    
    # Handle the case where we have multiple solution sets
    if isinstance(all_solutions, list) and len(all_solutions) > 0:
        # Check if we have a pair of solutions where one is positive and one is negative
        positive_solutions = []
        
        for sol in all_solutions:
            try:
                if all(is_positive_value(val) for val in sol.values()):
                    positive_solutions.append(sol)
            except TypeError:
                # Skip if we can't determine if the value is positive (e.g., complex)
                continue
        
        # If we have positive solutions, use only those
        if positive_solutions:
            chosen_solution = positive_solutions[0]
            solution_text = "After solving this system of equations, the positive solution is: \n"
            for var, val in chosen_solution.items():
                formatted_val = format_value(val)
                solution_text += f"\nValue of {var} = {formatted_val}"
                solution_values[str(var)] = formatted_val
        else:
            # If no positive solutions, just use the first solution
            chosen_solution = all_solutions[0]
            for var, val in chosen_solution.items():
                formatted_val = format_value(val)
                solution_text += f"\nValue of {var} = {formatted_val}"
                solution_values[str(var)] = formatted_val
            
            if len(all_solutions) > 1:
                solution_text += f"\n\nNote: This system has {len(all_solutions)} solution sets, but none with all positive values."
    
    # Handle the case of a single solution dictionary
    elif isinstance(all_solutions, dict) and all_solutions:
        for var, val in all_solutions.items():
            formatted_val = format_value(val)
            solution_text += f"\nValue of {var} = {formatted_val}"
            solution_values[str(var)] = formatted_val
    
    # No solutions found
    else:
        solution_text += "\nNo solution found for this system of equations."
    
    print(solution_values)
    
    return solution_text, solution_values
    

# Your existing functions here...
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

#Extract variables from the equation
def extract_variables(equations):
    variable_set = set()
    for equation in equations:
        found_vars = re.findall(r'[a-zA-Z]', equation)
        variable_set.update(found_vars)
    return tuple(sorted(variable_set))

#Retrieving the final equations
def determine_final_equations(equations, variables):
    final_equations = []
    for equation in reversed(equations):
        if any(var in equation for var in variables):
            final_equations.append(equation)
        if len(final_equations) == len(variables):
            break
    return list(reversed(final_equations))


#Finding the most similar examples to the user's prompt
def find_most_similar_examples(input_text, examples, n=3):
        embeddings=OpenAIEmbeddings(api_key=api_key)
        db=Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
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

# Load textbook examples from CSV
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

def get_textbook_examples():
    global textbook_examples
    if textbook_examples is None:
        textbook_examples = load_examples_from_csv("dataset.csv")
    return textbook_examples

@app.route('/chat-started', methods=["POST"])
def chat_started():
    data = request.get_json()
    chat_id = data.get('chat_id')
    f = open("temp_file","r")
    user_id = int(f.read())
    f.close()
    if not chat_id:
        return {"error": "Chat ID missing"}, 400
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO problem_solutions (user_id,chat_id, problem_solution) VALUES (%s,%s,NULL)",
            (user_id,chat_id)  
        )
        print("record inserted")
        conn.commit()  
        cur.close()
        return {"message": "Chat started successfully", "chat_id": chat_id}, 201
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/get-all-chats', methods=["POST", "GET"])
def get_all_chats():
    print('In here')
    try:
        f = open("temp_file", "r")
        user_id = int(f.read())
        f.close()
        print(user_id)
        
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
            
        cur = conn.cursor()
        
        # Get chat timestamps with their chat_ids
        cur.execute(f"SELECT chat_id, created_at FROM problem_solutions WHERE user_id={user_id} ORDER BY chat_id")
        rows = cur.fetchall()
        
        # Create a dictionary mapping chat_id to timestamp
        timestamps_dict = {}
        for row in rows:
            chat_id, timestamp = row
            # Format timestamp as string if needed
            if timestamp:
                timestamps_dict[chat_id] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
        
        # Get max chat_id
        cur.execute(f"SELECT MAX(chat_id) FROM problem_solutions WHERE user_id={user_id}")
        max_chat_id = cur.fetchone()[0]
        
        return jsonify({
            "chats": max_chat_id,
            "timestamps": timestamps_dict
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-chat-history',methods=["POST","GET"])
def get_chat_history():
    data = request.get_json()
    chat_id = int(data.get("chat_id"))
    try:
        cur = conn.cursor()
        f = open("temp_file","r")
        user_id = int(f.read())
        f.close()
        cur.execute("SELECT problem_solution FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id,user_id))
        rows = cur.fetchall()
        problems = []
        solutions = []
        response = {"problem": problems, "solution": solutions}
        for row in rows:
            print(f"Row with chat id {chat_id}", row)
            if row[0] is None:
                return jsonify(response), 200  
            if isinstance(row[0], dict):
                problem_solution = row[0]  
            else:
                problem_solution = json.loads(row[0])  
            problems.append(problem_solution.get("problem"))
            solutions.append(problem_solution.get("solution"))
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/download',methods=["POST","GET"])
def download(filename="download_material.pdf"):
    print("Inside download")
    chat_id = int(request.args.get('chat_id'))
    f = open("temp_file","r")
    user_id = int(f.read())
    f.close()
    cur = conn.cursor()
    cur.execute("SELECT problem_solution FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id,user_id))
    rows = cur.fetchall()
    problems = []
    solutions = []
    response = {"problem": problems, "solution": solutions}
    for row in rows:
        if row[0] is None:
            response = {
                "error": "Invalid request",
                "message": "The provided parameters are incorrect",
                "status": 400
            }
            return jsonify(response), 400
        if isinstance(row[0], dict):
            problem_solution = row[0]  
        else:
            problem_solution = json.loads(row[0])  
        ni = []
        ni.append("Question: "+problem_solution.get("problem"))
        ni.append("Answer: "+problem_solution.get("solution"))
        problems.append(ni)
    response = rows[0]
    pdf_path = os.path.join(os.getcwd(), filename)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter  # Get page dimensions

    def draw_page_border():
        """Draw a border around the entire page."""
        c.setLineWidth(2)  # Border thickness
        margin = 20  # Margin from edge
        c.rect(margin, margin, width - 2 * margin, height - 2 * margin)

    def add_header(first_page=False):
        """Add title and date only on the first page if requested"""
        if first_page:
            # Add Title (Centered)
            title = "CalcMate Solutions"
            c.setFont("Helvetica", 16)  # Removed bold
            title_width = c.stringWidth(title, "Helvetica", 16)
            c.drawString((width - title_width) / 2, height - 40, title)
            
            # Add Space & Date (Left-Aligned)
            c.setFont("Helvetica", 12)
            current_date = datetime.datetime.now().strftime("%d.%m.%Y")
            c.drawString(50, height - 110, f"Downloaded on: {current_date}")
            return height - 140  # Start position after header
        else:
            return height - 50  # Start position with no header

    print(problems)
    # Render each problem on its own page
    is_first_page = True

    for problem in problems:
        # Start a new page for each problem
        draw_page_border()  # Add border to the page
        
        # Add header only to first page
        y_position = add_header(is_first_page)
        is_first_page = False
        
        # Initialize position for content
        c.setFont("Helvetica", 12)
        line_height = 20
        
        # Render the problem content
        for item in problem:
            # Clean HTML tags first
            pattern = r'<[^>]*>'
            cleaned_item = re.sub(pattern, '', item)
            
            # Handle Question part
            if "Question:" in item:
                # Extra spacing before questions
                y_position -= line_height
                c.setFont("Helvetica-Bold", 12)
                
                # Split question and content
                parts = cleaned_item.split("Question:", 1)
                question_content = parts[1].strip()
                
                # Draw question label
                c.drawString(100, y_position, "Question:")
                y_position -= line_height
                
                # Draw question content
                wrapped_lines = textwrap.wrap(question_content, width=70)
                for line in wrapped_lines:
                    c.drawString(120, y_position, line)
                    y_position -= line_height
                
                # Reset font
                c.setFont("Helvetica", 12)
                
            # Handle Answer part
            elif "Answer:" in item:
                # Extra spacing before answers
                y_position -= line_height * 1.5
                c.setFont("Helvetica-Bold", 12)
                
                # Draw answer label
                c.drawString(100, y_position, "Answer:")
                y_position -= line_height
                c.setFont("Helvetica", 12)
                
                # Get the content after "Answer:"
                answer_content = cleaned_item.replace("Answer:", "").strip()
                
                # Check if this is a structured math problem or a simple explanation
                if "Problem Steps:" in answer_content:
                    # This is a structured math problem with steps
                    
                    # Get the intro text (before Problem Steps)
                    intro_parts = answer_content.split("Problem Steps:", 1)
                    intro = intro_parts[0].strip()
                    
                    # Draw intro if it exists
                    if intro:
                        wrapped_lines = textwrap.wrap(intro, width=70)
                        for line in wrapped_lines:
                            c.drawString(120, y_position, line)
                            y_position -= line_height
                        y_position -= line_height * 0.5
                    
                    # Handle Problem Steps section
                    # Draw Problem Steps header
                    y_position -= line_height * 0.5
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(120, y_position, "Problem Steps:")
                    y_position -= line_height
                    c.setFont("Helvetica", 12)
                    
                    # Extract steps content
                    steps_content = ""
                    if "Equations:" in answer_content:
                        steps_parts = answer_content.split("Problem Steps:", 1)[1]
                        steps_content = steps_parts.split("Equations:", 1)[0].strip()
                    else:
                        steps_content = answer_content.split("Problem Steps:", 1)[1]
                    
                    # Process each step separately
                    step_pattern = r'Step\d+\.'
                    steps = re.split(step_pattern, steps_content)
                    step_numbers = re.findall(step_pattern, steps_content)
                    
                    # Skip empty first part if exists
                    if steps and not steps[0].strip():
                        steps = steps[1:]
                    
                    # Process each step
                    for i, step in enumerate(steps):
                        if i < len(step_numbers):
                            step_text = step_numbers[i] + step
                        else:
                            step_text = step
                            
                        # Split by newlines to preserve manual line breaks
                        step_lines = step_text.strip().split('\n')
                        for step_line in step_lines:
                            wrapped_lines = textwrap.wrap(step_line.strip(), width=65)
                            for j, line in enumerate(wrapped_lines):
                                # First line of first part gets the step number
                                if j == 0 and step_line == step_lines[0]:
                                    c.drawString(140, y_position, line)
                                else:
                                    # Subsequent lines are indented further
                                    c.drawString(150, y_position, line)
                                y_position -= line_height
                        
                        # Extra space after each step
                        y_position -= line_height * 0.3
                    
                    # Handle Equations section
                    if "Equations:" in answer_content:
                        # Draw Equations header
                        y_position -= line_height * 0.5
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(120, y_position, "Equations:")
                        y_position -= line_height
                        c.setFont("Helvetica", 12)
                        
                        # Extract equations content
                        equations_content = ""
                        if "Solution:" in answer_content:
                            eq_parts = answer_content.split("Equations:", 1)[1]
                            equations_content = eq_parts.split("Solution:", 1)[0].strip()
                        else:
                            equations_content = answer_content.split("Equations:", 1)[1]
                        
                        # Split potential multiple equations that are on the same line
                        equation_text = equations_content.strip()
                        
                        # Use regex to find equation patterns (a digit followed by colon)
                        # \d matches any digit, and we're only matching a single digit before the colon
                        pattern = r'(\d:)'
                        matches = re.finditer(pattern, equation_text)
                        start_positions = [m.start() for m in matches]
                        
                        equation_items = []
                        
                        # If we found equation numbering patterns
                        if start_positions:
                            # Add one more position for the end of the string
                            start_positions.append(len(equation_text))
                            
                            # Extract each equation using the positions
                            for i in range(len(start_positions) - 1):
                                start = start_positions[i]
                                end = start_positions[i+1]
                                equation_items.append(equation_text[start:end].strip())
                        else:
                            # If no patterns found, split by newlines
                            equation_items = equation_text.split('\n')
                        
                        # Display each equation
                        for eq in equation_items:
                            eq = eq.strip()
                            if eq:
                                # Just display the equation as is, with the single-digit number
                                c.drawString(140, y_position, eq)
                                y_position -= line_height
                                
                                # Extra space between equations
                                y_position -= line_height * 0.3
                    # Handle Solution section
                    if "Solution:" in answer_content:
                        # Draw Solution header
                        y_position -= line_height * 0.5
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(120, y_position, "Solution:")
                        y_position -= line_height
                        c.setFont("Helvetica", 12)
                        
                        # Extract solution content
                        solution_content = answer_content.split("Solution:", 1)[1].strip()
                        
                        # Process "The value of" statements separately
                        value_statements = solution_content.split('\n')
                        for statement in value_statements:
                            statement = statement.strip()
                            if statement:
                                wrapped_lines = textwrap.wrap(statement, width=65)
                                for line in wrapped_lines:
                                    c.drawString(140, y_position, line)
                                    y_position -= line_height
                                
                                # Add a small space between value statements
                                if "The value of" in statement:
                                    y_position -= line_height * 0.2
                
                else:
                    # This is a simple explanation answer (not a structured math problem)
                    # Format it as a regular paragraph with proper indentation
                    
                    # Split by newlines to preserve manual line breaks
                    paragraphs = answer_content.split('\n\n')
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            wrapped_lines = textwrap.wrap(paragraph.strip(), width=70)
                            for line in wrapped_lines:
                                c.drawString(120, y_position, line)
                                y_position -= line_height
                            
                            # Add space between paragraphs
                            y_position -= line_height * 0.5
            
            else:
                # For other content, use standard formatting
                wrapped_lines = textwrap.wrap(cleaned_item, width=75)
                for line in wrapped_lines:
                    # Check if we need a new page
                    if y_position < 50:
                        c.showPage()  # Create new page
                        draw_page_border()  # Add border to new page
                        y_position = height - 50  # Reset position at top of new page
                    
                    c.drawString(100, y_position, line)
                    y_position -= line_height
            
            # Check if we need a new page before the next item
            if y_position < 50:
                c.showPage()  # Create new page
                draw_page_border()  # Add border to new page
                y_position = height - 50  # Reset position at top of new page
        
        c.showPage()  # Finish this page and move to the next

    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route('/solve-llama',methods=["POST"])
def solve_llama():
    try:
        problem_dict = request.get_json()
        chat_id=problem_dict.get('chat_id')
        problem = problem_dict.get('problem')
        problem_in_list = [problem]
        new_vectorized = vectorizer.transform(problem_in_list)
        prediction = model.predict(new_vectorized)
        print(prediction)
        f = open("temp_file","r")
        user_id = int(f.read())
        f.close()
        if prediction[0] == 1:
            example_template = """Q. {problem} A. {reasoning}"""
            prefix = """ You are a Math tutor with a STRICT LIMITATION: you CANNOT solve equations.
                        Your role is to convert word problems into mathematical equations with clear steps.
                        For any math problem:
                            - Convert it ONLY to mathematical equations with numbered steps
                            - Identify variables clearly
                            - Explain the reasoning for forming each equation
                            - DO NOT solve the equations or show any numerical answers
                            - DO NOT perform any calculations
                            - STOP after stating the final equations

            If the user asks a follow-up question or has doubts, answer accordingly without solving equations.

                        """
            suffix = """
            When converting a word problem, follow EXACTLY these steps:

            1. Identify and clearly define all variables
            2. Set up equations based on the conditions in the problem
            3. Explain your reasoning for each equation
            4. Format all equations using Python syntax:
            - Use ** for powers (e.g., x**2 instead of x²)
            - Use * for multiplication (e.g., 3*x instead of 3x)
            - Use / for division
            5. End with 'Final Equations:' followed by the system of equations

            IMPORTANT: DO NOT SOLVE THE EQUATIONS. Your task is ONLY to set them up.
            DO NOT provide numerical values for any variables.
            DO NOT show any steps involving quadratic formula, substitution, or algebraic manipulation.
            DO NOT include any statements like "The answer is" or "Therefore, x equals".

            Q. {wordproblem}
            """
            few_shot_prompt = FewShotPromptTemplate(
                        examples=[],
                        example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template),
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["wordproblem"],
            )
            csv_file_path = "dataset.csv"
            textbook_examples = get_textbook_examples()
            similar_examples = find_most_similar_examples(problem, textbook_examples)
            print(similar_examples)
            few_shot_prompt.examples = similar_examples
            llm_chain = LLMChain(llm=llm_ollama, prompt=few_shot_prompt)
            output = llm_chain.run(problem)
            if "Final Equations:" in output:
                equations = output.split('Final Equations:')[-1].strip()
                steps = output.split('Final Equations:')[0].strip()
            elif "Final Equation:" in output:
                equations = output.split('Final Equation:')[-1].strip()
                steps = output.split('Final Equation:')[0]
            else:
                equations = ""
                steps = output.strip()
            variable_labels = {}
    
            step1_match = re.search(r'Step1\.\s*(.*?)(?:Step\d+\.|$)', steps, re.DOTALL)
            
            if step1_match:
                full_step1 = step1_match.group(1).strip()
                
                # Process grouped variables from Step1
                grouped_vars = re.findall(r"Let\s+([a-zA-Z_]\w*(?:\s*(?:,|,?\s+and)\s+[a-zA-Z_]\w*)+)\s+be(?:\s+the)?\s+(\w+)\s+(\w+)s?(?:,|\.|$)", full_step1)
                
                if grouped_vars:  # Only process if we found matches
                    for group_match in grouped_vars:
                        vars_str = group_match[0]
                        quantity_word = group_match[1]
                        noun = group_match[2]
                
                        if noun[-1] == 's':
                            noun = noun[:len(noun)-1]
                        
                        # Extract individual variables from the group, handling both commas and "and"
                        vars_in_group = re.findall(r'([a-zA-Z_]\w*)', vars_str)
                        vars_in_group = [var for var in vars_in_group if var.lower() not in ["and", "or"]]
                        
                        # Map number words to their ordinal forms
                        number_words = {
                            "two": ["first", "second"],
                            "three": ["first", "second", "third"],
                            "four": ["first", "second", "third", "fourth"],
                            "five": ["first", "second", "third", "fourth", "fifth"],
                            "six": ["first", "second", "third", "fourth", "fifth", "sixth"]
                        }
                        
                        ordinals = number_words.get(quantity_word.lower())
                        
                        # If we have ordinals matching the quantity, use them
                        if ordinals and len(vars_in_group) <= len(ordinals):
                            for i, var in enumerate(vars_in_group):
                                variable_labels[var] = f"{ordinals[i]} {noun}"
                        else:
                            # Fallback: Use positional numbering
                            for i, var in enumerate(vars_in_group):
                                position = i + 1
                                suffix = "th"
                                if position == 1:
                                    suffix = "st"
                                elif position == 2:
                                    suffix = "nd"
                                elif position == 3:
                                    suffix = "rd"
                                variable_labels[var] = f"{position}{suffix} {noun}"
                
                # Expanded pattern to include both "be" and "represent"
                matches = re.findall(
                    r"(?:Let )?(\w+) (?:be|represent) (?:the )?(.+?)(?:,| and|\.|$)|" + 
                    r"(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", 
                    full_step1
                )
                
                for match in matches:
                    if match[0] and re.match(r'^[a-zA-Z_]\w*$', match[0]):
                        var = match[0]
                        desc = match[1]
                        if var not in variable_labels:
                            variable_labels[var] = desc.strip()
                    elif match[2] and re.match(r'^[a-zA-Z_]\w*$', match[2]):
                        var = match[2]
                        desc = match[3]
                        if var not in variable_labels:
                            variable_labels[var] = desc.strip()
            
            # Fallback method if Step1 parsing fails
            if not variable_labels:
                try:
                    matches = re.findall(
                        r"(?:Let )?(\w+) (?:be|represent) (?:the )?(.+?)(?:,| and|\.|$)|" + 
                        r"(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", 
                        steps
                    )
                    flag=False
                    for match in matches:
                        if match[0]:
                            flag = True
                    if not flag:
                        raise Exception("nothing in matches")
                except:
                    matches = re.findall(r"(?:Let\s+)?(\w+)\s*=\s*([^.,]+?)(?:,| and|\.|$)", steps)
                
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
            valid_equations = [i.replace("^","**") for i in valid_equations]
            distinct_symbols = tuple(set(symbols_found))
            if valid_equations:
                solution_text, solution_values = solve_equation(valid_equations, distinct_symbols)
            else:
                solution_text = "No valid equations found."
            formatted_steps = standardize_output(steps)
            new_formatted_steps=f"<div>{formatted_steps}</div>"
            formatted_equations = "<br>".join([f"{i+1}: {eq}" for i, eq in enumerate(valid_equations)])
            # Add this check before creating the solution text
            solution_text = "<br>".join(
                [f"The value of {var} ( {variable_labels[var]} ) is {solution_values.get(var, 'unknown')}." 
                for var in variable_labels 
                # Only include if it's an actual variable in our equations
                if var in distinct_symbols]
            )

            print(solution_text)
            output = output + solution_text
            memory.save_context({"wordproblem": problem}, {"output": output})
            print("---------------------------")
            print(output)
            print(distinct_symbols) 
            print(variable_labels)
            print(solution_text)
            print("-------------------------------")
            final_solution = (
                "Here is a solution to your problem..<br><br>"
                f"<u>Problem Steps:</u><br>{new_formatted_steps}<br><br>"
                f"<u>Equations:</u><br>{formatted_equations}<br><br>"
                f"<u>Solution:</u><br>{solution_text}"
            )
            cur=conn.cursor()
            problem_solution=json.dumps({'problem':problem,'solution':final_solution})
            cur.execute("SELECT * FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
            row = cur.fetchone()
            if row[2] is None:
                cur.execute("UPDATE problem_solutions SET problem_solution = %s WHERE chat_id = %s AND user_id = %s", 
                            (problem_solution, chat_id, user_id))
                print("Record updated")
            else:
                cur.execute("INSERT INTO problem_solutions (user_id, chat_id, problem_solution) VALUES (%s, %s, %s)", 
                            (user_id, chat_id, problem_solution))
                print("Record inserted")
 
            conn.commit()
            return jsonify({"solution": final_solution})
        else:
            chatHistory = memory.chat_memory
            preffix = """
                        You are a Math tutor named CalcMate and answer the follow up questions for the problem that you have solved 
                        in the chat history provided. 
                    """
            suffix = """
                        Chat History: {chat_history}
                        {wordproblem}
                        Keep the answers to the doubts short and crisp.
                    """
            temp = preffix + suffix
            prompt_template = PromptTemplate(
                    template= temp,
                    input_variables=["wordproblem"]
                )
            llm_chain = LLMChain(llm=openai_llm, prompt=prompt_template, memory = memory)
            llm_chain.prompt = prompt_template
            output = llm_chain.run({'wordproblem':problem})
            output = output.replace("Human:","")
            output = output.replace("AI:","")
            output = output.replace("Ai:","")
            memory.save_context({"wordproblem": problem}, {"output": output})
            cur=conn.cursor()
            cur.execute("SELECT * FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
            row = cur.fetchone()
            problem_solution=json.dumps({'problem':problem,'solution':output})
            if row[2] is None:
                cur.execute("UPDATE problem_solutions SET problem_solution = %s WHERE chat_id = %s AND user_id = %s", 
                            (problem_solution, chat_id, user_id))
                print("Record updated")
            else:
                cur.execute("INSERT INTO problem_solutions (user_id, chat_id, problem_solution) VALUES (%s, %s, %s)", 
                            (user_id, chat_id, problem_solution))
                print("Record inserted")

            conn.commit()
            return jsonify({"solution": output})
        
          
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"solution": "Error in solving problem"})
@app.route('/solve-gpt', methods=["POST"])
def solve_gpt():
    try:
        problem_dict = request.get_json()
        chat_id=problem_dict.get('chat_id')
        problem = problem_dict.get('problem')
        problem_in_list = [problem]
        new_vectorized = vectorizer.transform(problem_in_list)
        prediction = model.predict(new_vectorized)
        print(prediction)
        f = open("temp_file","r")
        user_id = int(f.read())
        f.close()
        if prediction[0] == 1:
            example_template = """Q. {problem} A. {reasoning}"""
            prefix = """ You are a Math tutor with a STRICT LIMITATION: you CANNOT solve equations.
                        Your role is to convert word problems into mathematical equations with clear steps.
                        For any math problem:
                            - Convert it ONLY to mathematical equations with numbered steps
                            - Identify variables clearly
                            - Explain the reasoning for forming each equation
                            - DO NOT solve the equations or show any numerical answers
                            - DO NOT perform any calculations
                            - STOP after stating the final equations

            If the user asks a follow-up question or has doubts, answer accordingly without solving equations.

                        """
            suffix = """
            When converting a word problem, follow EXACTLY these steps:

            1. Identify and clearly define all variables
            2. Set up equations based on the conditions in the problem
            3. Explain your reasoning for each equation
            4. Format all equations using Python syntax:
            - Use ** for powers (e.g., x**2 instead of x²)
            - Use * for multiplication (e.g., 3*x instead of 3x)
            - Use / for division
            5. End with 'Final Equations:' followed by the system of equations

            IMPORTANT: DO NOT SOLVE THE EQUATIONS. Your task is ONLY to set them up.
            DO NOT provide numerical values for any variables.
            DO NOT show any steps involving quadratic formula, substitution, or algebraic manipulation.
            DO NOT include any statements like "The answer is" or "Therefore, x equals".

            Q. {wordproblem}
            """
            few_shot_prompt = FewShotPromptTemplate(
                        examples=[],
                        example_prompt=PromptTemplate(input_variables=["problem", "reasoning"], template=example_template),
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["wordproblem"],
            )
            csv_file_path = "dataset.csv"
            textbook_examples = get_textbook_examples()
            similar_examples = find_most_similar_examples(problem, textbook_examples)
            print(similar_examples)
            few_shot_prompt.examples = similar_examples
            llm_chain = LLMChain(llm=openai_llm, prompt=few_shot_prompt)
            output = llm_chain.run(problem)
            if "Final Equations:" in output:
                equations = output.split('Final Equations:')[-1].strip()
                steps = output.split('Final Equations:')[0].strip()
            elif "Final Equation:" in output:
                equations = output.split('Final Equation:')[-1].strip()
                steps = output.split('Final Equation:')[0]
            else:
                equations = ""
                steps = output.strip()
            variable_labels = {}
    
            step1_match = re.search(r'Step1\.\s*(.*?)(?:Step\d+\.|$)', steps, re.DOTALL)
            
            if step1_match:
                full_step1 = step1_match.group(1).strip()
                
                # Process grouped variables from Step1
                grouped_vars = re.findall(r"Let\s+([a-zA-Z_]\w*(?:\s*(?:,|,?\s+and)\s+[a-zA-Z_]\w*)+)\s+be(?:\s+the)?\s+(\w+)\s+(\w+)s?(?:,|\.|$)", full_step1)
                
                if grouped_vars:  # Only process if we found matches
                    for group_match in grouped_vars:
                        vars_str = group_match[0]
                        quantity_word = group_match[1]
                        noun = group_match[2]
                
                        if noun[-1] == 's':
                            noun = noun[:len(noun)-1]
                        
                        # Extract individual variables from the group, handling both commas and "and"
                        vars_in_group = re.findall(r'([a-zA-Z_]\w*)', vars_str)
                        vars_in_group = [var for var in vars_in_group if var.lower() not in ["and", "or"]]
                        
                        # Map number words to their ordinal forms
                        number_words = {
                            "two": ["first", "second"],
                            "three": ["first", "second", "third"],
                            "four": ["first", "second", "third", "fourth"],
                            "five": ["first", "second", "third", "fourth", "fifth"],
                            "six": ["first", "second", "third", "fourth", "fifth", "sixth"]
                        }
                        
                        ordinals = number_words.get(quantity_word.lower())
                        
                        # If we have ordinals matching the quantity, use them
                        if ordinals and len(vars_in_group) <= len(ordinals):
                            for i, var in enumerate(vars_in_group):
                                variable_labels[var] = f"{ordinals[i]} {noun}"
                        else:
                            # Fallback: Use positional numbering
                            for i, var in enumerate(vars_in_group):
                                position = i + 1
                                suffix = "th"
                                if position == 1:
                                    suffix = "st"
                                elif position == 2:
                                    suffix = "nd"
                                elif position == 3:
                                    suffix = "rd"
                                variable_labels[var] = f"{position}{suffix} {noun}"
                
                # Expanded pattern to include both "be" and "represent"
                matches = re.findall(
                    r"(?:Let )?(\w+) (?:be|represent) (?:the )?(.+?)(?:,| and|\.|$)|" + 
                    r"(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", 
                    full_step1
                )
                
                for match in matches:
                    if match[0] and re.match(r'^[a-zA-Z_]\w*$', match[0]):
                        var = match[0]
                        desc = match[1]
                        if var not in variable_labels:
                            variable_labels[var] = desc.strip()
                    elif match[2] and re.match(r'^[a-zA-Z_]\w*$', match[2]):
                        var = match[2]
                        desc = match[3]
                        if var not in variable_labels:
                            variable_labels[var] = desc.strip()
            
            # Fallback method if Step1 parsing fails
            if not variable_labels:
                try:
                    matches = re.findall(
                        r"(?:Let )?(\w+) (?:be|represent) (?:the )?(.+?)(?:,| and|\.|$)|" + 
                        r"(\w+)\s*=\s*(.+?)(?:,| and|\.|$)", 
                        steps
                    )
                    flag=False
                    for match in matches:
                        if match[0]:
                            flag = True
                    if not flag:
                        raise Exception("nothing in matches")
                except:
                    matches = re.findall(r"(?:Let\s+)?(\w+)\s*=\s*([^.,]+?)(?:,| and|\.|$)", steps)
                
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
            valid_equations = [i.replace("^","**") for i in valid_equations]
            distinct_symbols = tuple(set(symbols_found))
            if valid_equations:
                solution_text, solution_values = solve_equation(valid_equations, distinct_symbols)
            else:
                solution_text = "No valid equations found."
            formatted_steps = standardize_output(steps)
            new_formatted_steps=f"<div>{formatted_steps}</div>"
            formatted_equations = "<br>".join([f"{i+1}: {eq}" for i, eq in enumerate(valid_equations)])
            # Add this check before creating the solution text
            solution_text = "<br>".join(
                [f"The value of {var} ( {variable_labels[var]} ) is {solution_values.get(var, 'unknown')}." 
                for var in variable_labels 
                # Only include if it's an actual variable in our equations
                if var in distinct_symbols]
            )

            print(solution_text)
            output = output + solution_text
            memory.save_context({"wordproblem": problem}, {"output": output})
            print("---------------------------")
            print(output)
            print(distinct_symbols)
            print(variable_labels)
            print(solution_text)
            print("-------------------------------")
            final_solution = (
                "Here is a solution to your problem..<br><br>"
                f"<u>Problem Steps:</u><br>{new_formatted_steps}<br><br>"
                f"<u>Equations:</u><br>{formatted_equations}<br><br>"
                f"<u>Solution:</u><br>{solution_text}"
            )
            cur=conn.cursor()
            problem_solution=json.dumps({'problem':problem,'solution':final_solution})
            cur.execute("SELECT * FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
            row = cur.fetchone()
            if row[2] is None:
                cur.execute("UPDATE problem_solutions SET problem_solution = %s WHERE chat_id = %s AND user_id = %s", 
                            (problem_solution, chat_id, user_id))
                print("Record updated")
            else:
                cur.execute("INSERT INTO problem_solutions (user_id, chat_id, problem_solution) VALUES (%s, %s, %s)", 
                            (user_id, chat_id, problem_solution))
                print("Record inserted")
 
            conn.commit()
            return jsonify({"solution": final_solution})
        else:
            chatHistory = memory.chat_memory
            preffix = """
                        You are a Math tutor and answer the follow up questions for the problem that you have solved 
                        in the chat history provided. 
                    """
            suffix = """
                        Chat History: {chat_history}
                        {wordproblem}
                        Keep the answers to the doubts short and crisp.
                    """
            temp = preffix + suffix
            prompt_template = PromptTemplate(
                    template= temp,
                    input_variables=["wordproblem"]
                )
            llm_chain = LLMChain(llm=openai_llm, prompt=prompt_template, memory = memory)
            llm_chain.prompt = prompt_template
            output = llm_chain.run({'wordproblem':problem})
            output = output.replace("Human:","")
            output = output.replace("AI:","")
            output = output.replace("Ai:","")
            memory.save_context({"wordproblem": problem}, {"output": output})
            cur=conn.cursor()
            cur.execute("SELECT * FROM problem_solutions WHERE chat_id = %s AND user_id = %s", (chat_id, user_id))
            row = cur.fetchone()
            problem_solution=json.dumps({'problem':problem,'solution':output})
            if row[2] is None:
                cur.execute("UPDATE problem_solutions SET problem_solution = %s WHERE chat_id = %s AND user_id = %s", 
                            (problem_solution, chat_id, user_id))
                print("Record updated")
            else:
                cur.execute("INSERT INTO problem_solutions (user_id, chat_id, problem_solution) VALUES (%s, %s, %s)", 
                            (user_id, chat_id, problem_solution))
                print("Record inserted")

            conn.commit()
            return jsonify({"solution": output})
        
          
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"solution": "Error in solving problem"})
if __name__ == '__main__':
    app.run(debug=True, port=5000)