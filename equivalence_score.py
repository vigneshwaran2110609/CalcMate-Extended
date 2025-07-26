import re
from sympy import symbols, Eq, solve, sympify

def identify_variables(equations):
    variables = sorted(set(re.findall(r'[a-zA-Z]+', ' '.join(equations))))
    return variables

def create_variable_mapping(variables):
    if len(variables) == 1:
        standard_vars = ['x']
    elif len(variables) == 2:
        standard_vars = ['x', 'y']
    elif len(variables) == 3:
        standard_vars = ['x', 'y', 'z']
    else:
        return {}
    
    var_mapping = {}
    for original_var, standard_var in zip(variables, standard_vars):
        if original_var not in standard_vars:
            var_mapping[original_var] = standard_var
        else:
            var_mapping[original_var] = original_var
    return var_mapping

def preprocess_equation(equation, var_mapping):
    equation = re.sub(r'[^a-zA-Z0-9+\-*/=().^]', '', equation)
    
    for original_var, standard_var in var_mapping.items():
        equation = equation.replace(original_var, standard_var)

    # Ensure multiplication is handled
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    equation = re.sub(r'(\d)\s*\(', r'\1*(', equation)
    return equation

def generate_canonical_forms(equations, var_mapping):
    sympy_eqs = []
    for equation in equations:
        equation = preprocess_equation(equation, var_mapping)
        if '=' in equation:
            lhs, rhs = equation.split('=')
            sympy_eqs.append(Eq(sympify(lhs), sympify(rhs)))
    return sympy_eqs

def solve_equations(equations, var_mapping):
    standardized_vars = symbols(' '.join(var_mapping.values()))
    sympy_eqs = generate_canonical_forms(equations, var_mapping)
    
    if not sympy_eqs:
        return {}, sympy_eqs
    
    solution = solve(sympy_eqs, standardized_vars)
    solution_values = {}
    
    if isinstance(solution, dict):
        for var, value in solution.items():
            solution_values[str(var)] = value
    elif isinstance(solution, list):
        for sol in solution:
            for var, val in zip(standardized_vars, sol):
                solution_values[str(var)] = val
    else:
        return {}, sympy_eqs
    
    return solution_values, sympy_eqs

def compare_solutions(input_solution, expected_solution):
    """ Compares two solutions dictionaries to ensure they are equivalent. """
    if input_solution == expected_solution:
        return True
    
    # Allow for small numerical differences
    for var in input_solution:
        if var in expected_solution:
            if isinstance(input_solution[var], (int, float)) and isinstance(expected_solution[var], (int, float)):
                if abs(input_solution[var] - expected_solution[var]) > 1e-6:  # tolerance level for floats
                    return False
            elif input_solution[var] != expected_solution[var]:
                return False
        else:
            return False
    return True

def calculate_equivalence_score(input_text, expected_text):
    try:
        input_eqns = [eq.strip() for eq in re.split(r',|and', input_text)]
        expected_eqns = [eq.strip() for eq in re.split(r',|and', expected_text)]
        
        input_eqns = sorted(input_eqns)
        expected_eqns = sorted(expected_eqns)

        input_vars = identify_variables(input_eqns)
        expected_vars = identify_variables(expected_eqns)

        input_var_mapping = create_variable_mapping(input_vars)
        expected_var_mapping = create_variable_mapping(expected_vars)

        print(input_var_mapping,expected_var_mapping)

        input_solution, _ = solve_equations(input_eqns, input_var_mapping)
        expected_solution, _ = solve_equations(expected_eqns, expected_var_mapping)

        print(input_solution)

        if compare_solutions(input_solution, expected_solution):
            return 100
        else:
            return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0 


