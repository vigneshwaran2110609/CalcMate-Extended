from fuzzywuzzy import fuzz

def calculate_fuzzy_score(input_text, expected_equation):
    score = fuzz.token_set_ratio(input_text, expected_equation)
    return score

input_text='2x+3y=12,2x-3y=0'
expected_equation='x+y=5,x-y=1'

print(calculate_fuzzy_score(input_text,expected_equation))
