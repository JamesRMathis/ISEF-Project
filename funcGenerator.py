import random

def generate_random_function():
    # Generate a random function name
    function_name = "func_" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))

    # Generate parameters (0-3 parameters)
    num_parameters = random.randint(0, 3)
    parameters = ', '.join([f'param_{i}' for i in range(num_parameters)])

    # Combine the function definition with parameters
    function_definition = f'def {function_name}({parameters}):\n'

    # Generate function body with random statements (0-5 statements)
    num_statements = random.randint(3, 20)
    statements = '\n'.join([generate_random_statement() for _ in range(num_statements)])

    # Combine everything to form the complete function
    function_code = function_definition + f'{statements}\n'

    return function_code


def generate_random_statement(indentation_level=1):
    # Choose a random statement type
    statement_type = random.choice(["assignment", "conditional", "loop", "return"])

    if statement_type == "assignment":
        return generate_assignment_statement(indentation_level)
    elif statement_type == "conditional":
        return generate_conditional_statement(indentation_level)
    elif statement_type == "loop":
        return generate_loop_statement(indentation_level)
    elif statement_type == "return":
        return generate_return_statement(indentation_level)

def generate_assignment_statement(indentation_level):
    variable_name = f"var_{random.randint(1, 100)}"
    value = random.randint(1, 100)
    indentation = "    " * indentation_level
    return f"{indentation}{variable_name} = {value}"

def generate_conditional_statement(indentation_level):
    boolean = random.randint(0, 1)
    if boolean:
        condition = random.choice(["True", "False"])
    else:
        condition = generate_complex_condition()
    indentation = "    " * indentation_level
    return f"{indentation}if {condition}:\n{generate_random_statement(indentation_level + 1)}"

def generate_loop_statement(indentation_level):
    indentation = "    " * indentation_level
    loop_type = random.choice(["for", "while"])
    
    if loop_type == "for":
        loop_variable = f"i_{random.randint(1, 10)}"
        loop_range = random.randint(1, 10)
        return f"{indentation}for {loop_variable} in range({loop_range}):\n{generate_random_statement(indentation_level + 1)}"
    else:
        boolean = random.randint(0, 1)
        if boolean:
            condition = random.choice(["True", "False"])
        else:
            condition = generate_complex_condition()
        return f"{indentation}while {condition}:\n{generate_random_statement(indentation_level + 1)}"
    
def generate_complex_condition():
    # Generate a random mathematical expression as a condition
    left_operand = random.randint(-999999, 999999)
    right_operand = random.randint(-999999, 999999)
    operator = random.choice(["<", ">", "==", "!=", "<=", ">="])
    return f"{left_operand} {operator} {right_operand}"

def generate_return_statement(indentation_level):
    return_value = random.randint(1, 100)
    indentation = "    " * indentation_level
    return f"{indentation}return {return_value}"

# Example usage:
random_function = generate_random_function()
print(random_function)
