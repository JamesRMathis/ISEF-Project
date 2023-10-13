import random

code = ''

def generate_random_function():
    global code
    # Generate a random function name
    function_name = "func_" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))

    # Generate parameters (0-3 parameters)
    # num_parameters = random.randint(0, 3)
    # parameters = ', '.join([f'param_{i}' for i in range(num_parameters)])

    # Generate function body with random statements (0-5 statements)
    num_statements = random.randint(3, 20)
    code += f'def {function_name}():\n'
    for _ in range(num_statements):
        code += generate_random_statement() + '\n'

    return code


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
    # elif statement_type == "break":
    #     return generate_break_statement(indentation_level)

def generate_assignment_statement(indentation_level):
    vars = re.findall(r'var_\d*', code)

    new_var = random.choice([True, False])
    if new_var or len(vars) == 0:
        variable_name = f"var_{random.randint(1, 100)}"
        value = random.randint(1, 100)
        operator = '='
    else:
        variable_name = random.choice(vars)
        value = random.randint(-999999, 999999)
        operator = random.choice(["+=", "-=", "*=", "/="])

    indentation = "    " * indentation_level
    return f"{indentation}{variable_name} {operator} {value}"

def generate_conditional_statement(indentation_level):
    import re

    basicCondition = random.choices([True, False], weights=[1, 3])[0]
    if basicCondition:
        condition = random.choice(["True", "False"])
    else:
        vars = list(set(re.findall(r'var_\d*', code)))

        if len(vars) == 0:
            condition = generate_complex_condition()
        elif len(vars) == 1:
            condition = generate_complex_condition(left_operand=vars[0])
        elif len(vars) == 2:
            condition = generate_complex_condition(left_operand=vars[0], right_operand=vars[1])
        else:
            left_operand, right_operand = random.sample(vars, 2)
            condition = generate_complex_condition(left_operand=left_operand, right_operand=right_operand)
        
    indentation = "    " * indentation_level
    statements = [f'{indentation}if {condition}:']
    for _ in range(random.randint(1, 3)):
        statements.append(generate_random_statement(indentation_level + 1))
    return '\n'.join(statements)

def generate_loop_statement(indentation_level):
    indentation = "    " * indentation_level
    loop_type = random.choice(["for", "while"])
    should_break = random.choices([True, False], weights=[1, 3])[0]
    
    if loop_type == "for":
        loop_variable = f"i_{random.randint(1, 10)}"
        loop_range = random.randint(1, 10)
        statement = f"{indentation}for {loop_variable} in range({loop_range}):\n{generate_random_statement(indentation_level + 1)}"
        statement += f"\n{generate_break_statement(statement)}" if should_break else ""
        return statement
    else:
        basicCondition = random.choices([True, False], weights=[1, 3])[0]
        if basicCondition:
            condition = random.choice(["True", "False"])
        else:
            vars = list(set(re.findall(r'var_\d*', code)))
        
            if len(vars) == 0:
                condition = generate_complex_condition()
            elif len(vars) == 1:
                condition = generate_complex_condition(left_operand=vars[0])
            elif len(vars) == 2:
                condition = generate_complex_condition(left_operand=vars[0], right_operand=vars[1] if random.choices([0, 1], [3, 1]) else None)
            else:
                left_operand, right_operand = random.sample(vars, 2)
                condition = generate_complex_condition(left_operand=left_operand, right_operand=right_operand)

        statements = [f'{indentation}while {condition}:']
        for _ in range(random.randint(1, 3)):
            statements.append(generate_random_statement(indentation_level + 1))
        if should_break:
            statements.append(generate_break_statement('\n'.join(statements)))
        return '\n'.join(statements)
    
def generate_complex_condition(left_operand=None, right_operand=None, operator=None):
    # Generate a random mathematical expression as a condition
    left_operand = random.randint(-999999, 999999) if left_operand is None else left_operand
    right_operand = random.randint(-999999, 999999) if right_operand is None else right_operand
    operator = random.choice(["<", ">", "==", "!=", "<=", ">="]) if operator is None else operator
    return f"{left_operand} {operator} {right_operand}"

def generate_return_statement(indentation_level):
    return_value = random.randint(1, 100)
    indentation = "    " * indentation_level
    return f"{indentation}return {return_value}"

def generate_break_statement(function_code):
    # indentation = "    " * (indentation_level + 1)
    # return f"{indentation}break"
    # Split the function code into lines
    lines = function_code.splitlines()

    last_loop_indentation = 0
    # Iterate over the lines in reverse order to find the last for or while loop
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if 'for' in line or 'while' in line:
            # Return the indentation level of the last loop
            last_loop_indentation = (len(line) - len(line.lstrip())) // 4

    # Return None if no loops were found
    if last_loop_indentation == 0:
        return None

    indentation = "    " * (last_loop_indentation + 1)
    return f"{indentation}break"


def run_with_timeout(func, args=(), timeout=10):
    import threading
    
    # Define a function to run the target function with a timeout
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            self.result = func(*args)

    # Start the thread and wait for the timeout
    it = InterruptableThread()
    it.start()
    it.join(timeout)

    # Check if the thread is still alive
    if it.is_alive():
        return False

    # If the thread is not alive, return the result
    return True

for _ in range(1):
    import re
    import time

    random_function = generate_random_function()
    print(random_function)
    fn_name = re.findall(r'func_\w*', random_function)[0]
    start = time.perf_counter() * 1000
    exec(random_function)
    end = time.perf_counter() * 1000
    if run_with_timeout(eval(fn_name), timeout=3):
        print(f"Function ran successfully in {end - start} milliseconds")
        # with open('functions.txt', 'a') as file:
        #     file.write(random_function + '#1\n<sep>\n')
    else:
        print("Function timed out")
        with open('checking.txt', 'a') as file:
            file.write(random_function + '\n\n')