import random

def generate_random_function():
    # Generate a random function name
    function_name = "func_" + ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))

    # Generate parameters (0-3 parameters)
    # num_parameters = random.randint(0, 3)
    # parameters = ', '.join([f'param_{i}' for i in range(num_parameters)])

    # Combine the function definition with parameters
    function_definition = f'def {function_name}():\n'

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



for _ in range(1):
    import re
    import importlib

    random_function = generate_random_function()
    fn_name = re.findall(r'func_\w*', random_function)[0]
    with open(f"./checking.py", "w") as file:
        file.write(random_function)

    module = importlib.import_module("checking")
    function = getattr(module, fn_name)
    print(function)







# def run_with_timeout(func, args=(), timeout=10):
#     import threading
    
#     # Define a function to run the target function with a timeout
#     class InterruptableThread(threading.Thread):
#         def __init__(self):
#             threading.Thread.__init__(self)
#             self.result = None

#         def run(self):
#             self.result = func(*args)

#     # Start the thread and wait for the timeout
#     it = InterruptableThread()
#     it.start()
#     it.join(timeout)

#     # Check if the thread is still alive
#     if it.is_alive():
#         # If it is, interrupt it and raise an exception
#         it._Thread__stop()
#         return False

#     # If the thread is not alive, return the result
#     return True

# import threading



# x = 0
# while x < 1:
#     function_code = generate_random_function()

#     # Split the function code into a list of lines
#     lines = function_code.splitlines()

#     # Remove the first line
#     lines = lines[1:]

#     # Remove a tab from each line after the first
#     for i in range(len(lines)):
#         lines[i] = lines[i].replace('    ', '', 1)

#         import re
#         lines[i] = re.sub(r'return \d*', 'exit()', lines[i])

#     # Join the modified lines back into a single string
#     modified_code = '\n'.join(lines)

#     class InterruptableThread(threading.Thread):
#         def __init__(self):
#             threading.Thread.__init__(self)
#             self.result = None

#         def run(self):
#             self.result = eval(modified_code)

#     # Start the thread and wait for the timeout
#     it = InterruptableThread()
#     it.start()
#     it.join(10)

#     # Check if the thread is still alive
#     if it.is_alive():
#         # If it is, interrupt it and raise an exception
#         it._Thread__stop()
        
#         with open('check.txt', 'a') as file:
#             file.write(function_code + '\n')
#         print('might not halt')
#     else:
#         with open('functions.txt', 'a') as file:
#             file.write(function_code + '\n<sep>\n')
#         print('halts')


#     x += 1


# # with open("functions.txt", "a") as file:
# #     file.write(random_function)
