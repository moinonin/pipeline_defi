import re
import ast, sys
import numpy as np

# Specify the path to your text file
file_path = 'results_file.txt'

# Read the entire content of the file
with open(file_path, 'r') as file:
    content = file.read()
    #print(content)
    

# Use a regular expression to find all array-like strings
# This pattern looks for text starting with '[[' and ending with ']]'
array_strings = re.findall(r'\[\[.*?\]\]', content, re.DOTALL)

best_matrix = None
best_center_value = None

# Iterate over all found array strings
for array_str in array_strings:
    try:
        # Safely convert the string to a Python list (of lists)
        matrix_list = ast.literal_eval(array_str)
        matrix = np.array(matrix_list)
        # Check if the matrix is at least 3x3 to have a middle element
        if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 3:
            continue
        
        # Determine the indices for the middle row and column
        mid_row = matrix.shape[0] // 2
        mid_col = matrix.shape[1] // 2
        
        center_value = matrix[mid_row, mid_col]

        # Update best_matrix if this center value is the largest so far
        if best_center_value is None or center_value > best_center_value:
            best_center_value = center_value
            best_matrix = matrix

            
    except Exception as e:
        # Handle any conversion errors gracefully
        print(f"Error parsing matrix: {e}")

if best_matrix is not None:
    print("Matrix with the largest center value:")
    print(best_matrix)
    print(f"Center value: {best_center_value}")
else:
    print("No valid 3x3 (or larger) confusion matrices found.")
