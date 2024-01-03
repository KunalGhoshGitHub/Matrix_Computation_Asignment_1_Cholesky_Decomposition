# PLEASE RUN THE CODE INSIDE AN EMPTY FOLDER TO AVOID ANY CONFLICTS

# Code:
"Kunal_Ghosh_AE_291_Assignment_1.ipynb"
"Kunal_Ghosh_AE_291_Assignment_1.py"

# Input files:
"A.csv" (Coefficient Matrix)
"b.csv" (RHS of the system of equations)
"A_*.csv" (Examples)
"b_*.csv" (Examples)

# Output files:
"x.csv" (Solution of Ax = b)

# Latex file:
Kunal_Ghosh_AE_291_Assignment_1.tex

# Report:
Kunal_Ghosh_AE_291_Assignment_1.pdf

# To compile the code:
make 

# To execute the code:
make execute
(NOTE: May require root access)
(NOTE: It will try to open the Jupyter Notebook in your default browser. You have to press "Restart & Run all" in the "Kernel" tab after opening the Jupyter Notebook.)
(NOTE: "make execute" command may NOT work on all platforms. The code can be opened manually.)

# To open the code manually:
open a terminal
Navigate to the directory where the code and input files are kept
enter the following commands to the terminal:
jupyter-notebok
Then click on the file "Kunal_Ghosh_AE_291_Assignment_1.ipynb"

# To delete the generated files:
make clean
(CAUTION: It will remove the following files from the folder: A.csv b.csv x.csv x_NON_UNIQUE_SOLUTION.csv)

# Make file:
"MakeFile"

(NOTE: Coding is done in Jupyter-Notebook using Python 3.9.13)
