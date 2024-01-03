.PHONY: start execute clean

all: start

code = Kunal_Ghosh_AE_291_Assignment_1.ipynb

start: 
	@echo "We are running the make file.... \n"

execute: ${code}
	@echo "We are opening the Jupyter Notebook in your default web browser.... \n"
	jupyter-notebook ${code}
	
clean:
	@echo "Removing the generated csv files.... \n"
	rm A.csv b.csv x.csv x_NON_UNIQUE_SOLUTION.csv
