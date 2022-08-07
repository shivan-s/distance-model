.PHONY: run
run: 
	pipenv run python main.py

.PHONY: srun
run: 
	pipenv run streamlit run main.py
