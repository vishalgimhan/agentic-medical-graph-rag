install: ## Install dependencies
	pip install -r requirements.txt

init-graph: ## Create Neo4j constraints and indexes
	python scripts/init_graph.py

seed-graph: ## Populate Neo4j with drug interaction data
	python scripts/seed_graph.py

clean-graph: ## Delete all graph data
	python scripts/init_graph.py --clean