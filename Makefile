
install:
	conda env create -f environment.yml
	conda activate ai-project-311
update:
	conda env update -f environment.yml

train-rule-based:
	APPROACH=rule_based STAGE=train python ./evaluation.py
test-rule-based:
	APPROACH=rule_based STAGE=test python ./evaluation.py

train-node-to-vec:
	APPROACH=node_to_vec STAGE=train python ./evaluation.py
test-node-to-vec:
	APPROACH=node_to_vec STAGE=test python ./evaluation.py

train-rules-with-similarity:
	APPROACH=rules_with_similarity STAGE=train python ./evaluation.py
test-rules-with-similarity:
	APPROACH=rules_with_similarity STAGE=test python ./evaluation.py

train-no-ml:
	APPROACH=no_ml STAGE=train python ./evaluation.py
test-no-ml:
	APPROACH=no_ml STAGE=test python ./evaluation.py

train-edge-gnn:
	APPROACH=edge_gnn STAGE=train python ./evaluation.py
test-dege-gnn:
	APPROACH=edge_gnn STAGE=test python ./evaluation.py
