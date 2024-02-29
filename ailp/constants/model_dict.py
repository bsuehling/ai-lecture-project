from ailp.approaches import NoMLModel, PredictionModel, RuleBasedModel, NodeToVecModel, RulesWithSimilarityModel

MODEL_DICT: dict[str, PredictionModel] = {"no_ml": NoMLModel }
MODEL_DICT["rule_based"] = RuleBasedModel
MODEL_DICT["node_to_vec"] = NodeToVecModel
MODEL_DICT["rules_with_similarity"] = RulesWithSimilarityModel
