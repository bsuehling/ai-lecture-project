from ailp.approaches import (
    EdgeGnnModel,
    NodeToVecModel,
    NoMLModel,
    PredictionModel,
    RuleBasedModel,
    RulesWithSimilarityModel,
)

MODEL_DICT: dict[str, PredictionModel] = {
    "edge_gnn": EdgeGnnModel,
    "node_to_vec": NodeToVecModel,
    "no_ml": NoMLModel,
    "rule_based": RuleBasedModel,
    "rules_with_similarity": RulesWithSimilarityModel,
}
