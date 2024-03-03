from ailp.approaches import (
    NodeToVecModel,
    NoMLModel,
    PredictionModel,
    RuleBasedModel,
    RulesWithSimilarityModel,
)

MODEL_DICT: dict[str, PredictionModel] = {
    "no_ml": NoMLModel,
    "rule_based": RuleBasedModel,
    "node_to_vec": NodeToVecModel,
    "rules_with_similarity": RulesWithSimilarityModel,
}
