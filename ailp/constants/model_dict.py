from ailp.approaches import EdgeGnnModel, NoMLModel, PredictionModel

MODEL_DICT: dict[str, PredictionModel] = {
    "no_ml": NoMLModel,
    "edge_gnn": EdgeGnnModel,
}
