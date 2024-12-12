import json
from typing import List

def update_metrics(
    file_path: str,
    model_name: str,
    temperature: float,
    bleu_scores: List[float],
    rouge1_scores: List[float],
    rouge2_scores: List[float],
    rougeL_scores: List[float],
    bert_scores: List[float]
):
    """
    Update metrics in a JSON file for a given model and temperature with new scores.
    
    Parameters:
    - file_path (str): Path to the JSON file.
    - model_name (str): Name of the model (e.g., "Vanilla", "CoT", "Fewshot").
    - temperature (float): Temperature value (must be 0.1, 0.5, or 0.9).
    - bleu_scores (List[float]): List of BLEU scores to add.
    - rouge1_scores (List[float]): List of ROUGE-1 scores to add.
    - rouge2_scores (List[float]): List of ROUGE-2 scores to add.
    - rougeL_scores (List[float]): List of ROUGE-L scores to add.
    - bert_scores (List[float]): List of BERT scores to add.
    
    Raises:
    - ValueError: If the temperature is not 0.1, 0.5, or 0.9.
    """
    valid_temperatures = {0.1, 0.5, 0.9}
    
    if temperature not in valid_temperatures:
        raise ValueError(f"Temperature {temperature} is not valid. Must be one of {valid_temperatures}.")
    
    # Load or initialize the metrics dictionary
    try:
        with open(file_path, "r") as file:
            metrics_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        metrics_data = {}

    # Ensure the model name exists
    if model_name not in metrics_data:
        metrics_data[model_name] = {}

    # Ensure the temperature key exists for the model
    if str(temperature) not in metrics_data[model_name]:
        metrics_data[model_name][str(temperature)] = {
            "BLEU": [],
            "ROUGE-1": [],
            "ROUGE-2": [],
            "ROUGE-L": [],
            "BERT": []
        }

    # Append the new scores to the respective lists
    metrics_data[model_name][str(temperature)]["BLEU"].extend(bleu_scores)
    metrics_data[model_name][str(temperature)]["ROUGE-1"].extend(rouge1_scores)
    metrics_data[model_name][str(temperature)]["ROUGE-2"].extend(rouge2_scores)
    metrics_data[model_name][str(temperature)]["ROUGE-L"].extend(rougeL_scores)
    metrics_data[model_name][str(temperature)]["BERT"].extend(bert_scores)

    # Save back the updated structure
    with open(file_path, "w") as file:
        json.dump(metrics_data, file, indent=4)

    print(f"Updated metrics for model '{model_name}' at temperature {temperature} in {file_path}.")
