# PromptMed: Enhancing Clinical Text Summarization Using DSPy

## Overview
This project builds on the work from the [StanfordMIMI/clin-summ](https://github.com/StanfordMIMI/clin-summ) repository to evaluate various prompting techniques for clinical text summarization. The goal is to improve the performance of summarization techniques using DSPy and compare the results against those in the original paper.

## Project Goals
1. Experiment with Prompting: Test various prompting techniques (zero-shot, one-shot, few-shot, Chain of Thought, etc.) using DSPy.
2. Evaluate Performance: Use the same evaluation metrics as the original paper (BLEU, ROUGE, and BERTScore) to assess the quality of generated summaries.
3. Compare Results: Determine if DSPy improves performance and identify which prompting technique works best.

## Dataset
We use the dialogue dataset from the original `clin-summ` repository, specifically the test split for evaluation purposes. We use the train split only for examples in prompting setups that require examples.

## Pipeline
The project follows these steps:
1. Prompting using DSPy: Various prompting techniques are applied to generate summaries of doctor-patient dialogues.
2. Metrics Calculation: BLEU, ROUGE, and BERTScore are computed to evaluate the quality of the generated summaries.
3. Document Results: Results are stored and analyzed to determine whether DSPy outperforms existing techniques from the paper.

## How to Use
### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/AI-AmrIbrahim/PromptMed.git
    ```
2. Navigate to the project directory:
   ```bash
   cd PromptMed
   ```
3. Install the required dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```
### Using `summarization_eval.py`
A script to evaluate summarization models using various configurations (vanilla, Chain of Thought (CoT), or control prompts).

Features:
- Evaluate Summarization Models: Supports vanilla, cot (Chain of Thought), and control prompts.
- Metrics Calculation: Computes BLEU, ROUGE (ROUGE-1, ROUGE-2, ROUGE-L), and BERT F1 scores.
- Supports Few-shot Learning: Configurable with training data (few-shot mode).
- Flexible Temperature Setting: Run evaluations at specific temperatures (e.g., 0.1, 0.5, 0.9).
- CSV Export: Optionally save evaluation results to a CSV file.

Example Usage
```python
from scripts.summarization_eval import SummarizationEvaluator

# Initialize evaluator
evaluator = SummarizationEvaluator(model_name="openrouter/openai/gpt-4o-mini")

# Evaluate using dspy (Vanilla mode)
results_df = evaluator.evaluate_dspy(
    testset_examples=testset, 
    mode="vanilla", 
    temperature=0.1, 
    compilation_mode="zero-shot", 
    save_to_csv=True
)

# Evaluate using a control prompt
results_control = evaluator.evaluate_control_prompt(
    testset=testset, 
    temperature=0.5, 
    save_to_csv=True
)
```
### Using `update_metrics.py`
A utility to append or update metrics data for different models in metrics.json file. Currenly only supports metrics [BLEU, ROUGE, BERT scores] and temperatures [0.1, 0.5, 0.9].

Example Usage
```python
from scripts.update_metrics import update_metrics

# Define metrics to update
bleu_scores = [15.2, 16.4, 14.8]
rouge1_scores = [20.1, 21.0, 19.9]
rouge2_scores = [8.5, 9.2, 8.0]
rougeL_scores = [18.4, 19.0, 18.2]
bert_scores = [75.3, 76.1, 74.9]

# Update JSON metrics
update_metrics(
    file_path="metrics.json",
    model_name="Vanilla",
    temperature=0.1,
    bleu_scores=bleu_scores,
    rouge1_scores=rouge1_scores,
    rouge2_scores=rouge2_scores,
    rougeL_scores=rougeL_scores,
    bert_scores=bert_scores
)
```
## Acknowledgments
Groupmates:
- Solha Park (Stony Brook University Data Science PhD Candidate)
- David Zhao (Stony Brook University Data Science PhD Candidate)
