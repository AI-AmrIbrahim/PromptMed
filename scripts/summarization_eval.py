import evaluate
import pandas as pd
import numpy as np
import dspy
from bert_score import score as bert_score

class SummarizationEvaluator:
    def __init__(self, model_name='openrouter/openai/gpt-4o-mini'):
        self.model_name = model_name

    @staticmethod
    def compute_scores(predicted, target):
        """Compute BLEU, ROUGE, and BERT scores for a given prediction and target."""
        scores = {}

        # BLEU
        bleu = evaluate.load("sacrebleu")
        bleu.add(prediction=predicted, reference=[target])
        bleu_results = bleu.compute(smooth_method="floor", smooth_value=0)
        scores['BLEU'] = bleu_results['score']

        # ROUGE
        rouge = evaluate.load('rouge')
        scores_rouge = rouge.compute(predictions=[predicted], references=[target])
        scores['ROUGE-1'] = scores_rouge['rouge1'] * 100
        scores['ROUGE-2'] = scores_rouge['rouge2'] * 100
        scores['ROUGE-L'] = scores_rouge['rougeL'] * 100

        # BERT
        P, R, F1 = bert_score([predicted], [target], lang="en", rescale_with_baseline=True)
        scores['BERT'] = F1.mean().item() * 100

        return scores

    def evaluate_dspy(self, testset_examples, mode="vanilla", temperature=0.1, compilation_mode="zero-shot", k=None, trainset_dict=None, save_to_csv=False):
        print(f"Evaluating with temperature: {temperature}, mode: {mode}, and compilation mode: {compilation_mode}")

        # Set up language model
        lm = dspy.LM(self.model_name, temperature=temperature)
        dspy.configure(lm=lm)

        # Define the Summarization Program
        class Summarization(dspy.Module):
            def __init__(self, mode="vanilla"):
                if mode == "vanilla":
                    self.summarize = dspy.Predict("inputs: str -> summary: str", cache=False)
                elif mode == "cot":
                    self.summarize = dspy.ChainOfThought("inputs: str -> summary: str", cache=False)
                else:
                    raise ValueError("Invalid mode. Choose 'vanilla' or 'cot'.")

            def forward(self, inputs: str):
                result = self.summarize(inputs=inputs)
                return result.summary if hasattr(result, 'summary') else result

        program = Summarization(mode=mode)

        if compilation_mode == "few-shot":
            if k is None or trainset_dict is None:
                raise ValueError("Few-shot mode requires 'k' and 'trainset_dict' to be provided.")
            fewshot = dspy.LabeledFewShot(k=k)
            program = fewshot.compile(program, trainset=trainset_dict)

        # Initialize results
        results = []

        for example in testset_examples:
            inputs = example.inputs  # Original text (to summarize)
            target = example.target  # Ideal summary (reference)

            predicted = program(inputs)

            # Compute metrics
            scores = self.compute_scores(predicted, target)

            results.append({
                "Inputs": inputs,
                "Target": target,
                "Predicted": predicted,
                **scores
            })

            # Log progress
            print(f"Processed example with input: {inputs[:30]}... - BLEU: {scores['BLEU']:.2f}, ROUGE-L: {scores['ROUGE-L']:.2f}, BERT: {scores['BERT']:.2f}")

        # Store results in a DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV if requested
        if save_to_csv:
            filename = f"dspy_{mode}_{temperature}.csv"
            results_df.to_csv(filename, index=False)
            print(f"Metrics saved to {filename}.")

        return results_df

    def evaluate_control_prompt(self, testset, temperature, save_to_csv=False):
        print("Evaluating using control prompts.")

        # Set up language model
        lm = dspy.LM(self.model_name, temperature=temperature)
        dspy.configure(lm=lm)

        results = []

        print(f"Processing temperature: {temperature}")

        for idx, row in testset.iterrows():
            input_text = row['inputs']
            target_summary = row['target']

            # Define the prompt
            prompt = f"""You are an expert medical professional.
Summarize the patient/doctor dialogue into an assessment and plan.
input: {input_text}
summary:"""

            # Generate the summary
            predicted_summary = lm(prompt)[0]

            # Compute metrics
            scores = self.compute_scores(predicted_summary, target_summary)

            # Log progress
            print(f"Processed idx {idx + 1}/{len(testset)} at temp {temperature} - BLEU: {scores['BLEU']:.2f}, ROUGE-L: {scores['ROUGE-L']:.2f}, BERT: {scores['BERT']:.2f}")

            # Store results
            results.append({
                "Temperature": temperature,
                "Index": idx,
                "Prediction": predicted_summary,
                "Target": target_summary,
                **scores
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV if requested
        if save_to_csv:
            filename = f"control_{temperature}.csv"
            results_df.to_csv(filename, index=False)
            print(f"Metrics saved to {filename}.")

        return results_df
