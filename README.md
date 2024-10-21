# **PromptMed: Enhancing Clinical Text Summarization Using DSPy**

### **Overview**
This project builds on the work from the [StanfordMIMI/clin-summ](https://github.com/StanfordMIMI/clin-summ) repository to evaluate various **prompting techniques** for clinical text summarization. The goal is to improve the performance of summarization techniques using **DSPy** and compare the results against those in the original paper.

### **Project Goals**
1. **Experiment with Prompting**: Test various prompting techniques (zero-shot, one-shot, few-shot, Chain of Thought, etc.) using DSPy.
2. **Evaluate Performance**: Use the same evaluation metrics as the original paper (BLEU, ROUGE, BERTScore, and MEDCON) to assess the quality of generated summaries.
3. **Compare Results**: Determine if DSPy improves performance and identify which prompting technique works best.

### **Dataset**
We use the **dialogue dataset** from the original `clin-summ` repository, specifically the **test split** for evaluation purposes. We use the **train split** only for examples in prompting setups that require examples.

### **Pipeline**
The project follows these steps:
1. **Prompting using DSPy**: Various prompting techniques are applied to generate summaries of doctor-patient dialogues.
2. **Metrics Calculation**: BLEU, ROUGE, BERTScore, and MEDCON are computed to evaluate the quality of the generated summaries.
3. **Document Results**: Results are stored and analyzed to determine whether DSPy outperforms existing techniques from the paper.

### **Installation**
1. Clone the repository:
    ```bash
    git clone https://github.com/AI-AmrIbrahim/PromptMed.git
    cd PromptMed
    ```

2. (Optional) Obtain a UMLS license for using the **UMLSScorer**:
    - Follow the instructions in the [UMLSScorer setup](https://github.com/StanfordMIMI/clin-summ/blob/main/README.md#umls-scorer).
