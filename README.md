# GPT-2 Small Indirect Prompt Injection Analysis

An exploration into the mechanics of indirect prompt injection in `gpt2-small`, using activation patching and attention analysis to understand model behavior.

## Project Goal

The primary goal of this project was to investigate the internal mechanics of an indirect prompt injection attack on the `gpt2-small` language model. The aim was to move beyond simply observing the vulnerability and instead use mechanistic interpretability techniques to identify the specific components (the "circuit") responsible for causing the model to disregard its initial instructions and follow a malicious one embedded in its input.

## Methodology

*   **Model:** `gpt2-small`
*   **Core Library:** `transformer_lens`
*   **Key Techniques:**
    *   **Attention Pattern Analysis:** Observing where the model "looks" when processing a prompt to find correlations between attention and behavior.
    *   **Activation Patching (Causal Intervention):** Surgically intervening in the model's computation by replacing specific activation values from a "clean" run with those from a "malicious" run to prove causation.

## File Descriptions

*   `injection_test.py`: Confirms the baseline vulnerability of `gpt2-small` to a basic indirect prompt injection attack.
*   `analyze_attention.py`: Analyzes attention patterns in the final layer to identify heads that focus on the malicious instruction.
*   `patching_test.py`: Tests the causal role of a single attention head (L11H4) via activation patching.
*   `patching_multiple_heads.py`: Expands the patching experiment to a group of suspected heads in the final layer.
*   `patching_residual_stream.py`: Performs a more powerful intervention by patching the entire residual stream output of the final layer.
*   `project_summary.md`: A detailed report summarizing the project's objectives, experimental phases, and overall conclusions.

## Setup

To run these experiments, you need to install the required Python libraries:

```bash
pip install torch transformers transformer_lens
```

## How to Run

Each Python script is self-contained and can be run directly from the command line:

```bash
python injection_test.py
python analyze_attention.py
python patching_test.py
python patching_multiple_heads.py
python patching_residual_stream.py
```

## Conclusion

Our systematic investigation led to a significant conclusion: The circuit in `gpt2-small` responsible for this indirect prompt injection is not a simple, localized mechanism. It is a subtle and distributed process where the final instruction to "hijack" the model's behavior is not fully resolved until the very end of the forward pass, likely involving the final layer normalization (`ln_final`) and the unembedding matrix (`W_U`).
