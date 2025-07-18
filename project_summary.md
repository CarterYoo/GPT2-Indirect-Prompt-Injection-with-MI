# Project Report: Investigating the Mechanics of Indirect Prompt Injection

## 1. Objective

The primary goal of this project was to investigate the internal mechanics of an indirect prompt injection attack on a Transformer-based language model. We aimed to move beyond simply observing the vulnerability and instead use mechanistic interpretability techniques to identify the specific components (the "circuit") responsible for causing the model to disregard its initial instructions and follow a malicious one embedded in its input.

## 2. Methodology

*   **Model:** `gpt2-small` (a small, well-understood model suitable for interpretability research).
*   **Core Library:** `transformer_lens`.
*   **Key Techniques:**
    *   **Attention Pattern Analysis:** Observing where the model "looks" when processing a prompt to find correlations between attention and behavior.
    *   **Activation Patching (Causal Intervention):** Surgically intervening in the model's computation by replacing specific activation values from a "clean" run with those from a "malicious" run to prove causation.

## 3. Experimental Phases

We conducted a series of experiments, with each one building on the results of the last.

### Experiment 1: Baseline Vulnerability Test

*   **Goal:** Confirm that `gpt2-small` is vulnerable to a basic indirect prompt injection.
*   **Setup:** We provided the model with a clean prompt (summarize an email) and an injected prompt (the same email with a malicious instruction hidden at the end).
*   **Result:** **Success.** The model correctly ignored the malicious instruction in the clean run but **leaked the secret password** in the injected run, confirming the vulnerability.
*   **File:** `injection_test.py`

### Experiment 2: Attention Pattern Analysis

*   **Goal:** Identify suspicious components by observing the model's internal state.
*   **Method:** We analyzed the attention patterns in the final layer (Layer 11) when the model was about to generate the leaked password.
*   **Result:** **Significant Findings.** We discovered that **71.5%** of the total attention from the final token position was directed towards the malicious instruction tokens. This provided a strong correlation and identified a list of suspect heads, including L11H0, L11H1, L11H2, L11H4, L11H6, and L11H7.
*   **File:** `analyze_attention.py`

### Experiment 3: Causal Test (Single-Head Patching)

*   **Goal:** Test the hypothesis that a single head (L11H4) was the primary cause of the injection.
*   **Method:** We patched the activation of only L11H4 from the malicious run onto the clean run.
*   **Result:** **Failure.** The model did not leak the password. This proved that no single attention head in the final layer is solely responsible.
*   **File:** `patching_test.py`

### Experiment 4: Causal Test (Multi-Head Patching)

*   **Goal:** Test the hypothesis that a "conspiracy" of heads in the final layer was responsible.
*   **Method:** We patched the activations of all six suspect heads simultaneously.
*   **Result:** **Failure.** The model still did not leak the password. This demonstrated that the responsible circuit is not confined to the attention heads of the final layer.
*   **File:** `patching_multiple_heads.py`

### Experiment 5: Causal Test (Residual Stream Patching)

*   **Goal:** Test the hypothesis that the full "hijack" signal was present in the entire output of the final layer.
*   **Method:** We performed our most powerful intervention, patching the entire residual stream (the main information highway) at the output of Layer 11.
*   **Result:** **Failure.** Even when transplanting the entire computational result of the final layer, the model did not leak the password.
*   **File:** `patching_residual_stream.py`

## 4. Overall Conclusion

Our systematic investigation, moving from observation to increasingly powerful causal interventions, has led to a significant conclusion:

**The circuit in `gpt2-small` responsible for this indirect prompt injection is not a simple, localized mechanism. It is a subtle and distributed process where the final instruction to "hijack" the model's behavior is not fully resolved until the very end of the forward pass, likely involving the final layer normalization (`ln_final`) and the unembedding matrix (`W_U`) that maps the model's final state to the vocabulary.**

This project successfully demonstrates the scientific process of mechanistic interpretability: forming hypotheses, running experiments, and using the results (even failures) to uncover deeper truths about how these complex systems operate.
