# Closed-Loop Adaptive Angular Steering for Agentic LLMs

This repository implements **Closed-Loop Adaptive Angular Steering**, an inference-time behavioral control pipeline designed for agentic scientific Large Language Models (LLMs) (e.g., Gemma).

## Theoretical Foundation
Rather than retraining or performing static additive activation engineering, this project models safety and refusal behaviors as latent nuisance variables within the high-dimensional activation space. This approach is grounded in the Deep Rendering Mixture Model (DRMM) framework[cite: 1]. The DRMM explicitly captures variations in data by modeling them as latent task nuisance variables[cite: 1]. 

## Architecture
The architecture couples lightweight representation compression with a multi-layer Proportional-Integral-Derivative (PID) feedback controller to dynamically scale and apply norm-preserving 2D angular rotations. Because deep networks inherently learn to disentangle factors of variation across multiple levels of abstraction[cite: 1], we can manage these behavioral nuances directly within the latent space during inference, bypassing the need for computationally expensive retraining via backpropagation[cite: 1].

1. **Representation Compression Bottleneck:** Activations are projected into a $k$-dimensional PCA subspace, acting as a regularizer[cite: 1] to strip spurious high-frequency jailbreak features.
2. **Layer-Wise PID Feedback Controller:** Regulates a multi-layer steering trajectory using discrete proportional, integral, and derivative terms to mitigate overshoot and semantic drift.
3. **Adaptive Norm-Preserving Rotation:** Rotates the activation in a 2D plane spanned by the nuisance vector and the orthogonal component, strictly preserving the $L_2$ norm.

## Evaluation
The system is evaluated on autonomous scientific workflows to prove robustness against adversarial exploitation (WMDP benchmark) while preserving complex reasoning capabilities (MMLU benchmark).
