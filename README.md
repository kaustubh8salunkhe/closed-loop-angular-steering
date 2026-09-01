# Closed-Loop Adaptive Angular Steering for Agentic LLMs

This repository implements **Closed-Loop Adaptive Angular Steering**, an inference-time behavioral control pipeline designed for agentic scientific Large Language Models (LLMs) (e.g., Gemma).

## Theoretical Foundation
Rather than retraining or performing static additive activation engineering, this project models safety and refusal behaviors as latent nuisance variables within the high-dimensional activation space. This approach is grounded in the Deep Rendering Mixture Model (DRMM) framework. The DRMM explicitly captures variations in data by modeling them as latent task nuisance variables. 

## Architecture
The architecture couples lightweight representation compression with a multi-layer Proportional-Integral-Derivative (PID) feedback controller to dynamically scale and apply norm-preserving 2D angular rotations. Because deep networks inherently learn to disentangle factors of variation across multiple levels of abstraction, we can manage these behavioral nuances directly within the latent space during inference, bypassing the need for computationally expensive retraining via backpropagation.

1. **Representation Compression Bottleneck:** Activations are projected into a $k$-dimensional PCA subspace, acting as a regularizer[cite: 1] to strip spurious high-frequency jailbreak features.
2. **Layer-Wise PID Feedback Controller:** Regulates a multi-layer steering trajectory using discrete proportional, integral, and derivative terms to mitigate overshoot and semantic drift.
3. **Adaptive Norm-Preserving Rotation:** Rotates the activation in a 2D plane spanned by the nuisance vector and the orthogonal component, strictly preserving the $L_2$ norm.

## Evaluation
The system is evaluated on autonomous scientific workflows to prove robustness against adversarial exploitation (WMDP benchmark) while preserving complex reasoning capabilities (MMLU benchmark).


## Methodological Pipeline
This project unifies the geometric insights of Representation Engineering with classical control-theoretic dynamics to form a comprehensive inference-time steering pipeline. The architecture is structured across four distinct phases:

1. **Phase 1: Nuisance Variable Extraction:**
Rooted in the Deep Rendering Mixture Model (DRMM), modeling the safety behaviors and refusal states as latent task nuisance variables. By analyzing the model from a top-down representational view, we extract a one-dimensional "refusal direction" using a difference-in-means calculation over contrastive pairs of harmful and harmless instructions.  

2. **Phase 2: 2D Steering Plane Construction:**
Standard activation addition and directional ablation operate by scaling or removing components, which can disrupt activation norms and degrade coherence. To prevent this, we construct a fixed two-dimensional steering subspace. We perform Principal Component Analysis (PCA) on candidate feature directions across layers and use Gram-Schmidt orthogonalization to establish an orthonormal basis consisting of the target feature direction and its primary axis of variation. 

3. **Phase 3: Closed-Loop PID Control:**
Treating the activation steering as a continuous-time dynamical system governed by a state-space model rather than relying on static algebraic manipulation. Because standard steering acts as a simple Proportional (P) controller that is prone to steady-state errors, we implement a multi-layer Proportional-Integral-Derivative (PID) feedback controller: 

Proportional (P): Reacts immediately to errors to align activations with the target semantic direction.  

Integral (I): Accumulates past errors to remove residual bias across transformer layers.  

Derivative (D): Dampens rapid activation changes to mitigate the overshooting typically caused by the integral term.  

4. **Phase 4: Adaptive Angular Rotation:**
The output of the PID controller modulates a dynamic rotation angle $\theta$. Rather than applying this rotation uniformly, we use an adaptive conditional mask based on the activation's projection onto the feature direction. The model's intermediate activations are rotated strictly within the 2D subspace, and only when they are positively aligned with the nuisance direction. This strictly preserves the $L_2$ norm of the activation and minimizes interference with unrelated semantic features. 
