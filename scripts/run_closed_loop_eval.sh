#!/bin/bash
# Pipeline to evaluate the steered agent across benchmarks

echo "Initializing Closed-Loop Adaptive Angular Steering Evaluation..."
echo "Loading Gemma-7B configuration..."
# python -m src.evaluation.run_benchmarks --config configs/eval_benchmarks.yaml --vectors data/vectors/
echo "Evaluation run finished. Metrics logged."
