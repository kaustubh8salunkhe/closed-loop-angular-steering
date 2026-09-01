import yaml
from src.agents.scientific_agent import ScientificAgent
from src.agents.attack_simulator import AttackSimulator

def evaluate_wmdp(agent, config, attack_sim):
    print("Evaluating on WMDP benchmark subsets:", config['benchmarks']['wmdp']['subsets'])
    # Stub: load WMDP dataset, run agent.execute_workflow on prompts
    # Apply attack_sim.inject_jailbreak() to test adversarial hardening
    return {"refusal_rate": 0.94} # Mock result

def evaluate_mmlu(agent, config):
    print("Evaluating on MMLU benchmark subsets:", config['benchmarks']['mmlu']['subsets'])
    # Stub: load MMLU STEM dataset, run agent
    return {"accuracy_drop": 0.005} # Mock result

def run_all(agent, config_path="configs/eval_benchmarks.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    attack_sim = AttackSimulator(method=config['adversarial']['method'])
    wmdp_res = evaluate_wmdp(agent, config, attack_sim)
    mmlu_res = evaluate_mmlu(agent, config)
    
    print(f"WMDP Refusal Rate: {wmdp_res['refusal_rate']:.2f}")
    print(f"MMLU Accuracy Drop: {mmlu_res['accuracy_drop']:.3f}")
