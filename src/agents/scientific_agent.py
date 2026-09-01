class ScientificAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = "You are a scientific autonomous agent with expertise in biology and chemistry."

    def execute_workflow(self, task_description):
        # Stub for complex LangChain/AutoGen style loop
        prompt = f"{self.system_prompt}\nTask: {task_description}\nPlan and execute:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
