import torch
from transformers import LlamaConfig
from .llava_llama import LlavaLlamaForCausalLM  # Replace with the actual import path

##use python -m llava.model.language_model.param_inspector

def print_top_level_attributes(model):
    print("\nTop-level attributes of the model:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")

def print_model_parameters_and_modules(model):
    print("Listing all parameters related to attention:")
    for name, param in model.named_parameters():
        if 'attention' in name:
            print(f"{name}: {param.size()}")

    print("\nListing all modules, looking for attention mechanisms:")
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")

def print_modules(model):
    for name, module in model.named_modules():
        print(f"{name}: {module}")
    

def main():
    # Configure the model
    config = LlamaConfig()
    model = LlavaLlamaForCausalLM(config)
    # Inspect the model
   # print_top_level_attributes(model)
   # print_model_parameters_and_modules(model)
    print_modules(model)
    
    print(model.model.modules)
    layers = model.model.layers
    print('process finished')

if __name__ == "__main__":
    main()