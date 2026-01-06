import timm
import torch.nn as nn

def inspect_timm_model(model_name):
    print(f"--- Inspecting timm model: {model_name} ---")
    model = timm.create_model(model_name, pretrained=False, num_classes=256)
    
    # Print the last few modules
    modules = list(model.named_modules())
    print("\nLast 10 modules:")
    for name, module in modules[-10:]:
        print(f"  {name}: {type(module)}")
        
    # Check for common head names
    for head_name in ['head', 'fc', 'classifier', 'head.fc']:
        if hasattr(model, head_name):
            print(f"\nFound head: {head_name}")

if __name__ == "__main__":
    inspect_timm_model('fastvit_t8.apple_dist_in1k')
