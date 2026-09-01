import torch
import numpy as np
from sklearn.decomposition import PCA

def extract_nuisance_vectors_and_pca(model, tokenizer, loader, layer_range, k_components=256):
    nuisance_vectors = {}
    pca_matrices = {}
    
    # Store activations: layer_idx -> {'benign': [], 'malicious': []}
    activations = {l: {'benign': [], 'malicious': []} for l in layer_range}

    # Register hooks to capture activations
    handles = []
    for l in layer_range:
        def get_hook(layer_idx):
            def hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                activations[layer_idx]['current_batch'].append(hidden[:, -1, :].detach().cpu())
            return hook
        handles.append(model.model.layers[l].register_forward_hook(get_hook(l)))

    # Process contrastive pairs
    for benign, malicious in loader.get_batches():
        for prompt_type, prompts in [('benign', benign), ('malicious', malicious)]:
            for l in layer_range:
                activations[l]['current_batch'] = []
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                model(**inputs)
            
            for l in layer_range:
                batch_acts = torch.cat(activations[l]['current_batch'], dim=0)
                activations[l][prompt_type].append(batch_acts)

    # Cleanup hooks
    for handle in handles:
        handle.remove()

    # Calculate difference-of-means and PCA per layer
    for l in layer_range:
        mu_benign = torch.cat(activations[l]['benign'], dim=0).mean(dim=0)
        mu_malicious = torch.cat(activations[l]['malicious'], dim=0).mean(dim=0)
        
        # Nuisance vector
        diff = mu_malicious - mu_benign
        v_r = diff / torch.norm(diff)
        nuisance_vectors[l] = v_r.to(model.device)
        
        # Fit PCA on combined activations for bottleneck compression
        all_acts = torch.cat([torch.cat(activations[l]['benign'], dim=0), 
                              torch.cat(activations[l]['malicious'], dim=0)], dim=0).numpy()
        pca = PCA(n_components=k_components)
        pca.fit(all_acts)
        w_pca = torch.tensor(pca.components_, dtype=torch.float32).to(model.device)
        pca_matrices[l] = w_pca
        
    return nuisance_vectors, pca_matrices
