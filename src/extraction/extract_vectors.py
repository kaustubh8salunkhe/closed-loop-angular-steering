import torch
from sklearn.decomposition import PCA

def compute_steering_plane(activations_harmful, activations_harmless):
    """
    Extracts the feature direction and PCA plane as per Angular Steering.
    """
    candidate_directions = []
    
    # Calculate Candidate Directions via Difference-in-Means
    for layer_idx in activations_harmful.keys():
        mu_harmful = activations_harmful[layer_idx].mean(dim=0)
        mu_harmless = activations_harmless[layer_idx].mean(dim=0)
        d_i = mu_harmful - mu_harmless 
        candidate_directions.append(d_i)

    candidates_tensor = torch.stack(candidate_directions)

    # Select the target feature direction (e.g., from a stable middle layer)
    d_feat = candidate_directions[len(candidate_directions) // 2]
    d_feat = d_feat / torch.norm(d_feat)

    # Perform PCA on the candidate directions to find the second axis for the plane
    pca = PCA(n_components=1)
    pca.fit(candidates_tensor.cpu().numpy())
    d_pc0 = torch.tensor(pca.components_[0], device=d_feat.device, dtype=d_feat.dtype)

    # Gram-Schmidt orthogonalization to establish orthonormal basis (b1, b2)
    b1 = d_feat
    b2 = d_pc0 - (torch.dot(b1, d_pc0) * b1)
    b2 = b2 / torch.norm(b2)

    return b1, b2
