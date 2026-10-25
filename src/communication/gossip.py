import torch


def aggregate_models(model_i, neighbor_models: list, alpha: float = 0.8) -> dict:
    """
    Weighted gossip parameter aggregation:
        W_i^(t+1) = α * W_i^t + (1-α)/|N| * Σ_j W_j^t

    Only model parameters are exchanged — never raw patient data.
    """
    if not neighbor_models:
        return model_i.state_dict()

    n = len(neighbor_models)
    nb_weight = (1.0 - alpha) / n
    local_state = model_i.state_dict()
    neighbor_states = [m.state_dict() for m in neighbor_models]

    new_state = {}
    for key in local_state:
        p = local_state[key]
        if torch.is_floating_point(p):
            agg = p.clone().float() * alpha
            for ns in neighbor_states:
                agg += ns[key].float() * nb_weight
            new_state[key] = agg
        else:
            new_state[key] = p.clone()
    return new_state
