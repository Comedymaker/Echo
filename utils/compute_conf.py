import torch
import torch.nn.functional as F

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]

def compute_topk_mass(probs, k=3):
    topk_vals, _ = torch.topk(probs, k, dim=-1)
    return topk_vals.sum(dim=-1)  # [B]

def compute_js_divergence(probs_p, probs_q):
    m = 0.5 * (probs_p + probs_q)
    kl_p_m = torch.sum(probs_p * (torch.log(probs_p + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    kl_q_m = torch.sum(probs_q * (torch.log(probs_q + 1e-8) - torch.log(m + 1e-8)), dim=-1)
    js = 0.5 * (kl_p_m + kl_q_m)
    return js  # [B]

def compute_confidence_features(logits_s, logits_l, topk=3):
    probs_s = F.softmax(logits_s, dim=-1)  # [B, V]
    probs_l = F.softmax(logits_l, dim=-1)  # [B, V]

    entropy_s = compute_entropy(probs_s)  # [B]
    entropy_l = compute_entropy(probs_l)  # [B]

    topk_mass_s = compute_topk_mass(probs_s, k=topk)  # [B]
    topk_mass_l = compute_topk_mass(probs_l, k=topk)  # [B]

    js_div = compute_js_divergence(probs_s, probs_l)  # [B]

    conf_feat = torch.stack([
        entropy_s,
        entropy_l,
        topk_mass_s,
        topk_mass_l,
        js_div
    ], dim=-1)  # [B, 5]

    return conf_feat
