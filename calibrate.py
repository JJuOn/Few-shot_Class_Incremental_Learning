import torch

import numpy as np
import torch.nn.functional as F

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(torch.linalg.norm(query - base_means[i]).unsqueeze(0))
    dist = torch.cat(dist, dim=0)
    _, index = torch.topk(dist, k, dim=0)
    mean = torch.cat([base_means[i].unsqueeze(0) for i in index] + [query.unsqueeze(0)], dim=0)
    calibrated_mean = torch.mean(mean, dim=0)
    cov = torch.cat([base_cov[i].unsqueeze(0) for i in index], dim=0)
    calibrated_cov = torch.mean(cov, dim=0) + alpha

    return calibrated_mean, calibrated_cov

    