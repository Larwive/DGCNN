import torch
import torch.nn.functional as F

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def normalize_A(A, lmax=2):
    A = F.relu(A)
    N = A.shape[0]
    A = A * (torch.ones(N, N) - torch.eye(N, N)).to(device)
    A = A + A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N, N).to(device) - torch.matmul(torch.matmul(D, A), D)
    Lnorm = (2 * L / lmax) - torch.eye(N, N).to(device)
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).to(device))
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2 * L, support[-1], ) - support[-2]
            support.append(temp)
    return support
