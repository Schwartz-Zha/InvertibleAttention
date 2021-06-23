'''
Acquired from the official implementation repo of "Invertible Residual Networks"
https://github.com/jhjacobsen/invertible-resnet
'''


import torch


def power_series_matrix_logarithm_trace(Fx, x, k, n):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :return: Tr(Ln(I + df/dx))
    """
    # trace estimation including power series
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        # compute summand
        vectors = torch.stack(vectors, dim=1)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / torch.Tensor([j]).to(summand.device)
        else:
            trLn -= summand / torch.Tensor([j]).to(summand.device)
    trace = trLn.mean(dim=1).squeeze()
    return trace