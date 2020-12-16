import torch
import torch.nn as nn

from .py_utils import exkp, CTLoss, _neg_loss, convolution, residual

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(exkp):
    def __init__(self, db):
        n       = 5
        dims    = [128, 128, 192, 192, 192, 256]
        modules = [1, 1, 1, 1, 1, 2]
        out_dim = 1
        super(model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=dims[0]
        )

loss = CTLoss(focal_loss=_neg_loss)
