import torch


class NLL_OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio=0.9):
        super(NLL_OHEM, self).__init__()
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)