import torch.nn as nn
import torch.nn.functional as F


class SingleClassTverskyLoss(nn.Module):
    def __init__(self, alpha):
        '''
        :param alpha: the weight for focusing on FPs
        '''
        super(SingleClassTverskyLoss, self).__init__()

        assert 0 <= alpha <= 1
        self.alpha = alpha

        return

    def forward(self, predictions, targets, logger=None, activate=False):
        assert len(predictions.shape) == 4
        assert len(targets.shape) == 3
        assert predictions.shape[1] == 1

        batch_size = targets.shape[0]
        epsilon = 1

        if targets.device.type != 'cuda':
            targets = targets.cuda()
        targets = targets.float()

        if activate:
            predictions = F.sigmoid(predictions)

        predictions = predictions.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        TPs = (predictions * targets).sum(1)
        FPs = (predictions * (1 - targets)).sum(1)
        FNs = ((1 - predictions) * targets).sum(1)

        tversky_index = (TPs + epsilon) / (TPs + self.alpha * FPs + (1 - self.alpha) * FNs + epsilon)
        loss = 1 - tversky_index.sum() / batch_size

        log_message = 'batch_size: {}, m_TPs: {:<8d}, m_FPs: {:<8d}, m_FNs: {:<8d}, loss: {:.4f}'.format(
            batch_size,
            int(TPs.mean().item()),
            int(FPs.mean().item()),
            int(FNs.mean().item()),
            loss.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        return loss

    def get_name(self):

        return 'SingleClassTverskyLoss'
