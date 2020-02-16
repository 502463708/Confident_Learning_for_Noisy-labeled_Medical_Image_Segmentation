import torch.nn as nn
import torch.nn.functional as F


class SingleClassDiceLoss(nn.Module):
    def __init__(self):
        super(SingleClassDiceLoss, self).__init__()

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

        intersection = (predictions * targets).sum(1)
        union = predictions.sum(1) + targets.sum(1)

        score = 2. * (intersection + epsilon) / (union + epsilon)
        loss = 1 - score.sum() / batch_size

        log_message = 'batch_size: {}, m_foreground: {:<8d}, m_background: {:<8d}, m_I: {:<8d}, m_U: {:<8d}, loss: {:.4f}'.format(
            batch_size,
            int((targets == 1).sum().item() / batch_size),
            int((targets == 0).sum().item() / batch_size),
            int(intersection.sum().item() / batch_size),
            int(union.sum().item() / batch_size),
            loss.item())

        if logger is not None:
            logger.write_and_print(log_message)
        else:
            print(log_message)

        return loss

    def get_name(self):

        return 'SingleClassDiceLoss'
