import torch
import torch.nn as nn

class LinearScheduler:
    """
    Simple linear scheduler for a value from start to end over total_epochs.
    """
    def __init__(self, start_value, end_value, total_epochs):
        self.start = start_value
        self.end = end_value
        self.total = max(total_epochs, 1)

    def get(self, epoch):
        e = min(max(epoch, 0), self.total)
        return self.start + (self.end - self.start) * (e / self.total)

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label
        self.apply_patch = Patch(config).apply_patch
        self.ignore_index= config.train.ignore_label
        #self.feature_extractor = feature_extractor
        self.gamma=0.7

        # schedulers
        E1 = config.attack.stage1_epochs
        E2 = config.attack.stage2_epochs
        self.gamma_sched = LinearScheduler(config.attack.gamma_start,
                                          config.attack.gamma_end, E1)
        self.beta_sched  = LinearScheduler(config.attack.beta_start,
                                          config.attack.beta_end,  E2)
        self.current_epoch = 0
        self.register_buffer('ema_kl', torch.zeros(1, device=self.device))

        # hyper-params
        self.margin = getattr(config.attack, 'margin', 0.1)
        self.lambda_ent = getattr(config.attack, 'lambda_ent', 0.1)
        self.eta = getattr(config.attack, 'eta', 0.5)
        self.use_feat_div = getattr(config.attack, 'use_feat_div', False)

    def compute_loss_transegpgd_stage1(self, pred, target, clean_pred):
        """
        Stage 1: emphasize hard-to-attack pixels (correctly predicted ones).
        """
        N, C, H, W = pred.shape
        pred_softmax = F.softmax(pred, dim=1)
        target_flat = target.view(-1)
        pred_label = pred_softmax.argmax(dim=1)

        # Flatten for per-pixel comparison
        pred_label_flat = pred_label.view(-1)
        correct_mask = (pred_label_flat == target_flat) & (target_flat != self.ignore_index)
        incorrect_mask = (pred_label_flat != target_flat) & (target_flat != self.ignore_index)

        loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none').view(-1)

        total_pixels = float(correct_mask.sum() + incorrect_mask.sum() + 1e-8)

        loss_weighted = (1 - self.gamma) * loss[correct_mask].sum() + \
                        self.gamma * loss[incorrect_mask].sum()

        return loss_weighted / total_pixels

    def compute_loss_transegpgd_stage2(self, pred, target, clean_pred):
        """
        Stage 2: emphasize high-transferability pixels (large KL divergence from clean prediction).
        """
        pred_softmax = F.softmax(pred, dim=1)
        clean_softmax = F.softmax(clean_pred, dim=1)

        kl_div = F.kl_div(pred_softmax.log(), clean_softmax, reduction='none').sum(1)  # (N, H, W)
        kl_mean = kl_div[target != self.ignore_index].mean()

        high_transfer_mask = (kl_div > kl_mean) & (target != self.ignore_index)
        low_transfer_mask = (kl_div <= kl_mean) & (target != self.ignore_index)

        loss = F.cross_entropy(pred,target, ignore_index=self.ignore_index, reduction='none')

        total_pixels = float(high_transfer_mask.sum() + low_transfer_mask.sum() + 1e-8)

        loss_weighted = (1 - self.beta) * loss[high_transfer_mask].sum() + \
                        self.beta * loss[low_transfer_mask].sum()

        return loss_weighted / total_pixels
