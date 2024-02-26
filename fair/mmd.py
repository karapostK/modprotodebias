import math

import torch
from torch import nn


class MMD(nn.Module):
    def __init__(self, default_class: int = 1):
        super(MMD, self).__init__()
        self.default_class = default_class

    def forward(self, embeddings, labels):
        # # mask = [0 for label in labels if label != 1]
        # # labels = torch.tensor(mask).to(labels.device)
        labels[labels != self.default_class] = 0
        if self.default_class != 1:
            labels[labels == self.default_class] = 1
        loss = self.mmd_loss(embeddings, labels, kernel_mul=4, kernel_num=4, fix_sigma=True)
        return loss

    def mmd_loss(self, embeds, domain_labels, kernel_mul, kernel_num, fix_sigma):
        loss = torch.tensor(0., device=domain_labels.device)
        # split into source and target samples
        unique_dl = domain_labels.unique()
        # if a batch with samples of only one domain is encountered - return 0 as loss
        if len(unique_dl) == 1:
            return loss
        src_mask = domain_labels == unique_dl[0]
        tgt_mask = domain_labels == unique_dl[1]
        # for all embeds calculate the mmd loss and sum them together

        # for embed in embeds:
        # try:
        src_embed = embeds[domain_labels == 0]
        tgt_embed = embeds[domain_labels == 1]
        embed_loss = self.mmd(src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma)
        loss += embed_loss
        # except:
        #     return loss
        return loss

    def gaussian_kernel(self, src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma):
        """Given source and target embeddings calculates kernel matrix based on given specific parameters."""
        if torch.mean(torch.abs(src_embed) + torch.abs(tgt_embed)) <= 1e-7:
            print("Warning: feature representations tend towards zero. "
                  "Consider decreasing 'da_lambda' or using lambda schedule.")
        n_samples = int(src_embed.size()[0] + tgt_embed.size()[0])
        total = torch.cat([src_embed, tgt_embed], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)  # shift bandwidth to the left
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / (bandwidth_temp + 1e-5)) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd(self, src_embed, tgt_embed, kernel_mul, kernel_num, fix_sigma):
        src_batch_size = src_embed.size(0)
        tgt_batch_size = tgt_embed.size(0)
        # handle case when source and target are not of same size
        # we extend both to the fixed size 'batch size', because sizes should not vary between batches
        batch_size = src_batch_size + tgt_batch_size
        src_repeats = math.ceil(batch_size / src_batch_size)
        tgt_repeats = math.ceil(batch_size / tgt_batch_size)
        src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
        tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]
        kernels = self.gaussian_kernel(src_embed_rep, tgt_embed_rep, kernel_mul, kernel_num, fix_sigma)
        # use different parts of the kernel matrix to calculate final loss
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)

        return loss
