import math
import random
import numpy as np
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
import time
import contextlib


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d, attention_mask=None):
    if attention_mask != None:
        attention_mask = attention_mask.unsqueeze(-1)
        d *= attention_mask
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    #     print("_l2_normalize, BEFORE:{} \t AFTER:{}".format(d.size(), d_reshaped.size()))
    return d


def _emb_norm(emb):
    e_reshaped = emb.view(emb.shape[0], -1, *(1 for _ in range(emb.dim() - 2)))
    enorm = torch.norm(e_reshaped, dim=1, keepdim=False) + 1e-8
    #     print("BEFORE:{} \t AFTER:{}".format(emb.size(), e_reshaped.size()))
    #     print("enorm:{}, {}".format(enorm.size(), enorm[:10]))
    return enorm


class VaSCL_Pturb(nn.Module):
    def __init__(self, xi=0.1, eps=1, ip=1, uni_criterion=None, bi_criterion=None):
        """VaSCL_Pturb on Transformer embeddings
            :param xi: hyperparameter of VaSCL_Pturb (default: 10.0)
            :param eps: hyperparameter of VaSCL_Pturb (default: 1.0)
            :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VaSCL_Pturb, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.delta = 1e-08

        self.uni_criterion = uni_criterion
        self.bi_criterion = bi_criterion
        print("\n VaSCL_Pturb on embeddings, xi:{}, eps:{} \n".format(xi, eps))

    def forward(self, model, inputs, hard_indices):
        #         print(inputs.size(), "\n", _emb_norm(inputs)[:5])
        with torch.no_grad():
            cnst = model.contrast_logits(inputs)

        # prepare random unit tensor
        d = torch.rand(inputs.shape).sub(0.5).to(inputs.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                cnst_hat = model.contrast_logits(inputs + self.xi * d)

                adv_cnst = self.uni_criterion(cnst, cnst_hat, hard_indices)
                adv_distance = adv_cnst['lds_loss']

                adv_distance.backward(retain_graph=True)
                d = _l2_normalize(d.grad)
                model.zero_grad()

        cnst_hat = model.contrast_logits(inputs + self.eps * d)
        adv_cnst = self.bi_criterion(cnst, cnst_hat, hard_indices)
        return adv_cnst


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, topk=16):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.topk = topk
        print(f"\n PosConLoss with temperature={temperature}, \t topk={topk}\n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        Ng = neg.sum(dim=-1)
        loss_pos = (- torch.log(pos / (Ng + pos))).mean()
        return {"loss": loss_pos}


class VaSCL_NUniDir(nn.Module):
    def __init__(self, temperature=0.05):
        super(VaSCL_NUniDir, self).__init__()
        self.temperature = temperature
        print(f"\n VaSCL_NUniDir \n")

    def forward(self, features_1, features_2, hard_indices=None):
        device = features_1.device
        batch_size = features_1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        neg = torch.exp(torch.mm(features_2, features_1.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(batch_size, -1)

        hard_mask = torch.zeros_like(neg, dtype=torch.int32).to(device)
        hard_mask = hard_mask.scatter_(1, hard_indices, 1) > 0
        hardneg = neg.masked_select(hard_mask).view(batch_size, -1)

        Ng = hardneg.sum(dim=-1)
        loss_pos = (- torch.log(pos / (Ng + pos))).mean()
        return {"lds_loss": loss_pos}


class VaSCL_NBiDir(nn.Module):
    def __init__(self, temperature=0.05):
        super(VaSCL_NBiDir, self).__init__()
        self.temperature = temperature
        print(f"\n VaSCL_NBiDir \n")

    def forward(self, features_1, features_2, hard_indices=None):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        hard_mask = torch.zeros(int(neg.shape[0] / 2), int(neg.shape[1] / 2), dtype=torch.int32).to(device)
        hard_mask = hard_mask.scatter_(1, hard_indices, 1) > 0
        hard_mask = hard_mask.repeat(2, 2)
        hardneg = neg.masked_select(hard_mask).view(2 * batch_size, -1)

        Ng = hardneg.sum(dim=-1)
        loss_pos = (- torch.log(pos / (Ng + pos))).mean()
        return {"lds_loss": loss_pos}


class VaSCL(SequentialRecommender):
    def __init__(self, config, dataset):
        super(VaSCL, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']
        self.temperature = config['temperature']
        self.eps = config['eps']
        self.topk = config['VaSCL_topk']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.contrast_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.inner_size, bias=False)
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.paircon_loss = ContrastiveLoss(temperature=self.temperature, topk=self.topk).cuda()

        self.uni_criterion = VaSCL_NUniDir(temperature=self.temperature).cuda()
        self.bi_criterion = VaSCL_NBiDir(temperature=self.temperature).cuda()
        self.perturb_embd = VaSCL_Pturb(xi=self.eps, eps=self.eps, uni_criterion=self.uni_criterion,
                                        bi_criterion=self.bi_criterion).cuda()

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1

    def CSE_forward(self, item_seq1, item_seq_len1, item_seq2, item_seq_len2, topk=16):
        seq_output_1 = self.forward(item_seq1, item_seq_len1)
        seq_output_2 = self.forward(item_seq2, item_seq_len2)

        inner_prod = torch.mm(seq_output_1, seq_output_1.t().contiguous())

        # estimate the neighborhood of input example
        batch_size = item_seq1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(seq_output_1.device)
        inner_prod_neg = inner_prod.masked_select(~mask).view(batch_size, -1)
        topk_inner, hard_indices_unidir = torch.topk(inner_prod_neg, k=topk, dim=-1)

        cnst_feat1, cnst_feat2 = self.contrast_logits(seq_output_1, seq_output_2)
        return seq_output_1, hard_indices_unidir, cnst_feat1, cnst_feat2

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_output = self.forward(item_seq, item_seq_len)
        embeddings, hard_indices, feat1, feat2 = self.CSE_forward(
            item_seq, item_seq_len, item_seq.clone(), item_seq_len.clone()
        )
        ''' Sequential Recommendation Loss '''
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(embeddings * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(embeddings * neg_items_emb, dim=-1)  # [B]
            sr_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(embeddings, test_item_emb.transpose(0, 1))
            sr_loss = self.loss_fct(logits, pos_items)
        losses = self.paircon_loss(feat1, feat2)
        vcl_loss = losses["loss"]
        # losses['vcl_loss'] = vcl_loss.item()

        # if self.eps > 0:
        assert self.eps > 0
        lds_losses = self.perturb_embd(self, embeddings.detach(), hard_indices)
        # losses.update(lds_losses)
        lds_loss = lds_losses["lds_loss"]
        # losses['optimized_loss'] = loss
        loss = sr_loss + vcl_loss + lds_loss
        return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
