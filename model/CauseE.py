import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import helper


class SupervisedProd2vec(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super().__init__()

        self.num_users = n_user
        self.num_products = n_item
        self.device = args.device
        self.l2_pen = 0
        self.batch_size = args.batch_size
        self.decay = eval(args.regs)[0]
        self.cf_pen = 1
        self.norm_adj = norm_adj
        self.cf_distance = 'l1'
        self.cf_loss = 0

        self.user_embeddings = nn.Embedding(self.num_users, args.embed_size)
        self.user_b = nn.Parameter(torch.zeros(self.num_users))
        self.product_embeddings = nn.Embedding(self.num_products, args.embed_size)
        self.prod_b = nn.Parameter(torch.zeros(self.num_products))

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.tensor(0.00000001))
        label1 = torch.ones([self.batch_size, 1])
        label0 = torch.zeros([self.batch_size, 1])
        self.label = torch.cat([label1, label0], dim=0).to(self.device)

    def forward(self, user_list, pos_items, neg_items, drop_flag=False, test_flag=False):
        if test_flag:
          return 0
        if neg_items != []:
            user_list = user_list + user_list
            item_list = pos_items + neg_items
        else:
            user_list = user_list
            item_list = pos_items
        user_list_ = torch.tensor(user_list).to(self.device)
        item_list_ = torch.tensor(item_list).to(self.device)
        user_embed = self.user_embeddings(user_list_)
        user_bias_embed = self.user_b[user_list]
        product_embed = self.product_embeddings(item_list_)
        prod_bias_embed = self.prod_b[item_list]
        control = helper.compute_2i_regularization_id(item_list, self.num_products // 2)
        control = torch.tensor(control).to(self.device)
        control_embed = self.product_embeddings(control)
        if self.cf_distance == "l1":
            # print("Using L1 difference between treatment and control embeddings")
            cf_loss = torch.mean(torch.sum(torch.abs(product_embed - control_embed), dim=1))

        elif self.cf_distance == "l2":
            # print("Using L2 difference between treatment and control embeddings")
            cf_loss = torch.sqrt(torch.sum((product_embed / torch.norm(product_embed, dim=1,
                                                                        keepdim=True) - control_embed / torch.norm(
                control_embed, dim=1, keepdim=True)) ** 2))

        elif self.cf_distance == "cos":
            # print("Using Cosine difference between treatment and control embeddings")
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cf_loss = 1 - cos(product_embed / torch.norm(product_embed, dim=1, keepdim=True),
                              control_embed / torch.norm(control_embed, dim=1, keepdim=True))
            cf_loss = torch.mean(cf_loss)
        self.cf_loss = cf_loss
        emb_logits = self.alpha * (user_embed * product_embed).sum(1).view(-1, 1)
        logits = (prod_bias_embed + user_bias_embed).view(-1, 1) + self.global_bias
        logits += emb_logits
        prediction = torch.sigmoid(logits)

        return prediction, logits, user_embed, product_embed

    def rating(self, i, user_list, i_g):
        user_list_ = torch.tensor(user_list).to(self.device)
        i_g_ = torch.tensor(i_g).to(self.device)
        user_embed = self.user_embeddings(user_list_)
        user_bias_embed = self.user_b[user_list]
        product_embed = self.product_embeddings(i_g_)
        prod_bias_embed = self.prod_b[i_g]

        emb_logits = self.alpha * torch.matmul(user_embed, product_embed.t())
        logits = prod_bias_embed.view(1,len(i_g)) + user_bias_embed.view(len(user_list), 1) + self.global_bias
        print(emb_logits.shape, logits.shape)
        logits += emb_logits
        prediction = torch.sigmoid(logits)
        return prediction

    def create_losses(self, logits, user_embed, product_embed):
        log_loss = F.binary_cross_entropy_with_logits(logits, self.label)
        reg_term = self.decay * (user_embed.norm() ** 2 + product_embed.norm() ** 2)/self.batch_size
        reg_term_biases = self.decay * (self.prod_b.norm() ** 2 + self.user_b.norm() ** 2)/self.batch_size
        factual_loss = log_loss + reg_term + reg_term_biases
        loss = factual_loss + (self.cf_pen * self.cf_loss)
        mse_loss = F.mse_loss(torch.sigmoid(logits), self.label)

        return loss, mse_loss, log_loss


