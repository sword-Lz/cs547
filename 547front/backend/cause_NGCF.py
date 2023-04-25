import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.utility import helper


class NGCF_cause(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF_cause, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.cf_pen = 1
        self.norm_adj = norm_adj
        self.cf_distance = "l1"

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        self.cause = args.cause
        self.embedding_dict, self.weight_dict = self.init_weight()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(
            self.device
        )
        label1 = torch.ones([self.batch_size, 1])
        label0 = torch.zeros([self.batch_size, 1])
        self.label = torch.cat([label1, label0], dim=0).to(self.device)
        self.cf_loss = 0.0

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.n_user, self.emb_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.n_item, self.emb_size))
                ),
            }
        )

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update(
                {
                    "W_gc_%d"
                    % k: nn.Parameter(
                        initializer(torch.empty(layers[k], layers[k + 1]))
                    )
                }
            )
            weight_dict.update(
                {
                    "b_gc_%d"
                    % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))
                }
            )
            weight_dict.update(
                {
                    "W_bi_%d"
                    % k: nn.Parameter(
                        initializer(torch.empty(layers[k], layers[k + 1]))
                    )
                }
            )
            weight_dict.update(
                {
                    "b_bi_%d"
                    % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))
                }
            )

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1.0 / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        regularizer = (
            torch.norm(users) ** 2
            + torch.norm(pos_items) ** 2
            + torch.norm(neg_items) ** 2
        ) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_cause_loss(self, users, pos_items, neg_items, logits, prediction):
        items = torch.cat([pos_items, neg_items], 0)
        log_loss = F.binary_cross_entropy(prediction, self.label)

        # Adding the regularizer term on user vct and prod vct and their bias terms
        # reg_term = self.l2_pen * (torch.norm(users, 2) + torch.norm(items, 2))
        # reg_term_biases = self.l2_pen * (torch.norm(self.prod_b, 2) + torch.norm(self.user_b, 2))
        # factual_loss = log_loss  # + reg_term + reg_term_biases
        # bpr_loss
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        regularizer = (
            torch.norm(users) ** 2
            + torch.norm(pos_items) ** 2
            + torch.norm(neg_items) ** 2
        ) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        # Adding the counter-factual loss

        cause_loss = self.cf_pen * self.cf_loss  # Imbalance loss
        loss = self.cause * cause_loss + (1 - self.cause) * mf_loss + emb_loss
        mse_loss = nn.MSELoss()(self.label, prediction)
        return loss, log_loss, cause_loss

    def rating(self, all_embeddings, u_g, i_g):
        u_g_embeddings = all_embeddings[: self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user :, :]
        u_g_embeddings = u_g_embeddings[u_g]
        pos_i_g_embeddings = i_g_embeddings[i_g]
        logits = torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

        # logits = 0

        return F.sigmoid(logits)

    def forward(self, users, pos_items, neg_items, drop_flag=True, test_flag=False):
        if neg_items != []:
            users_cause = users + users
        else:
            users_cause = users
        A_hat = (
            self.sparse_dropout(
                self.sparse_norm_adj, self.node_dropout, self.sparse_norm_adj._nnz()
            )
            if drop_flag
            else self.sparse_norm_adj
        )

        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], 0
        )

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = (
                torch.matmul(side_embeddings, self.weight_dict["W_gc_%d" % k])
                + self.weight_dict["b_gc_%d" % k]
            )

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(
                bi_embeddings,
                self.weight_dict["W_bi_%d" % k] + self.weight_dict["b_bi_%d" % k],
            )

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings
            )

            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[: self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user :, :]

        u_g_embeddings_cause = u_g_embeddings[users_cause, :]
        # u_g_embeddings_cause = torch.cat([u_g_embeddings, u_g_embeddings])
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]
        if test_flag:
            return all_embeddings
        control = helper.compute_2i_regularization_id(
            pos_items + neg_items, self.n_item // 2
        )

        control_embed = i_g_embeddings[control, :]
        i_g_embeddings = i_g_embeddings[pos_items + neg_items, :]
        emb_logits = torch.sum(
            u_g_embeddings_cause * i_g_embeddings, dim=1, keepdim=True
        )

        if self.cf_distance == "l1":
            # print("Using L1 difference between treatment and control embeddings")
            cf_loss = torch.mean(
                torch.sum(torch.abs(i_g_embeddings - control_embed), dim=1)
            )

        elif self.cf_distance == "l2":
            # print("Using L2 difference between treatment and control embeddings")
            cf_loss = torch.sqrt(
                torch.sum(
                    (
                        i_g_embeddings / torch.norm(i_g_embeddings, dim=1, keepdim=True)
                        - control_embed / torch.norm(control_embed, dim=1, keepdim=True)
                    )
                    ** 2
                )
            )

        elif self.cf_distance == "cos":
            # print("Using Cosine difference between treatment and control embeddings")
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cf_loss = 1 - cos(
                i_g_embeddings / torch.norm(i_g_embeddings, dim=1, keepdim=True),
                control_embed / torch.norm(control_embed, dim=1, keepdim=True),
            )
            cf_loss = torch.mean(cf_loss)
        self.cf_loss = cf_loss

        # logits = 0
        logits = emb_logits

        prediction = F.sigmoid(logits)
        return (
            all_embeddings,
            u_g_embeddings[users, :],
            i_g_embeddings,
            pos_i_g_embeddings,
            neg_i_g_embeddings,
            logits,
            prediction,
        )
