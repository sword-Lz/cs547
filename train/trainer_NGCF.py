import torch
import sys
from utility.helper import *
from utility.batch_test import *
from time import time
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import logging


def trainer_NGCF(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # from dataset.load_data import rcs_dataset
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    from model.NGCF import NGCF
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    writer = SummaryWriter(snapshot_path + '/log')
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    iter_num = 0
    for epoch in range(args.max_epochs):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        iterator = tqdm(range(n_batch), ncols=100)
        for idx in iterator:
            users, pos_items, neg_items = data_generator.sample()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss, mse_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss_loger.append(batch_loss.item())
            iter_num += 1
            iterator.set_postfix(
                {'total_loss': batch_loss.item(), 'mf_loss': batch_mf_loss.item(), 'emb_loss': batch_emb_loss.item(), 'mse_loss':mse_loss.item()})
            logging.info('iteration %d : loss : %f, mf_loss: %f, emb_loss: %f, mse_loss: %f'% (iter_num, batch_loss.item(), batch_mf_loss.item(), batch_emb_loss.item(), mse_loss.item()))
            if (idx + 1) % 1000 == 0:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(model, users_to_test, drop_flag=False)

                rec_loger.append(ret['recall'])
                ndcg_loger.append(ret['ndcg'])
                logging.info(
                    'iteration' + str(iter_num) + ',recall:' + str(ret['recall']) + ',ndcg:' + str(ret['ndcg'])+',mse:' + str(ret['mse']) + ',log:' + str(ret['log_loss']))


        if (epoch + 1) % 3 != 0:
            continue
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), snapshot_path + '/' + str(epoch) + '.pkl')
            print('save the weights in path: ', snapshot_path + '/' + str(epoch) + '.pkl')

    record(snapshot_path, np.array(loss_loger), np.array(rec_loger), np.array(ndcg_loger))
    writer.close()
    return 'finished'


def trainer_cause_NGCF(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # from dataset.load_data import rcs_dataset
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    from model.cause_NGCF import NGCF_cause
    model = NGCF_cause(data_generator.n_users,
                       data_generator.n_items,
                       norm_adj,
                       args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    writer = SummaryWriter(snapshot_path + '/log')
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    iter_num = 0
    for epoch in range(args.max_epochs):
        t1 = time()

        n_batch = data_generator.n_train // args.batch_size + 1
        iterator = tqdm(range(n_batch), ncols=100)
        for idx in iterator:
            users, pos_items, neg_items = data_generator.sample()
            all_emb, u_g_embeddings, i_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, logits, prediction= model(
                users,
                pos_items,
                neg_items,
                drop_flag=args.node_dropout_flag, test_flag=False)

            batch_loss, batch_mse_loss, cause_loss = model.create_cause_loss(u_g_embeddings,
                                                                             pos_i_g_embeddings,
                                                                             neg_i_g_embeddings,
                                                                             logits, prediction)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            iter_num += 1
            iterator.set_postfix(
                {'total_loss': batch_loss.item(), 'mse_loss': batch_mse_loss.item(), 'cause_loss': cause_loss.item()})
            logging.info('iteration %d : loss : %f, mse_loss: %f, cause_loss: %f, cf_loss: %f' % (
            iter_num, batch_loss.item(), batch_mse_loss.item(), cause_loss.item(), model.cf_loss.item()))
            if (iter_num + 1) % 1000 == 0:
                users_to_test = list(data_generator.test_set.keys())
                ret = test_cause(model, users_to_test, drop_flag=False, batch_test_flag=False)
                loss_loger.append(batch_loss)
                rec_loger.append(ret['recall'])
                ndcg_loger.append(ret['ndcg'])
                logging.info(
                    'iteration' + str(iter_num) + ',recall:' + str(ret['recall']) + ',ndcg:' + str(ret['ndcg'])+',mse:' + str(ret['mse']) + ',log:' + str(ret['log_loss']))

        if (epoch + 1) % 3 != 0:
            continue
        users_to_test = list(data_generator.test_set.keys())
        ret = test_cause(model, users_to_test, drop_flag=False)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), snapshot_path + '/' + str(epoch) + '.pkl')
            print('save the weights in path: ', snapshot_path + '/' + str(epoch) + '.pkl')

    record(snapshot_path, loss_loger, rec_loger, ndcg_loger)
    writer.close()
    return 'finished'
def trainer_cause(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # from dataset.load_data import rcs_dataset
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    from model.CauseE import SupervisedProd2vec
    model = SupervisedProd2vec(data_generator.n_users,
                       data_generator.n_items,
                       norm_adj,
                       args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    writer = SummaryWriter(snapshot_path + '/log')
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    iter_num = 0
    for epoch in range(args.max_epochs):
        t1 = time()

        n_batch = data_generator.n_train // args.batch_size + 1
        iterator = tqdm(range(n_batch), ncols=100)
        for idx in iterator:
            users, pos_items, neg_items = data_generator.sample()
            prediction, logits, user_embed, product_embed = model(
                users,
                pos_items,
                neg_items,
                test_flag=False)

            batch_loss, batch_mse_loss, batch_log_loss = model.create_losses(logits, user_embed, product_embed)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            iter_num += 1
            iterator.set_postfix(
                {'total_loss': batch_loss.item(), 'mse_loss': batch_mse_loss.item(), 'log_loss': batch_log_loss.item()})
            logging.info('iteration %d : loss : %f, mse_loss: %f, log_loss: %f' % (
            iter_num, batch_loss.item(), batch_mse_loss.item(), batch_log_loss.item()))
            if (iter_num + 1) % 1000 == 0:
                users_to_test = list(data_generator.test_set.keys())
                ret = test_cause(model, users_to_test, drop_flag=False, batch_test_flag=False)
                loss_loger.append(batch_loss)
                rec_loger.append(ret['recall'])
                ndcg_loger.append(ret['ndcg'])
                logging.info(
                    'iteration' + str(iter_num) + ',recall:' + str(ret['recall']) + ',ndcg:' + str(ret['ndcg'])+',mse:' + str(ret['mse']) + ',log:' + str(ret['log_loss']))

        if (epoch + 1) % 3 != 0:
            continue
        users_to_test = list(data_generator.test_set.keys())
        ret = test_cause(model, users_to_test, drop_flag=False)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), snapshot_path + '/' + str(epoch) + '.pkl')
            print('save the weights in path: ', snapshot_path + '/' + str(epoch) + '.pkl')

    record(snapshot_path, loss_loger, rec_loger, ndcg_loger)
    writer.close()
    return 'finished'