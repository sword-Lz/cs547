"""
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
"""
import backend.utility.metrics as metrics
from backend.utility.parser import parse_args_NGCF
from backend.dataset.load_data import *
import multiprocessing
import heapq
import numpy as np
import time
import tqdm
import random
from torch import nn
import torch.nn.functional as F

# cores = multiprocessing.cpu_count() // 2
cores = 8
args = parse_args_NGCF()
Ks = eval(args.Ks)

data_generator = rcs_dataset(
    path=args.data_path + args.dataset, batch_size=args.batch_size
)
CR = data_generator.cr
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = 4 * args.batch_size
ITEM_NUM = (ITEM_NUM // BATCH_SIZE) * BATCH_SIZE


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    score = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)

        else:
            r.append(0)
        score.append(item_score[i].item())
    auc = 0.0
    return r, auc, score


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks, K_max_item_score):
    precision, recall, ndcg, hit_ratio, mse, log_loss = [], [], [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))
        mse.append(metrics.mse_lift(r, K_max_item_score, CR, K))
        log_loss.append(metrics.log_lift(r, K_max_item_score, CR, K))

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "hit_ratio": np.array(hit_ratio),
        "auc": auc,
        "mse": np.array(mse),
        "log_loss": np.array(log_loss),
    }


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == "part":
        r, auc, K_max_item_score = ranklist_by_heapq(
            user_pos_test, test_items, rating, Ks
        )
    else:
        r, auc, K_max_item_score = ranklist_by_sorted(
            user_pos_test, test_items, rating, Ks
        )

    return get_performance(user_pos_test, r, auc, Ks, K_max_item_score)


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
    }

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(1):
        # start = u_batch_id * u_batch_size
        # end = (u_batch_id + 1) * u_batch_size
        user_batch = random.choices(test_users, k=4096)
        # user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(
                        user_batch, item_batch, [], drop_flag=False
                    )
                    i_rate_batch = (
                        model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                    )
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(
                        user_batch, item_batch, [], drop_flag=True
                    )
                    i_rate_batch = (
                        model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                    )

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, _ = model(
                    user_batch, item_batch, [], drop_flag=False
                )
                rate_batch = (
                    model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                )
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(
                    user_batch, item_batch, [], drop_flag=True
                )
                rate_batch = (
                    model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                )

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        # count += len(batch_result)

        for re in batch_result:
            result["precision"] += re["precision"] / n_test_users
            result["recall"] += re["recall"] / n_test_users
            result["ndcg"] += re["ndcg"] / n_test_users
            result["hit_ratio"] += re["hit_ratio"] / n_test_users
            result["auc"] += re["auc"] / n_test_users
            result["mse"] += re["mse"] / n_test_users
            result["log_loss"] += re["log_loss"] / n_test_users
    # assert count == n_test_users
    pool.close()
    return result


def test_cause(model, users_to_test, drop_flag=False, batch_test_flag=False):
    result = {
        "precision": np.zeros(len(Ks)),
        "recall": np.zeros(len(Ks)),
        "ndcg": np.zeros(len(Ks)),
        "hit_ratio": np.zeros(len(Ks)),
        "auc": 0.0,
        "mse": 0.0,
        "log_loss": 0.0,
    }

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size

    count = 0
    # user_t = tqdm.tqdm(range(n_user_batchs))
    for u_batch_id in range(1):
        # start = u_batch_id * u_batch_size
        # end = (u_batch_id + 1) * u_batch_size
        user_batch = random.choices(test_users, k=20)
        # user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size
            rate_batch = np.zeros(shape=(len(user_batch), n_item_batchs * i_batch_size))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                a = time.time()
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = list(range(i_start, i_end))

                if drop_flag == False:
                    (
                        u_g_embeddings,
                        i_g_embeddings,
                        pos_i_g_embeddings,
                        _,
                        logits,
                        prediction,
                    ) = model(user_batch, item_batch, [], drop_flag=False)
                    i_rate_batch = prediction.detach().cpu()
                else:
                    (
                        u_g_embeddings,
                        i_g_embeddings,
                        pos_i_g_embeddings,
                        _,
                        logits,
                        prediction,
                    ) = model(user_batch, item_batch, [], drop_flag=True)
                    i_rate_batch = prediction.detach().cpu()
                    i_rate_batch = (
                        model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                    )

                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]
                b = time.time()
                print(b - a)

            # assert i_count == ITEM_NUM

        else:
            # all-item test
            item_batch = list(range(ITEM_NUM))
            if drop_flag == False:
                all_emb = model(
                    user_batch, item_batch, [], drop_flag=False, test_flag=True
                )
                rate_batch = (
                    model.rating(all_emb, user_batch, item_batch).detach().cpu()
                )
                # rate_batch = prediction.detach().cpu()
            else:
                all_emb = model(
                    user_batch, item_batch, [], drop_flag=True, test_flag=True
                )
                rate_batch = (
                    model.rating(all_emb, user_batch, item_batch).detach().cpu()
                )
                # rate_batch = prediction.detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_batch)
        # print(len(rate_batch))
        # batch_result = []
        # for bru in tqdm.tqdm(user_batch_rating_uid):
        #   batch_result.append(test_one_user(bru))
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        # count += len(batch_result)

        for re in batch_result:
            result["precision"] += re["precision"] / n_test_users
            result["recall"] += re["recall"] / n_test_users
            result["ndcg"] += re["ndcg"] / n_test_users
            result["hit_ratio"] += re["hit_ratio"] / n_test_users
            result["auc"] += re["auc"] / n_test_users
            result["mse"] += re["mse"] / n_test_users
            result["log_loss"] += re["log_loss"] / n_test_users

    # assert count == n_test_users
    pool.close()
    return result
