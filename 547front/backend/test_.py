from backend.utility.batch_test import *
from backend.utility.parser import parse_args_NGCF


def predict(input):
    args = parse_args_NGCF()
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    from backend.cause_NGCF import NGCF_cause

    args.dataset = "yelp2018"
    args.device = "cpu"
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    model = NGCF_cause(
        data_generator.n_users, data_generator.n_items, norm_adj, args
    ).to(args.device)

    model.load_state_dict(torch.load("./backend/47.pkl", map_location="cpu"))
    user_batch = [input]
    ITEM_NUM = data_generator.n_items
    item_batch = list(range(ITEM_NUM))

    seed = 1234
    torch.manual_seed(seed)
    # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    all_emb = model(user_batch, item_batch, [], drop_flag=False, test_flag=True)
    rate_batch = model.rating(all_emb, user_batch, item_batch).detach().cpu()
    item_score = {}
    rate_batch = rate_batch.tolist()[0]
    for i in item_batch:
        item_score[i] = rate_batch[i]

    K_max = max(Ks)

    K_max_item_score = heapq.nlargest(10, item_score, key=item_score.get)
    print(K_max_item_score)
    return K_max_item_score
