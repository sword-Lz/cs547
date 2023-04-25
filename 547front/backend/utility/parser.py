import argparse


def parse_args_NGCF():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument(
        "--data_path", nargs="?", default="./backend/data/", help="Input data path."
    )
    parser.add_argument("--proj_path", nargs="?", default="", help="Project path.")
    parser.add_argument(
        "--dataset",
        nargs="?",
        default="yelp2018",
        help="Choose a dataset from {gowalla, yelp2018, amazon-book}",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Interval of evaluation."
    )
    parser.add_argument("--max_epochs", type=int, default=400, help="Number of epoch.")
    parser.add_argument(
        "--max_iterations", type=int, default=100000, help="Number of epoch."
    )
    parser.add_argument("--embed_size", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[64,64,64]",
        help="Output sizes of every layer",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")

    parser.add_argument("--regs", nargs="?", default="[1e-5]", help="Regularizations.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")

    parser.add_argument(
        "--model_type",
        nargs="?",
        default="ngcf",
        help="Specify the name of model (ngcf).",
    )
    parser.add_argument(
        "--adj_type",
        nargs="?",
        default="norm",
        help="Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.",
    )

    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument(
        "--node_dropout_flag",
        type=int,
        default=1,
        help="0: Disable node dropout, 1: Activate node dropout",
    )
    parser.add_argument(
        "--node_dropout",
        nargs="?",
        default="[0.1]",
        help="Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.",
    )
    parser.add_argument(
        "--mess_dropout",
        nargs="?",
        default="[0.1,0.1,0.1]",
        help="Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.",
    )

    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[20, 40, 60, 80, 100]",
        help="Output sizes of every layer",
    )

    parser.add_argument(
        "--save_flag",
        type=int,
        default=1,
        help="0: Disable model saver, 1: Activate model saver",
    )

    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    parser.add_argument(
        "--report",
        type=int,
        default=0,
        help="0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--cause", type=int, default=5, help="cause_portial")
    parser.add_argument("--model", type=str, default="cause_NGCF", help="cause_portial")
    return parser.parse_args()
