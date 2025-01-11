import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', "--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    
    # about dataset
    parser.add_argument('-data', "--dataset", type=str, default="fmnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument("--noniid_s", type=int, default=20)
    parser.add_argument("--local_size", type=int, default=600)

    # about model
    parser.add_argument('-mn', "--model_name", type=str, default="cnn")
    
    # about algorithm
    parser.add_argument('-algo', "--algorithm", type=str, default="FedPAC_M")

    # about client
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")    

    # about traing parameters
    parser.add_argument('-lbs', "--batch_size", type=int, default=50)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.02,
                        help="Local learning rate")
    parser.add_argument( "--local_learning_rate_head", type=float, default=0.1,
                        help="Local learning rate head")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    
    # about other parameters
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    
    # about FedPAC_Modify
    parser.add_argument('--plocal_steps', type=int, default=5, help="head epoch")

    args = parser.parse_args()
    return args