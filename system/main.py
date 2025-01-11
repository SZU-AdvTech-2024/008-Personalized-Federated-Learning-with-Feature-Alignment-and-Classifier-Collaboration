import time
import copy
import torch.nn as nn
# import local file
from config import args_parser
from flcore.models import CNN_1
from flcore.models import CNN_2
from flcore.models import BaseHeadSplit
from flcore.serverpac import FedPAC
from flcore.serverpac_modify import FedPAC_M

if __name__ == "__main__":
    args = args_parser()

    start = time.time()

    # Generate args.model
    if args.dataset in ['cifar10','cinic10']:
        args.num_classes = 10
        args.local_learning_rate=0.02
        args.model= CNN_2(num_classes=args.num_classes).to(args.device)
    elif args.dataset == 'fmnist':
        args.num_classes = 10
        args.local_learning_rate = 0.01
        args.model= CNN_1(num_classes=args.num_classes).to(args.device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        args.model=CNN_1(num_classes=args.num_classes).to(args.device)
    else:
        raise NotImplementedError
    
    print(args.model)

    # Create server and clients
    print("Creating server and clients ...")

    args.head = copy.deepcopy(args.model.fc2)
    args.model.fc2 = nn.Identity()
    args.model = BaseHeadSplit(args.model, args.head)

    # Select Algorithm
    if args.algorithm == 'FedPAC':
        server = FedPAC(args)
    elif args.algorithm == 'FedPAC_M':
        server = FedPAC_M(args)

    # Start train
    server.train()

    # Finish train
    print(f"\ntime cost: {time.time()-start}s.")
    print("All done!")


