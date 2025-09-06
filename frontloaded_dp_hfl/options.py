import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    # dataset and model
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='name of the dataset: mnist, cifar10, fashionmnist'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lenet',
        help='name of model. mnist: lenet; cifar10: cnn_complex_2, cnn_complex_3'
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        default=1,
        help='input channels. mnist: 1, cifar10: 3'
    )
    parser.add_argument(
        '--output_channels',
        type=int,
        default=10,
        help='output channels'
    )

    # nn training hyper-parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--num_communication',
        type=int,
        default=200,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=6,
        help='number of local updates (tau_1)'
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=2,
        help='number of edge aggregations (tau_2)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=1.0,
        help='lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.0,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='verbose for print progress bar'
    )

    # federated learning settings
    parser.add_argument(
        '--iid',
        type=int,
        default=1,
        help='distribution of the data: 1, 0, -2(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=0,
        help='distribution of the data under edges, 1(edgeiid), 0(edgeniid); used only when iid = -2'
    )
    parser.add_argument(
        '--cfrac',
        type=float,
        default=0.2,
        help='fraction of participated clients'
    )
    parser.add_argument(
        '--efrac',
        type=float,
        default=1.0,
        help='fraction of participated edges'
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=100,
        help='number of all available clients'
    )
    parser.add_argument(
        '--num_edges',
        type=int,
        default=5,
        help='number of edges'
    )

    # reproducibility / environment
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data',
        help='dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type=int,
        default=0,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default=2,
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to be selected, e.g., 0, 1, 2, 3'
    )

    # model switches
    parser.add_argument(
        '--mtl_model',
        type=int,
        default=0,
        help='enable MTL model (1) or not (0)'
    )
    parser.add_argument(
        '--global_model',
        type=int,
        default=1,
        help='enable global model (1) or not (0)'
    )
    parser.add_argument(
        '--local_model',
        type=int,
        default=0,
        help='enable local model (1) or not (0)'
    )

    # privacy mode + hyperparameters
    parser.add_argument(
        '--mode',
        type=str,
        default='CP-NP',
        help='mode to be selected: "baseline", "CG-NG", "CG-NP", "CP-NP"'
    )
    parser.add_argument(
        '--clip',
        type=float,
        default=2.0,
        help='L2 clipping value for DP modes'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.01,
        help='standard deviation of Gaussian noise'
    )

    # edge/server learning rate for CP-NP deltas
    parser.add_argument(
        '--eta',
        type=float,
        default=1.0,
        help='edge learning rate when applying averaged deltas (CP-NP); 1.0 = FedAvg'
    )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args