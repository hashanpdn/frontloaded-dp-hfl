# Flow of the algorithm
# Client update (t1) -> Edge Aggregate (t2) -> Cloud Aggregate (t3)

from options import args_parser
from tensorboardX import SummaryWriter
import torch
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn import cifar_cnn_2conv, cifar_cnn_3conv


def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[] for _ in range(10)]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis

def get_edge_class(args, edges, clients):
    edge_class = [[] for _ in range(5)]
    for (i, edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')

def initialize_edges_iid(num_edges, clients, args, client_class_dis):
    """
    Designed for 10*L users, 1-class per user; distribution among edges is iid.
    10 clients per edge, each edge has 10 classes.
    """
    edges = []
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
            assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edge = Edge(id=eid,
                    cids=assigned_clients_idxes,
                    shared_layers=copy.deepcopy(clients[0].model.shared_layers))
        [edge.client_register(clients[c]) for c in assigned_clients_idxes]
        edge.all_trainsample_num = sum(edge.sample_registration.values())
        edge.refresh_edgeserver()
        edges.append(edge)

    # last edge takes leftovers
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print(f"label {label} is empty")
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edge = Edge(id=eid,
                cids=assigned_clients_idxes,
                shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    [edge.client_register(clients[c]) for c in assigned_clients_idxes]
    edge.all_trainsample_num = sum(edge.sample_registration.values())
    edge.refresh_edgeserver()
    edges.append(edge)
    return edges

def initialize_edges_niid(num_edges, clients, args, client_class_dis):
    """
    Designed for 10*L users, 1-class per user; per-edge class coverage is 5 classes.
    """
    edges = []
    label_ranges = [[0,1,2,3,4],[1,2,3,4,5],[5,6,7,8,9],[6,7,8,9,0]]
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        label_range = label_ranges[eid]
        for _ in range(2):
            for label in label_range:
                if len(client_class_dis[label]) > 0:
                    assigned_client_idx = np.random.choice(client_class_dis[label], 1, replace=False)
                    client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
                else:
                    # dynamic backup: choose any non-empty label
                    non_empty = [l for l in range(10) if len(client_class_dis[l]) > 0]
                    if not non_empty:
                        raise RuntimeError("No clients left to assign.")
                    backup = int(np.random.choice(non_empty, 1, replace=False)[0])
                    assigned_client_idx = np.random.choice(client_class_dis[backup], 1, replace=False)
                    client_class_dis[backup] = list(set(client_class_dis[backup]) - set(assigned_client_idx))
                for idx in assigned_client_idx:
                    assigned_clients_idxes.append(idx)
        edge = Edge(id=eid,
                    cids=assigned_clients_idxes,
                    shared_layers=copy.deepcopy(clients[0].model.shared_layers))
        [edge.client_register(clients[c]) for c in assigned_clients_idxes]
        edge.all_trainsample_num = sum(edge.sample_registration.values())
        edge.refresh_edgeserver()
        edges.append(edge)

    # last edge takes leftovers
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print(f"label {label} is empty")
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edge = Edge(id=eid,
                cids=assigned_clients_idxes,
                shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    [edge.client_register(clients[c]) for c in assigned_clients_idxes]
    edge.all_trainsample_num = sum(edge.sample_registration.values())
    edge.refresh_edgeserver()
    edges.append(edge)
    return edges

def all_clients_test(server, clients, cids, device):
    # push server state to clients, then evaluate locally
    for cid in cids:
        server.send_to_client(clients[cid])
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge

def fast_all_clients_test(v_test_loader, global_nn, device):
    global_nn.eval()
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        else:
            raise ValueError(f"Model {args.model} not implemented for mnist")
    elif args.dataset == 'cifar10':
        if args.model == 'cnn_complex_2':
            global_nn = cifar_cnn_2conv(output_dim=10, inter_dim=200)
        elif args.model == 'cnn_complex_3':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        else:
            raise ValueError(f"Model {args.model} not implemented for cifar")
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")
    return global_nn

def HierFAVG(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device selection (robust)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device(f"cuda:{args.gpu}")
        # cuDNN determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    print(f'Using device {device}')

    FILEOUT = (
        f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_"
        f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}"
        f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}"
        f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}"
        f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
    )
    writer = SummaryWriter(comment=FILEOUT)

    # build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print(f"train dataloader {i} distribution")
            print(distribution)
        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader, args)
            print(f"test dataloader {i} distribution (size {test_size})")
            print(distribution)

    # initialize clients
    clients = [
        Client(id=i,
               train_loader=train_loaders[i],
               test_loader=test_loaders[i],
               args=args,
               device=device)
        for i in range(args.num_clients)
    ]

    # broadcast initial shared weights
    init_params = list(clients[0].model.shared_layers.parameters())
    nc = len(init_params)
    for client in clients:
        user_params = list(client.model.shared_layers.parameters())
        for j in range(nc):
            user_params[j].data[:] = init_params[j].data[:]

    # initialize edges and assign clients
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)

    if args.iid == -2:
        client_class_dis = get_client_class(args, clients)
        if args.edgeiid == 1:
            edges = initialize_edges_iid(num_edges=args.num_edges,
                                         clients=clients,
                                         args=args,
                                         client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            edges = initialize_edges_niid(num_edges=args.num_edges,
                                          clients=clients,
                                          args=args,
                                          client_class_dis=client_class_dis)
        else:
            raise ValueError("--edgeiid must be 0 or 1 when --iid == -2")
    else:
        # random assignment
        for i in range(args.num_edges):
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list(set(cids) - set(selected_cids))
            edge = Edge(id=i,
                        cids=selected_cids,
                        shared_layers=copy.deepcopy(clients[0].model.shared_layers))
            [edge.client_register(clients[cid]) for cid in selected_cids]
            edge.all_trainsample_num = sum(edge.sample_registration.values())
            edge.refresh_edgeserver()
            edges.append(edge)

    # initialize cloud
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    cloud.refresh_cloudserver()

    # NN for global validation
    global_nn = initialize_global_nn(args).to(device)

    # training
    selected_enum = max(int(args.num_edges * args.efrac), 1)
    selected_cnum = max(int(clients_per_edge * args.cfrac), 1)

    for num_comm in tqdm(range(args.num_communication)):
        print(f'num_comm: {num_comm}')
        cloud.refresh_cloudserver()
        edge_sample_cum = [0] * args.num_edges
        edge_sample_avg = [0] * args.num_edges

        # edge selection
        selected_eids = np.sort(np.random.choice(args.num_edges, selected_enum, replace=False))
        selected_edges = [edges[eid] for eid in selected_eids]

        training_loss = [0.0] * args.num_edge_aggregation

        for num_edgeagg in range(args.num_edge_aggregation):
            print(f'num_edgeagg: {num_edgeagg}')
            edge_loss = [0.0] * args.num_edges
            edge_sample = [0] * args.num_edges
            correct_all = 0.0
            total_all = 0.0

            for edge in selected_edges:
                edge.refresh_edgeserver()
                client_losses = []

                # build sampling weights on-the-fly aligned to edge.cids
                weights = np.array([len(clients[cid].train_loader.dataset) for cid in edge.cids], dtype=float)
                weights = weights / weights.sum() if weights.sum() > 0 else None

                assert selected_cnum <= len(edge.cids), \
                    f"selected_cnum ({selected_cnum}) > clients on edge {edge.id} ({len(edge.cids)})"

                selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace=False,
                                                 p=weights)

                # register & train selected clients
                for selected_cid in selected_cids:
                    edge.client_register(clients[selected_cid])

                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()

                    client_loss = clients[selected_cid].local_update(
                        num_iter=args.num_local_update,
                        device=device,
                        mode=args.mode,
                        clip=args.clip,
                        sigma=args.sigma
                    )
                    client_losses.append(client_loss)
                    clients[selected_cid].send_to_edgeserver(edge)

                # edge-specific weighted loss (aligns with registration order)
                reg_values = list(edge.sample_registration.values())
                edge_loss[edge.id] = sum([l * s for l, s in zip(client_losses, reg_values)]) / max(1, sum(reg_values))
                edge_sample[edge.id] = sum(reg_values)

                # aggregate at edge and evaluate (edge-wide)
                edge.aggregate(args)
                correct, total = all_clients_test(edge, clients, edge.cids, device)
                correct_all += correct
                total_all += total

            # end per-edge iteration
            edge_sample_cum = [a + b for a, b in zip(edge_sample_cum, edge_sample)]
            all_loss = sum(edge_loss[eid] for eid in selected_eids) / max(1, len(selected_eids))
            training_loss[num_edgeagg] = all_loss
            avg_acc = (correct_all / total_all) if total_all > 0 else 0.0

            writer.add_scalar('Partial_Avg_Train_loss',
                              all_loss,
                              num_comm * args.num_edge_aggregation + num_edgeagg + 1)
            writer.add_scalar('All_Avg_Test_Acc_edgeagg',
                              avg_acc,
                              num_comm * args.num_edge_aggregation + num_edgeagg + 1)

        # update edge sample counts (averaged over t2 rounds; keep float)
        edge_sample_avg = [x / float(args.num_edge_aggregation) for x in edge_sample_cum]
        for i in range(args.num_edges):
            edges[i].all_trainsample_num = edge_sample_avg[i]

        # register edges at cloud
        for edge in edges:
            cloud.edge_register(edge=edge)

        # send only selected edges to cloud for this round
        for edge in selected_edges:
            edge.send_to_cloudserver(cloud)

        # cloud aggregate & broadcast
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        # global validation
        global_nn.load_state_dict(state_dict=copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        avg_acc_v = (correct_all_v / total_all_v) if total_all_v > 0 else 0.0
        avg_train_loss = round((sum(training_loss) / max(1, args.num_edge_aggregation)), 4)
        print(f'avg_loss_a: {avg_train_loss} and avg_acc_v: {avg_acc_v}')

        writer.add_scalar('All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)

    writer.close()
    print(f"The final training loss is {avg_train_loss} and final test acc is {avg_acc_v}")

def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()