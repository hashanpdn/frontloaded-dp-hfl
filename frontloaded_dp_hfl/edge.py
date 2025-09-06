import copy
from average import average_weights, average_weights_edge

class Edge:
    def __init__(self, id, cids, shared_layers):
        self.id = id
        self.cids = cids

        # Buffers (cleared each round)
        self.receiver_buffer = {}         # client_id -> state_dict (baseline / CG-*)
        self.params_receiver_buffer = {}  # client_id -> delta state_dict (CP-NP)

        # Participation bookkeeping
        self.id_registration = []         # ordered list of participating client_ids (this round)
        self.sample_registration = {}     # client_id -> train sample count (this round)

        self.all_trainsample_num = 0
        self.shared_layers = shared_layers
        self.shared_state_dict = copy.deepcopy(shared_layers.state_dict())
        self.clock = []

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.params_receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        # Maintain a deterministic order for aggregation alignment
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict=None, cparams_state_dict=None):
        """
        Accepts either full weights (baseline / CG-*) or a noisy delta (CP-NP).
        If both are provided, prefer the delta (privacy-safe) and ignore full weights.
        """
        if (cshared_state_dict is None) and (cparams_state_dict is None):
            # nothing to store; silently ignore or raise if you prefer
            return None

        if cparams_state_dict is not None:
            # CP-NP branch: store deltas only (privacy-safe)
            self.params_receiver_buffer[client_id] = cparams_state_dict
            # Do NOT store cshared_state_dict here
        else:
            # Baseline / CG-* branch: store full weights
            self.receiver_buffer[client_id] = cshared_state_dict
        return None

    # def aggregate(self, args):
    #     """
    #     Weighted aggregation using per-client sample counts.
    #     Baseline / CG-* : FedAvg on full weights via average_weights(...)
    #     CP-NP          : FedAvg on noisy deltas via average_weights_edge(...), applied to current model
    #     """
    #     # Build aligned lists (same client ordering) and filter to those who actually sent updates.
    #     cids = [
    #         cid for cid in self.id_registration
    #         if (args.mode == "CP-NP" and cid in self.params_receiver_buffer)
    #         or (args.mode in ("baseline", "CG-NG", "CG-NP") and cid in self.receiver_buffer)
    #     ]
    #     if not cids:
    #         raise ValueError(f"[Edge {self.id}] No client updates received for mode {args.mode}.")

    #     sample_num = [self.sample_registration[cid] for cid in cids]

    #     if args.mode in ("baseline", "CG-NG", "CG-NP"):
    #         received_list = [self.receiver_buffer[cid] for cid in cids]
    #         self.shared_state_dict = average_weights(w=received_list, s_num=sample_num)

    #     elif args.mode == "CP-NP":
    #         delta_list = [self.params_receiver_buffer[cid] for cid in cids]
    #         # Apply averaged delta to the current shared state
    #         self.shared_state_dict = average_weights_edge(
    #             e_model=copy.deepcopy(self.shared_state_dict),
    #             w=delta_list,
    #             s_num=sample_num
    #         )
    #     else:
    #         raise ValueError(f"Invalid mode '{args.mode}'. Expected one of: baseline, CG-NG, CG-NP, CP-NP.")
    
    def aggregate(self, args):
        """
        Weighted aggregation using per-client sample counts.

        - baseline / CG-NG / CG-NP: FedAvg on FULL WEIGHTS via average_weights(...)
        - CP-NP: FedAvg on NOISY DELTAS via average_weights_edge(...), applied to current edge model
                (optionally damped by args.eta; eta=1.0 reproduces FedAvg on deltas)
        """
        if args.mode in ("baseline", "CG-NG", "CG-NP"):
            # Keep only clients that actually sent full weights
            cids = [cid for cid in self.id_registration if cid in self.receiver_buffer]
            if not cids:
                raise ValueError(f"[Edge {self.id}] No full-weight updates received for mode {args.mode}.")
            sample_num    = [self.sample_registration[cid] for cid in cids]
            received_list = [self.receiver_buffer[cid] for cid in cids]
            self.shared_state_dict = average_weights(w=received_list, s_num=sample_num)

        elif args.mode == "CP-NP":
            # Keep only clients that actually sent deltas
            cids = [cid for cid in self.id_registration if cid in self.params_receiver_buffer]
            if not cids:
                raise ValueError(f"[Edge {self.id}] No delta updates received for CP-NP.")
            sample_num = [self.sample_registration[cid] for cid in cids]
            delta_list = [self.params_receiver_buffer[cid] for cid in cids]
            self.shared_state_dict = average_weights_edge(
                e_model=self.shared_state_dict,   # current edge model weights
                w=delta_list,                     # list of client deltas
                s_num=sample_num,                 # sample weights
                eta=getattr(args, "eta", 1.0)     # optional damping; defaults to 1.0
            )

        else:
            raise ValueError(f"Invalid mode '{args.mode}'. Expected one of: baseline, CG-NG, CG-NP, CP-NP.")
    
    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        cloud.receive_from_edge(
            edge_id=self.id,
            eshared_state_dict=copy.deepcopy(self.shared_state_dict)
        )
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None