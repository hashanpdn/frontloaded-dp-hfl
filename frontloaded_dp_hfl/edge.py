"""
Edge node for hierarchical FL.

Receives client updates (full weights for baseline/CG-*, or noisy deltas for CP-NP),
aggregates them with sample-size weighting, and exchanges models with clients/cloud.
"""

import copy
from average import average_weights, average_weights_edge


class Edge:
    def __init__(self, id, cids, shared_layers):
        self.id = id
        self.cids = cids

        # Per-round buffers
        self.receiver_buffer = {}         # client_id -> full weights (baseline/CG-*)
        self.params_receiver_buffer = {}  # client_id -> deltas (CP-NP)

        # Participation bookkeeping
        self.id_registration = []         # ordered client IDs (this round)
        self.sample_registration = {}     # client_id -> sample count (this round)

        self.all_trainsample_num = 0
        self.shared_layers = shared_layers
        self.shared_state_dict = copy.deepcopy(shared_layers.state_dict())
        self.clock = []

    def refresh_edgeserver(self):
        """Clear per-round buffers and registrations."""
        self.receiver_buffer.clear()
        self.params_receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        """Register a participating client and record its sample count."""
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict=None, cparams_state_dict=None):
        """Store either full weights (baseline/CG-*) or a delta (CP-NP)."""
        if cparams_state_dict is not None:
            self.params_receiver_buffer[client_id] = cparams_state_dict
        elif cshared_state_dict is not None:
            self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def aggregate(self, args):
        """
        Weighted aggregation.
          - baseline / CG-NG / CG-NP: FedAvg on full weights.
          - CP-NP: apply averaged deltas to current model (optionally damped by args.eta).
        """
        if args.mode in ("baseline", "CG-NG", "CG-NP"):
            cids = [cid for cid in self.id_registration if cid in self.receiver_buffer]
            if not cids:
                raise ValueError(f"[Edge {self.id}] No full-weight updates for mode {args.mode}.")
            s_num = [self.sample_registration[cid] for cid in cids]
            w_list = [self.receiver_buffer[cid] for cid in cids]
            self.shared_state_dict = average_weights(w=w_list, s_num=s_num)

        elif args.mode == "CP-NP":
            cids = [cid for cid in self.id_registration if cid in self.params_receiver_buffer]
            if not cids:
                raise ValueError(f"[Edge {self.id}] No delta updates for CP-NP.")
            s_num = [self.sample_registration[cid] for cid in cids]
            d_list = [self.params_receiver_buffer[cid] for cid in cids]
            self.shared_state_dict = average_weights_edge(
                e_model=self.shared_state_dict,
                w=d_list,
                s_num=s_num,
                eta=getattr(args, "eta", 1.0),
            )
        else:
            raise ValueError(f"Invalid mode '{args.mode}'. Expected: baseline, CG-NG, CG-NP, CP-NP.")

    def send_to_client(self, client):
        """Push current edge model to a client."""
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        """Push current edge model to the cloud."""
        cloud.receive_from_edge(edge_id=self.id, eshared_state_dict=copy.deepcopy(self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        """Update edge model with cloud-aggregated state."""
        self.shared_state_dict = shared_state_dict
        return None
