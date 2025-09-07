"""
Cloud aggregator for hierarchical FL.

Collects per-edge models (full weights), performs weighted FedAvg using each
edge’s total sample count, and broadcasts the aggregated global model back
to all edges.
"""

import copy
from average import average_weights


class Cloud:
    def __init__(self, shared_layers):
        self.receiver_buffer = {}       # edge_id -> state_dict (full weights)
        self.shared_state_dict = copy.deepcopy(shared_layers.state_dict())
        self.id_registration = []       # ordered participating edge_ids (this round)
        self.sample_registration = {}   # edge_id -> total train samples (this round)
        self.clock = []

    def refresh_cloudserver(self):
        """Clear per-round buffers and registrations."""
        self.receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        """Register an edge and record its current total sample count."""
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        """Store a snapshot of the edge’s model weights for aggregation."""
        self.receiver_buffer[edge_id] = copy.deepcopy(eshared_state_dict)
        return None

    def aggregate(self, args):
        """FedAvg across registered edges weighted by their sample counts."""
        edge_ids = [eid for eid in self.id_registration if eid in self.receiver_buffer]
        if not edge_ids:
            raise ValueError("[Cloud] No edge updates received this round.")

        received_dict = [self.receiver_buffer[eid] for eid in edge_ids]
        sample_num    = [self.sample_registration[eid] for eid in edge_ids]
        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        """Broadcast the aggregated global model to an edge."""
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None
