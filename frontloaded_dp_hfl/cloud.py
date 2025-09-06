import copy
from average import average_weights

class Cloud:
    def __init__(self, shared_layers):
        self.receiver_buffer = {}       # edge_id -> state_dict (full weights)
        self.shared_state_dict = copy.deepcopy(shared_layers.state_dict())
        self.id_registration = []       # ordered list of participating edge_ids (this round)
        self.sample_registration = {}   # edge_id -> total train samples for that edge (this round)
        self.clock = []

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        # Maintain deterministic order alignment
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        # Snapshot to avoid mutability issues
        self.receiver_buffer[edge_id] = copy.deepcopy(eshared_state_dict)
        return None

    def aggregate(self, args):
        # Build aligned lists using the registered order
        edge_ids = [eid for eid in self.id_registration if eid in self.receiver_buffer]
        if not edge_ids:
            raise ValueError("[Cloud] No edge updates received this round.")

        received_dict = [self.receiver_buffer[eid] for eid in edge_ids]
        sample_num    = [self.sample_registration[eid] for eid in edge_ids]

        self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None