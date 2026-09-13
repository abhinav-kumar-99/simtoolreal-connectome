import torch
import copy
from torch.utils.data import Dataset


class PPODataset(Dataset):

    def __init__(
        self,
        batch_size,
        minibatch_size,
        is_discrete,
        is_rnn,
        device,
        seq_length,
        logical_minibatch_size=None,
    ):

        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.logical_minibatch_size = logical_minibatch_size
        self.device = device
        self._refresh_slices(self.batch_size)
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        total_games = self.batch_size // self.seq_length
        self.game_indexes = torch.arange(total_games, dtype=torch.long, device=self.device)
        self.flat_indexes = torch.arange(total_games * self.seq_length, dtype=torch.long, device=self.device).reshape(total_games, self.seq_length)

        self.special_names = ['rnn_states']

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict
        if values_dict is not None and 'returns' in values_dict:
            self._refresh_slices(len(values_dict['returns']))

    def _refresh_slices(self, total_size):
        logical_size = self.logical_minibatch_size or self.minibatch_size
        logical_batches = total_size // logical_size
        if logical_batches < 1:
            raise ValueError(
                f"dataset size {total_size} is smaller than logical minibatch "
                f"size {logical_size}"
            )
        self.slices = []
        for logical_index in range(logical_batches):
            logical_start = logical_index * logical_size
            logical_end = (
                total_size
                if logical_index == logical_batches - 1
                else (logical_index + 1) * logical_size
            )
            logical_count = logical_end - logical_start
            micro_start = logical_start
            while micro_start < logical_end:
                micro_end = min(micro_start + self.minibatch_size, logical_end)
                if (micro_end - micro_start) % self.seq_length != 0:
                    raise ValueError(
                        "microbatch remainder must preserve complete sequences: "
                        f"{micro_end - micro_start} samples for seq_length "
                        f"{self.seq_length}"
                    )
                self.slices.append(
                    {
                        "start": micro_start,
                        "end": micro_end,
                        "zero_grad": micro_start == logical_start,
                        "optimizer_step": micro_end == logical_end,
                        "loss_scale": (micro_end - micro_start) / logical_count,
                    }
                )
                micro_start = micro_end
        self.length = len(self.slices)

    def update_mu_sigma(self, mu, sigma):	    
        start = self.last_range[0]	           
        end = self.last_range[1]	
        self.values_dict['mu'][start:end] = mu	
        self.values_dict['sigma'][start:end] = sigma 

    def __len__(self):
        return self.length

    def _get_item_rnn(self, idx):
        item_slice = self.slices[idx]
        start = item_slice["start"]
        end = item_slice["end"]
        gstart = start // self.seq_length
        gend = end // self.seq_length
        self.last_range = (start, end)
        self.last_zero_grad = item_slice["zero_grad"]
        self.last_optimizer_step = item_slice["optimizer_step"]
        self.last_loss_scale = item_slice["loss_scale"]

        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names:
                if isinstance(v, dict):
                    v_dict = {kd:vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    if v is not None:
                        input_dict[k] = v[start:end]
                    else:
                        input_dict[k] = None
        
        rnn_states = self.values_dict['rnn_states']
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in rnn_states]

        return input_dict

    def _get_item(self, idx):
        item_slice = self.slices[idx]
        start = item_slice["start"]
        end = item_slice["end"]
        self.last_range = (start, end)
        self.last_zero_grad = item_slice["zero_grad"]
        self.last_optimizer_step = item_slice["optimizer_step"]
        self.last_loss_scale = item_slice["loss_scale"]
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]
                
        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample



class DatasetList(Dataset):
    def __init__(self):
        self.dataset_list = []

    def __len__(self):
        return self.dataset_list[0].length * len(self.dataset_list)

    def add_dataset(self, dataset):
        self.dataset_list.append(copy.deepcopy(dataset))

    def clear(self):
        self.dataset_list = []

    def __getitem__(self, idx):
        ds_len = len(self.dataset_list)
        ds_idx = idx % ds_len
        in_idx = idx // ds_len
        return self.dataset_list[ds_idx].__getitem__(in_idx)
