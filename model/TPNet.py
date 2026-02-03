import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import math

from losses import Huber2DLoss
from losses import CELoss
from metrics import BrierMinFDE
from metrics import MR
from metrics import MinADE
from metrics import MinFDE
from modules import Backbone
from modules import MapEncoder

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_vehicle_endpoints


class TPNet(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 front_loss_weight: float = 1.0,
                 **kwargs) -> None:
        super(TPNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.front_loss_weight = front_loss_weight

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pos_duration=pos_duration,
            pred_duration=pred_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()

        self.brier_minFDE = BrierMinFDE()
        self.minADE = MinADE()
        self.minFDE = MinFDE()
        self.MR = MR()

        self.test_traj_output = dict()
        self.test_prob_output = dict()
        self.test_front_output = dict()

    def forward(self, data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def _generate_front_target(self, data: Batch, target_traj: torch.Tensor, target_mask: torch.Tensor):
        """
        Generate front midpoint target trajectories.
        
        Args:
            data: Batch data
            target_traj: Center point target trajectory [(N1,...Nb),H,F,2]
            target_mask: Target mask [(N1,...Nb),H,F]
        
        Returns:
            front_target_traj: Front midpoint target trajectory [(N1,...Nb),H,F,2]
        """
        # Get vehicle length and heading for future steps
        vehicle_length = data['agent']['vehicle_length']  # [N]
        heading_full = data['agent']['heading_full']  # [N, num_steps]
        
        num_agents = target_traj.size(0)
        num_historical = self.num_historical_steps
        num_future = self.num_future_steps
        
        # Initialize front target
        front_target_traj = torch.zeros_like(target_traj)
        
        # For each historical step h, the target is future positions from h+1 to h+num_future
        for h in range(num_historical):
            future_start = h + 1
            future_end = h + 1 + num_future
            
            if future_end <= heading_full.size(1):
                future_headings = heading_full[:, future_start:future_end]  # [N, F]
            else:
                # Pad with last known heading
                available = heading_full.size(1) - future_start
                if available > 0:
                    future_headings = torch.zeros(num_agents, num_future, device=heading_full.device)
                    future_headings[:, :available] = heading_full[:, future_start:]
                    future_headings[:, available:] = heading_full[:, -1:].expand(-1, num_future - available)
                else:
                    future_headings = heading_full[:, -1:].expand(-1, num_future)
            
            # Compute front midpoint: center + (length/2) * direction
            half_length = (vehicle_length / 2.0).unsqueeze(-1)  # [N, 1]
            
            for f in range(num_future):
                cos_h = torch.cos(future_headings[:, f])
                sin_h = torch.sin(future_headings[:, f])
                
                front_target_traj[:, h, f, 0] = target_traj[:, h, f, 0] + half_length.squeeze(-1) * cos_h
                front_target_traj[:, h, f, 1] = target_traj[:, h, f, 1] + half_length.squeeze(-1) * sin_h
        
        return front_target_traj

    def training_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output, front_propose, front_output = self(data)
        
        target_traj, target_mask = generate_target(
            position=data['agent']['position'],
            mask=data['agent']['visible_mask'],
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps
        )  # [(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        # Generate front midpoint targets
        front_target_traj = self._generate_front_target(data, target_traj, target_mask)

        # Center trajectory errors for mode selection
        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)
        best_mode_index = errors.argmin(dim=-1)
        
        # Get best mode trajectories for center
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]
        
        # Get best mode trajectories for front
        front_best_propose = front_propose[torch.arange(front_propose.size(0))[:, None], torch.arange(front_propose.size(1))[None, :], best_mode_index]
        front_best_output = front_output[torch.arange(front_output.size(0))[:, None], torch.arange(front_output.size(1))[None, :], best_mode_index]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps], self.num_visible_steps)
        targ_mask = target_mask[predict_mask]
        
        # Center trajectory
        traj_pro = traj_best_propose[predict_mask]
        traj_ref = traj_best_output[predict_mask]
        targ = target_traj[predict_mask]
        
        # Front midpoint trajectory
        front_pro = front_best_propose[predict_mask]
        front_ref = front_best_output[predict_mask]
        front_targ = front_target_traj[predict_mask]
        
        prob = prob_output[predict_mask]
        label = best_mode_index[predict_mask]

        # Center losses
        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        
        # Front midpoint losses
        front_reg_loss_propose = self.reg_loss(front_pro[targ_mask], front_targ[targ_mask])
        front_reg_loss_refine = self.reg_loss(front_ref[targ_mask], front_targ[targ_mask])
        
        # Probability loss
        prob_loss = self.prob_loss(prob, label)
        
        # Combined loss with adjustable front loss weight
        loss = (reg_loss_propose + reg_loss_refine + 
                self.front_loss_weight * (front_reg_loss_propose + front_reg_loss_refine) + 
                prob_loss)
        
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_front_loss_propose', front_reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_front_loss_refine', front_reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_prob_loss', prob_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output, front_propose, front_output = self(data)
        
        target_traj, target_mask = generate_target(
            position=data['agent']['position'],
            mask=data['agent']['visible_mask'],
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps
        )

        # Generate front midpoint targets
        front_target_traj = self._generate_front_target(data, target_traj, target_mask)

        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)
        best_mode_index = errors.argmin(dim=-1)
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]
        
        front_best_propose = front_propose[torch.arange(front_propose.size(0))[:, None], torch.arange(front_propose.size(1))[None, :], best_mode_index]
        front_best_output = front_output[torch.arange(front_output.size(0))[:, None], torch.arange(front_output.size(1))[None, :], best_mode_index]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps], self.num_visible_steps)
        targ_mask = target_mask[predict_mask]
        traj_pro = traj_best_propose[predict_mask]
        traj_ref = traj_best_output[predict_mask]
        front_pro = front_best_propose[predict_mask]
        front_ref = front_best_output[predict_mask]
        prob = prob_output[predict_mask]
        targ = target_traj[predict_mask]
        front_targ = front_target_traj[predict_mask]
        label = best_mode_index[predict_mask]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        front_reg_loss_propose = self.reg_loss(front_pro[targ_mask], front_targ[targ_mask])
        front_reg_loss_refine = self.reg_loss(front_ref[targ_mask], front_targ[targ_mask])
        prob_loss = self.prob_loss(prob, label)
        
        loss = (reg_loss_propose + reg_loss_refine + 
                self.front_loss_weight * (front_reg_loss_propose + front_reg_loss_refine) + 
                prob_loss)
        
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_front_loss_propose', front_reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_front_loss_refine', front_reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_prob_loss', prob_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        # Validation metrics (unchanged - based on center point)
        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
        agent_prob = prob_output[agent_index, -1]  # [N,K]
        agent_targ = target_traj[agent_index, -1]  # [N,F,2]
        fde = torch.norm(agent_traj[:, :, -1, :2] - agent_targ[:, -1, :2].unsqueeze(1), p=2, dim=-1)  # [N,K]
        best_mode_index = fde.argmin(dim=-1)  # [N]
        agent_traj_best = agent_traj[torch.arange(num_agents), best_mode_index]  # [N,F,2]
        self.brier_minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2], agent_prob[torch.arange(num_agents), best_mode_index])
        self.minADE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.MR.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_brier_minFDE', self.brier_minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)

    def test_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output, front_propose, front_output = self(data)
        
        prob_output = prob_output ** 2
        prob_output = prob_output / prob_output.sum(dim=-1, keepdim=True)

        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
        agent_front = front_output[agent_index, -1]  # [N,K,F,2]
        agent_prob = prob_output[agent_index, -1]  # [N,K]

        for i in range(num_agents):
            scenario_name = data['scenario_name'][i]
            case_id = data['case_id'][i].item()
            id_key = f"{scenario_name}_{case_id}"
            traj = agent_traj[i].cpu().numpy()
            front = agent_front[i].cpu().numpy()
            prob = agent_prob[i].tolist()

            self.test_traj_output[id_key] = traj
            self.test_front_output[id_key] = front
            self.test_prob_output[id_key] = prob

    def on_test_end(self):
        import os
        import numpy as np
        output_path = './test_output'
        os.makedirs(output_path, exist_ok=True)
        
        # Save predictions
        for key, traj in self.test_traj_output.items():
            np.save(os.path.join(output_path, f'{key}_traj.npy'), traj)
            np.save(os.path.join(output_path, f'{key}_front.npy'), self.test_front_output[key])
            np.save(os.path.join(output_path, f'{key}_prob.npy'), np.array(self.test_prob_output[key]))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('TPNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=20)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--pos_duration', type=int, default=20)
        parser.add_argument('--pred_duration', type=int, default=20)
        parser.add_argument('--a2a_radius', type=float, default=80)  # Updated default
        parser.add_argument('--l2a_radius', type=float, default=80)  # Updated default
        parser.add_argument('--num_visible_steps', type=int, default=2)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=3)  # Updated default
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--front_loss_weight', type=float, default=1.0)  # New parameter
        return parent_parser
