import os
import torch
import logging
import numpy as np
import torch.nn as nn
from genotypes import normal_candidates
from operations import FactorizedReduce, ReLUConvBN
from mixed_edge import MixedEdge


class Cell(nn.Module):
    def __init__(self, steps, multiplier, prev_prev_chn, prev_chn, curr_chn, reduction, prev_reduction, mode, affine):
        super(Cell, self).__init__()

        self.curr_chn = curr_chn

        # if prev_cell is reduction, feature map from prev_cell should be reduce to fit in
        if prev_reduction:
            self.preprocess0 = FactorizedReduce(prev_prev_chn, curr_chn, affine)
        else:
            self.preprocess0 = ReLUConvBN(prev_prev_chn, curr_chn, 1, 1, 0, affine)
        self.preprocess1 = ReLUConvBN(prev_chn, curr_chn, 1, 1, 0, affine)
        self._steps = steps
        self._multiplier = multiplier

        self.edges = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                self.edges.append(MixedEdge(curr_chn, stride, mode, normal_candidates, affine))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.edges[offset+j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, net_mode, data_mode, prune_direct, affine, skip_r_num, pool_r_num, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._steps = steps
        self.layers = layers
        self.skip_r_num = skip_r_num
        self.pool_r_num = pool_r_num

        curr_chn = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, curr_chn, 3, padding=1, bias=False),
            nn.BatchNorm2d(curr_chn)
        )

        prev_prev_chn, prev_chn, curr_chn = curr_chn, curr_chn, C
        self.cells = nn.ModuleList()
        prev_reduction = False
        for idx in range(layers):
            if idx in [layers//3, 2*layers//3]:
                curr_chn *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, prev_prev_chn, prev_chn, curr_chn, reduction, prev_reduction, net_mode, affine)
            self.cells.append(cell)
            prev_reduction = reduction
            prev_prev_chn, prev_chn = prev_chn, multiplier * curr_chn

        self.globalpooling = nn.AdaptiveAvgPool2d(1)

        if data_mode == 'supervised' or data_mode == 'byol':
            self.classifier = nn.Linear(prev_chn, num_classes)
        elif data_mode == 'SimCLR':
            self.classifier = nn.Sequential(nn.Linear(prev_chn, prev_chn), nn.ReLU(), nn.Linear(prev_chn, 128))
        else:
            NotImplementedError
        
        # pruning settings
        if prune_direct == 'forward':
            self.cell_prune_list = sorted(list(range(len(self.cells))), reverse=True)
        elif prune_direct == 'backward':
            self.cell_prune_list = list(range(len(self.cells)))

    def alpha_params(self):
        for name, param in self.named_parameters():
            if 'alpha' in name:
                yield param
    
    def weight_params(self):
        for name, param in self.named_parameters():
            if 'alpha' not in name and 'binary_gates' not in name:
                yield param
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.globalpooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def reset_binary_gates(self, skip_drop_prob, pool_drop_prob, warmup):
        for i, cell in enumerate(self.cells):
            for mixed_edge in cell.edges:
                mixed_edge.binarize(skip_drop_prob, pool_drop_prob, warmup)
    
    def unused_ops_off(self):
        self._unused_ops = []
        for cell in self.cells:
            cell_unused = []
            for edge in cell.edges:
                edge_unused = {}
                involved_index = edge.active_index + edge.inactive_index
                for i in range(edge.n_choices):
                    if i not in involved_index:
                        edge_unused[i] = edge._ops[i]
                        edge._ops[i] = None
                cell_unused.append(edge_unused)
            self._unused_ops.append(cell_unused)

    def unused_ops_back(self):
        for cell_idx, cell in enumerate(self.cells):
            cell_unused = self._unused_ops[cell_idx]
            for edge_idx, edge in enumerate(cell.edges):
                edge_unused = cell_unused[edge_idx]
                for i in edge_unused:
                    edge._ops[i] = edge_unused[i]
        self._unused_ops = None
    
    def set_arch_param_grad(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.set_arch_param_grad()
    
    def rescale_updated_arch_param(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.rescale_updated_arch_param()
    
    def set_chosen_op_active(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.set_chosen_op_active()
    
    def freeze_one_cell(self, idx):
        for cell_id, cell in enumerate(self.cells):
            if cell_id == idx:
                for edge in cell.edges:
                    edge.freeze()
    
    def progressive_prune(self, args, epoch):
        assert self.cell_prune_list is not None, print('prune direct is not defined')
        prune_epochs = np.linspace(args.warmup_epochs + args.search_epochs, args.warmup_epochs + args.search_epochs + args.prune_epochs, self.layers).astype(int).tolist()
        if epoch in prune_epochs:
            prune_id = self.cell_prune_list.pop()
            self.freeze_one_cell(prune_id)

    def final_prune(self, add_restrict):
        node2idx = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12, 13]}

        def _prune_one_cell(weight):
            cell_arch = []
            skip_num = 0
            pool_num = 0

            for node_id in range(4):
                node_weight = weight[node2idx[node_id]]
                probs, ops = torch.max(node_weight, dim=-1)
                
                node_list = []
                for edge_id, (prob, op) in enumerate(list(zip(probs, ops))):
                    node_list.append((op.item(), node_id, edge_id, prob.item()))

                for op, node_id, edge_id, prob in sorted(node_list, key=lambda x: x[-1], reverse=True)[:2]:
                    if op == 0:
                        skip_num += 1
                    if op == 1 or op == 2:
                        pool_num += 1
                    cell_arch.append((normal_candidates[op], node_id, edge_id, prob))
            return cell_arch, skip_num, pool_num
        
        def _drop_op(weight, cell_arch, ops_to_drop):
            for op_to_drop in ops_to_drop:
                _, drop_node_id, drop_edge_id, _ = op_to_drop
                drop_edge_range = node2idx[drop_node_id]
                node_weight = weight[drop_edge_range]
                
                for _, node_id, edge_id, _ in cell_arch:
                    if node_id == drop_node_id and edge_id != drop_edge_id:
                        node_weight[edge_id] = 0
                        
                node_weight[:, :3] = 0
                new_edge_id, new_op = map(lambda x: x.item() if len(x) == 1 else x[0].item(), torch.where(node_weight == node_weight.max()))
                cell_arch = [(normal_candidates[new_op], drop_node_id, new_edge_id, torch.max(node_weight).item()) if op == op_to_drop else op for op in cell_arch]
            return cell_arch

        model_arch = []
        for cell in self.cells:
            cell_alpha = []
            for edge in cell.edges:
                cell_alpha.append(edge.alpha.data)
            cell_alpha = torch.stack(cell_alpha, dim=0)
            weight = torch.softmax(cell_alpha, dim=-1).data.cpu()
            cell_arch, skip_num, pool_num = _prune_one_cell(weight)

            if add_restrict:
                if skip_num > self.skip_r_num:
                    skip_num_to_drop = skip_num - self.skip_r_num
                    skip_to_drop = sorted(cell_arch, key=lambda x: x[-1] if x[0]=='skip_connect' else 1)[:skip_num_to_drop]
                    cell_arch = _drop_op(weight, cell_arch, skip_to_drop)
                if pool_num > self.pool_r_num:
                    pool_num_to_drop = pool_num - self.pool_r_num
                    pool_to_drop = sorted(cell_arch, key=lambda x: x[-1] if x[0]=='max_pool_3x3' or x[0] == 'avg_pool_3x3' else 1)[:pool_num_to_drop]
                    cell_arch = _drop_op(weight, cell_arch, pool_to_drop)
            
            cell_arch = [(op[0], op[2]) for op in cell_arch]
            model_arch.append(cell_arch)
        return model_arch

    def get_cells_arch(self, epoch, add_restrict):
        model_arch = self.final_prune(add_restrict)

        if add_restrict:
            logging.info('restrict')
        else:
            logging.info('without restrict')

        for idx, arch in enumerate(model_arch):
            if add_restrict:
                logging.info('Epoch_{}_r_Cell{}:{}'.format(epoch, idx, arch))
            else:
                logging.info('Epoch_{}_wr_Cell{}:{}'.format(epoch, idx, arch))
    
    def save_arch_weights(self, save_path, note):
        if isinstance(note, int):
            with open(os.path.join(save_path, 'net.config'), 'a') as f:
                f.write('Epoch_{}:'.format(note) + str(self.final_prune(add_restrict=True)) + '\n')
            torch.save(self.state_dict(), os.path.join(save_path, 'Epoch_{}.pth.tar'.format(note)))
        elif isinstance(note, str):
            with open(os.path.join(save_path, 'net.config'), 'a') as f:
                f.write('{}:'.format(note) + str(self.final_prune(add_restrict=True)) + '\n')
            torch.save(self.state_dict(), os.path.join(save_path, '{}.pth.tar'.format(note)))
        else:
            NotImplementedError


class NetworkImageNet(nn.Module):
    def __init__(self, init_channel, num_classes, layers, net_mode, data_mode, prune_direct, affine, skip_r_num, pool_r_num, steps=4, multiplier=4, stem_multiplier=3):
        super(NetworkImageNet, self).__init__()
        self._steps = steps
        self.layers = layers
        self.skip_r_num = skip_r_num
        self.pool_r_num = pool_r_num

        C = init_channel
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        prev_prev_chn, prev_chn, curr_chn = C, C, C
        self.cells = nn.ModuleList()
        prev_reduction = True
        for idx in range(layers):
            if idx in [layers//3, 2*layers//3]:
                curr_chn *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, prev_prev_chn, prev_chn, curr_chn, reduction, prev_reduction, net_mode, affine)
            prev_reduction = reduction
            self.cells.append(cell)
            prev_prev_chn, prev_chn = prev_chn, multiplier * curr_chn

        self.globalpooling = nn.AdaptiveAvgPool2d(1)

        if data_mode == 'supervised' or data_mode == 'byol':
            self.classifier = nn.Linear(prev_chn, num_classes)
        elif data_mode == 'SimCLR':
            self.classifier = nn.Sequential(nn.Linear(prev_chn, prev_chn), nn.ReLU(), nn.Linear(prev_chn, 128))
        else:
            NotImplementedError
        
        # pruning settings
        if prune_direct == 'forward':
            self.cell_prune_list = sorted(list(range(len(self.cells))), reverse=True)
        elif prune_direct == 'backward':
            self.cell_prune_list = list(range(len(self.cells)))

    def alpha_params(self):
        for name, param in self.named_parameters():
            if 'alpha' in name:
                yield param
    
    def weight_params(self):
        for name, param in self.named_parameters():
            if 'alpha' not in name and 'binary_gates' not in name:
                yield param
    
    def forward(self, x):
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.globalpooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def reset_binary_gates(self, skip_drop_prob, pool_drop_prob, warmup):
        for i, cell in enumerate(self.cells):
            for mixed_edge in cell.edges:
                mixed_edge.binarize(skip_drop_prob, pool_drop_prob, warmup)
    
    def unused_ops_off(self):
        self._unused_ops = []
        for cell in self.cells:
            cell_unused = []
            for edge in cell.edges:
                edge_unused = {}
                involved_index = edge.active_index + edge.inactive_index
                for i in range(edge.n_choices):
                    if i not in involved_index:
                        edge_unused[i] = edge._ops[i]
                        edge._ops[i] = None
                cell_unused.append(edge_unused)
            self._unused_ops.append(cell_unused)

    def unused_ops_back(self):
        for cell_idx, cell in enumerate(self.cells):
            cell_unused = self._unused_ops[cell_idx]
            for edge_idx, edge in enumerate(cell.edges):
                edge_unused = cell_unused[edge_idx]
                for i in edge_unused:
                    edge._ops[i] = edge_unused[i]
        self._unused_ops = None
    
    def set_arch_param_grad(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.set_arch_param_grad()
    
    def rescale_updated_arch_param(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.rescale_updated_arch_param()
    
    def set_chosen_op_active(self):
        for cell in self.cells:
            for edge in cell.edges:
                edge.set_chosen_op_active()
    
    def freeze_one_cell(self, idx):
        for cell_id, cell in enumerate(self.cells):
            if cell_id == idx:
                for edge in cell.edges:
                    edge.freeze()
    
    def progressive_prune(self, args, epoch):
        assert self.cell_prune_list is not None, print('prune direct is not defined')
        prune_epochs = np.linspace(args.warmup_epochs + args.search_epochs, args.warmup_epochs + args.search_epochs + args.prune_epochs, self.layers).astype(int).tolist()
        if epoch in prune_epochs:
            prune_id = self.cell_prune_list.pop()
            self.freeze_one_cell(prune_id)

    def final_prune(self, add_restrict):
        node2idx = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12, 13]}

        def _prune_one_cell(weight):
            cell_arch = []
            skip_num = 0
            pool_num = 0

            for node_id in range(4):
                node_weight = weight[node2idx[node_id]]
                probs, ops = torch.max(node_weight, dim=-1)
                
                node_list = []
                for edge_id, (prob, op) in enumerate(list(zip(probs, ops))):
                    node_list.append((op.item(), node_id, edge_id, prob.item()))

                for op, node_id, edge_id, prob in sorted(node_list, key=lambda x: x[-1], reverse=True)[:2]:
                    if op == 0:
                        skip_num += 1
                    if op == 1 or op == 2:
                        pool_num += 1
                    cell_arch.append((normal_candidates[op], node_id, edge_id, prob))
            return cell_arch, skip_num, pool_num
        
        def _drop_op(weight, cell_arch, ops_to_drop):
            for op_to_drop in ops_to_drop:
                _, drop_node_id, drop_edge_id, _ = op_to_drop
                drop_edge_range = node2idx[drop_node_id]
                node_weight = weight[drop_edge_range]
                
                for _, node_id, edge_id, _ in cell_arch:
                    if node_id == drop_node_id and edge_id != drop_edge_id:
                        node_weight[edge_id] = 0
                        
                node_weight[:, :3] = 0
                new_edge_id, new_op = map(lambda x: x.item() if len(x) == 1 else x[0].item(), torch.where(node_weight == node_weight.max()))
                cell_arch = [(normal_candidates[new_op], drop_node_id, new_edge_id, torch.max(node_weight).item()) if op == op_to_drop else op for op in cell_arch]
            return cell_arch

        model_arch = []
        for cell in self.cells:
            cell_alpha = []
            for edge in cell.edges:
                cell_alpha.append(edge.alpha.data)
            cell_alpha = torch.stack(cell_alpha, dim=0)
            weight = torch.softmax(cell_alpha, dim=-1).data.cpu()
            cell_arch, skip_num, pool_num = _prune_one_cell(weight)

            if add_restrict:
                if skip_num > self.skip_r_num:
                    skip_num_to_drop = skip_num - self.skip_r_num
                    skip_to_drop = sorted(cell_arch, key=lambda x: x[-1] if x[0]=='skip_connect' else 1)[:skip_num_to_drop]
                    cell_arch = _drop_op(weight, cell_arch, skip_to_drop)
                if pool_num > self.pool_r_num:
                    pool_num_to_drop = pool_num - self.pool_r_num
                    pool_to_drop = sorted(cell_arch, key=lambda x: x[-1] if x[0]=='max_pool_3x3' or x[0] == 'avg_pool_3x3' else 1)[:pool_num_to_drop]
                    cell_arch = _drop_op(weight, cell_arch, pool_to_drop)
            
            cell_arch = [(op[0], op[2]) for op in cell_arch]
            model_arch.append(cell_arch)
        return model_arch

    def get_cells_arch(self, epoch, add_restrict):
        model_arch = self.final_prune(add_restrict)

        if add_restrict:
            logging.info('restrict')
        else:
            logging.info('without restrict')

        for idx, arch in enumerate(model_arch):
            if add_restrict:
                logging.info('Epoch_{}_r_Cell{}:{}'.format(epoch, idx, arch))
            else:
                logging.info('Epoch_{}_wr_Cell{}:{}'.format(epoch, idx, arch))
    
    def save_arch_weights(self, save_path, note):
        if isinstance(note, int):
            with open(os.path.join(save_path, 'net.config'), 'a') as f:
                f.write('Epoch_{}:'.format(note) + str(self.final_prune(add_restrict=True)) + '\n')
            torch.save(self.state_dict(), os.path.join(save_path, 'Epoch_{}.pth.tar'.format(note)))
        elif isinstance(note, str):
            with open(os.path.join(save_path, 'net.config'), 'a') as f:
                f.write('{}:'.format(note) + str(self.final_prune(add_restrict=True)) + '\n')
            torch.save(self.state_dict(), os.path.join(save_path, '{}.pth.tar'.format(note)))
        else:
            NotImplementedError
