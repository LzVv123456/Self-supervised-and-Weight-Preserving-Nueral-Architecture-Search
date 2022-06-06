import torch
import torch.nn as nn
from operations import *
from genotypes import normal_candidates
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, architecture, C_prev_prev, C_prev, C, reduction, reduction_prev, affine):
        super(Cell, self).__init__()
        concat = range(2, 6)
        self.curr_chn = C
        self.affine = affine

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine)
        op_names, indices = zip(*architecture)
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, self.affine)
            if 'pool' in name:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        if num_classes == 200:
            self.features = nn.Sequential(
                nn.Conv2d(C, 64, 3, padding=1, bias=False),
                nn.AvgPool2d(2, stride=2, padding=0, count_include_pad=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 1, bias=False),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Linear(768, num_classes)
        elif num_classes == 1000:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Linear(768, num_classes)
        else:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

class Network(nn.Module):

    def __init__(self, init_channel, num_classes, architecture, auxiliary, affine, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()

        layers = len(architecture)
        self.arch = architecture
        self.auxiliary = auxiliary
        curr_chn = stem_multiplier * init_channel
        self.stem = nn.Sequential(
            nn.Conv2d(3, curr_chn, 3, padding=1, bias=False),
            nn.BatchNorm2d(curr_chn)
        )
        prev_prev_chn, prev_chn, curr_chn = curr_chn, curr_chn, init_channel
        self.cells = nn.ModuleList()
        reduction_prev = False
        for idx in range(layers):
            if idx in [layers // 3, 2 * layers // 3]:
                curr_chn *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(architecture[idx], prev_prev_chn, prev_chn, curr_chn, reduction, reduction_prev, affine)
            reduction_prev = reduction
            self.cells.append(cell)
            prev_prev_chn, prev_chn = prev_chn, multiplier * curr_chn
            if idx == 2 * layers // 3:
                aux_chn = prev_chn

        if auxiliary:
            self.aux_head = AuxiliaryHead(aux_chn, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_chn, num_classes)

    def forward(self, input):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for idx, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if idx == 2 * len(self.cells) // 3:
                if self.auxiliary and self.training:
                    aux_logits = self.aux_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, aux_logits

    def load_weight(self, searched_dict):
        config_matrix = torch.zeros(len(self.arch), len(self.arch[0]), 2).int()
        for i in range(config_matrix.size(0)):
            for j in range(config_matrix.size(1)):
                
                if j < 2:
                    config_matrix[i][j][0] = self.arch[i][j][1]
                elif j < 4:
                    config_matrix[i][j][0] = self.arch[i][j][1] + 2
                elif j < 6:
                    config_matrix[i][j][0] = self.arch[i][j][1] + 5
                else:
                    config_matrix[i][j][0] = self.arch[i][j][1] + 9
                config_matrix[i][j][1] = normal_candidates.index(self.arch[i][j][0])
        config_matrix = config_matrix.tolist()

        model_dict = self.state_dict()
        corrected_dict = {}
        another_dict = {}

        for k, v in model_dict.items():
            if k in searched_dict.keys():
                corrected_dict[k] = searched_dict[k]
            elif 'aux' in k:
                corrected_dict[k] = v
            elif 'classifier' in k:
                corrected_dict[k] = v
            else:
                another_dict[k] = v
        
        cells_dict = {}
        for i in range(len(config_matrix)):
            for j in range(len(config_matrix[i])):

                edge_id, op_id = config_matrix[i][j]
                model_key = 'cells.{}._ops.{}'.format(i, j)
                searched_key = 'cells.{}.edges.{}._ops.{}'.format(i, edge_id, op_id)

                for k, v in searched_dict.items():
                    if searched_key in k:
                        new_key = k.replace(searched_key, model_key)
                        cells_dict[new_key] = v

        corrected_dict.update(cells_dict)

        for k, v in model_dict.items():
            if k not in corrected_dict.keys():
                corrected_dict[k] = v

        self.load_state_dict(corrected_dict)


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, architecture, auxiliary, affine):
        super(NetworkImageNet, self).__init__()
        self.auxiliary = auxiliary
        self._layers = len(architecture)
        self.arch = architecture
        self._auxiliary = auxiliary
        self.affine = affine

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

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(self._layers):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(self.arch[i], C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * self._layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes)
        
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
