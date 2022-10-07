import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import  os

from    learner import Learner
import    copy

class Meta_mini(nn.Module):
    def __init__(self, args, task_num, train_update_steps, test_update_steps, inner_lr, outer_lr, config, device):
        super(Meta_mini, self).__init__()

        self.update_lr = inner_lr
        self.meta_lr = outer_lr
        self.n_way = args.n_way
        self.k_spt = args.k_shot
        self.k_qry = args.k_qry
        self.task_num = task_num
        self.head = args.head
        self.update_steps = train_update_steps
        self.update_steps_test = test_update_steps

        self.net = Learner(config, 3, 84)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.device = device
        self.feature_norm = args.feature_norm
        
        self.IFR = args.IFR
        self.q_contrast = args.q_contrast
        
        self.order = args.order

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.train()
        
    def save_model(self, save_path):
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.meta_optim.state_dict(),
        }, save_path)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def set_last_layer_variance(self, model, var):
        for p1, p2 in model.named_parameters():
            if "vars.16" in p1 or "vars.17" in p1:
                p2.data = p2.data * var
              
    def anil_update(self, model, grad, lr):
        temp = list()
        for grad_, (p1, p2) in zip(grad, model.net.named_parameters()):
            if "vars.16" in p1 or "vars.17" in p1:
                temp.append(p2 - grad_*lr)
            else:
                temp.append(p2)
        return temp

    def forward_contrastive(self, x_spt, y_spt, x_qry, y_qry):

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        loss, loss_item = 0, 0

        for i in range(task_num):

            model = copy.deepcopy(self)
            logits = model.net(x_spt[i], vars=None, bn_training=True)
            loss_s = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss_s, model.net.parameters())
            with torch.no_grad():
                fast_weights = self.anil_update(model, grad, model.update_lr)
                logits_q = model.net(x_qry[i], fast_weights, bn_training=True).detach()

            if self.order == 1:
                with torch.no_grad():
                    logits_s, features_s = self.net.forward_feature(x_spt[i], vars=None, bn_training=True)
            elif self.order == 2:
                logits_s, features_s = self.net.forward_feature(x_spt[i], vars=None, bn_training=True)
            logits_q_zero, features_q = self.net.forward_feature(x_qry[i], vars=None, bn_training=True)
            
            if self.feature_norm == 1:
                features_s = torch.nn.functional.normalize(features_s, 2, 1)
                features_q = torch.nn.functional.normalize(features_q, 2, 1)

            # Contrastive term
            pred_q = torch.nn.functional.softmax(logits_q, 1)
            pred_s = torch.nn.functional.softmax(logits_s, 1)
            q_coeff = pred_q.T[y_spt[i]].detach()
            s_coeff = pred_s.T[y_qry[i]].detach().T

            A = y_spt[i].unsqueeze(1).repeat(1, self.k_qry*self.n_way)
            B = y_qry[i].unsqueeze(0).repeat(self.k_spt*self.n_way, 1)
            same_label = ((A==B)*1).detach()

            if self.q_contrast == 1 or self.head == "zero":
                coeffs = - (- q_coeff + same_label).detach() 
            else:
                coeffs = - ((pred_s@pred_q.T) - s_coeff - q_coeff + same_label).detach()
            Inner = (features_s@features_q.T)
            loss += (coeffs*Inner).mean() * self.update_lr / task_num
            loss_item += (coeffs*Inner).mean().item() * self.update_lr / task_num

            # Interference term for the encoder.
            if self.IFR == 0 or self.head == "zero":
                assert True
            else:
                W = self.net.vars[16].detach()
                A = torch.Tensor([0,1,2,3,4]).unsqueeze(1).repeat(1, self.k_qry*self.n_way).to(self.device)
                B = y_qry[i].unsqueeze(0).repeat(self.n_way, 1)
                C = ((A == B)*1).detach()
                Inner = (W@features_q.T)
                pred_q = torch.nn.functional.softmax(logits_q, 1)
                D =  pred_q.T.detach()
                loss += ((D-C)*Inner).mean()/task_num * self.n_way
                loss_item += ((D-C)*Inner).mean().item() /task_num * self.n_way

            # Outer loop update of linear layer
            if self.head == "zero":
                assert True
            elif self.order == 1:
                W1 = self.net.vars[16]
                q_coeff = pred_q.T[y_spt[i]].detach()
                A = y_qry[i].unsqueeze(1).repeat(1, self.n_way).T
                B = torch.Tensor([0,1,2,3,4]).unsqueeze(1).repeat(1, self.k_qry*self.n_way).to(self.device)
                same_label = ((A==B)*1).detach()
                if W1.grad is None:
                    W1.grad = self.update_lr * same_label @ features_q.detach() /task_num
                    print(W1.grad.data.sum())
                else:
                    W1.grad.data += self.update_lr * same_label @ features_q.detach() /task_num
            else:
                assert False

        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()

        return loss_item

    def forward_maml_one_step(self, x_spt, y_spt, x_qry, y_qry):
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = 0
        corrects = [0 for _ in range(self.update_steps + 1)]

        for i in range(task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            if self.order == 1:
                grad = torch.autograd.grad(loss, self.net.parameters())
            elif self.order == 2:
                grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph = True, create_graph=True)
            fast_weights = self.anil_update(self, grad, self.update_lr)
                
            logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            losses_q += F.cross_entropy(logits_q, y_qry[i]) / task_num
        loss_q = losses_q / task_num

        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs
    
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_steps_test + 1)]
        net = copy.deepcopy(self.net)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[0] += torch.eq(pred_q, y_qry).sum().item()

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            corrects[1] += torch.eq(pred_q, y_qry).sum().item()

        for k in range(1, self.update_steps_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                corrects[k + 1] += torch.eq(pred_q, y_qry).sum().item()
        del net
        accs = np.array(corrects) / querysz
        return accs

    def finetunning_zero(self, x_spt, y_spt, x_qry, y_qry):
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_steps_test + 1)]

        net = copy.deepcopy(self.net)
        
        self.set_last_layer_variance(net, 0)
        
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_steps_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
        del net
        accs = np.array(corrects) / querysz
        return accs
    
    def get_feature(self, x, y):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x.shape) == 4

        querysz = x.size(0)
        corrects = [0 for _ in range(self.update_steps_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = copy.deepcopy(self.net)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            features = net.get_feature(x, net.parameters(), bn_training=True)
            logits_q = net(x, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y).sum().item()
            corrects[0] = corrects[0] + correct
            
        return features, logits_q, pred_q, correct

def main():
    pass

if __name__ == '__main__':
    main()
