import torch
import numpy as np
from torch.optim import Optimizer
from torch.distributions import Bernoulli, Normal
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from collections import defaultdict
import math


class SGD(Optimizer):
    def __init__(self, params, lr, mu=0, nesterov=False, weight_decay=0, lrd=1):
        defaults = {'lr': lr, 'mu': mu, 'nesterov': nesterov, 'weight_decay': weight_decay, 'lrd': lrd}
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            if mu != 0 and 'v' not in group:
                group['v'] = []
                if nesterov:
                    group['theta'] = []
                for param in group['params']:
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        theta_param = torch.ones_like(param).mul_(param.data)
                        group['theta'].append(theta_param)
            for idx, param in enumerate(group['params']):
                param.grad.data -= weight_decay * param.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)
                if mu != 0:
                    v = group['v'][idx]
                    v = mu * v - lr * param.grad.data
                    group['v'][idx] = v
                    if nesterov:
                        group['theta'][idx] += lrd_mask * v
                        param.data = group['theta'][idx] + mu * v
                    else:
                        param.data += lrd_mask * v
                else:
                    param.data -= lrd_mask * lr * param.grad.data


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, nesterov=False, l2_reg=0, weight_decay=0, rectified=False, lrd=1, eps=1e-8):
        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'nesterov': nesterov, 'l2_reg': l2_reg,
                    'weight_decay': weight_decay, 'rectified': rectified, 'lrd': lrd, 'eps': eps}
        super(Adam, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            nesterov = group['nesterov']
            l2_reg = group['l2_reg']
            weight_decay = group['weight_decay']
            rectified = group['rectified']
            lrd_bernoulli = Bernoulli(probs=group['lrd'])
            eps = group['eps']
            if 'm' not in group and 'v' not in group:
                group['m'] = []
                group['v'] = []
                group['t'] = 1
                if nesterov:
                    group['prev_grad'] = []
                for param in group['params']:
                    group['m'].append(torch.zeros_like(param))
                    group['v'].append(torch.zeros_like(param))
                    if nesterov:
                        group['prev_grad'].append(torch.zeros_like(param))
            for idx, param in enumerate(group['params']):
                if l2_reg:
                    param.grad.data += l2_reg * param.data
                if nesterov:
                    grad = group['prev_grad'][idx]
                else:
                    grad = param.grad.data
                lrd_mask = lrd_bernoulli.sample(param.size()).to(param.device)
                m = group['m'][idx]
                v = group['v'][idx]
                t = group['t']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                if nesterov:
                    group['prev_grad'][idx] = param.grad.data
                if rectified:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho = rho_inf - 2 * t * beta2**t / (1 - beta2**t)
                    if rho >= 5:
                        numerator = (1 - beta2**t) * (rho - 4) * (rho - 2) * rho_inf
                        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
                        r = np.sqrt(numerator / denominator)
                        param.data += - lrd_mask * lr * r * m_hat / (torch.sqrt(v) + eps)
                    else:
                        param.data += - lrd_mask * lr * m_hat
                else:
                    param.data += - lrd_mask * lr * m_hat / (torch.sqrt(v_hat) + eps)
                if weight_decay:
                    param.data -= weight_decay * param.data
                group['m'][idx] = m
                group['v'][idx] = v
            group['t'] += 1


class RMSProp(Adam):
    def __init__(self, params, lr, beta2):
        super(RMSProp, self).__init__(params, lr, beta2=beta2, beta1=0)


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups

        self.counter = 0
        for group in optimizer.param_groups:
            group['phi'] = []
            for param in group['params']:
                phi_param = torch.ones_like(param).mul_(param.data)
                group['phi'].append(phi_param)
    def step(self):
        if self.counter == self.k:
            for group_idx, group in enumerate(self.param_groups):
                for idx, _ in enumerate(group['phi']):
                    theta = self.optimizer.param_groups[group_idx]['params'][idx].data
                    group['phi'][idx] = group['phi'][idx] + self.alpha * (theta - group['phi'][idx])
            self.counter = 0
        else:
            self.counter += 1
            self.optimizer.step()


class GradientNoise(Optimizer):
    def __init__(self, optimizer, eta=0.3, gamma=0.55):
        self.optimizer = optimizer
        self.eta = eta
        self.gamma = gamma
        self.t = 0
        self.param_groups = optimizer.param_groups
    def step(self):
        normal = torch.empty(1).normal_(mean=0, std=np.sqrt(self.eta/((1+self.t)**self.gamma)))\
            .to(self.optimizer.param_groups[0]['params'][0].device)
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data += normal
                self.optimizer.step()
                self.t += 1


class GradientDropout(Optimizer):
    def __init__(self, optimizer, grad_retain=0.9):
        self.optimizer = optimizer
        self.grad_retain = grad_retain
        self.grad_bernoulli = Bernoulli(probs=grad_retain)
        self.param_groups = optimizer.param_groups
    def step(self):
        for group_idx, group in enumerate(self.param_groups):
            for idx, param in enumerate(group['params']):
                grad_mask = self.grad_bernoulli.sample(param.size()).to(param.device)
                self.optimizer.param_groups[group_idx]['params'][idx].grad.data *= grad_mask
                self.optimizer.step()


class HessianFree(Optimizer):
    def __init__(self, params,lr=1,damping=0.5,delta_decay=0.95,cg_max_iter=100,use_gnm=True,verbose=False):
                self._params = self.param_groups[0]['params']
    def _gather_flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)
    def step(self, closure, b=None, M_inv=None):
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        cg_max_iter = group['cg_max_iter']
        damping = group['damping']
        use_gnm = group['use_gnm']
        verbose = group['verbose']
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        loss_before, output = closure()
        current_evals = 1
        state['func_evals'] += 1
        flat_params = parameters_to_vector(self._params)
        flat_grad = self._gather_flat_grad()
        if use_gnm:
            def A(x):
                return self._Gv(loss_before, output, x, damping)
        else:
            def A(x):
                return self._Hv(flat_grad, x, damping)
        if M_inv is not None:
            m_inv = M_inv()
            if m_inv.dim() == 1:
                m = (m_inv + damping) ** (-0.85)

                def M(x):
                    return m * x
            else:
                m = torch.inverse(m_inv + damping * torch.eye(*m_inv.shape))

                def M(x):
                    return m @ x
        else:
            M = None
        b = flat_grad.detach() if b is None else b().detach().flatten()
        if state.get('init_delta') is not None:
            init_delta = delta_decay * state.get('init_delta')
        else:
            init_delta = torch.zeros_like(flat_params)
        eps = torch.finfo(b.dtype).eps
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,M=M, max_iter=cg_max_iter,tol=1e1 * eps, eps=eps, martens=True)
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]
        vector_to_parameters(flat_params + delta, self._params)
        loss_now = closure()[0]
        current_evals += 1
        state['func_evals'] += 1
        if verbose:
            print("Loss before CG: {}".format(float(loss_before)))
            print("Loss before BT: {}".format(float(loss_now)))
        for (d, m) in zip(reversed(deltas[:-1][::2]), reversed(Ms[:-1][::2])):
            vector_to_parameters(flat_params + d, self._params)
            loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            M = m
            loss_now = loss_prev
        if verbose:
            print("Loss after BT:  {}".format(float(loss_now)))
        reduction_ratio = (float(loss_now)-float(loss_before)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3
        if reduction_ratio < 0:
            group['init_delta'] = 0
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)
        for _ in range(60):
            if float(loss_now) <= float(loss_before) + alpha * min_improv:
                break
            alpha *= beta
            vector_to_parameters(flat_params + alpha * delta, self._params)
            loss_now = closure()[0]
        else:
            alpha = 0.0
            loss_now = loss_before
        vector_to_parameters(flat_params + alpha * delta, self._params)
        return loss_now

    def _CG(self, A, b, x0, M=None, max_iter=50, tol=1.2e-6, eps=1.2e-7,
            martens=False):
        x = [x0]
        r = A(x[0]) - b
        if M is not None:
            y = M(r)
            p = -y
        else:
            p = -r
        res_i_norm = r @ r
        if martens:
            m = [0.5 * (r - b) @ x0]
        for i in range(max_iter):
            Ap = A(p)
            alpha = res_i_norm / ((p @ Ap) + eps)
            x.append(x[i] + alpha * p)
            r = r + alpha * Ap
            if M is not None:
                y = M(r)
                res_ip1_norm = y @ r
            else:
                res_ip1_norm = r @ r
            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm
            if martens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])
                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break
            if res_i_norm < tol or torch.isnan(res_i_norm):
                break
            if M is not None:
                p = - y + beta * p
            else:
                p = - r + beta * p
        return (x, m) if martens else (x, None)

    def _Hv(self, gradient, vec, damping):
        Hv = self._Rop(gradient, self._params, vec)
        return Hv.detach() + damping * vec

    def _Gv(self, loss, output, vec, damping):
        Jv = self._Rop(output, self._params, vec)
        gradient = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._Rop(gradient, output, Jv)

        JHJv = torch.autograd.grad(
            output, self._params, grad_outputs=HJv.reshape_as(output), retain_graph=True)

        return parameters_to_vector(JHJv).detach() + damping * vec
    @staticmethod
    def _Rop(y, x, v, create_graph=False):
        if isinstance(y, tuple):
            ws = [torch.zeros_like(y_i, requires_grad=True) for y_i in y]
        else:
            ws = torch.zeros_like(y, requires_grad=True)
        jacobian = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True)

        Jv = torch.autograd.grad(parameters_to_vector(
            jacobian), ws, grad_outputs=v, create_graph=create_graph)
        return parameters_to_vector(Jv)


class GAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), optimism=0.0, avg_sq_mode='weight',amsgrad_decay=1, weight_decay=0, l1_decay=0, late_weight_decay=True, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, optimism=optimism, amsgrad_decay=amsgrad_decay,weight_decay=weight_decay, l1_decay=l1_decay, late_weight_decay=late_weight_decay, eps=eps)
        self.avg_sq_mode = avg_sq_mode
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        if self.avg_sq_mode == 'global':
            exp_avg_sq_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                amsgrad_decay = group['amsgrad_decay']
                amsgrad = amsgrad_decay != 1
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['prev_shift'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq * (1 - amsgrad_decay), exp_avg_sq, out=max_exp_avg_sq)
                    if self.avg_sq_mode == 'global':
                        exp_avg_sq_list.append(max_exp_avg_sq.mean())
                else:
                    if self.avg_sq_mode == 'global':
                        exp_avg_sq_list.append(exp_avg_sq.mean())
        if self.avg_sq_mode == 'global':
            global_exp_avg_sq = np.mean(exp_avg_sq_list)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                amsgrad_decay = group['amsgrad_decay']
                amsgrad = amsgrad_decay != 1
                exp_avg = state['exp_avg']
                if self.avg_sq_mode == 'weight':
                    exp_avg_sq = state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq']
                elif self.avg_sq_mode == 'tensor':
                    exp_avg_sq = (state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq']).mean()
                elif self.avg_sq_mode == 'output':
                    exp_avg_sq = (state['max_exp_avg_sq'] if amsgrad else state['exp_avg_sq'])
                    exp_avg_sq = exp_avg_sq.view(exp_avg_sq.shape[0], -1).mean(-1)\
                        .view(exp_avg_sq.shape[0], *((exp_avg_sq.dim() - 1) * [1]))
                elif self.avg_sq_mode == 'global':
                    exp_avg_sq = global_exp_avg_sq

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg = exp_avg / bias_correction1
                exp_avg_sq = exp_avg_sq / bias_correction2
                if self.avg_sq_mode == 'weight' or self.avg_sq_mode == 'output':
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = math.sqrt(exp_avg_sq) + group['eps']
                late_weight_decay = group['late_weight_decay']
                if late_weight_decay:
                    exp_avg = exp_avg.div(denom)

                weight_decay = group['weight_decay']
                l1_decay = group['l1_decay']
                if weight_decay != 0:
                    exp_avg.add_(weight_decay, p.data)
                if l1_decay != 0:
                    exp_avg.add_(l1_decay, p.data.sign())

                if not late_weight_decay:
                    exp_avg = exp_avg.div(denom)

                lr = group['lr']
                optimism = group['optimism']
                if optimism != 0:
                    prev_shift = state['prev_shift']
                    p.data.sub_(optimism, prev_shift)
                    cur_shift = (-lr / (1 - optimism)) * exp_avg
                    prev_shift.copy_(cur_shift)
                    p.data.add_(cur_shift)
                else:
                    grad = exp_avg
                    p.data.add_(-lr, grad)
        return loss


class Padam(Optimizer):
    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, partial=1 / 4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, partial=partial)
        super(Padam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']
                partial = group['partial']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom ** (partial * 2))
        return loss

class AMSGrad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AMSGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AMSGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

class Adamax(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                bias_correction = 1 - beta1 ** state['step']
                clr = group['lr'] / bias_correction
                p.addcdiv_(exp_avg, exp_inf, value=-clr)
        return loss

class AdaBound(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,eps=1e-8, weight_decay=0, amsbound=False):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['amsbound']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                p.data.add_(-step_size)

        return loss

class AdaBoundW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['amsbound']

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)
        return loss