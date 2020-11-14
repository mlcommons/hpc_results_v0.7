import torch
import os
import ast
from apex.multi_tensor_apply import multi_tensor_applier

num_step = 1


def UpdateIndex2LayerIndex(input):
    layer_num = 301
    flag = 0
    conv = [1,4,10,14,18,23,27,31,36,40,44,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102\
,106,110,114,118,122,126,130,134,138,142,146,150,154,158,162,166,170,174,178,182,186,190\
,194,198,202,206,210,214,218,222,226,230,234,241,245,249,251,252,255,256,259,260,263,266\
,269,272,278,281,290,291,293,294,296]
    if type(input) is not int:
        raise Exception("input value error!")
    if input > layer_num:
        return layer_num - 1
    if input < 1:
        return -1
    for num in conv:
        if input == num:
            return input - 1
        elif input > num:
            flag = num
            continue
        else:
            return flag - 1
    return flag - 1


class FusedLAMB(torch.optim.Optimizer):

    """Implements LAMB algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused LAMB implements 2 fusions.

      * Fusion of the LAMB update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedLAMB`'s usage is identical to any ordinary Pytorch optimizer::

        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedLAMB` may be used with or without Amp.  If you wish to use :class:`FusedLAMB` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        use_nvlamb (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 amsgrad=False, adam_w_mode=True,
                 grad_averaging=True, set_grad_none=True,
                 max_grad_norm=1.0, use_nvlamb=False):
        if amsgrad:
            raise RuntimeError('FusedLAMB does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)
        super(FusedLAMB, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_lamb = amp_C.multi_tensor_lamb
        else:
            raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions')

        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb
        set_skip_layer_num_str ="{" + os.getenv("SET_STOP_LAYER_NUM") + "}"
        self.skip_layer_num  = ast.literal_eval(set_skip_layer_num_str)
        set_skip_layer_thr_str ="{" + os.getenv("SET_STOP_LAYER_THR") + "}"
        self.skip_layer_thr  = ast.literal_eval(set_skip_layer_thr_str)

        self.skip_layer_count = 1
        self.Average_num = 10
        self.layer_w_sum = 0
        self.layer_w_sum_count = 0
        self.BD_end_num = 0

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                #for p in group['params']:
                    p.grad = None
        else:
            super(FusedLAMB, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        global num_step

        global layer_num
        global cac_bd_layer_num
        global cac_bd_end_flag
        global t_lr
        global End_lr
        global End_ratio

        if num_step == 1:
            layer_num = 301
            cac_bd_layer_num = -1
            cac_bd_end_flag = {}
            self.braking_distance = int(os.getenv('CAC_BRAKING_DISTANCE',"-1"))
            self.grad_norms = {}
            ratio = 1.000
            self.skipped_idx = 0
            self.old_skipped_idx = 0
            t_lr = {}
            Layer = 0
            End_lr = float(os.getenv('CAC_FINISH_LR',"0.000003"))
            self.N_factor = int(os.getenv('CAC_BRAKING_FACTOR',"-1"))
            self.braking_count = {}
            mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", -1)
            if mpi_rank == "0":
                print("Braking_Distance:", self.braking_distance)
            for Layer in range(layer_num):
                t_lr[Layer] = 1.0
                cac_bd_end_flag[Layer] = 0
                if self.braking_distance > 0:
                    self.braking_count[Layer] = self.braking_distance

        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []

        for group in self.param_groups:
            if self.skip_layer_count <= len(self.skip_layer_num): 
            	p_skip_layer_count = self.skip_layer_count
            	p_skip = self.skip_layer_num[p_skip_layer_count]
            	p_thr = self.skip_layer_thr[p_skip_layer_count]
            else:
                p_skip = 99999
                p_thr = 99999

            # skip judge & set to
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                if i == p_skip:
                    self.layer_w_sum += torch.norm(p)
                    self.layer_w_sum_count += 1
                    if self.layer_w_sum_count == self.Average_num:
                        self.layer_w_sum = self.layer_w_sum / self.Average_num
                        if self.layer_w_sum < p_thr:
                            Skip_index = UpdateIndex2LayerIndex(p_skip)
                            cac_bd_layer_num = Skip_index
                            mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", -1)
                            if mpi_rank == "0":
                            	print("******Skip Layer: {}, {}".format(str(Skip_index), self.layer_w_sum))
                            self.skip_layer_count += 1
                            self.layer_w_sum_count = 0
                            self.layer_w_sum = 0
                        else:
                            self.layer_w_sum_count = 0
                            self.layer_w_sum = 0
#                mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK", -1)
#                if mpi_rank == "0":
#                    with open("param.csv", "a") as f:
#                        print("Dump, {}, {}, {}, {}, {}, {}, {}, {}".format(num_step, i, str(t_lr[i]), p_w, p_g, p_e, p_skip, p_thr), file=f)
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')
        num_step += 1

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_32], False)[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm,
                                             self._dummy_overflow_buf,
                                             [g_all_16], False)[0]

        # blend two grad norms to get global grad norm
        global_grad_norm = multi_tensor_applier(self.multi_tensor_l2norm,
                                                self._dummy_overflow_buf,
                                                [[g_norm_32, g_norm_16]],
                                                False)[0]
        max_grad_norm = self.defaults['max_grad_norm']

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for i, p in enumerate(group['params']):
            #for p in group['params']:
                if p.grad is None:
                    continue
#                if self.grad_norms[i] == 0.0:
#                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedLAMB does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedLAMB only support fp16 and fp32.')

            self.skipped_idx = layer_num - len(m_32)
            if cac_bd_layer_num != -1:
                for Layer in range(cac_bd_layer_num + 1):
                    if cac_bd_end_flag[Layer] == 0:
                        if self.braking_count[Layer] > 0:
                            self.braking_count[Layer] -= 1
                            if self.braking_count[Layer] >= 0:
                                ratio = (self.braking_distance - self.N_factor)/ self.braking_distance
                                t_lr[Layer] = ratio
                                m_32[Layer - self.skipped_idx] *= t_lr[Layer]
                                if self.braking_count[Layer] == 0:
                                    cac_bd_end_flag[Layer] = 1
                                    os.environ['CAC_STOP_LAYER_NUM'] = str(Layer + 1)

            if(len(g_16) > 0):
                multi_tensor_applier(self.multi_tensor_lamb,
                                     self._dummy_overflow_buf,
                                     [g_16, p_16, m_16, v_16],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.adam_w_mode,
                                     global_grad_norm,
                                     max_grad_norm,
                                     self.use_nvlamb)
            if(len(g_32) > 0):
                multi_tensor_applier(self.multi_tensor_lamb,
                                     self._dummy_overflow_buf,
                                     [g_32, p_32, m_32, v_32],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.adam_w_mode,
                                     global_grad_norm,
                                     max_grad_norm,
                                     self.use_nvlamb)

        return loss
