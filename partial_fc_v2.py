
import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize

def layer_norm(Z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu  = Z.mean(dim=1, keepdim=True)                # row mean [K, 1]
    std = torch.sqrt((Z - mu).pow(2).mean(dim=1, keepdim=True) + eps).detach() # row std  [K, 1]
    Z_bar = (Z - mu) / std         # Normalize
    return Z_bar


class my_CE(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
    
class my_PFC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False
    ):
        super(my_PFC, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        # self.fp16 = fp16
        self.fp16 = None
        if fp16:
            self.fp16 = torch.float16
        else:
            self.fp16 = torch.float32
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.num_sample: int = int(sample_rate * num_classes)
        self.classes = torch.arange(num_classes, dtype=torch.int32)
        self.mask = torch.ones(num_classes, dtype=torch.bool)
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise
            
        self.lambd = 0.5
        self.s = 25
        # reocrd norms
        self.avg_weight_norm = 0
        self.avg_embedding_norm = 0

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        with torch.no_grad():
            self.weight.copy_(0.4*normalize(self.weight))
            
        if self.sample_rate < 1:
            with torch.no_grad():
                positive, idx_ivs = torch.unique(labels, sorted=False, return_inverse=True)
                
                self.mask[positive] = False
                negative = self.classes[self.mask]
                self.mask[positive] = True
                perm = torch.randperm(negative.shape[0])
                
                index = torch.zeros(self.num_sample, dtype=torch.int32)
                index[:positive.shape[0]] = positive
                index[positive.shape[0]:] = negative[perm[:(self.num_sample - positive.shape[0])]]
                
            weight = self.weight[index]
            labels = idx_ivs
        else:
            weight = self.weight

        with torch.amp.autocast("cuda", self.fp16):
        # with torch.cuda.amp.autocast(self.fp16):
            # norm_embeddings = embeddings
            # norm_weight_activated = weight
            
            # norm_embeddings = normalize(embeddings)
            # norm_weight_activated = normalize(weight)
            
            # norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            
            epsilon_embeddings = embeddings.norm(dim=1, keepdim=True).clamp_min(10e-12).detach()
            # epsilon_weight = weight.norm(dim=1, keepdim=True).clamp_min(10e-12).detach()
            norm_embeddings = embeddings / epsilon_embeddings
            #norm_weight_activated = weight / epsilon_weight
            
            # std is detached
            # norm_embeddings = layer_norm(embeddings)
            # norm_weight_activated = layer_norm(weight)
            
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        # loss = self.cross_entropy(logits, labels)
        loss_ce = self.cross_entropy(logits, labels)
        loss_p = torch.mean((embeddings.norm(dim=1) - self.s)**2)
        loss = loss_ce + self.lambd * loss_p
        
        with torch.no_grad():
            self.avg_embedding_norm = 0.9*self.avg_embedding_norm + 0.1*embeddings.norm(dim=1).mean()
            self.avg_weight_norm = 0.9*self.avg_weight_norm + 0.1*weight.norm(dim=1).mean()
        
        return loss, loss_ce.detach().cpu(), loss_p.detach().cpu(), self.avg_embedding_norm, self.avg_weight_norm
    
class LoraFC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        bottle_neck: int,
        fp16: bool = False,
    ):
        super(LoraFC, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight0 = torch.nn.Parameter(torch.normal(0, 0.01, (bottle_neck, embedding_size)))
        self.weight1 = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, bottle_neck)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        # weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            # norm_embeddings0 = normalize(embeddings)
            # norm_weight_activated0 = normalize(self.weight0)
            # embeddings1 = linear(norm_embeddings0, norm_weight_activated0)
            embeddings1 = linear(embeddings, self.weight0)
            
            norm_embeddings1 = normalize(embeddings1)
            norm_weight_activated1 = normalize(self.weight1)
            logits = linear(norm_embeddings1, norm_weight_activated1)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
    
class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def sample(self, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
