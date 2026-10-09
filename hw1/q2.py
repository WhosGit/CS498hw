###Q2: Parameter Server###
###please implement PS method, using  parameter update&optimizor states all be held in rank0###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def server(params, opt, world):
    # ---- aggregate grads from workers ----
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous() #usage: tranfer a list tensor to one 1-D tensor
    ##here, you should generate one big 1-D tensor containing all parameters to make the transfer process easy
    agg = flat_grad.clone() #agg as a aggregated counter to record sum gradients

    #                                                                   #
    #                                                                   #
    # your code here: receive gradients form worker, and add them to agg#
    #                                                                   #
    #                                                                   #

    recv_requests = []
    recv_tensor = []
    for r in range(1, world):
        recv_tensor.append(torch.empty_like(flat_grad))
        recv_requests.append(dist.irecv(recv_tensor[-1], src=r))

    # Wait
    for req in recv_requests:
        req.wait()

    # Sum
    for r in range(1, world):
        agg += recv_tensor[r - 1]

    agg /= world  # average the gradients

    synced_grads = _unflatten_dense_tensors(agg, [p.grad for p in params])
    # ---- set averaged grads locally & step ----
    for g, s in zip([p.grad for p in params], synced_grads):
        g.copy_(s)
    opt.step()

    # ---- broadcast updated params for this subset ----
    flat_param = _flatten_dense_tensors([p.data for p in params]).contiguous()
    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D parameter tensor to all workers   #
    #                                                                   #
    #                                                                   #
    send_requests = []
    for r in range(1, world):
        send_requests.append(dist.isend(flat_param, dst=r))

    for req in send_requests:
        req.wait()

def worker(params):
    flat_grad = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    # ---- push grads to server ----

    #                                                                   #
    #                                                                   #
    # your code here: send packed 1-D gradient to server
    #                                                                   #
    #                                                                   #
    dist.isend(flat_grad, dst=0).wait()

    # ---- receive updated params, write into local model ----
    
    #                                                                   #
    #                                                                   #
    # your code here: please get correct 1-D packed parameter from server
    #           And then unpacked it and store in synced_params
    #                                                                   #
    recv_tensor = torch.empty_like(flat_grad)
    dist.irecv(recv_tensor, src=0).wait()
    synced_params = _unflatten_dense_tensors(recv_tensor, [p.data for p in params])

    # synced_params = None #you should  assign correct value for synced_params#


    # ---- syncronize the parameters ----
    for p, s in zip(params, synced_params):
        p.data.copy_(s)

def PS_grads_(model,world_size=None, rankid=None, opt=None):
    """
    Synchronous PS step:
      - Rank 0: receive grads, sum/avg, set grads, opt.step(), broadcast updated params.
      - Rank >0: send grads, receive updated params, write into local model.
    Only processes the subset of parameters with non-None grads.
    """
    world = world_size
    rank  = rankid
    
    # Fast path: single process
    if world == 1:
        opt.step()
        return

    #not necessary, as in most case, params won't be empty list...
    # Collect params that participated in this backward pass
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        # No grads this step; only server might still want to advance schedulers, etc.
        if rank == 0:
            opt.step()
        return

    if rank == 0:
        server(params, opt, world)

    else:
        worker(params)

    # Optional hard step boundary
    dist.barrier()