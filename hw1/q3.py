###Q3: allreduce###
###implement ring all-reduce with isend/irecv; built-in collectives are not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for i in range(world-1):
        send_chunk = chunks[(rank - i - 1) % world]

        send_req = dist.isend(send_chunk, dst=right)
        recv_req = dist.irecv(tmp, src=left)
        send_req.wait()
        recv_req.wait()

        chunks[(rank - i - 2) % world] += tmp
    

        
def all_gather(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    for i in range(world-1):
        send_chunk = chunks[(rank - i) % world]

        send_req = dist.isend(send_chunk, dst=right)
        recv_req = dist.irecv(tmp, src=left)
        send_req.wait()
        recv_req.wait()

        chunks[(rank - i - 1) % world].copy_(tmp)

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce average using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_flat = torch.zeros(chunk * world, dtype=flat.dtype, device=flat.device)
    padded_flat[:n] = flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    tmp = torch.empty_like(chunks[0])
    reduce_scatter(chunks, tmp, world, rank, left, right)
    all_gather(chunks, tmp, world, rank, left, right)

    # stitch & unpad  
    padded_flat /= world
    tensor.view(-1).copy_(padded_flat[:n])
    return