###Q4: Tensor-Parallel MLP###
###please implement weight partitioning, forward, and backward computation###
###use isend/irecv only in sum_across_ranks; built-in collectives are not allowed###
###do not use autograd or optimizers; see the assignment for contracts and README.md for running instructions###
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F


def gelu_derivative(z):
    ###provided helper: derivative of exact GeLU###
    ###do not modify this function###
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)


@torch.no_grad()
def shard_weights(w1, w2, rank, world_size):
    # ---- partition the intermediate features across ranks ----
    # w1 has shape [H, F]; w2 has shape [F, H]; F >= world_size.
    # If F is not divisible by world_size, assign one extra feature to
    # each of the first F % world_size ranks. Use contiguous rank order.
    # Return independent, contiguous w1/w2 shards; do not modify full weights.
    #                                                                   #
    # your code here: find this rank's feature range, including remainder #
    #                columns, and copy the matching w1/w2 shards          #
    #                                                                   #

    F = w1.shape[1]

    features_per_rank = F // world_size
    remainder = F % world_size
    start_idx = rank * features_per_rank + min(rank, remainder)

    end_idx = start_idx + features_per_rank + (1 if rank < remainder else 0)

    # Return the shards for this rank
    return w1[:, start_idx:end_idx].clone(), w2[start_idx:end_idx, :].clone()



@torch.no_grad()
def sum_across_ranks(tensor, rank, world_size):
    # ---- sum contributions and distribute the result ----
    # All ranks provide tensors with the same shape and dtype.
    # Return a fresh tensor containing the full SUM on every rank.
    # Do not change the input or divide by world_size.
    # With one rank, return an independent copy without communication.
    # Only this function may communicate: use isend/irecv and wait().
    #                                                                   #
    # your code here: handle the single-rank case                        #
    #                                                                   #

    if world_size == 1:
        return tensor.clone()

    # ---- rank 0: aggregate contributions, then send the sum ----
    #                                                                   #
    # your code here: include rank 0's input and receive from workers;    #
    #                send the completed sum to every worker              #
    #                                                                   #

    if rank == 0:
        # Initialize the sum 
        total_sum = tensor.clone()

        # Receive contributions from other ranks
        recv_requests = []
        recv_tensor = []
        for r in range(1, world_size):
            recv_tensor.append(torch.empty_like(tensor))
            recv_requests.append(dist.irecv(recv_tensor[-1], src=r))

        # Wait for all receive operations to complete
        for req in recv_requests:
            req.wait()

        # Sum the contributions from other ranks
        for r in range(1, world_size):
            total_sum += recv_tensor[r - 1]

        # Send the completed sum to all other ranks
        send_requests = []
        for r in range(1, world_size):
            send_requests.append(dist.isend(total_sum, dst=r))

        # Wait for all send operations to complete
        for req in send_requests:
            req.wait()

        return total_sum

    # ---- other ranks: send the local contribution, receive the sum ----
    #                                                                   #
    # your code here: send to rank 0 and receive the completed sum        #
    #                Wait before reading/reusing buffers or returning.   #
    #                                                                   #
    dist.isend(tensor, dst=0).wait()

    recv_tensor = torch.empty_like(tensor)
    req = dist.irecv(recv_tensor, src=0)
    req.wait()
    return recv_tensor


@torch.no_grad()
def mlp_forward(x, w1_local, w2_local, rank, world_size):
    # ---- compute this rank's part of the MLP ----
    # x: [T, H]; w1_local: [H, F_r]; w2_local: [F_r, H].
    # Use the default F.gelu (exact mode); do not modify the inputs.
    # Call sum_across_ranks once to obtain the full output [T, H].
    # Return (output, cache), with cache in this exact order:
    # (x, z, a, w1_local, w2_local). Cached references are allowed.
    # ---- local up-projection, GeLU, and down-projection ----
    #                                                                   #
    # your code here: compute z, a, and the local partial output          #
    #                                                                   #
    z = x @ w1_local
    a = F.gelu(z)
    local_output = a @ w2_local

    # ---- combine outputs and save values for backward ----
    #                                                                   #
    # your code here: call sum_across_ranks and return output with cache  #
    #                                                                   #
    output = sum_across_ranks(local_output, rank, world_size)
    cache = (x, z, a, w1_local, w2_local)
    return output, cache


@torch.no_grad()
def mlp_backward(grad_output, cache, rank, world_size):
    # ---- compute local gradients and synchronize the input gradient ----
    # grad_output is the same full dL/dY on every rank; do not scale by P.
    # Use gelu_derivative and the backward formulas in the assignment.
    # Call sum_across_ranks once for grad_x; weight gradients stay local.
    # Do not modify grad_output or the cache, including its saved weights.
    # Return (grad_x, grad_w1_local, grad_w2_local).
    # ---- unpack the cache and compute the local weight gradients ----
    #                                                                   #
    # your code here: compute grad_w2, grad_z, and grad_w1 manually        #
    #                Use the supplied gelu_derivative helper.            #
    #                                                                   #
    x, z, a, w1_local, w2_local = cache
    grad_w2_local = a.T @ grad_output
    grad_a = grad_output @ w2_local.T
    grad_z = grad_a * gelu_derivative(z)
    grad_w1_local = x.T @ grad_z

    # ---- combine input-gradient contributions ----
    #                                                                   #
    # your code here: compute local grad_x, sum it across ranks, and      #
    #                return grad_x with the two local weight gradients   #
    #                                                                   #
    grad_x = grad_z @ w1_local.T
    grad_x = sum_across_ranks(grad_x, rank, world_size)
    return grad_x, grad_w1_local, grad_w2_local
