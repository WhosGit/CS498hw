# CS 498 Assignment One — Short Answers

Name: Keyuan Hu
NetID: keyuanh3

## Q1 (1 point)
768/12=64 dim/head

## Q2 (1 point)
$qK^T: O(sd_k)$

$softmax: O(s)$

$\times V: O(sd_k)$

Total:$O(s)$ 
## Q3 (2 points)

(a)
$1024\times 1024\times 16=16777216$

(b)
4 times


## Q4 (2 points)

(a) MHA:
$2\times d_k\times h\times L =1998848$
(b) MQA:
All head share the same K and V

$2\times d_k\times L =15616$
## Q5 (4 points)

input token embedding matrix: $V\times d_m$

$W_Q,W_K,W_V,W_O: 4\times d_m^2 \times L$

MLP matrices: $2\times 4d_m^2 \times L$

Parameters: $5.2\times 10^{10}$

Memory (GiB):97GiB

## Q6 (4 points)

(a)
$W_1: 4096\times 4096$

$W_2: 4096\times 4096$

(b)
Before applying GeLU. This is because

$GeLU(A+B)\neq GeLU(A)+GeLU(B)$

(c)
$128\times 4096$

no. This is because 

$GeLU([A|B])=[GeLU(A)|GeLU(B)]$

and for the second matrix multiplication we only need to column partition of intermidiate matrix instead of a full matrix

(d)
All reduce summation. We need to sum over all the GPU result to get the output matrix $Y$ and we need to broadcast $Y$ to all GPU to carry on to next attention layer calculation.