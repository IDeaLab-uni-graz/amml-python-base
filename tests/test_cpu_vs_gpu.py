import torch
import time

#Initialisation of tensors
dim=8000

print('Initialisation of tensors')
start_time = time.time()
x=torch.randn(dim,dim)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
x=torch.randn((dim,dim), device=torch.device("cuda:0"))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

# Matrix Multiplication
dim=8000

print('Matrix multiplication')
x=torch.randn(dim,dim)
y=torch.randn(dim,dim)
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)


x=torch.randn(dim,dim,device=torch.device("cuda:0"))
y=torch.randn(dim,dim,device=torch.device("cuda:0"))
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

#Broadcasting
dim=8000

print('Broadcasting')
device=torch.device("cuda:0")

start_time = time.time()
torch.add(torch.randn(dim,1), torch.randn(dim))
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
torch.add(torch.randn(dim,1,device=device), torch.randn(dim,device=device))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

#Outer Product of tensors

dim=8000

print('Outer product of tensors')
device=torch.device("cuda:0")

start_time = time.time()
torch.outer(torch.randn(dim), torch.randn(dim))
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
torch.outer(torch.randn(dim,device=device), torch.randn(dim,device=device))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)