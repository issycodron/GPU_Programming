import numpy as np
import pyopencl as cl
import timeit

a_np = np.random.rand(5000000).astype(np.float32)
b_np = np.random.rand(5000000).astype(np.float32)


ctx = cl.create_some_context() #initialise system to use GPU
queue = cl.CommandQueue(ctx) #list of algorithms to be executed 

mf = cl.mem_flags #setting up GPU memory ready for CPU to use
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np) #arrays are read only when on the gpu
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)


prg = cl.Program(
    ctx,
    """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
}
""",
).build() #this creates program to run on GPU


res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes) #set up one last array for resulting array
knl = prg.sum #use this Kernel object for repeated calls

starttime = timeit.default_timer()
knl(queue, a_np.shape, None, a_g, b_g, res_g) #run the kernel on the GPU
gpu_time = timeit.default_timer() - starttime

res_np = np.empty_like(a_np) #create empty array on CPU
cl.enqueue_copy(queue, res_np, res_g) #copy result from GPU to CPU

starttime = timeit.default_timer()
a_np + b_np #run the same operation on the CPU to compare speed with GPU
cpu_time = timeit.default_timer() - starttime

print(f"The GPU ran {cpu_time / gpu_time} x faster than the CPU")

## Check on CPU with numpy
#print(res_np - (a_np + b_np))
#print(np.linalg.norm(res_np - (a_np + b_np)))
#assert np.allclose(res_np, a_np + b_np)
