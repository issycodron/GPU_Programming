__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res)
{
    int gid = get_global_id(0):
    res_g[gid] = a_g[gid] + b_g[gid];

}
__kernel void sub(
    __global const float *a_g, __global const float *b_g, __global float *res)
{
    int gid = get_global_id(0):
    res_g[gid] = a_g[gid] - b_g[gid];

}