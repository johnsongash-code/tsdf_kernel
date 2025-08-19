#include "common.cl"

__kernel void update_tsdf(
    __global const float3* points,
    __global volatile uint* tsdf_voxel,
    const float trunc_margin,
    const float16 transform,
    const float3 center,
    __global volatile uint* voxel_count
) {
    int gid = get_global_id(0);
    float3 p = points[gid]; // local point
    const int size = (int)VOXEL_SIZE;
    const float resolution = (float)VOXEL_RESOLUTION;
    float half_dimension = (float)size * resolution / 2.0f;

    if (fabs(p.x) >= half_dimension || fabs(p.y) >= half_dimension || fabs(p.z) >= half_dimension)
        return;
    float3 pw = (float3)(
        transform.s0 * p.x + transform.s4 * p.y + transform.s8 * p.z + transform.sc,
        transform.s1 * p.x + transform.s5 * p.y + transform.s9 * p.z + transform.sd,
        transform.s2 * p.x + transform.s6 * p.y + transform.sa * p.z + transform.se
    );

    float3 position = (float3)(transform.sc, transform.sd, transform.se);
    float ray_length = length(p);
    float3 ray_dir = normalize(pw - position);
    int start = max(1, (int)((ray_length - trunc_margin) / resolution));;
    int end = (ray_length + trunc_margin) / resolution;
    int half_size = size / 2;
    const ulong full_size = (ulong)VOXEL_SIZE;   
    for (int i = start; i <= end; i++) {
        float range = resolution * (float)i;
        float sdf = ray_length - range;
        if (sdf < -trunc_margin)
            continue;
        float3 wp = position + ray_dir * range;
        int3 mp = world_to_map(wp - center, resolution);
        if (abs(mp.x) >= half_size || abs(mp.y) >= half_size || abs(mp.z) >= half_size)
            continue;

        uint x = mp.x + half_size;
        uint y = mp.y + half_size;
        uint z = mp.z + half_size;
        uint idx = to_flat_index((uint3)(x, y, z), full_size);
        float tsdf = max(-1.0f, min(1.0f, (float)(sdf / trunc_margin)));

        atomic_update_voxel(&tsdf_voxel[idx], tsdf, MAX_VOXEL_WEIGHT);
        atomic_inc(voxel_count);
    }
}