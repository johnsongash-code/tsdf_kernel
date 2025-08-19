#include "common.cl"

__kernel void traverse_tsdf(
    __global const uint* tsdf_voxel,
    __global volatile uint3* coords,
    __global volatile uint* voxel_count,
    const float threshold,
    const uchar min_weight
) {
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    const ulong size = (ulong)VOXEL_SIZE;
    if (x >= size || y >= size || z >= size) return;
    
    const uint3 mp = (uint3)(x, y, z);
    ulong idx = to_flat_index(mp, size);
    half tsdf = unpack_half_from_uint(tsdf_voxel[idx]);
    uchar weight = unpack_u8_from_uint(tsdf_voxel[idx]);

    if (weight >= min_weight && fabs(tsdf) < threshold) {
        uint i = atomic_inc(voxel_count);
        coords[i] = mp;          
    }
}

__kernel void export_tsdf(
    __global const uint* tsdf_voxel,
    __global volatile half* tsdfs,
    __global volatile uint3* coords,
    __global volatile uint* voxel_count
) {
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    const ulong size = (ulong)VOXEL_SIZE;
    if (x >= size || y >= size || z >= size) return;
    
    const uint3 mp = (uint3)(x, y, z);
    ulong idx = to_flat_index(mp, size);
    uchar weight = unpack_u8_from_uint(tsdf_voxel[idx]);

    if (weight > 0) {
        half tsdf = unpack_half_from_uint(tsdf_voxel[idx]);
        // the value before increment
        uint i = atomic_inc(voxel_count);
        tsdfs[i] = tsdf;
        coords[i] = mp;          
    }
}

__kernel void reset_voxel(
    __global volatile uint* tsdf_voxel,
    const ulong min_index,
    const ulong max_index,
    __global volatile uint* voxel_count
) {
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    const ulong size = (ulong)VOXEL_SIZE;
    const uint3 mp = (uint3)(x, y, z);
    ulong idx = to_flat_index(mp, size);
    if (idx < min_index || idx >= max_index) return;
    
    // set default tsdf and weight
    atomic_update_voxel(&tsdf_voxel[idx], 1.0, 0);
    atomic_inc(voxel_count);
}

__kernel void shift_voxel(
    __global const uint* origin_voxel,
    __global volatile uint* tsdf_voxel,
    const int3 offset
) {
    ulong x = get_global_id(0);
    ulong y = get_global_id(1);
    ulong z = get_global_id(2);

    const ulong size = (ulong)VOXEL_SIZE;
    const int3 p = (int3)((int)x - offset.x, (int)y - offset.y, (int)z - offset.z);
    if (p.x < 0 || p.x >= size || p.y < 0 || p.y >= size || p.z < 0 || p.z >= size)
        return;
    const ulong idx = to_flat_index((uint3)(x, y, z), size);
    const ulong new_idx = to_flat_index(convert_uint3(p), size);
    atomic_set_voxel(&tsdf_voxel[new_idx], origin_voxel[idx]);
}