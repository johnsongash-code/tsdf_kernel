// work for OpenCL 1.2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef COMMON_H
#define COMMON_H

#ifndef VOXEL_RESOLUTION
#define VOXEL_RESOLUTION 1.0
#endif
#ifndef VOXEL_SIZE
#define VOXEL_SIZE 1
#endif
#ifndef MAX_VOXEL_WEIGHT
#define MAX_VOXEL_WEIGHT 1
#endif

// bit layout in unsigned int
// |      31 - 16     |   15 - 8     |    7 - 0     |
// |   tsdf (half)    | weight (u8)  | padding/u8  |

inline half unpack_half_from_uint(uint packed) {
    ushort tsdf_bits = (ushort)(packed >> 16);
    return as_half(tsdf_bits);
}

inline uchar unpack_u8_from_uint(uint packed) {
    return (uchar)((packed >> 8) & 0xFF);
}

inline uint pack_half_and_u8(half tsdf, uchar weight, uchar padding) {
    ushort tsdf_bits = as_ushort(tsdf); // convert half to bits
    return ((uint)tsdf_bits << 16) | ((uint)weight << 8) | (uint)padding;
}

inline int3 world_to_map(float3 p, float resolution) {
    float3 p_ = (float3)(trunc(p.x / resolution), trunc(p.y / resolution), trunc(p.z / resolution));
    return convert_int3(p_);
}

inline ulong to_flat_index(uint3 p, ulong size) {
    return p.z * size * size + p.y * size + p.x;
}

inline void atomic_update_voxel(
    __global volatile uint* ptr,
    float new_tsdf, uchar max_weight)
{
    uint old_val, new_val;
    do {
        old_val = *ptr;
        // Optionally read old components
        half old_tsdf = unpack_half_from_uint(old_val);
        uchar old_weight = unpack_u8_from_uint(old_val);

        float tsdf_updated = ((float)old_weight * (float)old_tsdf + 1.0f * new_tsdf) / ((float)old_weight + 1.0f);
        // Clamp result to [-1, 1]
        tsdf_updated = fmin(fmax(tsdf_updated, -1.0f), 1.0f);

        uchar new_weight = 0;
        if (old_weight + 1 < max_weight)
            new_weight = old_weight + 1;
        new_val = pack_half_and_u8((half)tsdf_updated, new_weight, 0);
    } while (atomic_cmpxchg(ptr, old_val, new_val) != old_val);
}

inline void atomic_set_voxel(__global volatile uint* ptr, const uint new_val)
{
    uint old_val;
    do {
        old_val = *ptr;
    } while (atomic_cmpxchg(ptr, old_val, new_val) != old_val);
}
#endif