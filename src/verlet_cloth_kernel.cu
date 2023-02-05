//#include "cutil_math.h"
#include <helper_math.h>
#include <helper_cuda.h>
#include "math_constants.h"

//#define USE_SMEM
#define BLOCKSIZE (128 + 2) * (128 + 2)

__device__ int2 nextNeigh(int n)
{
	if (n == 0)		return make_int2(-1, -1);
	if (n == 1)		return make_int2( 0, -1);
	if (n == 2)		return make_int2( 1, -1);
	if (n == 3)		return make_int2( 1,  0);
	if (n == 4)		return make_int2( 1,  1);
	if (n == 5)		return make_int2( 0,  1);
	if (n == 6)		return make_int2(-1,  1);
	if (n == 7)		return make_int2(-1,  0);

	if (n == 8)		return make_int2(-2, -2);
	if (n == 9)		return make_int2( 2, -2);
	if (n == 10)	return make_int2( 2,  2);
	if (n == 11)	return make_int2(-2,  2);
	
	return make_int2(0, 0);
}


///////////////////////////////////////////////////////////////////////////////
//! kernel for cloth simulating via verlet integration
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
__global__ void verlet(	float4 * pos_vbo, float4 * nor_vbo, float4 * g_pos_in, float4 * g_pos_old_in, float4 * g_pos_out, float4 * g_pos_old_out, 
							int side, float stiffness, float damp, float inverse_mass, int coll_primitives )
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = index % side; 
	int iy = index / side; 

	//g_pos[index] = make_float4(threadIdx.x, blockIdx.x, blockDim.x, index);
	//return;

#ifdef USE_SMEM
    __shared__ float4 smem_pos[BLOCKSIZE];
    __shared__ float4 smem_vel[BLOCKSIZE];

	int ix_smem = threadIdx.x % side;  
	int iy_smem = threadIdx.x / side; 

	smem_pos[threadIdx.x] = g_pos_in[index]; 
	smem_vel[threadIdx.x] = g_vel_in[index]; 

	for (int k = 0; k < 12; k++)
	{
		int2 coord = nextNeigh(k);
		int j = coord.x;
		int i = coord.y;

		if (((iy_smem + i) < 0) || ((iy_smem + i) > (side - 1)))
			continue;

		if (((ix_smem + j) < 0) || ((ix_smem + j) > (side - 1)))
			continue;

		int index_neigh_smem = (iy_smem + i) * side + ix_smem + j;
		int index_neigh = (iy + i) * side + ix + j;

		smem_pos[index_neigh_smem] = g_pos_in[index_neigh]; 
		smem_vel[index_neigh_smem] = g_vel_in[index_neigh]; 
	}

	__syncthreads();

	volatile float4 posData = smem_pos[threadIdx.x];    // ensure coalesced read
    volatile float4 velData = smem_vel[threadIdx.x];
#else
	volatile float4 posData = g_pos_in[index];    // ensure coalesced read
    volatile float4 posOldData = g_pos_old_in[index];
#endif


    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 pos_old = make_float3(posOldData.x, posOldData.y, posOldData.z);
	float3 vel = (pos - pos_old) / 0.01;
	
	// used for computation of the normal
	float3 normal = make_float3(0, 0, 0);
	float3 last_diff = make_float3(0, 0, 0);
	float iters = 0.0;

	float3 force = make_float3(0.0, -9.81, 0.0);
	float inv_mass = inverse_mass;
	if (index <= (side - 1.0))
		inv_mass = 0.f;

	float step = 1.0 / (side - 1.0);

	for (int k = 0; k < 12; k++)
	{
		int2 coord = nextNeigh(k);
		int j = coord.x;
		int i = coord.y;

#ifdef USE_SMEM
		if (((iy_smem + i) < 0) || ((iy_smem + i) > (side - 1)))
			continue;

		if (((ix_smem + j) < 0) || ((ix_smem + j) > (side - 1)))
			continue;

		int index_neigh_smem = (iy_smem + i) * side + ix_smem + j;

		volatile float4 pos_neighData = smem_pos[index_neigh_smem];
#else
		if (((iy + i) < 0) || ((iy + i) > (side - 1)))
			continue;

		if (((ix + j) < 0) || ((ix + j) > (side - 1)))
			continue;

		int index_neigh = (iy + i) * side + ix + j;

		volatile float4 pos_neighData = g_pos_in[index_neigh];
#endif
		float3 pos_neigh = make_float3(pos_neighData.x, pos_neighData.y, pos_neighData.z);

		float3 diff = pos_neigh - pos;

		float3 curr_diff = diff;	// curr diff is the normalized direction of the spring
		curr_diff = normalize(curr_diff);
		
		if ((iters > 0.0) && (k < 8))
		{
			float an = dot(curr_diff, last_diff);
			if (an > 0.0)
				normal += cross(last_diff, curr_diff);
		}	
		last_diff = curr_diff;

		float2 fcoord = make_float2(coord)* step;
		float rest_length = length(fcoord);

		force += (curr_diff * (length(diff) - rest_length)) * stiffness - vel * damp;
		if (k < 8)
			iters += 1.0;
	}

	normal = normalize(normal / -(iters - 1.0));

	float3 acc = make_float3(0, 0, 0);
	acc = force * inv_mass;

	// verlet
	float3 tmp = pos; 
	pos = pos * 2 - pos_old + acc * 0.01 * 0.01;
	pos_old = tmp;

	// collision with cylinders
	if (coll_primitives & 2)
	{
		float step_cyl = 0.2f;
		float radius_cyl = step_cyl / 2.f;
		float pos_cyl = step_cyl;
		for (int i = 0; i < 2; i++)
		{
			float3 center = make_float3(0, -pos_cyl, pos_cyl);

			float3 pos_coll = make_float3(0, pos.y, pos.z);
			if (length(pos_coll - center) < radius_cyl)
			{
				float3 coll_dir = normalize(pos_coll - center);
				pos = make_float3(pos.x, center.y, center.z) + coll_dir * radius_cyl;
			}
		
			pos_cyl += step_cyl;
		}
	}
	

	// collision with a sphere
	if (coll_primitives & 1)
	{
		float3 center = make_float3(0.5, -0.5, 0.25);
		float radius = 0.3;

		if (length(pos - center) < radius)
		{
			// collision
			float3 coll_dir = normalize(pos - center);
			pos = center + coll_dir * radius;
		}
	}

	// collision with a plane
	if (coll_primitives & 4)
	{
		if (pos.y  < -0.6)
		{
			pos.y = -0.6;
			pos_old += (pos - pos_old) * 0.03;
		}
	}

	__syncthreads();

	pos_vbo[index] = make_float4(pos.x, pos.y, pos.z, posData.w);
	nor_vbo[index] = make_float4(normal.x, normal.y, normal.z, 0.0);

	g_pos_out[index] = make_float4(pos.x, pos.y, pos.z, posData.w);
	g_pos_old_out[index] = make_float4(pos_old.x, pos_old.y, pos_old.z, posOldData.w);

}

