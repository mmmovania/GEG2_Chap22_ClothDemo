
int2 nextNeigh(int n)
{
	if (n == 0)		return (int2)(-1, -1);
	if (n == 1)		return (int2)( 0, -1);
	if (n == 2)		return (int2)( 1, -1);
	if (n == 3)		return (int2)( 1,  0);
	if (n == 4)		return (int2)( 1,  1);
	if (n == 5)		return (int2)( 0,  1);
	if (n == 6)		return (int2)(-1,  1);
	if (n == 7)		return (int2)(-1,  0);
	if (n == 8)		return (int2)(-2, -2);
	if (n == 9)		return (int2)( 2, -2);
	if (n == 10)	return (int2)( 2,  2);
	if (n == 11)	return (int2)(-2,  2);
}


__kernel void verlet_cloth( __global float4 * g_pos_vbo,		// positions in a vbo interoperating with opengl
							__global float4 * g_nor_vbo,		// normal vectors in a vbo interoperating with opengl
							__global float4 * g_pos_idata,		// input positions
							__global float4 * g_pos_old_idata,	// input "old" positions (positions at the previous state)
							__global float4 * g_pos_odata,		// output positions
							__global float4 * g_pos_old_odata,	// output "old" positions
							float stiffness,					// spring stiffness
							float damp,							// damping
							float inv_mass,						// inverse of the mass of each particle
							int side,							// size of the side of the cloth (in particles)
							int coll_primitives)				// collision primitives
{
	const int x = get_global_id(0);

	int index = x;

	int ix = index % side; 
	int iy = index / side; 

	float4 pos = g_pos_idata[index];
	pos.w = 0.0;
	float4 pos_old = g_pos_old_idata[index];
	pos_old.w = 0;
	float4 vel = (pos - pos_old) / 0.01f;

	float inverse_mass = inv_mass;
	if (index <= (side - 1))
		inverse_mass = 0.f;

	float step = 1.0f / (side - 1.0f);
	float4 force = (float4)(0.0f, -9.81f, 0.0f, 0.0f);

	// used for computation of the normal
	float4 normal = (float4)(0, 0, 0, 0);
	float4 last_diff = (float4)(0, 0, 0, 0);
	float iters = 0.0f;

	for (int k = 0; k < 12; k++)
	{
		int2 coord = nextNeigh(k);
		int j = coord.x;
		int i = coord.y;

		if (((iy + i) < 0) || ((iy + i) > (side - 1)))
			continue;

		if (((ix + j) < 0) || ((ix + j) > (side - 1)))
			continue;

		int index_neigh = (iy + i) * side + ix + j;
		float4 pos_neigh = g_pos_idata[index_neigh];
		pos_neigh.w = 0;
		float4 diff = pos_neigh - pos;

		float4 curr_diff = normalize(diff);	// curr diff is the normalized direction of the spring
		
		if ((iters > 0.0) && (k < 8))
		{
			float an = dot(curr_diff, last_diff);
			if (an > 0.0)
				normal += cross(last_diff, curr_diff);
		}	
		last_diff = curr_diff;
		
		float rest_length = length((float2)(coord.x, coord.y)) * step;

		force += (curr_diff * (length(diff) - rest_length)) * stiffness - vel * damp;
		force.w = 0;

		if (k < 8)
			iters += 1.0;
	}

	normal = normalize(normal / -(iters - 1.0f));

	float4 acc = force * inverse_mass;

	// verlet
	float4 tmp = pos;
	pos = pos * 2 - pos_old + acc * 0.01f * 0.01f;
	pos_old = tmp;

	// collision with cylinders
	if (coll_primitives & 2)
	{
		float step_cyl = 0.2f;
		float radius_cyl = step_cyl / 2.f;
		float pos_cyl = step_cyl;
		for (int i = 0; i < 2; i++)
		{
			float4 center = (float4)(0, -pos_cyl, pos_cyl, 0);

			float4 pos_coll = (float4)(0, pos.y, pos.z, 0);
			if (length(pos_coll - center) < radius_cyl)
			{
				float4 coll_dir = normalize(pos_coll - center);
				pos = (float4)(pos.x, center.y, center.z, 0) + coll_dir * radius_cyl;
			}
		
			pos_cyl += step_cyl;
		}
	}

	//  collision with sphere
	if (coll_primitives & 1)
	{
		float4 center = (float4)(0.5f, -0.5f, 0.25f, 0.0f);
		float radius = 0.3f;

		if (length(pos - center) < radius)
		{
			// collision
			float4 coll_dir = normalize(pos - center);
			pos = center + coll_dir * radius;
		}
	}

	// plane
	if (coll_primitives & 4)
	{
		if (pos.y  <= -0.6f)
		{
			pos.y = -0.6f;
			pos_old += (pos - pos_old) * 0.03f;
		}
	}
	
	pos.w = 1.0;
	g_pos_vbo[index] = pos;
	g_nor_vbo[index] = normal;
	g_pos_odata[index] = pos;
	g_pos_old_odata[index] = pos_old;

	return;
}

