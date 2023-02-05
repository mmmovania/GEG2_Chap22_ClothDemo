

uniform sampler2D	texUnit;	// positions
uniform sampler2D	texUnit2;	// positions "old" (at the previous time step)

uniform float		texsize;

uniform float stiffness;
uniform float damp;
uniform float time_step;
uniform float inv_mass;
uniform int collide_sphere;
uniform int collide_cylinders;
uniform int collide_plane;

vec2 nextNeigh(int n)
	{
		if (n == 0)	return vec2(-1, -1);
		if (n == 1)	return vec2( 0, -1);
		if (n == 2)	return vec2( 1, -1);

		if (n == 3)	return vec2( 1,  0);
		if (n == 4) return vec2( 1,  1);
		if (n == 5) return vec2( 0,  1);
	
		if (n == 6) return vec2(-1,  1);
		if (n == 7) return vec2(-1,  0);
		
		if (n == 8)	return vec2(-2, -2);
		if (n == 9) return vec2( 2, -2);
		if (n ==10) return vec2( 2,  2);
		if (n ==11) return vec2(-2,  2);
	}


void main(void)
{
	vec3 pos = texture2D(texUnit, gl_TexCoord[0].xy).xyz;
	vec3 pos_old = texture2D(texUnit2, gl_TexCoord[0].xy).xyz;
	vec3 vel = (pos - pos_old) / time_step;	// velocity according to verlet integration
	
	// used for computation of the normal
	vec3 normal = vec3(0, 0, 0);
	vec3 last_diff = vec3(0, 0, 0);
	float iters = 0.0;

	float step = 1.0 / (texsize - 1.0);

	float ix = floor(gl_TexCoord[0].x * texsize);
	float iy = floor(gl_TexCoord[0].y * texsize);
	float index = iy * texsize + ix;

	vec3 force = vec3(0, -9.81, 0);
	float inverse_mass = inv_mass;
	if (index <= (texsize - 1.0))
		inverse_mass = 0.0;

	for (int k = 0; k < 12; k++)
	{
		vec2 coord = nextNeigh(k);
		float j = coord.x;
		float i = coord.y;

		if (((iy + i) < 0.0) || ((iy + i) > (texsize - 1.0)))
			continue;

		if (((ix + j) < 0.0) || ((ix + j) > (texsize - 1.0)))
			continue;

		vec2 coord_neigh = vec2(ix + j, iy + i) / texsize;
		
		vec3 pos_neigh = texture2D(texUnit, coord_neigh).xyz;
		vec3 diff = pos_neigh - pos;
		vec3 curr_diff = normalize(diff);	// curr diff is the normalized direction of the spring
		
		if ((iters > 0.0) && (k < 8))
		{
			float an = dot(curr_diff, last_diff);
			if (an > 0.0)
				normal += cross(last_diff, curr_diff);
		}	
		last_diff = curr_diff;
		
		float rest_length = length(coord) * step;

		force += (curr_diff * (length(diff) - rest_length)) * stiffness - vel * damp;
		if (k < 8)
			iters += 1.0;
	}

	normal = normalize(normal / -(iters - 1.0));

	vec3 acc;
	acc = force * inverse_mass;

	// verlet
	vec3 tmp = pos;
	pos = pos * 2.0 - pos_old + acc * time_step * time_step;
	pos_old = tmp;
	
	// collision with cylinders
	if (collide_cylinders == 1)
	{
		float step_cyl = 0.2;
		float radius_cyl = step_cyl / 2.0;
		float pos_cyl = step_cyl;
		for (int i = 0; i < 2; i++)
		{
			vec3 center = vec3(0.0, -pos_cyl, pos_cyl);

			vec3 pos_coll = vec3(0.0, pos.y, pos.z);
			if (length(pos_coll - center) < radius_cyl)
			{
				vec3 coll_dir = normalize(pos_coll - center);
				pos = vec3(pos.x, center.y, center.z) + coll_dir * radius_cyl;
			}
		
			pos_cyl += step_cyl;
		}
	}
	
	// collision with a sphere
	if (collide_sphere == 1)
	{
		vec3 center = vec3(0.5, -0.5, 0.25);
		float radius = 0.3;
		if (length(pos - center) < radius)
		{
			// collision
			vec3 coll_dir = normalize(pos - center);
			pos = center + coll_dir * radius;
		}
	}
	
	// collision with plane
	if (collide_plane == 1)
	{
		if (pos.y  <= -0.6)
		{
			pos.y = -0.6;
			pos_old += (pos - pos_old) * 0.03;
		}
	}

	gl_FragData[0] = vec4(pos,		1.0);
	gl_FragData[1] = vec4(pos_old,	0.0);
	gl_FragData[2] = vec4(normal,	0.0);
}
