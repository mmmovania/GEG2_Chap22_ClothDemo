// GPGPU Cloth simulation using GLSL, OpenCL and CUDA
// by Marco Fratarcangeli - 2010
//   marco@fratarcangeli.net
//   http://www.fratarcangeli.net
//
// This software is provided 'as-is', without any express or
// implied warranty. In no event will the author be held
// liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute
// it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
//    you must not claim that you wrote the original software.
//    If you use this software in a product, an acknowledgment
//    in the product documentation would be appreciated but
//    is not required.
//
// 2. Altered source versions must be plainly marked as such,
//    and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any
//    source distribution.


#ifndef __PARTICLE_SYSTEM__
#define __PARTICLE_SYSTEM__


#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

#define checkCudaErrors(val) val

#include "reusable.h"
#include "timer.h"
#include <vector>
#include <oclUtils.h>

#define CUT_CHECK_ERROR_GL() assert(glGetError()==GL_NO_ERROR)
using namespace std;

#define SPHERE		1
#define CYLINDERS	2
#define PLANE		4

enum ComputationType { CPU, GLSL, OPENCL, CUDA, NOCOMPUTATION };
ComputationType computation = CPU;

// number of physical simulation steps for each rendering
int nbSimIters = 1;	

// default values for simulation time step and damping constant
#define TIME_STEP 0.01f	
#define DAMP 1.0

// externally defined methods for CUDA implementation 
extern void InitCuda(const int size);
extern void UploadCuda(float * positions, float * positions_old, const int size);
extern void VerletCuda(float4* pos_vbo, float4* nor_vbo, float * positions, float * positions_old, const int size, const int & side, const float & stiffness, const float & damp, const float & inverse_mass, const int & coll_primitives);
extern void DownloadCuda(float * positions, float * positions_old, const int size);
extern void ResetCuda();

enum VboType
{
    POSITION_OBJECT = 0,
    NORMAL_OBJECT = 1
};

// this class handle the particle system representing a squared piece of cloth
// it computes its dynamics using CPU, GLSL, OpenCL and CUDA
// it is indipendent from the rendering platform: in principle one can use this class to render the cloth in opengl, directx, ...
class ParticleSystem
{	
	double      time_step;

public:

	// ---------------------------------------------------------------------------------------------------
	// physical constants
	float stretch_stiffness;	// spring stiffness constant
	float damp;					// damping constant
	float inverse_mass;			// inverse of the mass of each particle in the system (if 0, it means the particle has infinite mass and can not move)
	// ---------------------------------------------------------------------------------------------------

	// ---------------------------------------------------------------------------------------------------
	// current state of the particle system
	vector<float> positions;		// current position of the particles  (used in Verlet integration and rendering)
	vector<float> positions_old;	// previous position of the particles (used in Verlet integration)
	vector<float> normals;			// normal of each particle (used for rendering purposes)
	// ---------------------------------------------------------------------------------------------------

	int collPrimitives;

	vector<float> texcoords;	// 2D texture coordinates of each particle on the cloth (used for rendering purposes)
	int nb_springs;				// number of springs involved in the simulation

	vector<short> faces;		// three short for each face, dim = 3 * nbfaces; it used as index buffer only for  rendering phase

	// ---------------------------------------------------------------------------------------------------
	// GLSL data structures
	int texture_size;						// size of textures which stores the state of the particle system
	struct FBOMRTstruct *fbo_0, *fbo_1;		// frame buffer objects; they host both the textures storing the input data, and the render targets where the output is written
	GLuint shaderProgramObject;				// the program id used to update the simulation with verlet integration
	// ---------------------------------------------------------------------------------------------------

	// ---------------------------------------------------------------------------------------------------
	// OPENCL data structures
	cl_program cpProgram;	cl_kernel ckKernel;					// program computing the simulation
	cl_mem hSourcePos, hSourcePosOld, hDestPos, hDestPosOld;	// memory buffers to store the input positions and output positions
	cl_mem vbo_cl[2];
	// ---------------------------------------------------------------------------------------------------

	// ---------------------------------------------------------------------------------------------------
	// cuda vbo variables
	GLuint vbo[2];
	struct cudaGraphicsResource * cuda_vbo_resource[2];
	// ---------------------------------------------------------------------------------------------------

	ParticleSystem()
	{
		fbo_0 = NULL;
		fbo_1 = NULL;

		vbo[POSITION_OBJECT] = 0;
		vbo[NORMAL_OBJECT] = 0;

		hSourcePos = NULL;		hSourcePosOld = NULL;
		hDestPos = NULL;		hDestPosOld = NULL;
		cpProgram = NULL;		ckKernel = NULL;
		vbo_cl[POSITION_OBJECT] = 0;
		vbo_cl[NORMAL_OBJECT] = 0;

		Reset();

		time_step = TIME_STEP;
		stretch_stiffness = 1.f;
		damp = DAMP;
		inverse_mass = 1.f;
	}

	~ParticleSystem()
	{
		Reset();			// reset the state of the particle system	
		ResetGMem();		// clean memory and buffers

		shutdownOpenCL();	// cleanup OpenCL
		cudaThreadExit();	// stop any cuda run-time (this is called anyway implicitely)
	}

	// reset the state of the particle system
	void Reset()
	{
		faces.clear();

		positions.clear();
		positions_old.clear();
		normals.clear();

		texcoords.clear();
	}

	// clear the global memory used for GLSL (textures), OpenCL and CUDA
	void ResetGMem()
	{
		// clear frame buffer ojects
		if (fbo_0 != NULL)
		{
			glDeleteFramebuffersEXT(1, &(fbo_0->fb));
			delete fbo_0;
		}
		if (fbo_1 != NULL)
		{
			glDeleteFramebuffersEXT(1, &(fbo_1->fb));
			delete fbo_1;
		}
		fbo_0 = NULL;
		fbo_1 = NULL;

		// clear opencl memory buffers and kernel
		if (hSourcePos != NULL)		freeArray(hSourcePos);
		if (hSourcePosOld != NULL)	freeArray(hSourcePosOld);
		if (hDestPos != NULL)		freeArray(hDestPos);
		if (hDestPosOld != NULL)	freeArray(hDestPosOld);
		if (vbo_cl[POSITION_OBJECT] != NULL)	freeClVbo(&(vbo_cl[POSITION_OBJECT]));
		if (vbo_cl[NORMAL_OBJECT] != NULL)		freeClVbo(&(vbo_cl[NORMAL_OBJECT]));
		if (ckKernel)				clReleaseKernel(ckKernel); 
		if (cpProgram)				clReleaseProgram(cpProgram);

		hSourcePos = NULL;			hSourcePosOld = NULL;
		hDestPos = NULL;			hDestPosOld = NULL;
		cpProgram = NULL;			ckKernel = NULL;
		vbo_cl[POSITION_OBJECT] = 0;
		vbo_cl[NORMAL_OBJECT] = 0;			

		// clear CUDA data strucutres
		deleteCudaVBO();
		ResetCuda();
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Create VBO
	////////////////////////////////////////////////////////////////////////////////
	void InitInteropVBO()
	{
		unsigned int size = texture_size * texture_size * 4 * sizeof(float);

		// create buffer objects
		glGenBuffersARB(2, vbo);
		CUT_CHECK_ERROR_GL();
		

		// initialize buffer objects
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[POSITION_OBJECT]);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, 0, GL_DYNAMIC_DRAW);
		if (computation == OPENCL)
			allocateClVbo(&(vbo_cl[POSITION_OBJECT]), size, &(vbo[POSITION_OBJECT]));
		CUT_CHECK_ERROR_GL();

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[NORMAL_OBJECT]);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, 0, GL_DYNAMIC_DRAW);
		if (computation == OPENCL)
			allocateClVbo(&(vbo_cl[NORMAL_OBJECT]), size, &(vbo[NORMAL_OBJECT]));
		CUT_CHECK_ERROR_GL();

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		CUT_CHECK_ERROR_GL();

		if (computation == CUDA)
		{
			// register the buffer objects with CUDA
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(cuda_vbo_resource[POSITION_OBJECT]),	vbo[POSITION_OBJECT],	cudaGraphicsMapFlagsWriteDiscard));
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(cuda_vbo_resource[NORMAL_OBJECT]),		vbo[NORMAL_OBJECT],		cudaGraphicsMapFlagsWriteDiscard));
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Delete VBO
	////////////////////////////////////////////////////////////////////////////////
	void deleteCudaVBO()
	{
		//glBindBufferARB(1, vbo);
		//CUT_CHECK_ERROR_GL();
		if (vbo[POSITION_OBJECT] == 0)
			return;	// there is nothing to do

		glDeleteBuffers(2, vbo);
		CUT_CHECK_ERROR_GL();

		// unregister this buffer object with CUDA, if needed
		if (cuda_vbo_resource[POSITION_OBJECT])
		{
			checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource[POSITION_OBJECT]));
			checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource[NORMAL_OBJECT]));
		}
		cuda_vbo_resource[POSITION_OBJECT] = 0;
		cuda_vbo_resource[NORMAL_OBJECT] = 0;
		vbo[POSITION_OBJECT] = 0;
		vbo[NORMAL_OBJECT] = 0;
	}

	GLuint GetInteropVbo(VboType type)
	{
		return vbo[type];
	}

	double GetTimeStep() {return time_step;}

	// it builds a squared cloth composed by nb x nb particles
	// the spatial dimensinos are [0, .., 1] x [0, .., 1] regardless the value of nb
	void BuildCloth(int nb, float _stiffness = 3000.f, float _damp = 1.f, float _inv_mass = 1.f)
	{
		Reset();
		ResetGMem();

		// Read and compile shader program(s)
		shaderProgramObject = loadAndCompileShaders("verlet_cloth.vs", "verlet_cloth.fs");
		glUseProgram(0);

		int nb_vertices = nb * nb;	// number of vertices

		for (int h = 0; h < nb; h++)
		{
			for (int w = 0; w < nb; w++)
			{
				positions.push_back(w / (double)(nb - 1));	// x
				positions.push_back(0);						// y
				positions.push_back(h / (double)(nb - 1));	// z
				positions.push_back(1);						// w
			}
		}

		int nbfaces = (nb - 1) * (nb - 1) * 2;		// number of triangular faces

		//   i+0,j+0 -- i+0,j+1
		//      |   \     |
		//      |    \    |
		//      |     \   |
		//      |      \  |
		//   i+1,j+0 -- i+1,j+1
		//
		for (int h = 0; h < nb - 1; h++)
		{
			for (int w = 0; w < nb - 1; w++)
			{
				faces.push_back(h * nb + w);
				faces.push_back((h + 1) * nb + w + 1);
				faces.push_back(h * nb + w + 1);

				faces.push_back(h * nb + w);
				faces.push_back((h + 1) * nb + w);
				faces.push_back((h + 1) * nb + w + 1);
			}
		}

		stretch_stiffness = _stiffness;
		damp = _damp;
		inverse_mass = _inv_mass;

		InitBuffers();
		InitGMem();
	}

	// initialize global memory for the actual computation of the physical simulation
	// only the memory needed from the current platform is initialized
	void InitGMem()
	{
		InitInteropVBO();// create interop VBO
		switch (computation)
		{
		case GLSL:		
			InitFBOs();
			break;
		case OPENCL:
			InitCL();
			break;
		case CUDA:
			InitCuda(positions.size());
			break;
		}


		// shows currently available video memory
		GLint total_mem_kb = 0;
		glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

		GLint cur_avail_mem_kb = 0;
		glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

		printf("total video memory: %d   currently available: %d\n", total_mem_kb, cur_avail_mem_kb);
	}


	// initialize the data buffer to store the state of the particle system (positions, etc.)
	// buffers shall be loaded in FBOs and used for GLSL computations
	void InitBuffers()
	{
		texture_size = nearest_pow(ceil(sqrt((float)positions.size() / 4.0)));
		int nb_particles = texture_size * texture_size;

		positions_old.assign(positions.begin(), positions.end());
		normals.assign(nb_particles * 4, 0.f);

		// texcoords is used by the fabric_plaid shader for visualization
		for (int i = 0; i < nb_particles; i++)
		{
			texcoords.push_back(positions[i * 4]);
			texcoords.push_back(positions[i * 4 + 2]);
		}

		// count how many springs will be simulated in this piece of cloth
		nb_springs = 0;
		for (int index = 0; index < nb_particles; index++)
		{
			int ix = index % texture_size; 
			int iy = index / texture_size; 

			for (int k = 0; k < 12; k++)
			{
				int i, j;
				nextNeigh(k, i, j);

				if (((iy + i) < 0) || ((iy + i) > (texture_size - 1)))
					continue;

				if (((ix + j) < 0) || ((ix + j) > (texture_size - 1)))
					continue;

				nb_springs++;
			}
		}

		// textures must be entirely filled, in case the side of the cloth is not a power of two
		// so, push back as many 0s as needed
		positions.insert(positions.end(), (nb_particles * 4) - positions.size(), 0.f);
		positions_old.insert(positions_old.end(), (nb_particles * 4) - positions_old.size(), 0.f);
		normals.insert(normals.end(), (nb_particles * 4) - normals.size(), 0.f);
	}

	// initialize the FBOs for GLSL computation
	void InitFBOs()
	{
		vector<float*> data;
		data.push_back(&positions[0]);
		data.push_back(&positions_old[0]);
		data.push_back(&normals[0]);
		fbo_0 = initFloatFBOMRT(texture_size, texture_size, data);
		fbo_1 = initFloatFBOMRT(texture_size, texture_size, data);

		// send now initialization parameters to shader program
		setUniforms(shaderProgramObject);
	}

	// initialize the memory buffers for OpenCL computation
	void InitCL()
	{
		// Memory Setup
		allocateArray(&hSourcePos,		positions.size() * sizeof(cl_float));
		allocateArray(&hSourcePosOld,	positions.size() * sizeof(cl_float));
		allocateArray(&hDestPos,		positions.size() * sizeof(cl_float));
		allocateArray(&hDestPosOld,		positions.size() * sizeof(cl_float));

		const char* clSourcefile = "verlet_cloth.cl";
		const char* clKernelName = "verlet_cloth";
		createProgramAndKernel(clSourcefile, clKernelName, cpProgram, ckKernel);

		copyArrayToDevice(hSourcePos, &(positions[0]), 0, positions.size() * sizeof(cl_float));
		copyArrayToDevice(hSourcePosOld, &(positions_old[0]), 0, positions.size() * sizeof(cl_float));
		cl_int ciErrNum = CL_SUCCESS;
		ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *)&(vbo_cl[POSITION_OBJECT]));
		ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *)&(vbo_cl[NORMAL_OBJECT]));

		float fStiffness = (float)stretch_stiffness;
		ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), (void *)&(fStiffness));
		oclCheckError(ciErrNum, CL_SUCCESS);

		float fDamp = (float)damp;
		ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(float), (void *)&(fDamp));
		oclCheckError(ciErrNum, CL_SUCCESS);

		ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(float), (void *)&(inverse_mass));
		oclCheckError(ciErrNum, CL_SUCCESS);

		ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(int), (void *)&(texture_size));
		oclCheckError(ciErrNum, CL_SUCCESS);

		ciErrNum |= clSetKernelArg(ckKernel,10, sizeof(int), (void *)&(collPrimitives));
		oclCheckError(ciErrNum, CL_SUCCESS);

		oclCheckError(ciErrNum, CL_SUCCESS);
	}


	// advance the simulation of one time step using the selected computation platform
	void TimeStep()
	{
		static ComputationType oldComputation = computation;
		static int oldCollPrimitives = collPrimitives;

		// if the computation platform has been changed, reset the memory and reinit
		if ((oldComputation != computation) || (oldCollPrimitives != collPrimitives))
		{
			oldComputation = computation;
			oldCollPrimitives = collPrimitives;
			Reset();
			BuildCloth(texture_size, stretch_stiffness, damp, inverse_mass);
		}

		// advance the simulation of one time step using the selected computing platform
		switch (computation)
		{
		case CPU:
			VerletCpu();	
			break;
		case GLSL:
			VerletGlsl();	
			break;
		case OPENCL:
			VerletCl();	
			break;
		case CUDA:

			// TODO check if cuda_vbo_resource is NULL
			float4 * pos, * nor;
			{
				PROFILE_SAMPLE("Upload");
				UploadCuda(&(positions[0]), &(positions_old[0]), positions.size());

			    // map OpenGL buffer object for writing from CUDA
				checkCudaErrors(cudaGraphicsMapResources(2, cuda_vbo_resource, 0));
				size_t num_bytes; 
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&pos, &num_bytes, cuda_vbo_resource[POSITION_OBJECT]));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&nor, &num_bytes, cuda_vbo_resource[NORMAL_OBJECT]));
//				printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
			}
			{
				PROFILE_SAMPLE("Comp");
				VerletCuda(pos, nor, &(positions[0]), &(positions_old[0]), positions.size(), texture_size, stretch_stiffness, damp, inverse_mass, collPrimitives);
			}
			{
				PROFILE_SAMPLE("Download");
			//	DownloadCuda(&(positions[0]), &(positions_old[0]), &(normals[0]), positions.size());
				checkCudaErrors(cudaGraphicsUnmapResources(2, cuda_vbo_resource, 0));
			}

			break;
		}
	}	

	// Reshape, set viewport
	// it is used by the GLSL computation
	// by using this, each particle corresponds to a pixel in the FBOs
	void reshape(int w, int h)
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0.0, (GLfloat) w, 0.0, (GLfloat) h);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glViewport(0, 0, w, h);
	}

	// load data for the GLSL program
	void setUniforms(GLuint s)
	{
		GLint loc;

		glUseProgram(s);
		loc = glGetUniformLocation(s, "texUnit");
		glUniform1i(loc, 0); // load texture unit 0 in texUnit
		loc = glGetUniformLocation(s, "texUnit2");
		glUniform1i(loc, 1); // load texture unit 1 in texUnit2

		// load the simulation parameters in the GLSL fragment shader
		loc = glGetUniformLocation(s, "texsize");
		glUniform1f(loc, texture_size);
		loc = glGetUniformLocation(s, "time_step");
		glUniform1f(loc, time_step);
		loc = glGetUniformLocation(s, "damp");
		glUniform1f(loc, damp);
		loc = glGetUniformLocation(s, "stiffness");
		glUniform1f(loc, stretch_stiffness);
		loc = glGetUniformLocation(s, "inv_mass");
		glUniform1f(loc, inverse_mass);
		loc = glGetUniformLocation(s, "collide_sphere");
		glUniform1i(loc, collPrimitives & SPHERE);
		loc = glGetUniformLocation(s, "collide_cylinders");
		glUniform1i(loc, (collPrimitives & CYLINDERS) >> 1);
		loc = glGetUniformLocation(s, "collide_plane");
		glUniform1i(loc, (collPrimitives & PLANE) >> 2);

		glUseProgram(0);
	}

	// this method advances the simulation of one time step using the CPU for computing the new state of the particle system
	// it is implemented as if it would run on a GPU architecture
	void VerletCpu()
	{
		PROFILE_SAMPLE("Comp");

		int nb_particles = texture_size * texture_size;

		Vector3 gravity(0, -9.81, 0);
		vector<float> positions_next;	// used to store the outcome of the numerical integration
		positions_next.assign(nb_particles * 4, 0.f);

		for (int index = 0; index < nb_particles; index++)
		{
			// fetch data from buffers
			Vector3 pos(	positions[index * 4],		positions[index * 4 + 1],		positions[index * 4 + 2]);
			Vector3 pos_old(positions_old[index * 4],	positions_old[index * 4 + 1],	positions_old[index * 4 + 2]);

			Vector3 vel = (pos - pos_old) / time_step;	// estimate velocity

			// ---------------------------------------------------------------------------------------------------
			// used to compute the normal of the vertex corresponding tot he particle
			Vector3 normal(0, 0, 0);
			Vector3 last_dir;
			int iters = 0;
			// ---------------------------------------------------------------------------------------------------

			Vector3 force;	// force due to the springs connected to this particle
			float inv_mass = inverse_mass;
			if ((index == 0) || (index <= (texture_size - 1)))	// the first row of particles have infinite mass (they are fixed)
				inv_mass = 0.f;

			int ix = index % texture_size; 
			int iy = index / texture_size; 

			Vector3 pos_rest(ix / (float)(texture_size - 1), 0,	iy / (float)(texture_size - 1));	// rest position of the particle

			// compute the force due to each spring connected to this particle
			// in the process, evaluate the normall as well
			for (int k = 0; k < 12; k++)
			{
				int i, j;
				nextNeigh(k, i, j);

				if (((iy + i) < 0) || ((iy + i) > (texture_size - 1)))
					continue;

				if (((ix + j) < 0) || ((ix + j) > (texture_size - 1)))
					continue;

				int index_neigh = (iy + i) * texture_size + ix + j;
				Vector3 pos_neigh(positions[index_neigh * 4], positions[index_neigh * 4 + 1], positions[index_neigh * 4 + 2]);
				Vector3 diff = pos_neigh - pos;

				Vector3 dir = diff;
				dir.Normalize();	// normalized direction of the spring

				if ((iters > 0) && (k < 8))
				{
					float angle = dir * last_dir;
					if (angle > 0)
						normal += last_dir ^ dir;
				}
				last_dir = dir;	// stored for the next iteration

				Vector3 pos_neigh_rest((ix + j) / (float)(texture_size - 1), 0,	(iy + i) / (float)(texture_size - 1));	// rest position of the neighbour particle
				float rest_length = (pos_neigh_rest - pos_rest).Length();	// rest length of the spring

				force += (dir * (diff.Length() - rest_length)) * stretch_stiffness - vel * damp * 0.5;
				if (k < 8)
					iters++;
			}

			normal = (normal / (float) -(iters - 1)).Normalize();


			// verlet integration
			Vector3 acc = (force + gravity) *  inv_mass;
			Vector3 tmp = pos;
			pos = pos * 2 - pos_old + acc * time_step * time_step;
			pos_old = tmp;


			// collision with cylinders
			if (collPrimitives & CYLINDERS)
			{
				float step = 0.2;
				for (float f = step; f < 1 - 2 * step; f += step)
				{
					Vector3 center(0, -f, f);
					float radius = step / 2.f;

					Vector3 pos_coll(0, pos.y, pos.z);
					if ((pos_coll - center).Length() < radius)
					{
						// collision
						Vector3 dir_coll = (pos_coll - center).Normalize();
						pos = Vector3(pos.x, center.y, center.z) + dir_coll * radius;
					}
				}
			}

			// collision with sphere
			if (collPrimitives & SPHERE)
			{
				Vector3 center(0.5, -0.5, 0.25);
				float radius = 0.3;
				if ((pos - center).Length() < radius)
				{
					// collision
					Vector3 dir_coll = (pos - center).Normalize();
					pos = center + dir_coll * radius;
				}
			}

			// collision with plane
			if (collPrimitives & PLANE)
			{
				if (pos.y  <= -0.6)
				{
					pos.y = -0.6;
					pos_old += (pos - pos_old) * 0.03;
				}
			}

			// write the results back to buffers
			positions_next[index * 4] = pos.x;		positions_next[index * 4 + 1] = pos.y;		positions_next[index * 4 + 2] = pos.z;
			positions_old[index * 4] = pos_old.x;	positions_old[index * 4 + 1] = pos_old.y;	positions_old[index * 4 + 2] = pos_old.z;
			normals[index * 4] = normal.x;			normals[index * 4 + 1] = normal.y;			normals[index * 4 + 2] = normal.z;
		}

		// finally, all the new positions have been computed => they can be stored as the current positions
		for (int i = 0; i < nb_particles; i++)
		{	positions[i * 4] = positions_next[i * 4]; 	positions[i * 4 + 1] = positions_next[i * 4 + 1]; positions[i * 4 + 2] = positions_next[i * 4 + 2]; }

		unsigned int size = texture_size * texture_size * 4 * sizeof(float);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[POSITION_OBJECT]);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, (GLvoid*)&(positions[0]), GL_DYNAMIC_DRAW);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[NORMAL_OBJECT]);
		glBufferDataARB(GL_ARRAY_BUFFER_ARB, size, (GLvoid*)&(normals[0]), GL_DYNAMIC_DRAW);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	}

	// this method advances the simulation of one time step using the GLSL platform for computing the new state of the particle system
	void VerletGlsl()
	{
		int viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// switch to viewport for computing
		reshape(texture_size, texture_size);

		// -------------------------------------------------------------------
		// compute!!
		static int nb_iters = 0;

		{
			PROFILE_SAMPLE("Upload");

			if ((nb_iters % 2) == 0)
				useFBOMRT(fbo_1, fbo_0, NULL);	// ping...
			else
				useFBOMRT(fbo_0, fbo_1, NULL);	// .. pong
		}

		{
			PROFILE_SAMPLE("Comp");
			applyFilter(shaderProgramObject, texture_size, texture_size);
			glFlush();
		}
		// -------------------------------------------------------------------

		{
			// copy the color buffers to vertex buffers to visualize the updated positions and normals
			// in this way the data never leave the gpu and execution is faster
			PROFILE_SAMPLE("Download");

			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT); 
			glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, vbo[POSITION_OBJECT]); 
			glReadPixels(0, 0, texture_size, texture_size, GL_RGBA, GL_FLOAT, 0); 

			glReadBuffer(GL_COLOR_ATTACHMENT2_EXT); 
			glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, vbo[NORMAL_OBJECT]); 
			glReadPixels(0, 0, texture_size, texture_size, GL_RGBA, GL_FLOAT, 0); 

			glReadBuffer(GL_NONE); 
			glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0 );
			CUT_CHECK_ERROR_GL();
		}

		// restore rendering viewport
		reshape(viewport[2], viewport[3]); 
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

		nb_iters++;
	}

	size_t uSnap(size_t a, size_t b)
	{
		return ((a % b) == 0) ? a : (a - (a % b) + b);
	}

	void VerletCl()
	{
		// set global and local work item dimensions
		const size_t _szLocalWorkSize = 128;
		const size_t _szGlobalWorkSize = uSnap(texture_size * texture_size, _szLocalWorkSize);

		if (positions.empty())	return;

		cl_int ciErrNum = CL_SUCCESS;
		static int iters = 0;

		{
			PROFILE_SAMPLE("Upload");

			if ((iters % 2) == 0)
			{
				ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *)&hSourcePos);
				ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void *)&hSourcePosOld);
				ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void *)&hDestPos);
				ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *)&hDestPosOld);
			}
			else
			{
				ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *)&hDestPos);
				ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void *)&hDestPosOld);
				ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void *)&hSourcePos);
				ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *)&hSourcePosOld);
			}
		}


		// Local memory
		//ciErrNum |= clSetKernelArg(ckKernel, 7, (szLocalWorkSize[0]+(2*16))*(szLocalWorkSize[1]+(2*16))*sizeof(int), NULL); // ???
		{
			PROFILE_SAMPLE("Comp");
			executeKernel(ckKernel, _szGlobalWorkSize, _szLocalWorkSize);
		}

		{
			PROFILE_SAMPLE("Download");

			unsigned int size = texture_size * texture_size * 4 * sizeof(cl_float);
			readGLBufferObject(vbo_cl[POSITION_OBJECT], vbo[POSITION_OBJECT], size);
			readGLBufferObject(vbo_cl[NORMAL_OBJECT], vbo[NORMAL_OBJECT], size);
		}

		iters++;
	}

private:

	// helper function to compute the nearest pover of two of an integer
	static unsigned int nearest_pow (unsigned int num)
	{
		unsigned int n = num > 0 ? num - 1 : 0;

		n |= n >> 1;
		n |= n >> 2;
		n |= n >> 4;
		n |= n >> 8;
		n |= n >> 16;
		n++;

		return n;
	}

	// return the 2D index of the n-th neighbour
	void nextNeigh(int n, int & i, int & j)
	{
		if (n == 0)	{j = -1; i = -1;}
		if (n == 1)	{j =  0; i = -1;}
		if (n == 2)	{j =  1; i = -1;}

		if (n == 3)	{j =  1; i =  0;}
		if (n == 4) {j =  1; i =  1;}
		if (n == 5) {j =  0; i =  1;}

		if (n == 6) {j = -1; i =  1;}
		if (n == 7) {j = -1; i =  0;}

		if (n == 8)	{j = -2; i = -2;}
		if (n == 9) {j =  2; i = -2;}
		if (n ==10) {j =  2; i =  2;}
		if (n ==11) {j = -2; i =  2;}
	}

};

#endif // __PARTICLE_SYSTEM__