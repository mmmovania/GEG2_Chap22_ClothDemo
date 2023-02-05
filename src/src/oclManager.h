
#ifndef PARTICLESYSTEM_COMMON_H
#define OCL_MANAGER_H

//#include <GL/glew.h>
//#include <oclUtils.h>
#include <CL/cl.h>
#include <helper_gl.h>

#include "vector3.h"

////////////////////////////////////////////////////////////////////////////////
// CPU/GPU common types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef cl_mem memHandle_t;


////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
extern "C" void startupOpenCL();
extern "C" void shutdownOpenCL(void);

extern "C" void createProgramAndKernel(const char * clSourcefile, const char * clKernelName, cl_program & cpProgram, cl_kernel & ckKernel);
extern "C" void executeKernel(cl_kernel & ckKernel, const size_t globalSize, const size_t localSize);

extern "C" void allocateArray(memHandle_t *memObj, size_t size);
extern "C" void freeArray(memHandle_t memObj);

extern "C" void allocateClVbo(memHandle_t *vbo_cl, size_t size, GLuint * vbo);
extern "C" void freeClVbo(cl_mem * vbo_cl);

extern "C" void copyArrayFromDevice(void *hostPtr, const memHandle_t memObj, size_t size);
extern "C" void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size);

extern "C" void registerGLBufferObject(unsigned int vbo);
extern "C" void unregisterGLBufferObject(unsigned int vbo);

extern "C" void readGLBufferObject(cl_mem & vbo_cl, unsigned int vbo, int size);

extern "C" void mapGLBufferObject(cl_mem * vbo_cl);
extern "C" void unmapGLBufferObject(cl_mem * vbo_cl);

#endif
