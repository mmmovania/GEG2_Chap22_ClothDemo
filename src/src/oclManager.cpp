// Utilities, OpenCL and system includes
#include <oclUtils.h>
#include "oclManager.h"


////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;


//Context initialization/deinitialization
void startupOpenCL()
{
    cl_device_id cdDevice;
    cl_int ciErrNum;
    
    // Get the NVIDIA platform
    shrLog("oclGetPlatformID...\n\n"); 
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get the devices
	shrLog("clGetDeviceIDs...\n\n"); 
	ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);

	// Create the context; 
    // This will fail if the requested device does not support context sharing with OpenGL
	shrLog("clCreateContext...\n\n"); 
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};

	//cl_context_properties props[] = // ONLY FOR WIN32
	//{
	//	CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
	//	CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
	//	CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
	//	0
	//};
//	cxGPUContext = clCreateContext(props, 1, &cdDevice, NULL, NULL, &ciErrNum);
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // List used device 
    shrLog("GPU Device being used:\n"); 
    oclPrintDevInfo(LOGBOTH, cdDevice);

    //Create a command-queue
    shrLog("clCreateCommandQueue...\n\n"); 
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    oclPrintDevName(LOGBOTH, cdDevice);
    shrLog("\n");
}


void shutdownOpenCL(void)
{
    cl_int ciErrNum;
    ciErrNum  = clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void createProgramAndKernel(const char * clSourcefile, const char * clKernelName, cl_program & cpProgram, cl_kernel & ckKernel)
{
    cl_int ciErrNum;

	// Program Setup
    size_t program_length;
    char * source = oclLoadProgSource(clSourcefile, "", &program_length);
    oclCheckError(source != NULL, shrTRUE);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,(const char **) &source, &program_length, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    free(source);

    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL/*"-cl-fast-relaxed-math"*/, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclVerlet.ptx");
//      Cleanup(EXIT_FAILURE); 
		shrLogEx(LOGBOTH | CLOSELOG, 0, "GPGPU cloth exiting...\nPress <Enter> to Quit\n");
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
		exit(EXIT_FAILURE);
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, clKernelName, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}


void executeKernel(cl_kernel & ckKernel, const size_t _szGlobalWorkSize, const size_t _szLocalWorkSize)
{
	cl_int ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &_szGlobalWorkSize, &_szLocalWorkSize, 0,0,0 );
    oclCheckError(ciErrNum, CL_SUCCESS);
}


//GPU buffer allocation
void allocateArray(memHandle_t *memObj, size_t size)
{
    cl_int ciErrNum;
    shrLog(" clCreateBuffer (GPU GMEM, %u bytes)...\n\n", size); 
    *memObj = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void allocateClVbo(memHandle_t *vbo_cl, size_t size, GLuint * vbo)
{
    cl_int ciErrNum;
    shrLog("create OpenCL buffer from GL VBO (GPU GMEM, %u bytes)...\n\n", size); 
    *vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
    //*vbo_cl = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, *vbo, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void freeArray(memHandle_t memObj)
{
    cl_int ciErrNum;
    ciErrNum = clReleaseMemObject(memObj);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void freeClVbo(cl_mem * vbo_cl)
{
   clReleaseMemObject(*vbo_cl);
}


//host<->device memcopies
void copyArrayFromDevice(void *hostPtr, memHandle_t memObj, size_t size)
{
    cl_int ciErrNum;
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size)
{
	cl_int ciErrNum;
	ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
	oclCheckError(ciErrNum, CL_SUCCESS);
}


void readGLBufferObject(cl_mem & vbo_cl, unsigned int vbo, int size)
{
	cl_int ciErrNum;
	// Explicit Copy 
	// map the PBO to copy data from the CL buffer via host
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);    

	// map the buffer object into client's memory
	float* ptr = (float*)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);

	ciErrNum = clEnqueueReadBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
	clFinish(cqCommandQueue);
	oclCheckError(ciErrNum, CL_SUCCESS);

	glUnmapBufferARB(GL_ARRAY_BUFFER_ARB); 
	oclCheckError(ciErrNum, CL_SUCCESS);
}


//// USE THESE ONLY IN A OPENCL CONTEXT WITH OPENGL INTEROPERABILITY ENABLED

//Map/unmap OpenGL buffer object to/from Compute buffer
void mapGLBufferObject(cl_mem * vbo_cl)
{
	cl_int ciErrNum;
	// map OpenGL buffer object for writing from OpenCL
	glFinish();
	ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, vbo_cl, 0,0,0);
	oclCheckError(ciErrNum, CL_SUCCESS);
}

void unmapGLBufferObject(cl_mem * vbo_cl)
{
	cl_int ciErrNum;
	// unmap buffer object
	ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, vbo_cl, 0,0,0);
	oclCheckError(ciErrNum, CL_SUCCESS);
	clFinish(cqCommandQueue);
}


