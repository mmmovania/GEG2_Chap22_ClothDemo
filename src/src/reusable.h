#ifndef _REUSABLES_
#define _REUSABLES_

#include <vector>
using namespace std;

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	//#include <gl/glew.h>
	//#include <gl/glut.h>
	#include <gl/glew.h>
	//#include <GL/gl.h>
#endif



#ifdef _MSC_VER
#pragma warning(disable:4305)
#pragma warning(disable:4244)
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#define _CRT_TERMINATE_DEFINED
#endif

// Shader utilities

char* readFile(char * filename);
GLuint loadAndCompileShaders(char *vsFilename, char *fsFilename);

// FBO utilities

// A structure to collect FBO info
typedef struct FBOstruct
{
	GLuint texid;
	GLuint fb;
	GLuint rb;
	int width, height;
} FBOstruct;

typedef struct FBOMRTstruct
{
	vector<GLuint> texid;
	GLuint fb;
	GLuint rb;
	int width, height;
} FBOMRTstruct;

GLuint initTexture(int side, float* data);
struct FBOstruct *initFBO(int width, int height, char *image);
struct FBOstruct *initFloatFBO(int width, int height, float *data);
struct FBOMRTstruct * initFloatFBOMRT(int width, int height, vector<float*> & data);	// fbo with multiple render targets
void useFBO(struct FBOstruct *out, struct FBOstruct *in1, struct FBOstruct *in2);
//void useFBOMRT(struct FBOMRTstruct *out, struct FBOstruct *in1, struct FBOstruct *in2);
void useFBOMRT(struct FBOMRTstruct *out, struct FBOMRTstruct *in1, struct FBOMRTstruct *in2);
void applyFilter(GLuint shaderProgramObject, int width, int height);

#endif
