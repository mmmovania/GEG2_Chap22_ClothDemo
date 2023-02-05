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


#ifdef _MSC_VER
#pragma warning(disable:4305)
#pragma warning(disable:4244)
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif
#include <helper_cuda.h>
#include <helper_gl.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <stdarg.h>


#define USE_PROFILE		// undef to disable profiling
#include <helper_cuda.h>

// Utilities, OpenCL and system includes
#include "oclManager.h"

#include "reusable.h"	// OpenGL, extensions and FBOs
#include "profile.h"
#include "timer.h"

#include <cuda_runtime.h>

//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>



#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <rendercheck_gl.h>

#include "particle_system.h"

bool isExtendedInfo = false;	// if true, verbose information is shown
bool isProfiling = false;		// if true, it activates profiling mode

bool isWireframe = true;		// if true, it activates wireframe visualization 

Vector3 mouse2D, mouse3D;		// position of the mouse in 2d (screen coordinates) and 3d (world coordinates)

// size of the window
int window_size_x = 0, window_size_y = 0;

// used for the trackball implementation
const double m_ROTSCALE = 90.0;
const double m_ZOOMSCALE = 0.008;
float fit_factor = 1.f;
Vector3 trackBallMapping(int x, int y);    // Utility routine to convert mouse locations to a virtual hemisphere
Vector3 lastPoint;                           // Keep track of the last mouse location
enum MovementType { ROTATE, ZOOM, NONE };  // Keep track of the current mode of interaction (which mouse button)
MovementType Movement;                     //    Left-mouse => ROTATE, Central-mouse => ZOOM

static GLdouble objectXform[4][4] = {
	{1.0, 0.0, 0.0, 0.0},
	{0.0, 1.0, 0.0, 0.0},
	{0.0, 0.0, 1.0, 0.0},
	{0.0, 0.0, 0.0, 1.0}
};

GLuint renderShaderProgram = -1;
GLint FPS = 60;		

ParticleSystem ps;	// this is the deformable body
bool isSimulationFreezed = false; // if true, freese the simulation
bool isProfilerShown = false;
double phys_computation_time = 0.0;
double avg_phys_computation_time = 0.0;
double tot_phys_computation_time = 0.0;
int nb_frames = 0;

// implementation of printf with GLUT
void glPrint(float* c, float x, float y, float z, const char *fmt, ...);

// handles key down event
void keyboardDown(unsigned char key, int x, int y) 
{
	switch(key) 
	{
	case '1':
		nbSimIters = 4;
		ps.BuildCloth(32, 55000, 1, 64.0 / (32.0 * 32.0));
		break;

	case '2':
		nbSimIters = 8;
		ps.BuildCloth(64, 180000, 1, 64.0 / (64.0 * 64.0));
		break;

	case '3':
		nbSimIters = 16;
		ps.BuildCloth(128, 650000, 2, 64.0 / (128.0 * 128.0));
		break;

	case '4':
		nbSimIters = 16;
		ps.BuildCloth(256, 1750000, 2.0, 64.0 / (256.0 * 256.0));
		break;

	case 'p':
	case 'P':
		isProfilerShown = !isProfilerShown;
		isProfiling = isProfilerShown;
		isExtendedInfo = isProfilerShown;
		break;

	case 'w':
	case 'W':
		isWireframe = !isWireframe;
		break;

	case 32: // space
		isSimulationFreezed = !isSimulationFreezed;
		break;

	case 'Q':
	case 'q':
	case  27:   // ESC
		exit(0);
		break;
	}

	if (isProfiling) 
		GetProfiler()->Reset();	

	nb_frames = 0;

	glutPostRedisplay();
}

// by pressing left and right cursors, the closed mesh inflates
void specialDown(int key, int x, int y)
{
	switch(key) 
	{
	case GLUT_KEY_LEFT:
		break;

	case GLUT_KEY_RIGHT:
		break;

	case GLUT_KEY_F1:
		computation = CPU;
		break;

	case GLUT_KEY_F2:
		computation = GLSL;
		break;

	case GLUT_KEY_F3:
		computation = OPENCL;
		break;

	case GLUT_KEY_F4:
		computation = CUDA;
		break;
	}

	if (isProfiling)
		GetProfiler()->Reset();	

	nb_frames = 0;
	glutPostRedisplay();
}


void reshape(int width, int height) 
{
	window_size_x = width;
	window_size_y = height;

	// Determine the new aspect ratio
	GLdouble gldAspect = (GLdouble) width/ (GLdouble) height;

	// Reset the projection matrix with the new aspect ratio.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(40.0, gldAspect, 0.01, 60.0);
	glTranslatef( -0.5, 0.0, -2.0 );

	// Set the viewport to take up the entire window.
	glViewport(0, 0, width, height);
}

// handles when a mouse button is pressed / released
void mouseClick(int button, int state, int x, int y) 
{
	mouse2D = Vector3(x, window_size_y - y, 0);

	if (state == GLUT_UP)
	{
		// Turn-off rotations and zoom.
		Movement = NONE;
		glutPostRedisplay();
		return;
	}

	switch (button)
	{
	case (GLUT_LEFT_BUTTON):

		// Turn on user interactive rotations.
		// As the user moves the mouse, the scene will rotate.
		Movement = ROTATE;

		// Map the mouse position to a logical sphere location.
		// Keep it in the class variable lastPoint.
		lastPoint = trackBallMapping( x, y );

		// Make sure we are modifying the MODELVIEW matrix.
		glMatrixMode( GL_MODELVIEW );
		break;

	case (GLUT_MIDDLE_BUTTON):

		// Turn on user interactive zooming.
		// As the user moves the mouse, the scene will zoom in or out
		//   depending on the x-direction of travel.
		Movement = ZOOM;

		// Set the last point, so future mouse movements can determine
		//   the distance moved.
		lastPoint.x = (double) x;
		lastPoint.y = (double) y;

		// Make sure we are modifying the PROJECTION matrix.
		glMatrixMode( GL_PROJECTION );

		break;

	case (GLUT_RIGHT_BUTTON):
		// enable picking of a particle
		Movement = NONE;
		break;
	}

	glutPostRedisplay();
}


// handle any necessary mouse movements through the trackball
void mouseMotion(int x, int y) 
{
	Vector3 direction;
	double pixel_diff;
	double rot_angle, zoom_factor;
	Vector3 curPoint;

	switch (Movement) 
	{
	case ROTATE :  // Left-mouse button is being held down
		{
			curPoint = trackBallMapping( x, y );  // Map the mouse position to a logical sphere location.
			direction = curPoint - lastPoint;
			double velocity = direction.Length();
			if( velocity > 0.0001 )
			{
				// Rotate about the axis that is perpendicular to the great circle connecting the mouse movements.
				Vector3 rotAxis;
				rotAxis = lastPoint ^ curPoint ;
				rot_angle = velocity * m_ROTSCALE;

				// We need to apply the rotation as the last transformation.
				//   1. Get the current matrix and save it.
				//   2. Set the matrix to the identity matrix (clear it).
				//   3. Apply the trackball rotation.
				//   4. Pre-multiply it by the saved matrix.
				glGetFloatv( GL_MODELVIEW_MATRIX, (GLfloat *) objectXform );
				glLoadIdentity();
				glRotatef( rot_angle, rotAxis.x, rotAxis.y, rotAxis.z );
				glMultMatrixf( (GLfloat *) objectXform );

				//  If we want to see it, we need to force the system to redraw the scene.
				glutPostRedisplay();
			}
			break;
		}
	case ZOOM :  // Right-mouse button is being held down
		//
		// Zoom into or away from the scene based upon how far the mouse moved in the x-direction.
		//   This implementation does this by scaling the eye-space.
		//   This should be the first operation performed by the GL_PROJECTION matrix.
		//   1. Calculate the signed distance
		//       a. movement to the left is negative (zoom out).
		//       b. movement to the right is positive (zoom in).
		//   2. Calculate a scale factor for the scene s = 1 + a*dx
		//   3. Call glScalef to have the scale be the first transformation.
		// 
		pixel_diff = y - lastPoint.y; 
		zoom_factor = 1.0 + pixel_diff * m_ZOOMSCALE;
		glScalef( zoom_factor, zoom_factor, zoom_factor );

		// Set the current point, so the lastPoint will be saved properly below.
		curPoint.x = (float) x;  curPoint.y = (float) y;  curPoint.z = 0;

		//  If we want to see it, we need to force the system to redraw the scene.
		glutPostRedisplay();
		break;
	}

	// Save the location of the current point for the next movement. 
	lastPoint = curPoint;	// in spherical coordinates
	mouse2D = Vector3(x, window_size_y - y, 0);	// in window coordinates
}

// draw the coordinate axes
void DrawAxes(double length)
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(length, length, length);

	glLineWidth(2.f);
	glBegin(GL_LINES);

	// x red
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(1.f, 0.f, 0.f);

	// y green
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 1.f, 0.f);

	// z blue
	glColor3f(0.f, 0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 1.f);

	glEnd();
	glLineWidth(1.f);

	glPopMatrix();
}

// draw information on the screen
void DrawInfo()
{
	// -- start Ortographic Mode
	glMatrixMode(GL_PROJECTION);					//Select the projection matrix
	glPushMatrix();									//Store the projection matrix
	glLoadIdentity();								//Reset the projection matrix
	glOrtho(0, window_size_x, window_size_y, 0, -1, 1);		//Set up an ortho screen

	glMatrixMode(GL_MODELVIEW);						//Select the modelview matrix
	glPushMatrix();									//Store the projection matrix

	glLoadIdentity();								//Reset the projection matrix

	float c[3] = {0.1, 0.1, 0.1};
	float y = 25;

	// Display results

	std::string comp;
	switch (computation)
	{
	case CPU:
		comp = "CPU";
		break;
	case GLSL:
		comp = "GLSL";
		break;
	case OPENCL:
		comp = "OpenCL";
		break;
	case CUDA:
		comp = "Cuda";
		break;
	}

	static double start = GetRealTimeInMS();
	int now = GetRealTimeInMS();

	float fps = 1000.0 / (now - start);

	if (isExtendedInfo)
	{
		glPrint(c, 10, y, 0, "[ 1 ] 32x32   [ 2 ] 64x64     [ 3 ] 128x128   [ 4 ] 256x256"); y += 20;
		glPrint(c, 10, y, 0, "[ F1 ] CPU    [ F2 ] GLSL     [ F3 ] OpenCL   [ F4 ] CUDA"); y += 20;
		glPrint(c, 10, y, 0, "dt: %.3f     fps: %.2f      phys_t: %.3f   avg: %.3f   tot: %.3f   frame: %d    nbIters: %d  comp: %s", 
			ps.GetTimeStep() * nbSimIters, fps, phys_computation_time, avg_phys_computation_time, tot_phys_computation_time, nb_frames, nbSimIters, comp.c_str());  y += 20;
		glPrint(c, 10, y, 0, "springs: %d  particles: %d   texsize: %d", ps.nb_springs, ps.positions.size() / 4, ps.texture_size);  y += 20; 
		glPrint(c, 10, y, 0, "[ P ] show / hide profiler"); y += 20;

		if (isProfiling)
		{
			if (isProfilerShown)
			{
				std::stringstream op;
				GetProfiler()->GetRootNode()->DisplayFlatStats(op);

				float x = 10;
				char msg[256] = " ";
				while (strlen(msg) > 0)
				{
					op.getline(msg, 256);
					glPrint(c, x, y, 0, "%s", msg); y += 20;
				}
			}
		}
	}
	else
	{
		glPrint(c, 10, y, 0, "[ 1 ] 32x32   [ 2 ] 64x64     [ 3 ] 128x128   [ 4 ] 256x256"); y += 20;
		glPrint(c, 10, y, 0, "[ F1 ] CPU    [ F2 ] GLSL     [ F3 ] OpenCL   [ F4 ] CUDA"); y += 20;
		glPrint(c, 10, y, 0, "springs: %d  particles: %d", ps.nb_springs, ps.positions.size() / 4);  y += 20; 
		glPrint(c, 10, y, 0, "phys_t: %.3f ms   avg: %.3f ms  frame: %d   comp: %s", 
			phys_computation_time, avg_phys_computation_time, nb_frames, comp.c_str());  y += 20;
		glPrint(c, 10, y, 0, "[ P ] profiling mode   [ W ] wireframe"); y += 20;
	}

	start = now;

	y = 550;
	glPrint(c, 10, y, 0, "[ Left click + drag ] : Rotate          [ Central click + drag ] : Zoom"); y += 25;
	glPrint(c, 10, y, 0, "[ Right click ] : Menu Collision"); y += 25;
	glPrint(c, 10, y, 0, "[ Space ] : Freeze / Continue the simulation "); y += 25;
	glPrint(c, 10, y, 0, "Coded by Marco Fratarcangeli for Game Engine Gems 2"); y += 25;

	glPopMatrix();									//Restore the old projection matrix
	glMatrixMode(GL_PROJECTION);					//Select the projection matrix
	glPopMatrix();									//Restore the old projection matrix
	glMatrixMode(GL_MODELVIEW);
	// -- end Ortographic mode
}

// draw the particle system
// note: I could put this inside ParticleSystem class,
// I preferred to put it here to make ParticleSystem independent from the rendering API

void DrawCollidingSphere()
{
	// collision sphere
	glPushMatrix();
	glTranslatef(0.5, -0.5, 0.25);
	glColor3f(1.f, 0.5f, 0.f);
	gluSphere(gluNewQuadric(), .295, 20, 20);
	glPopMatrix();
}

void DrawCollidingCylinders()
{
	float step = 0.2;
	float radius = step / 2.05f;
	glColor3f(0.3f, 0.8f, 0.8f);
	for (float f = step; f < 1 - 2 * step; f += step)
	{
		glPushMatrix();
		// TODO: check the memory leak here		
		glTranslatef(-0.25, -f, f);
		glRotatef(90, 0, 1, 0);	
		gluCylinder (gluNewQuadric(), radius, radius, 1.5, 20, 20);
		glPopMatrix();
	}
}

void DrawCollidingPlane()
{
	glColor3f(0.3f, 0.f, 0.8f);
	glPushMatrix();
	glTranslatef(0, -0.605, 0);
	glBegin(GL_QUADS);
	glVertex3f(0, 0, 0);
	glVertex3f(1, 0, 0);
	glVertex3f(1, 0, 1);
	glVertex3f(0, 0, 1);
	glEnd();
	glPopMatrix();
}

void DrawParticleSystem()
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glPushMatrix();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (ps.collPrimitives & SPHERE)		DrawCollidingSphere();
	if (ps.collPrimitives & CYLINDERS)	DrawCollidingCylinders();
	if (ps.collPrimitives & PLANE)		DrawCollidingPlane();

	// cheat the last normal
	int size = (int)ps.normals.size() / 4;
	for (int i = 0; i < 4; i++)
		ps.normals[(size - 1) * 4 + i] = ps.normals[(size - 2) * 4 + i];

	// draw the mesh
	glUseProgram(renderShaderProgram);
//	glUseProgram(0);

	//////////////////////////////////////////////////////////// render the cuda vbo
    // render from the vbo
	GLuint vbo_pos =  ps.GetInteropVbo(POSITION_OBJECT);
	GLuint vbo_nor =  ps.GetInteropVbo(NORMAL_OBJECT);
	if (vbo_pos)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo_pos);
		glVertexPointer(4, GL_FLOAT, 0, 0);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo_nor);
		glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glTexCoordPointer(2, GL_FLOAT, sizeof(float) * 2, &(ps.texcoords[0]));

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, (GLsizei)ps.faces.size(), GL_UNSIGNED_SHORT, &(ps.faces[0]));
		glUseProgram(0);

		if (isWireframe)
		{
			glColor3f(0.f, 0.f, 0.2f);
			glEnable(GL_POLYGON_OFFSET_LINE);
			glPolygonOffset(-1.0f, -1.0f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, (GLsizei)ps.faces.size(), GL_UNSIGNED_SHORT, &(ps.faces[0]));
			glDisable(GL_POLYGON_OFFSET_LINE);
		}

		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	glUseProgram(0);

	glPopMatrix();
}

// draw the scene
void draw() 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();
	glTranslatef(0.f, .5f, 0.f);
	glRotatef(45, 0, 1, 0);
	glRotatef(10, 1, 0, 0);
	glRotatef(6, 0, 0, 1);
	DrawAxes(0.15);

	glPushMatrix();
	glScalef(fit_factor, fit_factor, fit_factor);

	// draw the particle system
	DrawParticleSystem();
	glPopMatrix();

	glPopMatrix();

	DrawInfo();

	glutSwapBuffers();
}


void idle() { }

void initGL(int width, int height) 
{
	GLfloat light_diffuse_0[] = {0.6, 0.6, 0.65, 1.0};
	glLightfv (GL_LIGHT0, GL_DIFFUSE,	light_diffuse_0);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	reshape(width, height);

	glClearColor(1.f, 1.f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	renderShaderProgram = loadAndCompileShaders("fabric_plaid.vs", "fabric_plaid.fs");
}


void animation(int t)
{
	if (nb_frames == 0)
		tot_phys_computation_time = 0.0;

	//if (nb_frames > 599)
	//	isSimulationFreezed = true;
	//else
		nb_frames++;

	if (!isSimulationFreezed)
	{
		PROFILE_SAMPLE("TimeStep");

		double start = GetRealTimeInMS();

		if (isProfiling)
			ps.TimeStep();
		else
			for (int i = 0; i < nbSimIters; i++)
				ps.TimeStep();

		double end = GetRealTimeInMS();
		phys_computation_time = end - start;
		tot_phys_computation_time += phys_computation_time;

		assert(nb_frames != 0);
		avg_phys_computation_time = tot_phys_computation_time / (double) nb_frames;

	}

#ifdef USE_PROFILE  
	GetProfiler()->EndFrame();	
#endif

	glutPostRedisplay();

	glutTimerFunc((int) 1000/FPS, animation, 0);
}


void processMenuEvents(int option)
{
	ps.collPrimitives ^= option;

	if (ps.collPrimitives & SPHERE)
		glutChangeToMenuEntry(1, "SPHERE", SPHERE);
	else
		glutChangeToMenuEntry(1, "Sphere", SPHERE);

	if (ps.collPrimitives & CYLINDERS)
		glutChangeToMenuEntry(2, "CYLINDERS", CYLINDERS);
	else
		glutChangeToMenuEntry(2, "Cylinders", CYLINDERS);

	if (ps.collPrimitives & PLANE)
		glutChangeToMenuEntry(3, "PLANE", PLANE);
	else
		glutChangeToMenuEntry(3, "Plane", PLANE);
}

inline int _ConvertSMVer2Cores_local(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
	int current_device = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device = 0;
	int device_count = 0, best_SM_arch = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&device_count);
	// Find the best major SM Architecture GPU device
	while (current_device < device_count) {
		cudaGetDeviceProperties(&deviceProp, current_device);
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;
	while (current_device < device_count) {
		cudaGetDeviceProperties(&deviceProp, current_device);
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
			sm_per_multiproc = 1;
		}
		else {
			sm_per_multiproc = _ConvertSMVer2Cores_local(deviceProp.major, deviceProp.minor);
		}

		int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if (compute_perf > max_compute_perf) {
			// If we find GPU with SM major > 2, search only these
			if (best_SM_arch > 2) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {
					max_compute_perf = compute_perf;
					max_perf_device = current_device;
				}
			}
			else {
				max_compute_perf = compute_perf;
				max_perf_device = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}


int main(int argc, char** argv) 
{
	int width = 1024;
	int height = 768;

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("GPGPU Cloth simulation");

	GLenum res = glewInit();
	if (res != 0)
	{
		printf("GLEW failed to initialize, maybe old graphics card or drivers?\n");
		exit(-1);
	}

	glutKeyboardFunc(keyboardDown);
	glutSpecialFunc(specialDown);

	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(reshape);
	glutDisplayFunc(draw);  
	glutIdleFunc(idle);
	glutTimerFunc((int) 1000/FPS, animation, 0);
	glutIgnoreKeyRepeat(false); // process keys held down

	int menu = glutCreateMenu(processMenuEvents);
	glutAddMenuEntry("Sphere", SPHERE);
	glutAddMenuEntry("Cylinders", CYLINDERS);
	glutAddMenuEntry("Plane", PLANE);
	
	// attach the menu to the right button
	glutAttachMenu(GLUT_RIGHT_BUTTON);
	initGL(width, height);

	startupOpenCL();

//	cudaSetDevice( cutGetMaxGflopsDeviceId() ); 
	checkCudaErrors(cudaGLSetGLDevice(cutGetMaxGflopsDeviceId()));	// use device with highest Gflops/s

	nbSimIters = 4;
	ps.BuildCloth(32, 55000, .5, 64.0 / (32.0 * 32.0));
	

	glutMainLoop();

	return 0;
}

//
// Utility routine to calculate the 3D position of a 
// projected unit vector onto the xy-plane. Given any
// point on the xy-plane, we can think of it as the projection
// from a sphere down onto the plane. The inverse is what we
// are after.
//
Vector3 trackBallMapping(int x, int y)
{
	Vector3 v;
	double d;

	v.x = (2.0 * x - window_size_x) / window_size_x;
	v.y = (window_size_y - 2.0 * y) / window_size_y;
	v.z = 0.0;
	d = v.Length();
	d = (d < 1.0) ? d : 1.0;  // If d is > 1, then clamp it at one.
	v.z = sqrtf( 1.001 - d * d );  // project the line segment up to the surface of the sphere.

	v.Normalize();  // We forced d to be less than one, not v, so need to normalize somewhere.


	return v;
}



// Custom GL "Print" Routine
// needs glut
void glPrint(float* c, float x, float y, float z, const char *fmt, ...)
{
	if (fmt == NULL)	// If There's No Text
		return;			// Do Nothing

	char text[256];		// Holds Our String
	va_list ap;			// Pointer To List Of Arguments

	va_start(ap, fmt);								// Parses The String For Variables
	vsprintf(text,/* 256 * sizeof(char),*/ fmt, ap);	// And Converts Symbols To Actual Numbers
	va_end(ap);										// Results Are Stored In Text

	size_t len = strlen(text);

	if (c != NULL)
		glColor3fv(c);

	glRasterPos3f(x, y, z);
	for(size_t i = 0; i < len; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, text[i]);

}


