
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vViewVec;


void main(void)
{
	float scale = 1.0;
	
	gl_Position = ftransform();
   
	// Pass position to fragment shader
	vPosition = vec3(gl_MultiTexCoord0 * scale);

	// Eye-space lighting
	vNormal = gl_NormalMatrix * gl_Normal;
   
	vViewVec   = -vec3(gl_ModelViewMatrix * gl_Vertex);
}