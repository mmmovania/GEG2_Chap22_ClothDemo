
varying vec3 vPosition;
varying vec3 vNormal;
varying vec3 vViewVec;


// Returns a 3d grid pattern
float pattern(vec3 pos, float freq)
{
   vec3 fp = fract(pos * freq);
   fp *= (1.0 - fp);
   return dot(fp, vec3(1.0, 1.0, 1.0));
}



void main(void)
{
   vec4 color = vec4(1.00, 0.08, 0.5, 1.0);
// vec4 color = vec4(1.00, 0.205, 0.0, 1.0);
   vec3 pos = vPosition * 2.5;
   
   // Signed noise
	 float noisy = float(2.0 * noise1(pos) - 1.0);


   // Create the pattern
   float patt = 0.0;
   float freq = 1.47;
   float scale = 0.5;
   
   float noiseScale = 0.07;
   
	float sheen = 0.74 * 0.4;
	float furriness = 20.0;
	vec4 lightDir = vec4(.6, .6, -.514, 0) * 1.8;
	
 
	{ // i = 1
		patt += pattern(pos + noiseScale * noisy, freq) * scale;
		freq *= 2.0;
		scale *= 0.7;
	}
   
	{  // i = 2
		patt += pattern(pos + noiseScale * noisy, freq) * scale;
		freq *= 2.0;
		scale *= 0.7;
	}
  
   {  // i = 3
      patt += pattern(pos + noiseScale * noisy, freq) * scale;
      freq *= 2.0;
      scale *= 0.7;
   }

   {  // i = 4
      patt += pattern(pos + noiseScale * noisy, freq) * scale;
      freq *= 2.0;
      scale *= 0.7;
   }
   
   {  // i = 5
      patt += pattern(pos + noiseScale * noisy, freq) * scale;
      freq *= 2.0;
      scale *= 0.7;
   }


  {  // i= 6
      patt += pattern(pos + noiseScale * noisy, freq) * scale;
     freq *= 2.0;
      scale *= 0.7;
   }
   
   {  // i = 7
      patt += pattern(pos + noiseScale * noisy, freq) * scale;
      freq *= 2.0;
      scale *= 0.7;
   }

	// Apply some fabric style lighting
	float diffuse = 0.25 * (1.0 + dot(vNormal, vec3(lightDir.x, lightDir.y, -lightDir.z))); // this lightDir is -z of the d3d version.
	float cosView = clamp(dot(normalize(vViewVec), vNormal), 0.0, 1.0);
	float shine = pow(1.0 - cosView * cosView, furriness);

	gl_FragColor = (patt * color  + sheen * shine) * diffuse;
}