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



#ifndef __VECTOR3__
#define __VECTOR3__

#include <math.h>


// represents a vector in 3D
class Vector3
{
public:
	float x, y, z;		// coordinates

	Vector3()
	{
		x = 0; 
		y = 0; 
		z = 0;
	}

	Vector3(float _x, float _y, float _z)
	{ 
		x = _x; 
		y = _y; 
		z = _z;
	}


	// returns the euclidean norm of the vector
	inline float Length() const
	{
	    return sqrt( x * x + y * y + z * z );
	}
		
	// returns the squared euclidean norm of the vector
	inline float SquaredLength() const
	{
	    return x * x + y * y + z * z;
	}
		
	inline Vector3 & operator =( Vector3 const & v )
	{
			x = v.x; 
			y = v.y; 
			z = v.z;
			return *this;
	}

	// addition among vectors
	inline Vector3 operator + ( Vector3 const & v) const
	{
		return Vector3( x + v.x, y + v.y, z + v.z );
	}

	// addition among vectors
	inline Vector3 & operator += ( Vector3 const & v)
	{
		x += v.x;
		y += v.y;
		z += v.z;

		return *this;
	}

	// difference among vectors
	inline Vector3 & operator -= ( Vector3 const & v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;

		return *this;
	}

	// difference among vectors
	inline Vector3 operator - ( Vector3 const & v) const
	{
		return Vector3( x - v.x, y - v.y, z - v.z );
	}

	// multiplication by a scalar (float)
	inline Vector3 operator * ( const float s ) const
	{
		return Vector3( x * s, y * s, z * s);
	}

	// division by a scalar (float)
	inline Vector3 operator / ( const float s ) const
	{
		return Vector3( x / s, y / s, z / s);
	}

	// division by a scalar (float)
	inline Vector3 & operator /= ( const float s)
	{
		x /= s;
		y /= s;
		z /= s;

		return *this;
	}

	// Cross product
	inline Vector3 operator ^ ( Vector3 const & p ) const
	{
		return Vector3
		(
			y * p.z - z * p.y,
			z * p.x - x * p.z,
			x * p.y - y * p.x
		);
	}

	// Dot product
	inline float operator * ( Vector3 const & p ) const
	{
		return x * p.x + y * p.y + z * p.z;
	}

	// normalizes the vector lenght to unity
	inline Vector3 Normalize()
	{
		float length = Length();

		x /= length;
		y /= length;
		z /= length;

		return *this;
	}
};


#endif //__VECTOR3__