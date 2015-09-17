//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	cleap                                                                   //
//	A library for handling / processing / rendering 3D meshes.	        //
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Copyright © 2011 Cristobal A. Navarro.					//
//										//	
//	This file is part of cleap.						//
//	cleap is free software: you can redistribute it and/or modify		//
//	it under the terms of the GNU General Public License as published by	//
//	the Free Software Foundation, either version 3 of the License, or	//
//	(at your option) any later version.					//
//										//
//	cleap is distributed in the hope that it will be useful,		//
//	but WITHOUT ANY WARRANTY; without even the implied warranty of		//
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	    	//
//	GNU General Public License for more details.				//
//										//
//	You should have received a copy of the GNU General Public License	//
//	along with cleap.  If not, see <http://www.gnu.org/licenses/>. 		//
//										//
//////////////////////////////////////////////////////////////////////////////////

#ifndef _MDT_KERNEL_UTILS_H
#define _MDT_KERNEL_UTILS_H

#define CLEAP_POLY_TRIANGLE 	3
#define CLEAP_EPSILON_DELAUNAY 	0.0000001f
#define CLEAP_PI 		3.14159265f
// A helper macro to get a position
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
////////////////////////////////////////////////////////////////////////////////
/// Kernel -- init Array
////////////////////////////////////////////////////////////////////////////////
__global__ void cleap_kernel_init_array_int(int* array, int size, int value){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<size ){
        array[i] = value;
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Kernel -- Init two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void cleap_kernel_init_device_arrays_dual(int* array1, int* array2, int size, int value){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<size ){
        array1[i] = value;
        array2[i] = value;
    }
}
__device__ float cleap_d_dot_product( float3 u, float3 v ){
        return u.x*v.x + u.y*v.y + u.z*v.z;
}
__device__ float3 cleap_d_cross_product( float3 u, float3 v){
	return make_float3( u.y*v.z - v.y*u.z, u.x*v.z - v.x*u.z, u.x*v.y - v.x*u.y );
}
__device__ float cleap_d_magnitude( float3 v ){
        return __fsqrt_rn( powf(v.x, 2) + powf(v.y, 2) + powf(v.z,2));
}
__device__ float cleap_d_angle_vectors( float3 u, float3 v ){
	return atan2f( cleap_d_magnitude( cleap_d_cross_product(u, v) ), cleap_d_dot_product( u, v ) );
}
__device__ int cleap_d_geometry_angle_test(float4* mesh_data, GLuint* eab, int ta, int tb, float angle_limit){

	// vectors
	float3 u, v, n1, n2;
	float4 p, q;
	// triangle ta points
	p = mesh_data[eab[ta*CLEAP_POLY_TRIANGLE+0]];
	q = mesh_data[eab[ta*CLEAP_POLY_TRIANGLE+1]];
	u = make_float3( q.x - p.x, q.y - p.y, q.z - p.z); // first ta vector
	q = mesh_data[eab[ta*CLEAP_POLY_TRIANGLE+2]]; // q changes for the second vector
	v = make_float3( q.x - p.x, q.y - p.y, q.z - p.z); // second ta vector
	n1 = cleap_d_cross_product(u, v); // compute n1
	// triangle tb points
	p = mesh_data[eab[tb*CLEAP_POLY_TRIANGLE+0]];
	q = mesh_data[eab[tb*CLEAP_POLY_TRIANGLE+1]];
	u = make_float3( q.x - p.x, q.y - p.y, q.z - p.z); // first tb vector
	q = mesh_data[eab[tb*CLEAP_POLY_TRIANGLE+2]];
	v = make_float3( q.x - p.x, q.y - p.y, q.z - p.z); // second tb vector
	n2 =  cleap_d_cross_product(u,v); // compute n2

	return ((int) (fabs(cleap_d_angle_vectors( n1, n2 )) / angle_limit )); // 1 pass, 0 fail
}

//! ~~ 62 flop
__device__ int cleap_d_delaunay_test_3d(const float4* mesh_data, const int op1, const int op2, const int com_a, const int com_b, const float limit_angle ){

	float3 u, v, n1, n2;
	float4 p, q;
	// get two vectors of the first triangle
	p = mesh_data[op1];
	q = mesh_data[com_a];
	u = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	q = mesh_data[com_b];
	v = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	// compute angle
	float alpha = cleap_d_angle_vectors(u, v);
	// compute cross product for 3d filter
	n1 = cleap_d_cross_product(u, v);
	// the same for the other triangle
	p = mesh_data[op2];
	q = mesh_data[com_a];
	u = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	q = mesh_data[com_b];
	v = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	float beta = cleap_d_angle_vectors(u, v);
	n2 = cleap_d_cross_product(v, u);

	return ( (int)(fabs(alpha + beta)/CLEAP_PI - CLEAP_EPSILON_DELAUNAY) * (int)(limit_angle/fabs(cleap_d_angle_vectors( n1, n2 )))  ); //! + 12 flop

}
//! 39 flop
__device__ int cleap_d_delaunay_test_2d(const float4* mesh_data, const int op1, const int op2, const int com_a, const int com_b ){

	float3 u; // vector
	float4 p, q; // points
	// get two vectors of the first triangle
	p = mesh_data[op1];
	q = mesh_data[com_a];
	u = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	q = mesh_data[com_b];
	float alpha = cleap_d_angle_vectors(u, make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w) ); //! + 11 flop
	// the same for other triangle
	p = mesh_data[op2];
	q = mesh_data[com_a];
	u = make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w); //! + 5 flop
	q = mesh_data[com_b];
	float beta = cleap_d_angle_vectors(u, make_float3(q.x - p.x, q.y - p.y, q.z - p.z + 0.0*q.w*p.w)); //! + 11 flop

	return (int)(fabs(alpha + beta)/CLEAP_PI - CLEAP_EPSILON_DELAUNAY); //! + 7 flop

}

//! 39 flop
__device__ int cleap_d_delaunay_test_2d_det(const float4* mesh_data, const int op1, const int op2, const int com_a, const int com_b ){

	float det;
	float4 A = mesh_data[com_a];
	float4 B = mesh_data[com_b];
	float4 C = mesh_data[op1];
	float4 D = mesh_data[op2];

	float A11 = A.x - D.x; 
	float A12 = A.y - D.y; 
	float A13 = (A.x*A.x - D.x*D.x) + (A.y*A.y - D.y*D.y);
	float A21 = B.x - D.x;	
	float A22 = B.y - D.y;
	float A23 = (B.x*B.x - D.x*D.x) + (B.y*B.y - D.y*D.y);
	float A31 = C.x - D.x;
	float A32 = C.y - D.y;
	float A33 = (C.x*C.x - D.x*D.x) + (C.y*C.y - D.y*D.y);
	
	det = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31);

	return signbit(det);

}
#endif

