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

#ifndef _CLEAP_KERNEL_FIX_INVERTED_TRIANGLES_H
#define _CLEAP_KERNEL_FIX_INVERTED_TRIANGLES_H

#include "cleap_kernel_utils.cu"

#define CLEAP_TRIANGLE_ZERO_AREA_EPS 0.000001

__device__ __host__ float
hmod(float2 v){
    return sqrt(v.x*v.x+v.y*v.y);
}

#define DIST_EPS 0.000001

__device__ __host__ int
invertedTriangleTest(float4 op1, float4 op2, float4 e1, float4 e2)
{
    float2 v0 = distVec(e1, e2);
    float2 v2 = distVec(e1, op2);
    float2 v1 = distVec(e1, op1);

    double d = cross(v2, v0);
    double s = cross(v1, v0);
    double t = cross(v2, v1);

    // this is the case where two particles intersect each other, or said differently, both triangles get their area
    // close to zero
    if((abs(t)<CLEAP_TRIANGLE_ZERO_AREA_EPS) && (abs(s)<CLEAP_TRIANGLE_ZERO_AREA_EPS) && (abs(d)<CLEAP_TRIANGLE_ZERO_AREA_EPS)){
        //printf("v0: %f %f, v1: %f %f, v2: %f %f, d:%f, s:%f, t:%f\n",v0.x,v0.y,v1.x,v1.y,v2.x,v2.y,d,s,t);
        return -1;
    }

    return ((d < 0 && s <= 0 && t <= 0 && s+t >= d) ||
           (d > 0 && s >= 0 && t >= 0 && s+t <= d) ||
           (s < 0 && d <= 0 && -t <= 0 && d-t >= s) ||
           (s > 0 && d >= 0 && -t >= 0 && d-t <= s));
}

template<class T>
__device__ __host__ void swap(T& a, T& b){
    T aux=a;
    a=b;
    b=aux;
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: triangle fix :: exclussion & processing 2D
////////////////////////////////////////////////////////////////////////////////
template<unsigned int block_size>
__global__ void correctTrianglesKernel(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs, int* has_to_swap_vertices){
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];

        __syncthreads();

        if( b_shared_array[threadIdx.x].x != -1 ){
            int test = invertedTriangleTest( mesh_data[triangles[op_shared_array[threadIdx.x].x]], mesh_data[triangles[op_shared_array[threadIdx.x].y]], mesh_data[triangles[a_shared_array[threadIdx.x].x]], mesh_data[triangles[a_shared_array[threadIdx.x].y]]);
            if(test==-1){
                listo[0] = -1;
                has_to_swap_vertices[triangles[a_shared_array[threadIdx.x].y]]=triangles[a_shared_array[threadIdx.x].x];
                has_to_swap_vertices[triangles[a_shared_array[threadIdx.x].x]]=triangles[a_shared_array[threadIdx.x].y];
            }
            if(test>0) {
                listo[0] = (listo[0]==-1?-1:0);
                // exclusion part
                if( atomicExch( &(trireservs[a_shared_array[threadIdx.x].y/3]), i ) == -1 && atomicExch( &(trireservs[b_shared_array[threadIdx.x].y/3]), i ) == -1 ){ //!  + 8 flop
                    // proceed to flip the edges
                    trirel[a_shared_array[threadIdx.x].y/3] = b_shared_array[threadIdx.x].y/3; //! + 8 flop
                    trirel[b_shared_array[threadIdx.x].y/3] = a_shared_array[threadIdx.x].y/3; //! + 8 flop
                    // exchange necessary indexes
                    triangles[a_shared_array[threadIdx.x].x] = triangles[op_shared_array[threadIdx.x].y];
                    triangles[b_shared_array[threadIdx.x].y] = triangles[op_shared_array[threadIdx.x].x];
                    // update the indices of the flipped edge.
                    edges_a[i] = make_int2(op_shared_array[threadIdx.x].x, a_shared_array[threadIdx.x].x);
                    edges_b[i] = make_int2(b_shared_array[threadIdx.x].y, op_shared_array[threadIdx.x].y);
                    // update vertex indices
                    edges_n[i] = make_int2(triangles[op_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].x]);
                    // update oppposites indices
                    edges_op[i] = make_int2(a_shared_array[threadIdx.x].y, b_shared_array[threadIdx.x].x);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: triangle fix :: exclussion & processing 2D
////////////////////////////////////////////////////////////////////////////////
__global__ void repairTrianglesKernel(unsigned int* triangles, int* rotations,
                      int2* edge_idx, int2* edge_ta, int2* edge_tb, int2* edge_op,
                      unsigned int num_edges)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num_edges)
    {
        // use volatile variables, this forces register use. Sometimes manual optimization achieves better performance.
        volatile int2 e = edge_idx[tidx];
        volatile int2 ta = edge_ta[tidx];
        volatile int2 tb = edge_tb[tidx];
        volatile int2 op = edge_op[tidx];
        // if the t_a pair of indexes are broken
        if ((e.x != triangles[ta.x] || e.y != triangles[ta.y]))
        {
            // then repair them.
            int t_index = rotations[ ta.x/3 ];
            if (triangles[3*t_index+0] == e.x)
            {
                ta.x = 3*t_index+0;
                triangles[3*t_index+1] == e.y ? (ta.y = 3*t_index+1, op.x = 3*t_index+2) : (ta.y = 3*t_index+2, op.x = 3*t_index+1);
            }
            else if (triangles[3*t_index+1] == e.x)
            {
                ta.x = 3*t_index+1;
                triangles[3*t_index+0] == e.y ? (ta.y = 3*t_index+0, op.x = 3*t_index+2) : (ta.y = 3*t_index+2, op.x = 3*t_index+0);
            }
            else if (triangles[3*t_index+2] == e.x)
            {
                ta.x = 3*t_index+2;
                triangles[3*t_index+0] == e.y ? (ta.y = 3*t_index+0, op.x = 3*t_index+1) : (ta.y = 3*t_index+1, op.x = 3*t_index+0);
            }
        }
        if (tb.x != -1)
        {
            if ((e.x != triangles[tb.x] || e.y != triangles[tb.y]))
            {
                int t_index = rotations[ tb.x/3 ];
                if (triangles[3*t_index+0] == e.x)
                {
                    tb.x = 3*t_index+0;
                    triangles[3*t_index+1] == e.y ? (tb.y = 3*t_index+1, op.y = 3*t_index+2) : (tb.y = 3*t_index+2, op.y = 3*t_index+1);
                }
                else if(triangles[3*t_index+1] == e.x)
                {
                    tb.x = 3*t_index+1;
                    triangles[3*t_index+0] == e.y ? (tb.y = 3*t_index+0, op.y = 3*t_index+2) : (tb.y = 3*t_index+2, op.y = 3*t_index+0);
                }
                else if(triangles[3*t_index+2] == e.x)
                {
                    tb.x = 3*t_index+2;
                    triangles[3*t_index+0] == e.y ? (tb.y = 3*t_index+0, op.y = 3*t_index+1) : (tb.y = 3*t_index+1, op.y = 3*t_index+0);
                }
            }
        }
        edge_ta[tidx] = make_int2(ta.x, ta.y);
        edge_tb[tidx] = make_int2(tb.x, tb.y);
        edge_op[tidx] = make_int2(op.x, op.y);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: triangle fix transformation :: exclussion & processing 3D
////////////////////////////////////////////////////////////////////////////////
//TODO: this function isn't programmed yet
template<unsigned int block_size>
__global__ void cleap_kernel_triangle_fix_3d(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs){
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay triangle fix :: exclussion & processing 2D debug
////////////////////////////////////////////////////////////////////////////////
//! 2D --> 65 flop
//TODO: this function isn't programmed yet
template<unsigned int block_size>
__global__ void cleap_kernel_triangle_fix_2d_debug(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs, int* flips){
}

#endif