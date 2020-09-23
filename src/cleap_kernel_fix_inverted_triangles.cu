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

inline __device__ __host__
double2 distVec(float4 a, float4 b)
{
    return make_double2(b.x - a.x, b.y - a.y);
}

inline __device__ __host__ double cross(double2 u, double2 v)
{
    return u.x * v.y - v.x * u.y;
}

__device__ __host__ bool
invertedTriangleTest(float4 op1, float4 op2, float4 e1, float4 e2)
{
    double2 v0 = distVec(e1, e2);
    double2 v2 = distVec(e1, op2);
    double2 v1 = distVec(e1, op1);

    double d = cross(v2, v0);
    double s = cross(v1, v0);
    double t = cross(v2, v1);

    return (d < 0 && s <= 0 && t <= 0 && s+t >= d) ||
           (d > 0 && s >= 0 && t >= 0 && s+t <= d) ||
           (s < 0 && d <= 0 && -t <= 0 && d-t >= s) ||
           (s > 0 && d >= 0 && -t >= 0 && d-t <= s);
}

////////////////////////////////////////////////////////////////////////////////
/// Kernel -- Copy arrays
////////////////////////////////////////////////////////////////////////////////

template<unsigned int block_size>
__global__ void correctTrianglesKernel(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs){
    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];

        if( b_shared_array[threadIdx.x].x != -1 ){
            if( invertedTriangleTest( mesh_data[triangles[op_shared_array[threadIdx.x].x]], mesh_data[triangles[op_shared_array[threadIdx.x].y]], mesh_data[triangles[a_shared_array[threadIdx.x].x]], mesh_data[triangles[a_shared_array[threadIdx.x].y]]) ) {
                //if( cleap_d_delaunay_test_2d_det( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
                listo[0] = 0;
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
/// CLEAP::KERNEL:: delaunay transformation :: exclussion & processing 3D
////////////////////////////////////////////////////////////////////////////////
//! 2D --> 65 flop
template<unsigned int block_size>
__global__ void cleap_kernel_triangle_fix_3d(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs){

    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];
        if( b_shared_array[threadIdx.x].x != -1 ){
            //! 3D mode --> + 62 flop
            if( cleap_d_delaunay_test_3d( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y], 10.0f) > 0) {
                listo[0] = 0;
                // exclusion part
                if( atomicExch( &(trireservs[a_shared_array[threadIdx.x].y/3]), i ) == -1 && atomicExch( &(trireservs[b_shared_array[threadIdx.x].y/3]), i ) == -1 ){ //!  + 8 flop
                    // flip the edge
                    trirel[a_shared_array[threadIdx.x].y/3] = b_shared_array[threadIdx.x].y/3;
                    trirel[b_shared_array[threadIdx.x].y/3] = a_shared_array[threadIdx.x].y/3;
                    // exchange indices
                    triangles[a_shared_array[threadIdx.x].x] = triangles[op_shared_array[threadIdx.x].y];
                    triangles[b_shared_array[threadIdx.x].y] = triangles[op_shared_array[threadIdx.x].x];
                    // update flipped edge
                    edges_a[i] = make_int2(op_shared_array[threadIdx.x].x, a_shared_array[threadIdx.x].x);
                    edges_b[i] = make_int2(b_shared_array[threadIdx.x].y, op_shared_array[threadIdx.x].y);
                    edges_n[i] = make_int2(triangles[op_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].x]);
                    edges_op[i] = make_int2(a_shared_array[threadIdx.x].y, b_shared_array[threadIdx.x].x);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay transformation :: Repair
////////////////////////////////////////////////////////////////////////////////

__global__ void cleap_kernel_repair2(GLuint* triangles, int* trirel, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<edge_count ){
	// use volatile variables, this forces register use. Sometimes manual optimization achieves better performance.
        volatile int2 n = edges_n[i];
        volatile int2 a = edges_a[i];
        volatile int2 b = edges_b[i];
        volatile int2 op = edges_op[i];
	// if the t_a pair of indexes are broken
        if( (n.x != triangles[a.x] || n.y != triangles[a.y]) ){
	    // then repair them.
            int t_index = trirel[ a.x/3 ];
            if( triangles[3*t_index+0] == n.x ){
               a.x = 3*t_index+0;
               triangles[3*t_index+1] == n.y ? (a.y = 3*t_index+1, op.x = 3*t_index+2) : (a.y = 3*t_index+2, op.x = 3*t_index+1);
            }
            else if( triangles[3*t_index+1] == n.x ){
               a.x = 3*t_index+1;
               triangles[3*t_index+0] == n.y ? (a.y = 3*t_index+0, op.x = 3*t_index+2) : (a.y = 3*t_index+2, op.x = 3*t_index+0);
            }
            else if( triangles[3*t_index+2] == n.x ){
               a.x = 3*t_index+2;
               triangles[3*t_index+0] == n.y ? (a.y = 3*t_index+0, op.x = 3*t_index+1) : (a.y = 3*t_index+1, op.x = 3*t_index+0);
            }
        }
        if( b.x != -1 ){
            if( (n.x != triangles[b.x] || n.y != triangles[b.y]) ){
                int t_index = trirel[ b.x/3 ];
                if( triangles[3*t_index+0] == n.x ){
                   b.x = 3*t_index+0;
                   triangles[3*t_index+1] == n.y ? (b.y = 3*t_index+1, op.y = 3*t_index+2) : (b.y = 3*t_index+2, op.y = 3*t_index+1);
                }
                else if( triangles[3*t_index+1] == n.x ){
                   b.x = 3*t_index+1;
                   triangles[3*t_index+0] == n.y ? (b.y = 3*t_index+0, op.y = 3*t_index+2) : (b.y = 3*t_index+2, op.y = 3*t_index+0);
                }
                else if( triangles[3*t_index+2] == n.x ){
                   b.x = 3*t_index+2;
                   triangles[3*t_index+0] == n.y ? (b.y = 3*t_index+0, op.y = 3*t_index+1) : (b.y = 3*t_index+1, op.y = 3*t_index+0);
                }
            }
        }
        edges_a[i] = make_int2(a.x, a.y);
        edges_b[i] = make_int2(b.x, b.y);
        edges_op[i] = make_int2(op.x, op.y);
    }
}


////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay transformation :: exclussion & processing 2D debug
////////////////////////////////////////////////////////////////////////////////
//! 2D --> 65 flop
template<unsigned int block_size>
__global__ void cleap_kernel_triangle_fix_2d_debug(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs, int* flips){

    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];

        if( b_shared_array[threadIdx.x].x != -1 ){
		if( cleap_d_delaunay_test_2d( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
                listo[0] = 0;
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
                    atomicAdd(flips, 1);
                }
            }
        }
    }
}

template<unsigned int block_size>
__global__ void cleap_random_move_points_kernel(float4* mesh_data, float2* displacements, int vertexCount){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=vertexCount)return;
    mesh_data[i].x+=displacements[i].x;
    mesh_data[i].y+=displacements[i].y;
}

#endif