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

#ifndef _CLEAP_KERNEL_DELAUNAY_TRANSFORMATION_H
#define _CLEAP_KERNEL_DELAUNAY_TRANSFORMATION_H

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay transformation :: exclussion & processing 2D
////////////////////////////////////////////////////////////////////////////////
//! 2D --> 65 flop
template<unsigned int block_size>
__global__ void cleap_kernel_exclusion_processing_2d(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs){

    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];

        if( b_shared_array[threadIdx.x].x != -1 ){
		//if( cleap_d_delaunay_test_2d( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
		if( cleap_d_delaunay_test_2d_det( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
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
/// CLEAP::KERNEL:: delaunay transformation :: exclussion & processing 3D
////////////////////////////////////////////////////////////////////////////////
//! 2D --> 65 flop
template<unsigned int block_size>
__global__ void cleap_kernel_exclusion_processing_3d(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs){

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

__global__ void cleap_kernel_repair(GLuint* triangles, int* trirel, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count){

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
__global__ void cleap_kernel_exclusion_processing_2d_debug(float4* mesh_data, GLuint* triangles, int2 *edges_n, int2 *edges_a, int2 *edges_b, int2 *edges_op, int edge_count, int *listo, int* trirel, int* trireservs, int* flips){

    const int i = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    __shared__ int2 a_shared_array[block_size];
    __shared__ int2 b_shared_array[block_size];
    __shared__ int2 op_shared_array[block_size];
    if( i<edge_count ){
        a_shared_array[threadIdx.x] = edges_a[i];
        b_shared_array[threadIdx.x] = edges_b[i];
        op_shared_array[threadIdx.x] = edges_op[i];

        if( b_shared_array[threadIdx.x].x != -1 ){
		//if( cleap_d_delaunay_test_2d( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
		if( cleap_d_delaunay_test_2d_det( mesh_data, triangles[op_shared_array[threadIdx.x].x], triangles[op_shared_array[threadIdx.x].y], triangles[a_shared_array[threadIdx.x].x], triangles[a_shared_array[threadIdx.x].y]) > 0) {
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
#endif
