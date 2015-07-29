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

#ifndef _CLEAP_KERNEL_NORMALIZE_NORMALS_H_
#define _CLEAP_KERNEL_NORMALIZE_NORMALS_H_


////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: normalize normals
////////////////////////////////////////////////////////////////////////////////

__global__ void cleap_kernel_normalize_normals(float4* mesh_normals, GLuint num_vertices){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i<num_vertices ){
        float4 n = mesh_normals[i];
        float mod = sqrtf( n.x*n.x + n.y*n.y + n.z*n.z);
        n.x /= mod;
        n.y /= mod;
        n.z /= mod;
        mesh_normals[i] = make_float4(n.x, n.y, n.z, 1.0f);
    }
}


#endif

