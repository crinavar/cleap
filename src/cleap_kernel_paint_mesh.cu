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

#ifndef _MDT_KERNEL_PAINT_MESH_H
#define _MDT_KERNEL_PAINT_MESH_H

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: paint mesh
////////////////////////////////////////////////////////////////////////////////
__global__ void cleap_kernel_paint_mesh(float4* mesh_colors, int vertex_count, float r, float g, float b, float a ){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < vertex_count ){
          mesh_colors[i] = make_float4(r, g, b, a);
    }
}
#endif



