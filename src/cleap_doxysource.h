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

/** @mainpage User Manual & API Documentation
 * Welcome to cleap's reference manual. In this place you will find the documentation of the functions, data structures and primitives implemented in the library. Additionally, we provide
 * instructions on how to compile, link and run cleap with other programs.
 *
 * @section description Description:
 * cleap is a CUDA based library for handling and processing 3D meshes. 
 * It was created for the purpose of making our Computer Science research
 * accesable and usable by other developers. We hope that in the future this
 * library grows in functionalities and becomes useful to the people :).
 *
 *
 * @section features Features:
 *		- Compiler modularism (users develop their project with normal gcc and g++)
 *		- Parallel gpu accelerated functions:
 *			- Uniform mesh paintaing
 *			- Mesh normals normalization
 *			- Massive Delaunay Transformations or MDT ( <a href="http://eurocg11.inf.ethz.ch/abstracts/35.pdf"> A parallel GPU-based algorithm for Delaunay edge-flips</a> published at EuroCG 2011, Morschach, Switzerland, March 28–30, 2011).
 *
 *		- Abstraction on the mesh data structure.
 *		- Embeddable self rendering for OpenGL applications.
 *
 * @section project_homepage Project Homepage.
 * <a href="https://sourceforge.net/projects/cleap/"> https://sourceforge.net/projects/cleap/ </a>.
 * @section api_ref API Reference.
 * Complete API Documentation of functions and the mesh data structure is available on the html doc (From the top menu, go to Files->File Members).
 * Also, you can check the equivalent pdf manual "refman.pdf" from our projects site 
 * <a href="http://sourceforge.net/projects/cleap/files/refman.pdf/download"> site </a>.
 *
 * @section contents Contents:
 *	- @ref hardware_req
 *	- @ref dependencies
 * 	- @ref install_bin
 *	- @ref install_src
 *	- @ref api_ref
 *	- @ref example
 *	- @ref license
 *
 *
 * @section hardware_req Hardware requirements.
 *	- A CUDA enabled GPU with CC >= 1.1 (Geforce 9800GTX, GTX 2xx, GTX 4xx, GTX 5xx series, to name a few GPUs with Compute Capability >= 1.1)
 *	- GPU support for OpenGL 2.1 or higher.
 *
 * @section dependencies Dependencies.
 *	- Nvidia video driver CUDA capable.
 *	- CUDA runtime >= 3.0.
 *	- GLEW >= 1.5.0 (OpenGL Extension Wrangler).
 *	- cmake (OPTIONAl -- if you wish to re-compile the library).
 *
 *
 * @section install_bin Manual install from binaries.
 * 	- Download the binaries (only *.deb at the moment) from the project's web-site hosted by Sourceforge.
 *	- open a terminal and type:
 * @code
 * ~$ sudo dpkg -i <cleap_package_name>.deb
 * @endcode
 *	- If everything went ok, then the library should be installed inside /usr/local/ in the standard GNU way
 *
 * @section install_src Install from sources.
 *	- Check that you've got Nvidia's CUDA compiler installed and available from $PATH (run "nvcc --version" and see if you got at least CUDA 3.0).
 * 	- Download the sources package at the project's website hosted by Sourceforge. 
 *	  Optionally, you can get the latest build from "git clone git://git.code.sf.net/p/cleap/code cleap-code".
 * 	- Extract the library and go to its main directory, then perform the following commands (x.x is just for being generic):
 * @code
 * ~$ cd cleap-x.x
 * ~$ mkdir build
 * ~$ cd build
 * ~$ cmake ..
 * ~$ make
 * ~$ sudo make install
 * @endcode
 *	- If everything went ok, then the library should be installed inside /usr/local/ in the standard GNU way
 *
 *	- If you want to generate a *.deb package, then instead of
 * @code 
 * make
 * @endcode 
 * 	Use: 
 * @code 
 * make package
 * @endcode.
 *
 *
 *
 * @section compile_link Compile and link with cleap.
 * 	- make sure you set <tt>CFLAGS</tt> properly:
 * @code
 * CFLAGS = -I ${PREFIX}/include/cleap-x.y.z
 * @endcode
 * 	where most of the times ${PREFIX} is = <tt>/usr/local</tt> and x.y.z is the version you downloaded.
 * 	- For linking, make sure you set <tt>LDFLAGS</tt> properly:
 * @code
 * LDFLAGS = -L ${PREFIX}/lib -lcleap
 * @endcode
 *	- For example, if i download and install cleap-1.0.0 version, then i could create an application that uses and compiles with cleap:
 * @code
 * g++ my_application.cpp -o app -I /usr/local/include/cleap-1.0.0 -L /usr/local/lib -lcleap
 * @endcode
 * The same for a "C" only application:
 * @code
 * gcc main.c -o app -I /usr/local/include/cleap-1.0.0 -L /usr/local/lib -lcleap
 * @endcode
 * @section example Example program.
 * You can download the following example program from <a href="http://sourceforge.net/projects/cleap/files/cleap_example.tar.gz/download"> http://sourceforge.net/projects/cleap/files/cleap_example.tar.gz/download </a>.
 * @code
 * #include <stdio.h>
 * #include <cleap.h>
 * 
 * int main(int argc, char* argv[]){
 * 	cleap_mesh* m;
 * 	cleap_init_no_render();
 * 	m = cleap_load_mesh(argv[1]); // pass the mesh as program argument
 *	cleap_delaunay_transformation(m, CLEAP_MODE_2D);
 *	cleap_save_mesh(m, "outmesh.off");
 * }
 * @endcode
 * Mesh is included in the files.
 * @section license License.
 * Copyright © 2011 Cristobal A. Navarro.
 * This software is under the laws and terms of the GPL v3 License.
 */
