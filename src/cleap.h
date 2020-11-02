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

#ifndef CLEAP_H
#define CLEAP_H

/** cleap status result for many functionalities. values can be <tt>CLEAP_SUCCESS</tt>, <tt>CLEAP_FAILURE</tt>.
 */
typedef int CLEAP_RESULT;
/** macro status for for success.
 */
#define CLEAP_SUCCESS 	1
/** macro status for for failure.
 */
#define CLEAP_FAILURE 	-1
/** macro value for 2D mode.
 */
#define CLEAP_MODE_2D	2
/** macro value for 3D mode.
 */
#define CLEAP_MODE_3D	3
/** macro value for boolean <tt>true</tt>.
 */
#define CLEAP_TRUE	1
/** macro value for boolean <tt>false</tt>.
 */
#define CLEAP_FALSE	0


#ifdef __cplusplus
extern "C" {
#endif
	/** cleap mesh. It is the library's main structure.
	 * This structure hides all internals to the user and the library provides functions for all operations.
	 * An application should use pointers to this struct in order to handle one or multiple meshes simultaneusly and correctly.
	 */
	//struct _cleap_mesh;
	typedef struct _cleap_mesh cleap_mesh;
	/** Initializes cleap. By default chosses the highest GFlop GPU.
	 * @return <tt>CLEAP_SUCCESS</tt> if initialization was succesful.
	 */
	CLEAP_RESULT cleap_init();
	/** Initializes cleap, creates a dummy OpenGL context and initializes glew. This mode is for applications that only process and dont render anything.
	 * @return <tt>CLEAP_SUCCESS</tt> If initialization was succesful.
	 */
	CLEAP_RESULT cleap_init_no_render();
	/** Ends cleap => destroys OpenGL context too.
	 * @return <tt>CLEAP_SUCCESS</tt> If finalization was succesful.
	 */
	CLEAP_RESULT cleap_end();	
	/** Loads an OFF mesh from he given filename and creates a cleap_mesh instance from it. 
	 * @param filename the full or relative path to the file. For example: "/home/me/meshes/sphere.off".
	 * @return <tt>cleap_mesh*</tt> with it's status = CLEAP_SUCCESS inside, otherwise it will have status = CLEAP_FAILURE or just be null.
	 */
	cleap_mesh* cleap_load_mesh(const char *filename);
	/** Synchronizes mesh's host and device data assuming that device has the latest changes.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>CLEAP_SUCCESS</tt> if synchronization was succesful, otherwise it returns <tt>CLEAP_FAILURE</tt>.
	 */
	CLEAP_RESULT cleap_sync_mesh(cleap_mesh *m);
	/** Deletes and frees all host and device memory used for passed mesh.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>CLEAP_SUCCESS</tt> if all was freed succesfully, otherwise <tt>CLEAP_FAILURE</tt> .
	 */
	CLEAP_RESULT cleap_clear_mesh(cleap_mesh *m);
	/** Saves the mesh into the desired path. The mesh is automatically synced before saving.
	 * @param m a pointer of type cleap_mesh.
	 * @param filename the desired name of the file.
	 * @return <tt>CLEAP_SUCCESS</tt> if mesh was saved succesfully, otherwise <tt>CLEAP_FAILURE</tt> .
	 */
	CLEAP_RESULT cleap_save_mesh(cleap_mesh *m, const char *filename);
	/** Saves the mesh into the desired path, without any synchronization.
	 * @param m a pointer of type cleap_mesh.
	 * @param filename the desired name of the file.
	 * @return <tt>CLEAP_SUCCESS</tt> if mesh was saved succesfully, otherwise <tt>CLEAP_FAILURE</tt> .
	 */
	CLEAP_RESULT cleap_save_mesh_no_sync(cleap_mesh *m, const char *filename);
	/** Paints the mesh with the desired r, g, b, a
	 * @param m a pointer of type cleap_mesh.
	 * @param r the red color value.
	 * @param g the green color value.
	 * @param b the blue color value.
	 * @param a the alpha transparency value. [solid, transparent] <=> [1.0f, 0.0f].
	 * @return <tt>CLEAP_SUCCESS</tt> if mesh was painted succesfully, otherwise <tt>CLEAP_FAILURE</tt> .
	 */
	CLEAP_RESULT cleap_paint_mesh(cleap_mesh *m, float r, float g, float b, float a );
	/** Gets the number of vertices the mesh has.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>int >= 0</tt>.
	 */
	int cleap_get_vertex_count(cleap_mesh *m);
	/** Gets the number of edges the mesh has.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>int >= 0</tt>.
	 */
	int cleap_get_edge_count(cleap_mesh *m);
	/** Gets the number of faces the mesh has.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>int >= 0</tt>.
	 */
	int cleap_get_face_count(cleap_mesh *m);
	/** Prints the mesh vertex, edge and face data.
	 * @param m a pointer of type cleap_mesh.
	 */
	void cleap_print_mesh(cleap_mesh *m);
	/** Gets the x coordinate of the center of the mesh's bounding sphere.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>float</tt>.
	 */
	float cleap_get_bsphere_x(cleap_mesh *m);
	/** Gets the y coordinate of the center of the mesh's bounding sphere.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>float</tt>.
	 */
	float cleap_get_bsphere_y(cleap_mesh *m);
	/** Gets the z coordinate of the center of the mesh's bounding sphere.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>float</tt>.
	 */
	float cleap_get_bsphere_z(cleap_mesh *m);
	/** Gets the radius of the mesh's bounding sphere.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>float >= 0</tt>.
	 */
	float cleap_get_bsphere_r(cleap_mesh *m);
	/** Informs if the mesh is on wireframe mode or not.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>1</tt> if the mesh is on wireframe mode, otherwise <tt>0</tt>.
	 */
	int cleap_mesh_is_wireframe(cleap_mesh *m);
	/** Informs if the mesh is on solid mode or not.
	 * @param m a pointer of type cleap_mesh.
	 * @return <tt>1</tt> if the mesh is on solid mode, otherwise <tt>0</tt>.
	 */
	int cleap_mesh_is_solid(cleap_mesh *m);
	/** Sets the mesh's wireframe mode to enabled or disabled.
	 * @param m a pointer of type cleap_mesh.
	 * @param w the wireframe mode, <tt>1</tt> or <tt>0</tt>.
	 */
	void cleap_mesh_set_wireframe(cleap_mesh *m, int w);
	/** Sets the mesh's solid mode to enabled or disabled.
	 * @param m a pointer of type cleap_mesh.
	 * @param s the solid mode, <tt>1</tt> or <tt>0</tt>.
	 */
	void cleap_mesh_set_solid(cleap_mesh *m, int s);
	/** Render the given mesh with OpenGL buffers.
	 * @param m a pointer of type cleap_mesh. <tt>0</tt>.
	 * @result <tt>CLEAP_SUCCESS</tt> if succesful, otherwise <tt>CLEAP_FAILURE</tt>.
	 */
	CLEAP_RESULT cleap_render_mesh(cleap_mesh *m);
    /** Fix a 2D mesh whose triangles might be reverted.
     * @param m a pointer of type cleap_mesh.
     * @result <tt>CLEAP_SUCCESS</tt> if succesful, otherwise <tt>CLEAP_FAILURE</tt>.
     */
    CLEAP_RESULT cleap_fix_inverted_triangles(cleap_mesh* m);
	/** Fix a mesh whose triangles might be reverted.
	 * @param m a pointer of type cleap_mesh.
	 * @result <tt>CLEAP_SUCCESS</tt> if succesful, otherwise <tt>CLEAP_FAILURE</tt>.
	 */
	CLEAP_RESULT cleap_fix_inverted_triangles_mode(cleap_mesh* m, int mode);
    /** Transforms the mesh into a Delaunay one, via the iterative MDT method.
     * @param m a pointer of type cleap_mesh.
     * @param mode dimensional mode; <tt>CLEAP_MODE_2D</tt> or <tt>CLEAP_MODE_3D</tt>.
     * @result <tt>CLEAP_SUCCESS</tt> if succesful, otherwise <tt>CLEAP_FAILURE</tt>.
     */
    CLEAP_RESULT cleap_delaunay_transformation(cleap_mesh *m, int mode);
	/** Performs one iteration of the MDT method. Educational purposes.
	 * @param m a pointer of type cleap_mesh.
	 * @param mode dimensional mode; <tt>CLEAP_MODE_2D</tt> or <tt>CLEAP_MODE_3D</tt>.
	 * @result <tt>CLEAP_SUCCESS</tt> if succesful, otherwise <tt>CLEAP_FAILURE</tt>.
	 */
	int cleap_delaunay_transformation_interactive(cleap_mesh *m, int mode);
    /** Returns a copy of the OpenGL vertex buffer pointer.
     * @param m a pointer of type cleap_mesh.
     * @result The vertex buffer pointer, of type GLuint.
     */
    GLuint cleap_get_vertex_buffer(cleap_mesh *m);
#ifdef __cplusplus
}
#endif

#endif
