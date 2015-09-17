# cleap is a GPU-based library for handling, processing and rendering 3D meshes via CUDA and OpenGL.

# 1) Versions history:
	- version 0.3.2 (September, 2015)


# 2) Requirements:
	- CUDA Runtime Library
	- Nvidia GPU supporting CUDA
	- GLEW (with support of at least OpenGL 2.1)

# 3) License:
	cleap is open source and has GPL v3 license. Therefore it may be freely copied, modified, and redistributed under the
	copyright notices stated in the file COPYING.

# 4) Install:
	1) cd cleap
	2) mkdir build
	3) cd build
	4) cmake ..
	5) make
	6) sudo make install
	- after this, cleap will be installed into /usr/local 


# 5) Use the library
	(this steps suppose that prefix is left as PREFIX=/usr/local. If you choose another path, just change "/usr/local" to the new one)
	* configure $CFLAGS to include "-I /usr/local/include/cleap-x.y.z"
	* configure $LDFLAGS to include "-L /usr/local/lib"
	* make sure dynamic linking is set properly: 
		~$ set LD_LIBRARY_PATH = /usr/local/lib:$LD_LIBRARY_PATH

#) Documentation
    Please visit http://users.dcc.uchile.cl/~crinavar/doc/cleap/ for more documentation.
