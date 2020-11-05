#ifndef _CLEAP_PRIVATE_H_
#define _CLEAP_PRIVATE_H_

// gl headers
#include <GL/glew.h>

// c headers
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <locale.h>
#include <vector_functions.h>
#include <sys/time.h>
#include <map>
#include <vector>

#include "cleap.h"

struct cleap_vnc_data{
    float4 *v; /*!< Detailed description after the member */
    float4 *n; /*!< Detailed description after the member */
    float4 *c; /*!< Detailed description after the member */
};

struct cleap_edge_data{
    int2 *n;
    int2 *a;
    int2 *b;
    int2 *op;
};

struct cleap_device_mesh {
	struct cudaGraphicsResource *vbo_v_cuda, *vbo_n_cuda, *vbo_c_cuda, *eab_cuda;
	GLuint vbo_v, vbo_n, vbo_c, eab;
	int2 *d_edges_n, *d_edges_a, *d_edges_b, *d_edges_op;
	int *d_trirel, *d_trireservs, *d_listo;
	CLEAP_RESULT status;
};


struct _cleap_mesh {
    cleap_vnc_data vnc_data;
    cleap_edge_data edge_data;
    GLuint* triangles;
    int vertex_count, edge_count, face_count;
    float3 max_coords, min_coords;
    int processed_edges, wireframe, solid;
    cleap_device_mesh *dm;
    CLEAP_RESULT status;	// important flag!!

    std::vector<int*> associated_int_buffers;
    std::vector<float*> associated_float_buffers;
    std::vector<double*> associated_double_buffers;
    std::vector<int2*> associated_int2_buffers;
    std::vector<float2*> associated_float2_buffers;
    std::vector<double2*> associated_double2_buffers;
};


typedef struct{
	int id;
	int n1, n2;
	int a1, a2;
	int b1, b2;
	int op1, op2;
} _tmp_edge;

#ifdef __cplusplus
extern "C" {
#endif

// private functions
void 	_cleap_init_array_int(int* h_array, int size, int value);
void 	_cleap_init_device_array_int(int* d_array, int length, int value);
void 	_cleap_init_device_dual_arrays_int(int* d_array1, int* d_array2, int length, int value, dim3 &dimBlock, dim3 &dimGrid);
void 	_cleap_print_gpu_mem();
int  	_cleap_choose_best_gpu_id();
void 	_cleap_print_splash();
void 	_cleap_init_cuda();
void 	_cleap_start_timer();
double 	_cleap_stop_timer();
void 	_cleap_reset_minmax(_cleap_mesh* m);

CLEAP_RESULT _cleap_init_glew();
CLEAP_RESULT _cleap_device_load_mesh(_cleap_mesh* m);
CLEAP_RESULT _cleap_host_load_mesh(_cleap_mesh *m, const char *filename);
CLEAP_RESULT _cleap_normalize_normals(_cleap_mesh *m);

#ifdef __cplusplus
}
#endif
#endif
