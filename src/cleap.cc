#include "cleap_private.h"
// c++ headers
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include <tr1/unordered_map>

typedef std::pair<int, int> pairArco;


CLEAP_RESULT _cleap_generate_edges_hash(_cleap_mesh *m, FILE *off, float prog, float cont, float pbFraction){
	
	// IO:: parsing faces and edges
	int face_type, io_val;
	int face = 3;
	float3 normal;
	float3 v1,v2;
	cont=1.0f;

	// the hash for edges
	std::tr1::unordered_map<int, std::tr1::unordered_map<int, _tmp_edge> > root_hash;
	std::tr1::unordered_map<int, _tmp_edge>::iterator hit;

	std::vector<_tmp_edge*> edge_vector;

	_tmp_edge* aux_tmp_edge;
	int j_sec[3] = {0, 0, 1};
	int k_sec[3] = {1, 2, 2};
	int op_sec[3] = {2, 1, 0};
	int j ,k, op;
	
	for(int i=0; i<m->face_count; i++) {
		io_val = fscanf(off,"%d",&face_type);
		if( face_type == 3 ){
			// scan the three triangle indexes
			io_val = fscanf(off,"%d %d %d",&m->triangles[i*face_type], &m->triangles[i*face_type+1], &m->triangles[i*face_type+2]);
			//Building Edges
			for(int q=0; q<3; q++){
				j=j_sec[q], k=k_sec[q], op=op_sec[q];
				// always the higher first
				if( m->triangles[i*face_type+j] < m->triangles[i*face_type+k]){ 
					k = j;	
					j = k_sec[q]; 
				}
				// ok, first index already existed, check if the second exists or not
				std::tr1::unordered_map<int, _tmp_edge> *second_hash = &root_hash[m->triangles[i*face_type+j]];
				hit = second_hash->find(m->triangles[i*face_type+k]);
				if( hit != second_hash->end() ){
					// the edge already exists, then fill the remaining info
					aux_tmp_edge = &(hit->second);
					aux_tmp_edge->b1 = i*face_type+j;
					aux_tmp_edge->b2 = i*face_type+k;
					aux_tmp_edge->op2 = i*face_type+op;
				}
				else{
					// create a new edge	
					aux_tmp_edge = &(*second_hash)[m->triangles[i*face_type+k]];					// create the low value on secondary_hash	
					aux_tmp_edge->n1 = m->triangles[i*face_type+j];
					aux_tmp_edge->n2 = m->triangles[i*face_type+k];
					aux_tmp_edge->a1 = i*face_type+j;
					aux_tmp_edge->a2 = i*face_type+k;
					aux_tmp_edge->b1 = -1;
					aux_tmp_edge->b2 = -1;
					aux_tmp_edge->op1 = i*face_type+op;
					aux_tmp_edge->op2 = -1;
					aux_tmp_edge->id = edge_vector.size();
					edge_vector.push_back( aux_tmp_edge );
				}
			}
		}
		else{
		    printf("CLEAP::load_mesh::error IO00::mesh has other types of polygons, need triangle only\n");
		    free(m->vnc_data.v);
		    free(m->vnc_data.n);
		    free(m->vnc_data.c);
		    free(m->triangles);
		    fclose(off);
		    m->status = CLEAP_FAILURE;
		    return CLEAP_FAILURE;
		}

		float4 p1 = m->vnc_data.v[m->triangles[i*face]];
		float4 p2 = m->vnc_data.v[m->triangles[i*face+1]];
		float4 p3 = m->vnc_data.v[m->triangles[i*face+2]];
		v1 = make_float3( p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
		v2 = make_float3( p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
		normal.x =   (v1.y * v2.z) - (v2.y * v1.z);
		normal.y = -((v1.x * v2.z) - (v2.x * v1.z));
		normal.z =   (v1.x * v2.y) - (v2.x * v1.y);
		//printf("   Normal= (%f, %f, %f)\n", normal.x, normal.y, normal.z);

		//!Calculate Normals for this face
		m->vnc_data.n[m->triangles[i*face]].x += normal.x;
		m->vnc_data.n[m->triangles[i*face]].y += normal.y;
		m->vnc_data.n[m->triangles[i*face]].z += normal.z;

		m->vnc_data.n[m->triangles[i*face+1]].x += normal.x;
		m->vnc_data.n[m->triangles[i*face+1]].y += normal.y;
		m->vnc_data.n[m->triangles[i*face+1]].z += normal.z;

		m->vnc_data.n[m->triangles[i*face+2]].x += normal.x;
		m->vnc_data.n[m->triangles[i*face+2]].y += normal.y;
		m->vnc_data.n[m->triangles[i*face+2]].z += normal.z;

		// CANSKIP:: progress bar code, nothing important
		//if( i > pbFraction*cont ){
		//    prog += 0.25;
		//    cont += 25.0f;
		//    if( prog > 1.0 ){ prog = 1.0;}
		    //printf("%.0f%%...", prog*100.0); fflush(stdout);
		//}
	}
	m->processed_edges = 0;
	// CLEAP::MESH:: update the edge count, now after being calculated
	m->edge_count = edge_vector.size();
	// CLEAP::MESH:: malloc edge data
	m->edge_data.n = (int2*)malloc( sizeof(int2)*m->edge_count );
	m->edge_data.a = (int2*)malloc( sizeof(int2)*m->edge_count );
	m->edge_data.b = (int2*)malloc( sizeof(int2)*m->edge_count );
	m->edge_data.op = (int2*)malloc( sizeof(int2)*m->edge_count );
	// CLEAP::MESH:: put edge data into its final format that matches the _cleap_mesh structure
	for( int i=0; i<m->edge_count; i++ ){
	    m->edge_data.n[i] = make_int2(edge_vector[i][0].n1, edge_vector[i][0].n2);
	    m->edge_data.a[i] = make_int2(edge_vector[i][0].a1, edge_vector[i][0].a2);
	    m->edge_data.b[i] = make_int2(edge_vector[i][0].b1, edge_vector[i][0].b2);
	    m->edge_data.op[i] = make_int2(edge_vector[i][0].op1, edge_vector[i][0].op2);
	}
	edge_vector.clear();
	//printf("ok\n"); fflush(stdout);
}

CLEAP_RESULT _cleap_host_load_mesh(_cleap_mesh *m, const char* filename){

	int v_count, f_count, e_count, io_val;
	char line[255];
	setlocale(LC_NUMERIC, "POSIX");	// IO :: necessary for other languajes.
	FILE *off = fopen(filename,"r");
	if(!off){
		printf("CLEAP:load_mesh::error::cannot find file \"%s\"\n", filename);
		exit(1);
	}
	
	io_val = fscanf(off,"%s\n",line);
	if(io_val == EOF){
		printf("CLEAP::load_mesh::error:failed at reading line\n");
		m->status = CLEAP_FAILURE;		
		return CLEAP_FAILURE;
	}
	while(true){	// IO :: Ignore comments (line starting with '#')
		fgets(line,255,off);
		if (line[0]!='#')break;
	}
	sscanf(line,"%d %d %d\n",&v_count,&f_count,&e_count);
	m->vertex_count = v_count;
	m->edge_count = e_count;
	m->face_count = f_count;
	_cleap_reset_minmax(m);
	// CLEAP:: malloc host triangles array
	m->triangles = (GLuint*)malloc(sizeof(GLuint)*f_count*3);
	// CLEAP:: malloc vertex data => struct of arrays
	m->vnc_data.v = (float4*)malloc(sizeof(float4)*v_count);
	m->vnc_data.n = (float4*)malloc(sizeof(float4)*v_count);
	m->vnc_data.c = (float4*)malloc(sizeof(float4)*v_count);
	// CANSKIP::Progress bar code, can skip
	float prog = 0.0f; //progress 0.0 to 1.0
	float cont=25.0f;
	float pbFraction = (float)((float)v_count+(float)f_count)/(100.0f);
	//printf("CLEAP::load_mesh::reading...0%%...");
	// IO:: PARSE VERTEX DATA
	for(int i=0; i<v_count; i++) {

		io_val = fscanf(off,"%f %f %f\n",   &m->vnc_data.v[i].x,&m->vnc_data.v[i].y,&m->vnc_data.v[i].z);
		m->vnc_data.v[i].w = 1.0f;
		// normals
		m->vnc_data.n[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		// maximum values
		if (m->vnc_data.v[i].x > m->max_coords.x) m->max_coords.x=m->vnc_data.v[i].x;
		if (m->vnc_data.v[i].y > m->max_coords.y) m->max_coords.y=m->vnc_data.v[i].y;
		if (m->vnc_data.v[i].z > m->max_coords.z) m->max_coords.z=m->vnc_data.v[i].z;
		if (m->vnc_data.v[i].x < m->min_coords.x) m->min_coords.x=m->vnc_data.v[i].x;
		if (m->vnc_data.v[i].y < m->min_coords.y) m->min_coords.y=m->vnc_data.v[i].y;
		if (m->vnc_data.v[i].z < m->min_coords.z) m->min_coords.z=m->vnc_data.v[i].z;
		// CANSKIP:: progress bar code, nothing important
		if( i > pbFraction*cont ){
		    prog += 0.25;
		    cont += 25.0f;
		    //printf("%.0f%%...", prog*100.0); fflush(stdout);
		}
	}
	_cleap_generate_edges_hash(m, off, prog, cont, pbFraction);
	//----OPERAR CON LOS DATOS
	fclose(off);
	setlocale(LC_NUMERIC, "");
	//printf( "CLEAP::load_mesh::mesh => (v, e, f)  = (%i, %i, %i)\n", m->vertex_count, m->edge_count, m->face_count);
	m->status = CLEAP_SUCCESS;
	m->wireframe = 0;
	m->solid = 1;

	return CLEAP_SUCCESS;
}
