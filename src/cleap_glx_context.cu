//////////////////////////////////////////////////////////////////////////////////
//                                                                           	//
//	cleap                                                                   //
//	A library for handling / processing / rendering 3D meshes.	        //
//                                                                           	//
//////////////////////////////////////////////////////////////////////////////////
//										//
//	Part of this source code is courtesy of OpenGL.org wiki			//
// 	http://www.opengl.org/wiki/Tutorial:_OpenGL_3.0_Context_Creation_(GLX)  //
//										//
//////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <GL/glx.h>
 
#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
 
// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
int _cleap_is_extension_supported(const char *ext_list, const char *extension);
static bool _cleap_ctx_error_occurred = false;
static int _cleap_ctx_error_handler( Display *dpy, XErrorEvent *ev ){
    _cleap_ctx_error_occurred = true;
    return 0;
}
 
CLEAP_RESULT _cleap_create_glx_context(){

	Display *display = XOpenDisplay(0);
	if ( !display ){
		printf( "CLEAP::create_glx_context::Failed to open X display\n" );
		return CLEAP_FAILURE;
	}
	// get a matching FB config
	static int visual_attribs[] = {
		GLX_X_RENDERABLE    , True,
		GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
		GLX_RENDER_TYPE     , GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
		GLX_RED_SIZE        , 8,
		GLX_GREEN_SIZE      , 8,
		GLX_BLUE_SIZE       , 8,
		GLX_ALPHA_SIZE      , 8,
		GLX_DEPTH_SIZE      , 24,
		GLX_STENCIL_SIZE    , 8,
		GLX_DOUBLEBUFFER    , True,
		//GLX_SAMPLE_BUFFERS  , 1,
		//GLX_SAMPLES         , 4,
		None
	};

	int glx_major, glx_minor;
	// FBConfigs were added in GLX version 1.3.
	if ( !glXQueryVersion( display, &glx_major, &glx_minor ) || ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) ){
		printf( "CLEAP::create_glx_context::Invalid GLX version" );
		return CLEAP_FAILURE;
	}
	//printf( "Getting matching framebuffer configs\n" );
	int fbcount;
	GLXFBConfig *fbc = glXChooseFBConfig( display, DefaultScreen( display ), visual_attribs, &fbcount );
	if ( !fbc ){
		printf( "CLEAP::create_glx_context::Failed to retrieve a framebuffer config\n" );
		return CLEAP_FAILURE;
	}
	//printf( "Found %d matching FB configs.\n", fbcount );
	// Pick the FB config/visual with the most samples per pixel
	//printf( "Getting XVisualInfos\n" );
	int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
	int i;
	for ( i = 0; i < fbcount; i++ ){
		XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
		if ( vi ){
			int samp_buf, samples;
			glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
			glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );
			//printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d, SAMPLES = %d\n", i, vi -> visualid, samp_buf, samples );
			if ( best_fbc < 0 || samp_buf && samples > best_num_samp )
				best_fbc = i, best_num_samp = samples;
			if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
				worst_fbc = i, worst_num_samp = samples;
		}
		XFree( vi );
	}
	GLXFBConfig bestFbc = fbc[ best_fbc ];
	// Be sure to free the FBConfig list allocated by glXChooseFBConfig()
	XFree( fbc );
	// Get a visual
	XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
	//printf( "Chosen visual ID = 0x%x\n", vi->visualid );
	//printf( "Creating colormap\n" );
	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap( display, RootWindow( display, vi->screen ), vi->visual, AllocNone );
	swa.background_pixmap = None ;
	swa.border_pixel      = 0;
	swa.event_mask        = StructureNotifyMask;
	//printf( "Creating window\n" );
	Window win = XCreateWindow( display, RootWindow( display, vi->screen ), 
		              0, 0, 1, 1, 0, vi->depth, InputOutput, 
		              vi->visual, 
		              CWBorderPixel|CWColormap|CWEventMask, &swa );

	if ( !win ){
		printf( "CLEAP::create_glx_context::Failed to create window.\n" );
		return CLEAP_FAILURE;
	}
	// Done with the visual info data
	XFree( vi );
	//XStoreName( display, win, "Dummy GL 3.0 Window" );
	//printf( "Mapping window\n" );
	// commented this line, so the dummy window is invisible
	//XMapWindow( display, win );
	// Get the default screen's GLX extension list
	const char *glxExts = glXQueryExtensionsString( display, DefaultScreen( display ) );
	// NOTE: It is not necessary to create or make current to a context before
	// calling glXGetProcAddressARB
	glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
	glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
	glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );

	GLXContext ctx = 0;

	// Install an X error handler so the application won't exit if GL 3.0
	// context allocation fails.
	//
	// Note this error handler is global.  All display connections in all threads
	// of a process use the same error handler, so be sure to guard against other
	// threads issuing X commands while this code is running.
	_cleap_ctx_error_occurred = false;
	int (*oldHandler)(Display*, XErrorEvent*) = XSetErrorHandler(&_cleap_ctx_error_handler);
	// Check for the GLX_ARB_create_context extension string and the function.
	// If either is not present, use GLX 1.3 context creation method.
	if ( !_cleap_is_extension_supported( glxExts, "GLX_ARB_create_context" ) || !glXCreateContextAttribsARB ){
		printf( "glXCreateContextAttribsARB() not found ... using old-style GLX context\n" );
		ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
	}
	// If it does, try to get a GL 3.0 context!
	else{
		int context_attribs[] = {
			GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
			GLX_CONTEXT_MINOR_VERSION_ARB, 0,
			//GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
			None
		};
		//printf( "Creating context\n" );
		ctx = glXCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
		// Sync to ensure any errors generated are processed.
		XSync( display, False );
		if ( _cleap_ctx_error_occurred || !ctx ){
			// Couldn't create GL 3.0 context.  Fall back to newest old-style 2.x context.
			context_attribs[1] = 1;	// GLX_CONTEXT_MAJOR_VERSION_ARB = 1
			context_attribs[3] = 0;	// GLX_CONTEXT_MINOR_VERSION_ARB = 0
			_cleap_ctx_error_occurred = false;
			printf( "Failed to create GL 3.0 context ... using old-style GLX context\n" );
			ctx = glXCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
		}
	}
	// Sync to ensure any errors generated are processed.
	XSync( display, False );
	// Restore the original error handler
	XSetErrorHandler( oldHandler );
	if ( _cleap_ctx_error_occurred || !ctx ){
		printf( "CLEAP::create_glx_context::Failed to create an OpenGL context\n" );
		return CLEAP_FAILURE;
	}

	glXMakeCurrent( display, win, ctx );

	return CLEAP_SUCCESS;
}

CLEAP_RESULT _cleap_destroy_glx_context(){
	GLXContext ctx = glXGetCurrentContext();
	Display *display = glXGetCurrentDisplay();
	glXMakeCurrent( display, 0, 0 );
	glXDestroyContext( display, ctx );
	//XDestroyWindow( display, win );
	//XFreeColormap( display, cmap );
	XCloseDisplay( display );
	return CLEAP_SUCCESS;
}


int _cleap_is_extension_supported(const char *ext_list, const char *extension){
 
  const char *start;
  const char *where, *terminator;
 
  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if ( where || *extension == '\0' )
    return 0;
 
  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  for ( start = ext_list; ; ) {
	where = strstr( start, extension );

	if ( !where ){
		break;
	}
	terminator = where + strlen( extension );
	if ( where == start || *(where - 1) == ' ' ){
		if ( *terminator == ' ' || *terminator == '\0' ){
			return 1;
		}
	}
	start = terminator;
  }
 
  return 0;
}
