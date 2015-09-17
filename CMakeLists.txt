# -----------------------------------------------------------
# name of the project.
# -----------------------------------------------------------
project(cleap)









# ------------------------------------------------------
# set variables
# ------------------------------------------------------
set(CLEAP_VERSION_MAJOR 0)
set(CLEAP_VERSION_MINOR 3)
set(CLEAP_VERSION_PATCH 2)

set(CLEAP_SOURCES cleap.cu cleap.cc)
set(CLEAP_PUBLIC_HEADER cleap.h)
set(CLEAP_CONFIG_HEADER cleap_config.h)

set(CLEAP_SHORT_DESCRIPTION "A gpu-based library for handling and processing 3D meshes.")
set(CLEAP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(CLEAP_RESOURCES_DIR ${CMAKE_SOURCE_DIR}/resources)
set(CLEAP_OUTPUT_DIR "cleap-${CLEAP_VERSION_MAJOR}.${CLEAP_VERSION_MINOR}.${CLEAP_VERSION_PATCH}")
set(CLEAP_OUTPUT_NAME "cleap-${CLEAP_VERSION_MAJOR}.${CLEAP_VERSION_MINOR}.${CLEAP_VERSION_PATCH}")







# -----------------------------------------------------------
# minimum cmake version that includes FindCUDA.
# -----------------------------------------------------------
cmake_minimum_required(VERSION 2.8)












# -----------------------------------------------------------
# set path to custom modules
# -----------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake_modules)













# -----------------------------------------------------------
# include directories.
# -----------------------------------------------------------
include_directories(${CLEAP_SOURCE_DIR})













# -----------------------------------------------------------
# generate configure headers
# -----------------------------------------------------------
configure_file( ${CLEAP_SOURCE_DIR}/${CLEAP_CONFIG_HEADER}.in ${CLEAP_SOURCE_DIR}/${CLEAP_CONFIG_HEADER})
configure_file( Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile )










# -----------------------------------------------------------
# search for package dependencies
# -----------------------------------------------------------
find_package(CUDA 3.0)
find_package(OpenGL 2.1)
# GLEW
MESSAGE(STATUS "Looking for GLEW" )
find_package(GLEW)
if(GLEW_FOUND)
	# MESSAGE THAT IT IS FOUND
	MESSAGE(STATUS "Looking for GLEW -- found" )
else(GLEW_FOUND)		
	# MESSAGE THAT IT IS NOT FOUND
	MESSAGE(FATAL_ERROR "Looking for GLEW -- not found")
endif(GLEW_FOUND)











# -----------------------------------------------------------
# generate documentation. It's optional.
# -----------------------------------------------------------
include(FindDoxygen)
# check if doxygen was found
IF(DOXYGEN_EXECUTABLE)
	# MESSAGE THAT IT IS FOUND
	MESSAGE(STATUS "Generating cleap's documentation with Doxygen..." )
	# Executing Doxygen
	EXECUTE_PROCESS( COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile OUTPUT_QUIET)
	INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html ${CMAKE_BINARY_DIR}/doc/latex DESTINATION share/doc/${CLEAP_OUTPUT_DIR})
ELSE(DOXYGEN_EXECUTABLE)
	# WARNING MESSAGE
	message(STATUS "Warning: Doxygen is missing, skipping automatic documentation generation")
ENDIF(DOXYGEN_EXECUTABLE)










# ------------------------------------------------------------
# extra general options
# ------------------------------------------------------------
# verbose mode on
set(CUDA_VERBOSE_PTXAS ${VERBOSE})











# ------------------------------------------------------------
# add tree directories
# ------------------------------------------------------------
ADD_SUBDIRECTORY(${CLEAP_SOURCE_DIR})
ADD_SUBDIRECTORY(${CLEAP_RESOURCES_DIR})











# ------------------------------------------------------------
# package generation
# ------------------------------------------------------------
set(CPACK_GENERATOR "DEB")
set(CPACK_SOURCE_GENERATOR TGZ)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Cristobal A. Navarro")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "a gpu-based mesh processing library")
set(CPACK_PACKAGE_VENDOR "Cristobal A. Navarro")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_PACKAGE_VERSION_MAJOR ${CLEAP_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CLEAP_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CLEAP_VERSION_PATCH})
set(CPACK_SOURCE_IGNORE_FILES "artwork_src" "cleap_upload_doc.sh" "${CMAKE_BINARY_DIR}/*" "${CMAKE_BINARY_DIR}" ".git" "TODO")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CLEAP_OUTPUT_NAME}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CMAKE_PREFIX_PATH})
INCLUDE(CPack)
