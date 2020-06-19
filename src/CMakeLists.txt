# ----------------------------------------------------------
# set gpu architecture parameters
# ----------------------------------------------------------
SET(CUDA_NVCC_FLAGS "-arch=sm_60")



# ----------------------------------------------------------
# compiler options for using unordered_map (C and
# ----------------------------------------------------------
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fmessage-length=0")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)



# ----------------------------------------------------------
# create the shared library
# ----------------------------------------------------------
CUDA_ADD_LIBRARY(cleap ${CLEAP_SOURCES} SHARED)



# ----------------------------------------------------------
# compile for architecture (maybe this can be solved better)
# ----------------------------------------------------------
IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
  SET_TARGET_PROPERTIES(cleap PROPERTIES COMPILE_FLAGS "-fPIC")
ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )




# ------------------------------------------------------
# link library with its dependencies
# ------------------------------------------------------
target_link_libraries(cleap ${CUDA_LIBRARIES} ${GLEW_LIBRARY})



# ------------------------------------------------------
# library properties
# ------------------------------------------------------
set_target_properties(cleap
			PROPERTIES
			VERSION "${CLEAP_VERSION_MAJOR}.${CLEAP_VERSION_MINOR}.${CLEAP_VERSION_PATCH}")




# ------------------------------------------------------
# install library
# ------------------------------------------------------
INSTALL(TARGETS cleap DESTINATION lib)
INSTALL(FILES ${CLEAP_PUBLIC_HEADER} DESTINATION include/${CLEAP_OUTPUT_DIR})
INSTALL(FILES ${CLEAP_CONFIG_HEADER} DESTINATION lib/${CLEAP_OUTPUT_DIR}/include)
