# Install script for directory: /home/cristobal/Dropbox/dev/cleaplib/cleap/resources

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cleap-0.3.2" TYPE FILE FILES
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/cleap_doxy_logo.png"
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/cleap_doxy_logo_white.png"
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/cleap_icon_bw.png"
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/cleap_icon_wb.png"
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/cleap_wb.png"
    "/home/cristobal/Dropbox/dev/cleaplib/cleap/resources/gplv3.png"
    )
endif()

