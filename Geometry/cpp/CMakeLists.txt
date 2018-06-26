cmake_minimum_required (VERSION 2.6)
project(geometrylib)

if (APPLE)
   set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++ -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set compile flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}") # -lblitz")

# These are used to specify directories for build outputs
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

# Set the install prefix if not specified on command line
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "." CACHE PATH "..." FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Find all the library directories
file(GLOB libraries classes/*)
# Find all the library files
foreach(library ${libraries})
  #message(${library})
  file(GLOB these_sources ${library}/*cc)
  set(library_sources "${library_sources};${these_sources}")
  file(GLOB these_headers ${library}/*h)
  set(library_headers "${library_headers};${these_headers}")
endforeach()
# Set up build

#add_library(geometry STATIC ${library_sources})
add_library(geometry SHARED ${library_sources})
target_link_libraries(geometry) # netcdf netcdf_c++ gsl gslcblas) # blitz)
target_include_directories(geometry PUBLIC ${libraries})
# This allows the headers to be installed
set_target_properties(geometry PROPERTIES PUBLIC_HEADER "${library_headers}")
# Set up install path
install(TARGETS geometry DESTINATION lib PUBLIC_HEADER DESTINATION include)

# Find all the mains
#file(GLOB main_files mains/*cc)
#foreach(main ${main_files})
  #message(${main})
  # Just the name of file, without path or extension
#  get_filename_component(exe_name ${main} NAME_WE)
  # Set up build
#  add_executable(${exe_name} ${main})
#  target_link_libraries(${exe_name} geometry)
  # Set up install path
#  install(TARGETS ${exe_name} DESTINATION bin)
#endforeach()
