project(ExploringSfMWithOpenCV)
cmake_minimum_required(VERSION 2.8)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_definitions(-DCOMPILEDWITHC11)
	message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	add_definitions(-DCOMPILEDWITHC0X)
	message(STATUS "Using flag -std=c++0x.")
else()
	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(PCL REQUIRED)
if(EIGEN_INCLUDE_DIRS) # if compiling with PCL, it will bring Eigen with it
	message(STATUS "will use Eigen")
	include_directories(${EIGEN_INCLUDE_DIRS})
	add_definitions( -DUSE_EIGEN )
endif()

#set(SSBA_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/3rdparty/SSBA-3.0/build" CACHE PATH "Directory to find SSBA libs")
#set(SSBA_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/3rdparty/SSBA-3.0/build" CACHE PATH "Directory to find SSBA libs")
set(SSBA_LIBRARY_DIR /home/3rdParty/SSBA/)
set(OpenCV_DIR /home/qin/opencv-2.4.11/build)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
link_directories(
	${SSBA_LIBRARY_DIR}
	)

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib -L/opt/local/lib")
ENDIF(APPLE)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

link_directories(${SSBA_LIBRARY_DIR})
#链接库的绝对路径
link_directories(/home/qin/3rdParty/SSBA/build)
include_directories(${SSBA_LIBRARY_DIR}/../)

include_directories(/home/qin/3rdParty/SSBA)

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__SFM__DEBUG__" )

# Core functions of SfM
file(GLOB SFM_LIB_SOURCES	
	Distance.* 
	Triangulation.* 
	FindCameraMatrices.*
	MultiCameraDistance.*
	IDistance.*
	MultiCameraPnP.*
	Common.*
	IFeatureMatcher.*
	RichFeatureMatcher.*
	OFFeatureMatcher.*
	BundleAdjuster.*
	GPUSURFFeatureMatcher.*
	AbstractFeatureMatcher.*
	SfMUpdateListener.*
)

add_library( ExploringSfMLibrary ${SFM_LIB_SOURCES} )
if(MSVC)
	set_target_properties(ExploringSfMLibrary PROPERTIES COMPILE_FLAGS "/openmp")
	target_link_libraries(ExploringSfMLibrary V3D COLAMD )
endif()
if(APPLE)
	# default is to build SSBA as static libs...
	target_link_libraries(ExploringSfMLibrary 
		"${SSBA_LIBRARY_DIR}/libV3D.a" 
		"${SSBA_LIBRARY_DIR}/libCOLAMD.a"
		)
		
	set_target_properties(ExploringSfMLibrary PROPERTIES COMPILE_FLAGS "-fopenmp")
	set_target_properties(ExploringSfMLibrary PROPERTIES LINK_FLAGS "-fopenmp")
endif()


# UI part
add_executable(ExploringSfMExec 
	Visualization.cpp
	Visualization.h
	main.cpp
	)


target_link_libraries(ExploringSfMExec 
			ExploringSfMLibrary
			${OpenCV_LIBS}
			${PCL_LIBRARIES}
			V3D
			colamd
			)

