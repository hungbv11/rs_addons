cmake_minimum_required(VERSION 2.8.3)
project(rs_addons)
find_package(catkin REQUIRED robosherlock simtrack_nodes message_generation message_runtime)
################################################################################
## Constants for project                                                      ##
################################################################################
set(NAMESPACE rs_addons)
set(TYPESYSTEM_CPP_PATH ${PROJECT_SOURCE_DIR}/include/rs_addons/types)
set(TYPESYSTEM_XML_PATH ${PROJECT_SOURCE_DIR}/descriptors/typesystem)
set(ANNOTATOR_PATH      ${PROJECT_SOURCE_DIR}/descriptors/annotators)
set(ENGINE_PATH         ${PROJECT_SOURCE_DIR}/descriptors/analysis_engines)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake)
################################################################################
## Configure project configuration files                                      ##
################################################################################
configure_file(${RS_PROJECT_CONFIG} ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake @ONLY)
################################################################################
## Find all include directories                                               ##
################################################################################
find_include_dirs(RS_INCLUDE_DIRS_LIST)
catkin_package()
################################################################################
## Package dependencies                                                       ##
################################################################################
#Add required catkin packages here
#find_package(catkin REQUIRED COMPONENTS xxx)
#Add required system packages here
#find_package(xxx)
################################################################################
## Build                                                                      ##
################################################################################
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${RS_INCLUDE_DIRS_LIST}
        ${catkin_INCLUDE_DIRS}
        )
################################################################################
## Update analysis engines, typesystem and include all relevant files         ##
################################################################################
##Update xml list of annotators inside analysis engines
update_analysis_engines(robosherlock)
## generate classes from the typesystem xml files
generate_type_system(robosherlock)
#find all relevant files
find_additional_files()
################################################################################
## Sub Projects                                                               ##
################################################################################
#If you want to divide your projects into subprojects include the subdirectories
#each containing a CMakeLists.txt here
#add_subdirectory(src/xxx)

rs_add_library(rs_simtrack src/SimtrackDetection.cpp)
target_link_libraries(rs_simtrack rs_core)
find_package(aruco QUIET)

if(aruco_FOUND)
    message("OPTIONAL MODULE aruco FOUND")
    set(OPTIONAL_aruco aruco)
    add_definitions( -DARUCO_FOUND)
    message("Building AR Markers Detector")
    include_directories(${aruco_INCLUDE_DIRS})
    rs_add_library(rs_arMarkersDetector src/ARMarkersDetector.cpp)
    target_link_libraries(rs_arMarkersDetector rs_core ${aruco_LIBS})
endif(aruco_FOUND)
