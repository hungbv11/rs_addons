cmake_minimum_required(VERSION 2.8.3)
project(rs_addons)


find_package(simtrack_nodes QUIET)
if(simtrack_nodes_FOUND)
   message("OPTIONAL MODULE simtrack FOUND")
   set(OPTIONAL_simtrack simtrack_nodes)
endif(simtrack_nodes_FOUND)

find_package(Caffe QUIET)

if(Caffe_FOUND)
  message("OPTIONAL MODULE Caffe FOUND")
  set(OPTIONAL_Caffe Caffe)
  add_definitions( -DCAFFE_FOUND ${Caffe_DEFINITIONS})
endif(Caffe_FOUND)

find_package(catkin REQUIRED robosherlock 
	message_generation 
	message_runtime
	${OPTIONAL_simtrack}
	)
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

catkin_package(
  INCLUDE_DIRS ${RS_INCLUDE_DIRS_LIST}
  LIBRARIES ${RS_LIBS}
  CATKIN_DEPENDS robosherlock
  CFG_EXTRAS ${PROJECT_NAME}_config.cmake
)

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

if(simtrack_FOUND)
  rs_add_library(rs_simtrack src/SimtrackDetection.cpp)
  target_link_libraries(rs_simtrack rs_core)
  find_package(aruco QUIET)
endif(simtrack_FOUND)

if(aruco_FOUND)
    message("OPTIONAL MODULE aruco FOUND")
    add_definitions( -DARUCO_FOUND)
    message("Building AR Markers Detector")
    include_directories(${aruco_INCLUDE_DIRS})
    rs_add_library(rs_arMarkersDetector src/ARMarkersDetector.cpp)
    target_link_libraries(rs_arMarkersDetector rs_core ${aruco_LIBS})
endif(aruco_FOUND)


#if(Caffe_FOUND)
  add_library(rs_caffeProxyLib src/CaffeProxy.cpp)
  target_link_libraries(rs_caffeProxyLib ${catkin_LIBRARIES} ${Caffe_LIBRARIES} ${PCL_LIBRARIES})
 
  add_executable(extractfeats src/utils/ExtractFeatsFromCNNLayer)
  target_link_libraries(extractfeats rs_caffeProxyLib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  
  rs_add_library(rs_caffeClassification src/CaffeClassificationAnnotator.cpp)
  target_link_libraries(rs_caffeClassification rs_caffeProxyLib ${catkin_LIBRARIES})

  rs_add_library(rs_deCafClassifier src/DeCafClassifier.cpp)
  target_link_libraries(rs_deCafClassifier rs_caffeProxyLib ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#endif(Caffe_FOUND)
