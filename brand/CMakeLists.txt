cmake_minimum_required (VERSION 2.6)
project (brand)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package (OpenCV REQUIRED) 
include_directories(include ${OpenCV_INCLUDE_DIRS})
 
FILE(GLOB SRC_FILES src/brand.cpp)
FILE(GLOB HEADER_FILES include/brand.h)


add_library(${PROJECT_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})
install(TARGETS ${PROJECT_NAME} DESTINATION lib EXPORT ${PROJECT_NAME}-targets)
install(EXPORT ${PROJECT_NAME}-targets DESTINATION lib/${PROJECT_NAME})
target_link_libraries ( ${PROJECT_NAME} ${OpenCV_LIBS} )



