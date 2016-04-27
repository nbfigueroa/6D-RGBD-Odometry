 get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
 include(${SELF_DIR}/brand-targets.cmake)
 get_filename_component(BRAND_INCLUDE_DIRS "${SELF_DIR}/../../include/brand" ABSOLUTE)
