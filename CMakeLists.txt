cmake_minimum_required(VERSION 3.11.4)
project(CUDA_tests LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)

set(MY_SOURCE_FILES
    include/integer_bit.h
    src/integer_bit.cu
   )


# List header directories in project
set(MY_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

#Create dependancy list
set(DEPENDANCY_LIST "")
LIST(APPEND DEPENDANCY_LIST "my_cuda")

#------------------------------
#- Generic part
#------------------------------
# Search for CUDA files
foreach(SOURCE_FILE IN ITEMS ${MY_SOURCE_FILES})
    if(${SOURCE_FILE} MATCHES "(.*\).cu$")
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        enable_language(CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(MY_CPP_FLAGS ${MY_CPP_FLAGS} -std=c++11)
            set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--ptxas-options=-v -arch=sm_75 -gencode=arch=compute_75,code=sm_75")
        endif()
    endif()
endforeach(SOURCE_FILE)

set(DEPENDANCY_OBJECTS "")
set(LINKED_LIBRARIES "")
#iterate on dependancies
foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    message("${PROJECT_NAME} need dependancy ${DEPENDANCY_ITEM}")
    set(IS_IMPORTED FALSE)
    #Get information from dependancies if not already done
    if(NOT TARGET ${DEPENDANCY_ITEM})
        set(SUB_DEPENDANCY_OBJECTS "")
        set(SUB_LINKED_LIBRARIES "")
        add_subdirectory($ENV{QUICKY_REPOSITORY}/${DEPENDANCY_ITEM} ${CMAKE_CURRENT_BINARY_DIR}/${DEPENDANCY_ITEM})
        get_target_property(IS_IMPORTED ${DEPENDANCY_ITEM} IMPORTED)
        if(NOT ${IS_IMPORTED})
            list(APPEND DEPENDANCY_OBJECTS "$<TARGET_OBJECTS:${DEPENDANCY_ITEM}>" ${SUB_DEPENDANCY_OBJECTS})
        else()
            list(APPEND LINKED_LIBRARIES ${DEPENDANCY_ITEM})
        endif()
        list(APPEND LINKED_LIBRARIES ${SUB_LINKED_LIBRARIES})
    endif()

    # Complete include directories with directories coming from dependancy because of transitivity issues with Object libraries
    if(NOT ${IS_IMPORTED})
        set(MY_INCLUDE_DIRECTORIES ${MY_INCLUDE_DIRECTORIES} $<TARGET_PROPERTY:${DEPENDANCY_ITEM},INCLUDE_DIRECTORIES>)
    endif()

endforeach(DEPENDANCY_ITEM)

#Prepare targets
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(IS_DIRECTORY ${HAS_PARENT})
    message("Declare library ${PROJECT_NAME}")
    add_library(${PROJECT_NAME} OBJECT ${MY_SOURCE_FILES})
    set(SUB_DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS} PARENT_SCOPE)
    set(SUB_LINKED_LIBRARIES ${LINKED_LIBRARIES} PARENT_SCOPE)
else()
    #set(CMAKE_VERBOSE_MAKEFILE ON)
    # string(REPLACE " " ";" DEPENDANCY_OBJECTS ${DEPENDANCY_OBJECTS})
    message("Create target")
    add_executable(${PROJECT_NAME} ${MY_SOURCE_FILES} ${DEPENDANCY_OBJECTS} src/main_${PROJECT_NAME}.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -D$<UPPER_CASE:${PROJECT_NAME}>_SELF_TEST)
    message(Linked librarries ${LINKED_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} ${LINKED_LIBRARIES})
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -ansi -pedantic -g -O0 -DNDEBUG ${MY_CPP_FLAGS})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${MY_INCLUDE_DIRECTORIES})

foreach(DEPENDANCY_ITEM IN ITEMS ${DEPENDANCY_LIST})
    add_dependencies(${PROJECT_NAME} ${DEPENDANCY_ITEM})
endforeach(DEPENDANCY_ITEM)

#Force use of -std=c++11 instead of -std=gnu++11
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if(DEFINED CMAKE_CUDA_HOST_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS)
        if(NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            set_property(TARGET ${EXISTING_TARGET} PROPERTY COMPILE_OPTIONS
                         "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
                        )
        endif()
    endfunction()
    CUDA_CONVERT_FLAGS(${PROJECT_NAME})
endif(DEFINED CMAKE_CUDA_HOST_COMPILER)
#EOF
