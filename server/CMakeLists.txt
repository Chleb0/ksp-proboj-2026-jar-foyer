cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(ksp-proboj-2026-jar LANGUAGES CXX)

add_compile_options(-Wall -Wextra -Wpedantic)

# Setup external dependencies
include(FetchContent)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

# Add subdirectories
add_subdirectory(game)
add_subdirectory(helper)
add_subdirectory(map)
add_subdirectory(protocol)

# Create the executable
add_executable(server main.cpp)

# Link libraries
target_link_libraries(server PRIVATE
    game
    helper
    map
    protocol
)

# Optionally set C++ standard
target_compile_features(server PRIVATE cxx_std_23)
