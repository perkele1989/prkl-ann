cmake_minimum_required(VERSION 3.2...4.0)
project(prkl-ann)

add_library(prkl-ann STATIC "src/common.hpp" "src/common.cpp" "src/layer.hpp" "src/layer.cpp" "src/model.hpp" "src/model.cpp" "src/set.cpp" "src/set.hpp" "third_party/json.hpp")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(prkl-ann PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(prkl-ann PUBLIC "src" "third_party")
set_property(TARGET prkl-ann PROPERTY CXX_STANDARD 20)

add_executable(prkl-example "example/main.cpp")
target_include_directories(prkl-example PRIVATE "example")
set_property(TARGET prkl-example PROPERTY CXX_STANDARD 20)
target_link_libraries(prkl-example prkl-ann)

add_executable(prkl-train "apps/train.cpp")
target_include_directories(prkl-train PRIVATE "apps")
set_property(TARGET prkl-train PROPERTY CXX_STANDARD 20)
target_link_libraries(prkl-train prkl-ann)

add_executable(prkl-mnist-digits "apps/mnist-digits.cpp")
target_include_directories(prkl-mnist-digits PRIVATE "apps")
set_property(TARGET prkl-mnist-digits PROPERTY CXX_STANDARD 20)
target_link_libraries(prkl-mnist-digits prkl-ann)

add_executable(prkl-evaluate "apps/evaluate.cpp")
target_include_directories(prkl-evaluate PRIVATE "apps")
set_property(TARGET prkl-evaluate PROPERTY CXX_STANDARD 20)
target_link_libraries(prkl-evaluate prkl-ann)


add_executable(math-sandbox "apps/math-sandbox.cpp")
target_include_directories(math-sandbox PRIVATE "apps")
set_property(TARGET math-sandbox PROPERTY CXX_STANDARD 20)
target_link_libraries(math-sandbox prkl-ann)