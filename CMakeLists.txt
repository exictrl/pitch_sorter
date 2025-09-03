
cmake_minimum_required(VERSION 3.16)
project(pitch_sorter VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_executable(pitch_sorter
  src/main.cpp
)

# ---- libsndfile discovery (robust cross-env) ----
# Prefer CMake package from vcpkg/official builds:
find_package(sndfile CONFIG QUIET)

# Fallback to alternate config name used by some installers:
if(NOT sndfile_FOUND)
  find_package(SndFile CONFIG QUIET)
endif()

# Last resort: pkg-config module (редко на Windows, но пусть будет)
if(NOT sndfile_FOUND AND NOT TARGET SndFile::sndfile AND NOT TARGET SndFile::SndFile AND NOT TARGET sndfile::sndfile)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(sndfile IMPORTED_TARGET sndfile)
  endif()
endif()

# Select found imported target
if(TARGET sndfile::sndfile)
  target_link_libraries(pitch_sorter PRIVATE sndfile::sndfile)
elseif(TARGET SndFile::sndfile)
  target_link_libraries(pitch_sorter PRIVATE SndFile::sndfile)
elseif(TARGET SndFile::SndFile)
  target_link_libraries(pitch_sorter PRIVATE SndFile::SndFile)
elseif(TARGET PkgConfig::sndfile)
  target_link_libraries(pitch_sorter PRIVATE PkgConfig::sndfile)
else()
  message(FATAL_ERROR "libsndfile не найден. Используйте vcpkg или укажите путь к пакету (SndFile_DIR/CMAKE_PREFIX_PATH).")
endif()

# Отключаем исключения std::filesystem при копировании (поймаем сами)
target_compile_definitions(pitch_sorter PRIVATE _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
