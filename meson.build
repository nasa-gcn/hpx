project(
    'hpx',
    'cpp',
    meson_version: '1.4.0',
    default_options: 'cpp_std=c++14')

py_mod = import('python')
py = py_mod.find_installation(pure: false, modules:'numpy')

add_project_arguments(
    '-DNPY_TARGET_VERSION=NPY_1_23_API_VERSION',
    '-DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION',
    '-std=c++14',
    language: 'cpp')

add_project_dependencies(
    dependency('numpy'),
    dependency('CGAL'),
    dependency('openmp', required: false),
    language: 'cpp')

py.install_sources('lib/hpx/__init__.py', subdir: 'hpx')

py.extension_module('_core', 'lib/hpx/_core.cpp', subdir: 'hpx', install : true)
