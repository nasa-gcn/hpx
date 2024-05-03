project(
    'hpx',
    'cpp',
    meson_version: '>=1.5.0',
    default_options: 'cpp_std=c++14')

py = import('python').find_installation(pure: false)

add_project_arguments(
    '-DNPY_TARGET_VERSION=NPY_1_23_API_VERSION',
    '-DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION',
    language: 'cpp')

add_project_dependencies(
    dependency('numpy'),
    dependency('CGAL'),
    language: 'cpp')

py.install_sources(
    'lib/hpx/__init__.py',
    'lib/hpx/_core.pyi',
    subdir: 'hpx')

py.extension_module(
    '_core',
    'lib/hpx/_core.cpp',
    limited_api: '3.9',
    subdir: 'hpx',
    install: true)
