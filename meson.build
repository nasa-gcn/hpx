project(
    'hpx',
    'cpp',
    meson_version: '>=1.4.0',
    default_options: 'cpp_std=c++14')

py = import('python').find_installation(pure: false)

add_project_arguments(
    '-DNPY_TARGET_VERSION=NPY_1_23_API_VERSION',
    '-DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION',
    language: 'cpp')

add_project_dependencies(
    # FIXME: replace with dependency('numpy') once we depend on numpy>=2.0.0rc1
    declare_dependency(include_directories: include_directories(run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check: true).stdout().strip())),
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
