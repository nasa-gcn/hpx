#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Origin.h>
#include <CGAL/surface_neighbor_coordinates_3.h>
#include <cassert>

static void surface_interp_loop(char **args,
                                const npy_intp *dimensions,
                                const npy_intp *steps,
                                void *NPY_UNUSED(data))
{
    assert(steps[0] == 0);
    assert(steps[1] == 0);
    assert(dimensions[2] == 3);

    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
    typedef CGAL::Delaunay_triangulation_3<Kernel, CGAL::Fast_location> Delaunay;
    typedef Kernel::Point_3 Point;
    typedef Kernel::Vector_3 Vector;
    typedef Kernel::FT Coord_type;
    typedef std::vector<std::pair<Point, Kernel::FT>> Point_coordinate_vector;
    typedef std::map<Point, Coord_type, Kernel::Less_xyz_3> Point_value_map;
    typedef std::vector<Point> Point_list;
    typedef CGAL::Data_access<Point_value_map> Value_access;

    Point_list points;
    points.reserve(dimensions[1]);
    Point_value_map values;
    bool good = true;

    for (npy_intp i = 0; good && i < dimensions[1]; i++)
    {
        double xyz[3];
        for (int j = 0; good && j < 3; j++)
        {
            xyz[j] = *(double *)&args[0][i * steps[4] + j * steps[5]];
            if (!std::isfinite(xyz[j]))
                good = false;
        }

        if (good)
        {
            Point point(*(double *)&args[0][i * steps[4]],
                        *(double *)&args[0][i * steps[4] + steps[5]],
                        *(double *)&args[0][i * steps[4] + steps[5] * 2]);
            double value = *(double *)&args[1][i * steps[6]];
            points.push_back(point);
            values.insert(std::make_pair(point, value));
        }
    }

    if (!good)
    {
        for (npy_intp i = 0; i < dimensions[0]; i++)
            *(double *)&args[3][i * steps[3]] = NAN;
        return;
    }

    Delaunay delaunay(points.begin(), points.end());
    Value_access value_access(values);

    for (npy_intp i = 0; i < dimensions[0]; i++)
    {
        double xyz[3], result = NAN;
        good = true;
        for (int j = 0; good && j < 3; j++)
        {
            xyz[j] = *(double *)&args[2][i * steps[2] + j * steps[7]];
            if (!std::isfinite(xyz[j]))
                good = false;
        }
        if (good)
        {
            Point point(xyz[0], xyz[1], xyz[2]);
            Vector normal(point - CGAL::ORIGIN);
            Point_coordinate_vector coords;
            auto norm = CGAL::surface_neighbor_coordinates_3(
                            delaunay, point, normal, std::back_inserter(coords))
                            .second;
            result = CGAL::linear_interpolation(
                coords.begin(), coords.end(), norm, value_access);
        }
        *(double *)&args[3][i * steps[3]] = result;
    }
}

static PyUFuncGenericFunction surface_interp_loops[] = {surface_interp_loop};
static char surface_interp_name[] = "surface_interp";
static char surface_interp_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *no_ufunc_data[] = {NULL};

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_core",
};

PyMODINIT_FUNC PyInit__core(void)
{
    PyObject *module, *obj;
    int fail = 0;

    import_array();
    import_umath();

    module = PyModule_Create(&moduledef);
    if (!module)
        goto done;

    obj = PyUFunc_FromFuncAndDataAndSignature(
        surface_interp_loops, no_ufunc_data, surface_interp_types, 1, 3, 1,
        PyUFunc_None, surface_interp_name, NULL, 0, "(n,3),(n),(3)->()");
    fail = PyModule_AddObjectRef(module, surface_interp_name, obj);
    Py_XDECREF(obj);
    if (fail)
        goto done;

done:
    if (fail)
    {
        Py_DECREF(module);
        module = NULL;
    }
    return module;
}
