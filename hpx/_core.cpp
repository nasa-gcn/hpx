#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Origin.h>
#include <CGAL/surface_neighbor_coordinates_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Delaunay_triangulation_3<Kernel, CGAL::Fast_location> Delaunay;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef Kernel::FT Value;
typedef std::map<Point, Value, Kernel::Less_xyz_3> PointValueMap;

typedef struct
{
    PyObject_HEAD
        Delaunay *delaunay;
    PointValueMap *point_value_map;
} LinearSphericalInterpolator;

static PyObject *LinearSphericalInterpolator_ufunc;
static thread_local LinearSphericalInterpolator *LinearSphericalInterpolator_current;

static int LinearSphericalInterpolator_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
#define FAIL(msg)                                 \
    do                                            \
    {                                             \
        PyErr_SetString(PyExc_ValueError, (msg)); \
        goto fail;                                \
    } while (0)
#define GET_POINT(i, j) *(double *)PyArray_GETPTR2(points_array, (i), (j))
#define GET_VALUE(i) *(double *)PyArray_GETPTR1(values_array, (i))

    int result = -1;
    static const char *kws[] = {"points", "values", NULL};
    PyObject *points, *values;
    PyArrayObject *points_array = NULL, *values_array = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", const_cast<char **>(kws), &points, &values))
        goto fail;

    points_array = reinterpret_cast<PyArrayObject *>(PyArray_FROMANY(
        points, NPY_DOUBLE, 2, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED));
    if (!points_array)
        goto fail;
    values_array = reinterpret_cast<PyArrayObject *>(PyArray_FROMANY(
        values, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED));
    if (!values_array)
        goto fail;

    {
        npy_intp n = PyArray_DIM(points_array, 0);

        if (n != PyArray_DIM(values_array, 0))
            FAIL("points and values must have the same length");
        if (PyArray_DIM(points_array, 1) != 3)
            FAIL("points must have shape (n, 3)");

        for (npy_intp i = 0; i < n; i++)
            for (npy_intp j = 0; j < 3; j++)
                if (!std::isfinite(GET_POINT(i, j)))
                    FAIL("all elements of points must be finite");

        Py_BEGIN_ALLOW_THREADS
        PointValueMap *point_value_map = new std::map<Point, Value, Kernel::Less_xyz_3>();
        Delaunay *delaunay;
        {
            std::vector<Point> point_list;
            point_list.reserve(n);
            for (npy_intp i = 0; i < n; i++)
            {
                Point point(GET_POINT(i, 0), GET_POINT(i, 1), GET_POINT(i, 2));
                Value value = GET_VALUE(i);
                point_list.push_back(point);
                point_value_map->insert(std::make_pair(point, value));
            }
            delaunay = new Delaunay(point_list.cbegin(), point_list.cend());
        }

        auto obj = reinterpret_cast<LinearSphericalInterpolator *>(self);
        obj->delaunay = delaunay;
        obj->point_value_map = point_value_map;
        Py_END_ALLOW_THREADS
    }

    result = 0;
fail:
    Py_XDECREF(points_array);
    Py_XDECREF(values_array);
    return result;
}

static void LinearSphericalInterpolator_finalize(PyObject *self)
{
    auto obj = reinterpret_cast<LinearSphericalInterpolator *>(self);
    delete obj->delaunay;
    delete obj->point_value_map;
}

static PyObject *LinearSphericalInterpolator_call(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    LinearSphericalInterpolator_current = reinterpret_cast<LinearSphericalInterpolator *>(self);
    return PyObject_Call(LinearSphericalInterpolator_ufunc, args, kwargs);
}

static const char LinearSphericalInterpolator_name[] = "LinearSphericalInterpolator";
static PyTypeObject LinearSphericalInterpolator_type{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
                   .tp_basicsize = sizeof(LinearSphericalInterpolator),
    .tp_call = LinearSphericalInterpolator_call,
    .tp_finalize = LinearSphericalInterpolator_finalize,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_init = LinearSphericalInterpolator_init,
    .tp_name = LinearSphericalInterpolator_name,
    .tp_new = PyType_GenericNew,
};

static void LinearSphericalInterpolator_loop(char **args,
                                             const npy_intp *dimensions,
                                             const npy_intp *steps,
                                             void *NPY_UNUSED(data))
{
    CGAL::Data_access<PointValueMap> values(
        *LinearSphericalInterpolator_current->point_value_map);
    for (npy_intp i = 0; i < dimensions[0]; i++)
    {
        double xyz[3];
        bool good = true;
        for (npy_intp j = 0; j < 3; j++)
        {
            double component = *(double *)&args[0][i * steps[0] + j * steps[2]];
            if (std::isfinite(component))
            {
                xyz[j] = component;
            }
            else
            {
                good = false;
                break;
            }
        }
        double result;
        if (good)
        {
            Point point(xyz[0], xyz[1], xyz[2]);
            Vector normal(point - CGAL::ORIGIN);
            std::vector<std::pair<Point, Value>> coords;
            auto norm = CGAL::surface_neighbor_coordinates_3(
                            *LinearSphericalInterpolator_current->delaunay,
                            point, normal, std::back_inserter(coords))
                            .second;
            result = CGAL::linear_interpolation(
                coords.begin(), coords.end(), norm, values);
        }
        else
        {
            result = NAN;
        }
        *(double*)&args[1][i * steps[1]] = result;
    }
}

static PyUFuncGenericFunction LinearSphericalInterpolator_ufunc_loops[] = {LinearSphericalInterpolator_loop};
static const char LinearSphericalInterpolator_ufunc_types[] = {NPY_DOUBLE, NPY_DOUBLE};

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_core",
};

PyMODINIT_FUNC PyInit__core(void)
{
    import_array();
    import_umath();

    if (PyType_Ready(&LinearSphericalInterpolator_type))
        return NULL;

    LinearSphericalInterpolator_ufunc = PyUFunc_FromFuncAndDataAndSignature(
        LinearSphericalInterpolator_ufunc_loops, NULL,
        const_cast<char *>(LinearSphericalInterpolator_ufunc_types), 1, 1, 1,
        PyUFunc_None, "LinearSphericalInterpolator.__call__", NULL, 0, "(3)->()");
    if (!LinearSphericalInterpolator_ufunc)
        return NULL;

    PyObject *module = PyModule_Create(&moduledef);
    if (!module)
        return NULL;

    if (PyModule_AddObjectRef(
            module, LinearSphericalInterpolator_name,
            reinterpret_cast<PyObject*>(&LinearSphericalInterpolator_type)))
    {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
