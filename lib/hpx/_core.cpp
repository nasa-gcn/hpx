#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Origin.h>
#include <CGAL/surface_neighbor_coordinates_3.h>

// PyModule_AddObjectRef was added in Python 3.10.
// FIXME: Remove when we require Python >= 3.10.
static int AddObjectRef(PyObject *mod, const char *name, PyObject *value) {
    int result = PyModule_AddObject(mod, name, value);
    if (result) {
        Py_XDECREF(value);
    }
    return result;
}


typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Delaunay_triangulation_3<Kernel, CGAL::Fast_location> Delaunay;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef Kernel::FT Value;
typedef std::map<Point, Value, Kernel::Less_xyz_3> PointValueMap;

typedef struct
{
    // Generously sized reserved memory for PyObject_HEAD
    // (which isn't in the limited API).
    // FIXME: Remove when we require Python >= 3.12.
    char reserved[128];

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
            FAIL("points must have shape (npoints, 3)");

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

        // FIXME: replace with PyObject_GetTypeData
        // when we require Python >= 3.12.
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
    // FIXME: replace with PyObject_GetTypeData
    // when we require Python >= 3.12.
    auto obj = reinterpret_cast<LinearSphericalInterpolator *>(self);
    delete obj->delaunay;
    delete obj->point_value_map;
}

static PyObject *LinearSphericalInterpolator_call(
    PyObject *self, PyObject *args, PyObject *kwargs)
{
    // FIXME: replace with PyObject_GetTypeData
    // when we require Python >= 3.12.
    LinearSphericalInterpolator_current = reinterpret_cast<LinearSphericalInterpolator *>(self);
    return PyObject_Call(LinearSphericalInterpolator_ufunc, args, kwargs);
}

static const char LinearSphericalInterpolator_name[] = "LinearSphericalInterpolator";
static const char LinearSphericalInterpolator_doc[] = R"(
Piecewise linear interpolation of unstructured data on a unit sphere.

Parameters
----------
points : ndarray of floats, shape (npoints, 3)
    2-D array of the Cartesian coordinates of the sample points, which must
    be unit vectors. All elements of this array must be finite.
values : ndarray of floats, shape (npoints)
    Values of the function at the sample points.

Notes
-----
The interpolation method is analogous to SciPy's
:class:`~scipy.interpolation.LinearNDInterpolator` except that the points lie
on the surface of a sphere.

The interpolation is done using CGAL by finding the 3D Delaunay triangulation
of the sample points [1], finding surface natural neighbor coordinates at the
evaluation points [2], and performing linear interpolation [3].

This method is not as fast as we would like because CGAL constructs a miniature
2D Delaunay triangulation in the plane tangent to each evaluation point. The
CGAL 2D Triangluations on the Sphere [4] library may be promising but it does
not provide a readymade implementation of natural neighbor coordinates.

References
----------
.. [1] https://doc.cgal.org/latest/Triangulation_3/index.html#Triangulation_3Delaunay
.. [2] https://doc.cgal.org/latest/Interpolation/index.html#secsurface
.. [3] https://doc.cgal.org/latest/Interpolation/index.html#InterpolationLinearPrecisionInterpolation
.. [4] https://doc.cgal.org/latest/Triangulation_on_sphere_2/index.html

Examples
--------
>>> import numpy as np
>>> from astropy.coordinates import uniform_spherical_random_surface
>>> np.random.seed(1234)  # make the output reproducible
>>> npoints = 100
>>> points = uniform_spherical_random_surface(npoints).to_cartesian().xyz.value.T
>>> points
array([[ 0.30367159,  0.78890962,  0.53423326],
       [-0.65451629, -0.63115799,  0.41623072],
       ...
       [-0.22006045, -0.39686604,  0.89110647]])
>>> values = np.random.uniform(-1, 1, npoints)
values
array([ 0.95807764,  0.76246449,  0.25536384,  0.86097307,  0.44957991,
        ...
       -0.33998522,  0.21335043,  0.64431958,  0.25593013, -0.76415388])
>>> interp = LinearSphericalInterpolator(points, values)
>>> eval_points = uniform_spherical_random_surface(10).to_cartesian().xyz.value.T
>>> interp(eval_points)
array([-0.05681123, -0.14872424, -0.15398783,  0.24820993,  0.6796055 ,
        0.25451712, -0.08408354,  0.20886784,  0.22627028,  0.08563897])
)";

static PyType_Spec LinearSphericalInterpolator_typespec = {
    .name = LinearSphericalInterpolator_name,
    // FIXME: change to -sizeof(LinearSphericalInterpolator)
    // when we require Python >= 3.12.
    .basicsize = sizeof(LinearSphericalInterpolator),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot[]){
        {Py_tp_call, (void *) LinearSphericalInterpolator_call},
        {Py_tp_doc, (void *) LinearSphericalInterpolator_doc},
        {Py_tp_init, (void *) LinearSphericalInterpolator_init},
        {Py_tp_new, (void *) PyType_GenericNew},
        {Py_tp_finalize, (void *) LinearSphericalInterpolator_finalize},
        {0, NULL},
    }
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
        double result = NAN;
        for (npy_intp j = 0; j < 3; j++)
        {
            double component = *(double *)&args[0][i * steps[0] + j * steps[2]];
            if (!std::isfinite(component)) goto next;
            xyz[j] = component;
        }

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
next:
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

    LinearSphericalInterpolator_ufunc = PyUFunc_FromFuncAndDataAndSignature(
        LinearSphericalInterpolator_ufunc_loops, NULL,
        const_cast<char *>(LinearSphericalInterpolator_ufunc_types), 1, 1, 1,
        PyUFunc_None, "LinearSphericalInterpolator.__call__", NULL, 0, "(3)->()");
    if (!LinearSphericalInterpolator_ufunc)
        return NULL;

    PyObject *module = PyModule_Create(&moduledef);
    if (!module)
        return NULL;

    if (AddObjectRef(
            module, LinearSphericalInterpolator_name,
            PyType_FromSpec(&LinearSphericalInterpolator_typespec)))
    {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
