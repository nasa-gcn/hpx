#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/Origin.h>
#include <CGAL/surface_neighbor_coordinates_3.h>
#include <string>
#include <valarray>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Delaunay_triangulation_3<Kernel, CGAL::Fast_location> Delaunay;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef Kernel::FT Scalar;
typedef std::valarray<Scalar> Value;
typedef std::map<Point, Value, Kernel::Less_xyz_3> PointValueMap;

struct LinearSphericalInterpolator
{
    std::vector<npy_intp> value_dims;
    npy_intp value_size;
    void *data[1];
    Delaunay delaunay;
    PointValueMap point_value_map;
};

static PyArrayObject *ensure2d(PyArrayObject *arrayobject)
{
    PyArrayObject *result = NULL;
    if (arrayobject)
    {
        npy_intp dims[] = {PyArray_DIM(arrayobject, 0), -1};
        PyArray_Dims newshape = {dims, 2};
        result = reinterpret_cast<PyArrayObject *>(
            PyArray_Newshape(arrayobject, &newshape, NPY_ANYORDER));
        Py_DECREF(arrayobject);
    }
    return result;
}

template<class C>
static void copy_output(std::vector<npy_intp>::const_iterator value_dims_begin,
                        const std::vector<npy_intp>::const_iterator &value_dims_end,
                        C &results,
                        const npy_intp *strides,
                        char *out)
{
    if (value_dims_begin == value_dims_end) {
        *(double*)out = *results++;
    } else {
        for (npy_intp i = 0; i < *value_dims_begin; i ++, out += *strides) {
            copy_output(value_dims_begin + 1, value_dims_end, results, strides + 1, out);
        }
    }
}

static void LinearSphericalInterpolator_loop(char **args,
                                             const npy_intp *dimensions,
                                             const npy_intp *steps,
                                             void *data)
{
    auto interp = reinterpret_cast<LinearSphericalInterpolator*>(data);
    CGAL::Data_access<PointValueMap> values(interp->point_value_map);
    char *out = args[1];

    for (npy_intp i = 0; i < dimensions[0]; i++, out += steps[1])
    {
        Value results(NAN, interp->value_size);
        double xyz[3];
        for (npy_intp j = 0; j < 3; j++)
        {
            double component = *(double *)&args[0][i * steps[0] + j * steps[2]];
            if (!std::isfinite(component)) goto next;
            xyz[j] = component;
        }

        {
            Point point(xyz[0], xyz[1], xyz[2]);
            Vector normal(point - CGAL::ORIGIN);
            std::vector<std::pair<Point, Scalar>> coords;
            auto norm = CGAL::surface_neighbor_coordinates_3(
                            interp->delaunay,
                            point, normal, std::back_inserter(coords)).second;
            results = CGAL::linear_interpolation(
                coords.cbegin(), coords.cend(), norm, values);
        }

next:
        if (interp->value_size > 0)
        {
            auto result_iterator = std::begin(results);
            copy_output(interp->value_dims.cbegin(), interp->value_dims.cend(),
                        result_iterator, steps + 3, out);
        }
    }
}

static void LinearSphericalInterpolator_destroy(PyObject *capsule)
{
    delete reinterpret_cast<LinearSphericalInterpolator *>(
        PyCapsule_GetPointer(capsule, NULL));
}

static const PyUFuncGenericFunction LinearSphericalInterpolator_ufunc_loops[] = {
    LinearSphericalInterpolator_loop};
static const char LinearSphericalInterpolator_ufunc_types[] = {NPY_DOUBLE, NPY_DOUBLE};

static PyObject *LinearSphericalInterpolator_init(PyObject *module, PyObject *args, PyObject *kwargs)
{
#define FAIL(msg) do {PyErr_SetString(PyExc_ValueError, (msg)); goto fail;} while(0)
#define GET_DOUBLE_2D(array, i, j) *(double *)PyArray_GETPTR2((array), (i), (j))
#define GET_POINT(i, j) GET_DOUBLE_2D((points_array), (i), (j))
#define GET_VALUE(i, j) GET_DOUBLE_2D((values_array), (i), (j))

    static const char *kws[] = {"points", "values", NULL};
    PyObject *points, *values, *capsule, *result = NULL;
    PyArrayObject *points_array = NULL, *values_array = NULL;
    LinearSphericalInterpolator *interp;
    std::vector<npy_intp> value_dims;
    npy_intp npoints, nvalues;
    std::string signature;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", const_cast<char **>(kws), &points, &values))
        goto fail;

    points_array = reinterpret_cast<PyArrayObject *>(
        PyArray_FROMANY(points, NPY_DOUBLE, 2, 2,
                        NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED));
    if (!points_array)
        goto fail;

    values_array = reinterpret_cast<PyArrayObject *>(
        PyArray_FROMANY(values, NPY_DOUBLE, 1, 0,
                        NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED));
    if (!values_array)
        goto fail;

    npoints = PyArray_DIM(points_array, 0);
    if (npoints != PyArray_DIM(values_array, 0))
        FAIL("points and values must have the same length");
    if (PyArray_DIM(points_array, 1) != 3)
        FAIL("points must have shape (npoints, 3)");

    for (int i = 1; i < PyArray_NDIM(values_array); i ++)
        value_dims.push_back(PyArray_DIM(values_array, i));
    values_array = ensure2d(values_array);
    if (!values_array)
        goto fail;
    nvalues = PyArray_DIM(values_array, 1);

    for (npy_intp i = 0; i < npoints; i++)
        for (npy_intp j = 0; j < 3; j++)
            if (!std::isfinite(GET_POINT(i, j)))
                FAIL("all elements of points must be finite");

    Py_BEGIN_ALLOW_THREADS
    {
        interp = new LinearSphericalInterpolator{value_dims, nvalues};
        interp->data[0] = interp;
        std::vector<Point> point_list;
        point_list.reserve(npoints);
        for (npy_intp i = 0; i < npoints; i++)
        {
            Point point(GET_POINT(i, 0), GET_POINT(i, 1), GET_POINT(i, 2));
            Value value(nvalues);
            for (npy_intp j = 0; j < nvalues; j++)
            {
                value[j] = GET_VALUE(i, j);
            }
            point_list.push_back(point);
            interp->point_value_map.insert(std::make_pair(point, value));
        }
        interp->delaunay.insert(point_list.cbegin(), point_list.cend());
    }
    Py_END_ALLOW_THREADS

    capsule = PyCapsule_New(interp, NULL, LinearSphericalInterpolator_destroy);
    if (!capsule)
    {
        delete interp;
        goto fail;
    }

    signature = "(3)->(";
    {
        auto it = value_dims.cbegin();
        auto end = value_dims.cend();
        if (it != end)
            signature += std::to_string(*it++);
        for (; it != end; it++) {
            signature += ',';
            signature += std::to_string(*it);
        }
    }
    signature += ')';

    result = PyUFunc_FromFuncAndDataAndSignature(
        const_cast<PyUFuncGenericFunction*>(LinearSphericalInterpolator_ufunc_loops),
        interp->data, const_cast<char*>(LinearSphericalInterpolator_ufunc_types),
        1, 1, 1, PyUFunc_None, "LinearSphericalInterpolator_call", NULL, 0,
        signature.c_str());
    if (!result)
    {
        Py_DECREF(capsule);
        goto fail;
    }
    reinterpret_cast<PyUFuncObject*>(result)->obj = capsule;
fail:
    Py_XDECREF(points_array);
    Py_XDECREF(values_array);
    return result;
}

static const char LinearSphericalInterpolator_doc[] = R"(
Piecewise linear interpolation of unstructured data on a unit sphere.

Parameters
----------
points : ndarray of floats, shape (npoints, 3)
    2-D array of the Cartesian coordinates of the sample points, which must
    be unit vectors. All elements of this array must be finite.
values : ndarray of floats, shape (npoints, ...)
    Values of the function at the sample points. The length along the first
    dimension must be the same as the length of ``points``.

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

static PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_core",
    .m_methods = (PyMethodDef[]){
        {"LinearSphericalInterpolator", (PyCFunction)LinearSphericalInterpolator_init, METH_VARARGS | METH_KEYWORDS, LinearSphericalInterpolator_doc},
        {}}};

PyMODINIT_FUNC PyInit__core(void)
{
    import_array();
    import_ufunc();
    return PyModule_Create(&moduledef);
}
