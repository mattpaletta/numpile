from llvmlite.llvmpy.core import Type

from numpile.lang import TCon, TApp, ftv

int32 = TCon("Int32")
int64 = TCon("Int64")
float32 = TCon("Float")
double64 = TCon("Double")
void = TCon("Void")
array = lambda t: TApp(TCon("Array"), t)

array_int32 = array(int32)
array_int64 = array(int64)
array_double64 = array(double64)

pointer     = Type.pointer
int_type    = Type.int()
float_type  = Type.float()
double_type = Type.double()
bool_type   = Type.int(1)
void_type   = Type.void()
void_ptr    = pointer(Type.int(8))
struct_type = Type.struct([])

def array_type(elt_type):
    return Type.struct([
        pointer(elt_type),  # data
        int_type,           # dimensions
        pointer(int_type),  # shape
        'ndarray_' + str(elt_type), # name
    ])

int32_array = pointer(array_type(int_type))
int64_array = pointer(array_type(Type.int(64)))
double_array = pointer(array_type(double_type))

lltypes_map = {
    int32          : int_type,
    int64          : int_type,
    float32        : float_type,
    double64       : double_type,
    array_int32    : int32_array,
    array_int64    : int64_array,
    array_double64 : double_array
}


def to_lltype(ptype):
    return lltypes_map[ptype]


def determined(ty):
    return len(ftv(ty)) == 0
