import ctypes
import numpy as np
import string
import llvmlite.llvmpy.core as ll_core

from numpile import engine
from numpile.lang import TVar, TFun
from numpile.pytypes import array, int32, int64, int_type, double_type, \
    float_type, void_type, pointer


def naming():
    k = 0
    while True:
        for a in string.ascii_lowercase:
            yield ("'"+a+str(k)) if (k > 0) else (a)
        k = k+1


def mangler(fname, sig):
    return fname + str(hash(tuple(sig)))


_nptypemap = {
    'i': ctypes.c_int,
    'f': ctypes.c_float,
    'd': ctypes.c_double,
}


def wrap_module(sig, llfunc):
    pfunc = wrap_function(llfunc, engine)
    dispatch = dispatcher(pfunc)
    return dispatch


def wrap_function(func, engine):
    args = func.type.pointee.args
    ret_type = func.type.pointee.return_type
    ret_ctype = wrap_type(ret_type)
    args_ctypes = map(wrap_type, args)

    functype = ctypes.CFUNCTYPE(ret_ctype, *args_ctypes)
    fptr = engine.get_pointer_to_function(func)

    cfunc = functype(fptr)
    cfunc.__name__ = func.name
    return cfunc

def wrap_type(llvm_type):
    kind = type(llvm_type)
    if kind == int_type:
        ctype = getattr(ctypes, "c_int"+str(llvm_type.width))
    elif kind == double_type:
        ctype = ctypes.c_double
    elif kind == float_type:
        ctype = ctypes.c_float
    elif kind == void_type:
        ctype = None
    elif kind == pointer:
        pointee = llvm_type.pointee
        p_kind = pointee.kind
        if p_kind == int_type:
            width = pointee.width
            if width == 8:
                ctype = ctypes.c_char_p
            else:
                ctype = ctypes.POINTER(wrap_type(pointee))
        elif p_kind == void_type:
            ctype = ctypes.c_void_p
        else:
            ctype = ctypes.POINTER(wrap_type(pointee))
    elif kind == struct_type:
        struct_name = llvm_type.name.split('.')[-1]
        struct_name = struct_name.encode('ascii')
        struct_type = None

        if struct_type and issubclass(struct_type, ctypes.Structure):
            return struct_type

        if hasattr(struct_type, '_fields_'):
            names = struct_type._fields_
        else:
            names = ["field"+str(n) for n in range(llvm_type.element_count)]

        ctype = type(ctypes.Structure)(struct_name, (ctypes.Structure,),
                                       {'__module__': "numpile"})

        fields = [(name, wrap_type(elem))
                  for name, elem in zip(names, llvm_type.elements)]
        setattr(ctype, '_fields_', fields)
    else:
        raise Exception("Unknown LLVM type %s" % kind)
    return ctype


def wrap_ndarray(na):
    # For NumPy arrays grab the underlying data pointer. Doesn't copy.
    ctype = _nptypemap[na.dtype.char]
    _shape = list(na.shape)
    data = na.ctypes.data_as(ctypes.POINTER(ctype))
    dims = len(na.strides)
    shape = (ctypes.c_int * dims)(*_shape)
    return data, dims, shape


def wrap_arg(arg, val):
    if isinstance(val, np.ndarray):
        ndarray = arg._type_
        data, dims, shape = wrap_ndarray(val)
        return ndarray(data, dims, shape)
    else:
        return val


def dispatcher(fn):
    def _call_closure(*args):
        cargs = list(fn._argtypes_)
        pargs = list(args)
        rargs = list(map(wrap_arg, cargs, pargs))
        return fn(*rargs)
    _call_closure.__name__ = fn.__name__
    return _call_closure


class TypeInfer(object):

    def __init__(self):
        self.constraints = []
        self.env = {}
        self.names = naming()
        self.argtys = None
        self.retty = None

    def fresh(self):
        return TVar('$' + next(self.names))  # New meta type variable.

    def visit(self, node):
        name = "visit_%s" % type(node).__name__
        if hasattr(self, name):
            return getattr(self, name)(node)
        else:
            return self.generic_visit(node)

    def visit_Fun(self, node):
        arity = len(node.args)
        self.argtys = [self.fresh() for v in node.args]
        self.retty = TVar("$retty")
        for (arg, ty) in zip(node.args, self.argtys):
            arg.type = ty
            self.env[arg.id] = ty
        list(map(self.visit, node.body))
        return TFun(self.argtys, self.retty)

    def visit_Noop(self, node):
        return None

    def visit_LitInt(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_LitFloat(self, node):
        tv = self.fresh()
        node.type = tv
        return tv

    def visit_Assign(self, node):
        ty = self.visit(node.val)
        if node.ref in self.env:
            # Subsequent uses of a variable must have the same type.
            self.constraints += [(ty, self.env[node.ref])]
        self.env[node.ref] = ty
        node.type = ty
        return None

    def visit_Index(self, node):
        tv = self.fresh()
        ty = self.visit(node.val)
        ixty = self.visit(node.ix)
        self.constraints += [(ty, array(tv)), (ixty, int32)]
        return tv

    def visit_Prim(self, node):
        if node.fn == "shape#":
            return array(int32)
        elif node.fn == "mult#":
            tya = self.visit(node.args[0])
            tyb = self.visit(node.args[1])
            self.constraints += [(tya, tyb)]
            return tyb
        elif node.fn == "add#":
            tya = self.visit(node.args[0])
            tyb = self.visit(node.args[1])
            self.constraints += [(tya, tyb)]
            return tyb
        else:
            raise NotImplementedError

    def visit_Var(self, node):
        ty = self.env[node.id]
        node.type = ty
        return ty

    def visit_Return(self, node):
        ty = self.visit(node.val)
        self.constraints += [(ty, self.retty)]

    def visit_Loop(self, node):
        self.env[node.var.id] = int32
        varty = self.visit(node.var)
        begin = self.visit(node.begin)
        end = self.visit(node.end)
        self.constraints += [(varty, int32), (
            begin, int64), (end, int32)]
        list(map(self.visit, node.body))

    def generic_visit(self, node):
        raise NotImplementedError
