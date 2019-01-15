import sys
import numpy as np

from numpile import function_cache
from numpile.lang import TFun, TVar
from numpile.pytypes import array, int32, int64, double64, float32, determined
from numpile.solve import solve, apply, compose, unify, UnderDeteremined
from numpile.transformer import TypeInfer, mangler, wrap_module
from numpile.visitor import PythonVisitor


def typeinfer(ast):
    infer = TypeInfer()
    ty = infer.visit(ast)
    mgu = solve(infer.constraints)
    infer_ty = apply(mgu, ty)
    return infer_ty, mgu


def arg_pytype(arg):
    if isinstance(arg, np.ndarray):
        if arg.dtype == np.dtype('int32'):
            return array(int32)
        elif arg.dtype == np.dtype('int64'):
            return array(int64)
        elif arg.dtype == np.dtype('double'):
            return array(double64)
        elif arg.dtype == np.dtype('float'):
            return array(float32)
    elif isinstance(arg, int) & (arg < sys.maxsize):
        return int64
    elif isinstance(arg, float):
        return double64
    else:
        raise Exception("Type not supported: %s" % type(arg))


def specialize(ast, infer_ty, mgu):
    def _wrapper(*args):
        types = list(map(arg_pytype, list(args)))
        spec_ty = TFun(argtys=types, retty=TVar("$retty"))
        unifier = unify(infer_ty, spec_ty)
        specializer = compose(unifier, mgu)

        retty = apply(specializer, TVar("$retty"))
        argtys = [apply(specializer, ty) for ty in types]
        print('Specialized Function:', TFun(argtys, retty))

        if determined(retty) and all(map(determined, argtys)):
            key = mangler(ast.fname, argtys)
            # Don't recompile after we've specialized.
            if key in function_cache:
                return function_cache[key](*args)
            else:
                llfunc = codegen(ast, specializer, retty, argtys)
                pyfunc = wrap_module(argtys, llfunc)
                function_cache[key] = pyfunc
                return pyfunc(*args)
        else:
            raise UnderDeteremined()
    return _wrapper


def codegen(ast, specializer, retty, argtys):
    from numpile.emitter import LLVMEmitter

    cgen = LLVMEmitter(specializer, retty, argtys)
    mod = cgen.visit(ast)
    # cgen.function.verify()
    print(cgen.function)
    # print(target.emit_assembly(mod))
    return cgen.function


def autojit(fn):
    transformer = PythonVisitor()
    ast = transformer(fn)
    (ty, mgu) = typeinfer(ast)
    return specialize(ast, ty, mgu)
