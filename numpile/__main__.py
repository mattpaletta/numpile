import llvmlite.llvmpy.core as ll_core

# All these initializations are required for code generation!
from numpile.autojit import autojit
from numpile.solve import apply, solve
from numpile.transformer import TypeInfer

from numpile.pyast import pformat_ast
from numpile.visitor import PythonVisitor


def func(name, module, rettype, argtypes):
    func_type = ll_core.Type.function(rettype, argtypes, False)
    lfunc = ll_core.Function.new(module, func_type, name)
    entry_block = lfunc.append_basic_block("entry")
    builder = ll_core.Builder(entry_block)
    return lfunc, builder


if __name__ == "__main__":
    print("Starting llvm")

    def transform():
        def add(a, b):
            return a + b

        transformer = PythonVisitor()
        core = transformer(add)
        print(pformat_ast(core))

    def transform2():
        def count(n):
            a = 0
            for i in range(0, n):
                a += i
            return a

        transformer = PythonVisitor()
        core = transformer(count)
        print(pformat_ast(core))

    def constraint():
        def addup(n):
            x = 1
            for i in range(n):
                n += 1 + x
            return n

        transformer = PythonVisitor()
        core = transformer(addup)
        infer = TypeInfer()
        sig = infer.visit(core)

        print('Signature:%s \n' % sig)

        print('Constraints:')
        for (a, b) in infer.constraints:
            print(a, '~', b)

    def test_infer(fn):
        transformer = PythonVisitor()
        ast = transformer(fn)
        infer = TypeInfer()
        ty = infer.visit(ast)
        mgu = solve(infer.constraints)
        infer_ty = apply(mgu, ty)

        print('Unifier: ')
        for (a, b) in mgu.items():
            print(a + ' ~ ' + str(b))

        print('Solution: ', infer_ty)


    def test_solve():
        def dot2(a, b):
            c = 0
            n = a.shape[0]
            for i in range(n):
                c += a[i] * b[i]
            return c

        test_infer(dot2)

    def test_solve2():
        def addup(n):
            x = 1
            for i in range(n):
                n += 1 + x
            return n

        test_infer(addup)

    def test_solve3():
        def const(a, b):
            return a

        test_infer(const)


    def test_autojit():
        @autojit
        def add(a, b):
            return a + b

        a = 3.1415926
        b = 2.7182818
        print('Result:', add(a, b))

    #transform()
    #transform2()
    #constraint()
    #test_solve3()
    test_autojit()

    # @autojit
    # def dot(a, b):
    #     c = 0
    #     n = a.shape[0]
    #     for i in range(n):
    #         c += a[i] * b[i]
    #     return c
