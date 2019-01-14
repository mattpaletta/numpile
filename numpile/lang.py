from __future__ import print_function

import sys
import ast
import types
import ctypes
import inspect
import pprint
import string
from functools import reduce
from typing import List

import numpy as np

from textwrap import dedent
from collections import deque, defaultdict

import llvmlite.llvmpy.core as ll_core
import llvmlite.binding.executionengine as ll_ee
import llvmlite.binding as llvm
import llvmlite.binding.passmanagers as llp
# from llvm.core import Module, Builder, Function, Type, Constant

DEBUG = False

# All these initializations are required for code generation!
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()  # yes, even this one


class Var(ast.AST):
    _fields = ["id", "type"]

    def __init__(self, id, type = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id
        self.type = type


class Assign(ast.AST):
    _fields = ["ref", "val", "type"]

    def __init__(self, ref, val, type = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref = ref
        self.val = val
        self.type = type


class Return(ast.AST):
    _fields = ["val"]

    def __init__(self, val, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val = val


class Loop(ast.AST):
    _fields = ["var", "begin", "end", "body"]

    def __init__(self, var, begin, end, body, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.begin = begin
        self.end = end
        self.body = body


class App(ast.AST):
    _fields = ["fn", "args"]

    def __init__(self, fn, fn_args, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.args = fn_args


class Fun(ast.AST):
    _fields = ["fname", "args", "body"]

    def __init__(self, fname, fargs, body, **kwargs):
        super().__init__(**kwargs)
        self.fname = fname
        self.args = fargs
        self.body = body


class LitInt(ast.AST):
    _fields = ["n"]

    def __init__(self, n, type = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.type = type


class LitFloat(ast.AST):
    _fields = ["n"]

    def __init__(self, n, type = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.type = None


class LitBool(ast.AST):
    _fields = ["n"]

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n


class Prim(ast.AST):
    _fields = ["fn", "args"]

    def __init__(self, fn, fargs, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.args = fargs


class Index(ast.AST):
    _fields = ["val", "ix"]

    def __init__(self, val, ix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val = val
        self.ix = ix


class Noop(ast.AST):
    _fields = []


class TVar(object):
    _fields = ["s"]

    def __init__(self, s):
        self.s = s

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.s == other.s
        else:
            return False

    def __str__(self):
        return self.s

    __repr__ = __str__


class TCon(object):
    _fields = ["s"]

    def __init__(self, s):
        self.s = s

    def __eq__(self, other):
        if isinstance(other, TCon):
            return self.s == other.s
        else:
            return False

    def __hash__(self):
        return hash(self.s)

    def __str__(self):
        return self.s

    __repr__ = __str__


class TApp(object):
    _fields = ["a", "b"]

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if isinstance(other, TApp):
            return (self.a == other.a) & (self.b == other.b)
        else:
            return False

    def __hash__(self):
        return hash((self.a, self.b))

    def __str__(self):
        return str(self.a) + " " + str(self.b)

    __repr__ = __str__


class TFun(object):
    _fields = ["argtys", "retty"]

    def __init__(self, argtys: List, retty):
        assert isinstance(argtys, list)
        self.argtys = argtys
        self.retty = retty

    def __eq__(self, other):
        if isinstance(other, TFun):
            return (self.argtys == other.argtys) & (self.retty == other.retty)
        else:
            return False

    def __str__(self):
        return str(self.argtys) + " -> " + str(self.retty)

    __repr__ = __str__

def ftv(x):
    if isinstance(x, TCon):
        return set()
    elif isinstance(x, TApp):
        return ftv(x.a) | ftv(x.b)
    elif isinstance(x, TFun):
        # TODO: Update this to py3
        return reduce(set.union, map(ftv, x.argtys)) | ftv(x.retty)
    elif isinstance(x, TVar):
        return {x}


def is_array(ty):
    return isinstance(ty, TApp) and ty.a == TCon("Array")
