#!/usr/bin/env python3


# import click

from dataclasses import dataclass
from typing import List
import random
import itertools


class Type:
    pass


@dataclass
class Scalar(Type):
    c_repr: str


@dataclass
class Struct(Type):
    name: str
    members: List[Type]

    @property
    def c_repr(self):
        lines = [ "struct {} {{".format(self.name) ]
        for i, typ in enumerate(self.members):
            lines.append('  {:15} memb{};'.format(typ.c_repr, i))
        lines.append("};")
        return "\n".join(lines)


scalar_types = [
    Scalar('char'),
    Scalar('short int'),
    Scalar('int'),
    Scalar('long long int'),
    Scalar('char*'),
    Scalar('float'),
    Scalar('double'),
]


def gen_type():
    yield from scalar_types


def gen():
    return itertools.permutations(gen_type())


def test_c_code(arg_types):
    args_s = ', '.join([
        '{} arg{}'.format(t.c_repr, i)
        for i, t in enumerate(arg_types)
    ])

    return '\n'.join([
        'int func({}) {{'.format(args_s),
        '  asm(',
        '    ";; target = %0"',
        '    : /* outputs */',
        '    : "X" (arg2) /* inputs */',
        '    : /* clobbers */',
        '  );',
        '',
        '  return 123;',
        '}',
    ])


from tempfile import NamedTemporaryFile
from pprint import pprint
import subprocess

def compile(source):
    import shlex

    with NamedTemporaryFile(mode='w', suffix='.c', delete_on_close=False) as srcf:
        with NamedTemporaryFile(mode='r+', suffix='.S') as asmf:
            srcf.write(source)
            srcf.close()
        
            cmd = ['cc', '-march=native', '-O3', '-S', srcf.name, '-o', asmf.name]
            subprocess.call(cmd)

            asmf.seek(0)
            asm = asmf.read()
            return asm

   
def main():
    from pprint import pprint
    from collections import defaultdict

    for arg_types in gen():
        src = test_c_code(arg_types)
        # print('--- src')
        # print(src)
        # print('--- compile')
        asm = compile(src)

        key_line = next(line for line in asm.splitlines() if ';; target = ' in line)
        t = (arg_types[0].c_repr, arg_types[1].c_repr, arg_types[2].c_repr, key_line)
        counter[t] += 1
        print('{:20}, {:20}, {:20} {}'.format(
            arg_types[0].c_repr,
            arg_types[1].c_repr,
            arg_types[2].c_repr,
            key_line,
        ))

        
if __name__ == "__main__":
    main()

