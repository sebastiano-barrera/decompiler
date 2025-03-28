#!/usr/bin/env python3


# import click

from dataclasses import dataclass
from typing import List
from tempfile import NamedTemporaryFile
from pprint import pprint
import subprocess
import random
import itertools
import sys


class Type:
    @property
    def decl(self):
        return ''


@dataclass
class Scalar(Type):
    c_repr: str

    def decorate_var(self, name):
        return f'{self.c_repr} {name}'
        


@dataclass
class Struct(Type):
    name: str
    members: List[Type]

    @property
    def c_repr(self):
        lines = [ "struct {} {{".format(self.name) ]
        for i, typ in enumerate(self.members):
            name = f'memb{i}'
            decl = typ.decorate_var(name)
            lines.append(f'  {decl};')
        lines.append("};")
        return "\n".join(lines)

    @property
    def decl(self):
        return self.c_repr

    def decorate_var(self, var_name):
        return f'struct {self.name} {var_name}'

@dataclass
class Array(Type):
    element_type: Type
    count: int

    def decorate_var(self, var_name):
        sub = self.element_type.decorate_var(var_name)
        return f'{sub}[{self.count}]'


scalar_types = [
    Scalar('char'),
    Scalar('short int'),
    Scalar('int'),
    Scalar('long long int'),
    Scalar('void*'),
    Scalar('float'),
    Scalar('double'),
]


def gen_pre_targets():
    yield Scalar('long long int')
    yield Scalar('void*')
    yield Scalar('double')


def gen_targets():
    yield Scalar('char')
    yield Scalar('short int')
    yield Scalar('int')
    yield Scalar('long long int')
    yield Scalar('void*')
    yield Scalar('float')
    yield Scalar('double')
    yield from gen_structs()


def gen_seq(seq_len):
    for seq_pre in itertools.combinations_with_replacement(gen_pre_targets(), seq_len - 1):
        for target in gen_targets():
            yield list(seq_pre) + [target]


def gen():
    for seq_len in range(1, 10):
        for seq in gen_seq(seq_len):
            yield seq


def gen_struct_member():
    yield Scalar('int')
    yield Scalar('long long int')
    yield Scalar('float')
    yield Scalar('double')
    yield Scalar('void*')
    yield Array(Scalar('short'), 3)
    yield Array(Scalar('void*'), 5)

def gen_structs():
    for memb_count in range(5):
        for memb_types in itertools.combinations(gen_struct_member(), memb_count):
            yield Struct(name='s', members=memb_types)
                
     
def test_c_code(name, arg_types):
    args_s = ', '.join([
        t.decorate_var('arg') if i == (len(arg_types) - 1) else ''
        for i, t in enumerate(arg_types)
    ])

    return '\n'.join(
        [ t.decl for t in arg_types ] + [
            'int {}({}) {{'.format(name, args_s),
            '  asm(',
            '    ";; target = %0"',
            '    : /* outputs */',
            '    : "X" (arg) /* inputs */',
            '    : /* clobbers */',
            '  );',
            '',
            '  return 123;',
            '}',
        ])


def compile(source):
    import shlex

    with NamedTemporaryFile(mode='w', suffix='.c', delete_on_close=False) as srcf:
        with NamedTemporaryFile(mode='r+', suffix='.S') as asmf:
            srcf.write(source)
            srcf.close()
        
            cmd = ['cc', '-march=native', '-O3', '-S', srcf.name, '-o', asmf.name]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)

            asmf.seek(0)
            asm = asmf.read()
            return asm

   
def main():
    from pprint import pprint
    from collections import defaultdict

    for arg_types in gen():
        src = test_c_code(name='func', arg_types=arg_types)

        # print('--- src')
        # print(src)
        # print('--- compile')
        try:
            asm = compile(src)
        except subprocess.CalledProcessError as err:
            print('failed to compile the following program:')
            for line in src.splitlines():
                print('>', line)
            print(err)
            print(err.output.decode('utf8'))
            sys.exit(1)

        key_line = next(line for line in asm.splitlines() if ';; target = ' in line)
        t = tuple(arg_types + [key_line])
        print('{} -> {}'.format(
            ', '.join(t.c_repr for t in arg_types),
            key_line,
        ))


        
if __name__ == "__main__":
    main()

