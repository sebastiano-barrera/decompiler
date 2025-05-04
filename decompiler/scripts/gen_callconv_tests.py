#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Type:
    @property
    def decl(self) -> str:
        return ''
    def iter_targets(self, self_name):
        raise NotImplementedError()
    def decorate_var(self, decl):
        raise NotImplementedError()


@dataclass(frozen=True)
class Scalar(Type):
    c_repr: str

    def decorate_var(self, decl):
        return f'{self.c_repr} {decl}'

    def iter_targets(self, self_name):
        yield self_name, self


@dataclass(frozen=True)
class Struct(Type):
    name: str
    member_types: Sequence[Type]

    @property
    def c_repr(self):
        lines = [ "struct {} {{".format(self.name) ]
        for i, (name, typ) in enumerate(self.members):
            decl = typ.decorate_var(name)
            lines.append(f'  {decl};')
        lines.append("};")
        return "\n".join(lines)

    @property
    def decl(self):
        return self.c_repr

    def decorate_var(self, decl):
        return f'struct {self.name} {decl}'

    @property
    def member_names(self):
        for i in range(len(self.member_types)):
            yield f'member{i}'

    @property
    def members(self):
        return list(zip(self.member_names, self.member_types))

    def iter_targets(self, self_name):
        yield self_name, self
        for name, ty in self.members:
            yield from ty.iter_targets(f'{self_name}.{name}')


@dataclass(frozen=True)
class Array(Type):
    element_type: Type
    count: int

    def decorate_var(self, decl):
        sub = self.element_type.decorate_var(decl)
        return f'{sub}[{self.count}]'

    def iter_targets(self, self_name):
        for i in range(self.count):
            yield f'{self_name}[{i}]', self.element_type


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
    yield Scalar('void*')
    yield Scalar('double')
    # just one big struct, just to put something in memory
    yield big_struct


def gen_target_types():
    return [
        Scalar('char'),
        Scalar('short int'),
        Scalar('int'),
        Scalar('long long int'),
        Scalar('void*'),
        Scalar('float'),
        Scalar('double'),
        small_struct,
        small_struct_floats,
        big_struct,
    ]

small_struct = Struct(
    name='small',
    member_types=tuple([
        Scalar('void*'),
        Scalar('float'),
        Scalar('uint8_t'),
    ])
)

small_struct_floats = Struct(
    name='small_xmms',
    member_types=tuple([
        Scalar('float'),
        Scalar('double'),
    ])
)

big_struct = Struct(
    name='big',
    member_types=tuple([
        Scalar('float'),
        Scalar('double'),
        Scalar('void*'),
        Scalar('uint8_t'),
        Array(
            count=3,
            element_type=Scalar('uint8_t'),
        ),
    ])
)

def gen_seqs_with_len(seq_len):
    seq = [Scalar('void*') for _ in range(seq_len)]

    if seq_len >= 2:
        seq[seq_len - 2] = small_struct

    for target_type in gen_target_types():
        seq[-1] = target_type
        yield seq

def gen_seqs():
    # for seq_len 1..=6 we expect mostly passing via registers;
    # going up to 8 ensure some usage of the stack even for integers
    for seq_len in range(1, 8):
        yield from gen_seqs_with_len(seq_len)

def attach_names(seq):
    return [
        (f"arg{i}", ty)
        for i, ty in enumerate(seq)
    ]

def gen():
    for seq in gen_seqs():
        named = attach_names(seq)
        target_name, target_ty = named[-1]
        for tgt_name, tgt_ty in target_ty.iter_targets(target_name):
            yield (tgt_name, tgt_ty, named)


def func_code(func_name, target_name, target_ty, named_args):
    args_s = ', '.join([
        arg_ty.decorate_var(arg_name)
        for arg_name, arg_ty in named_args
    ])
    func_declarator = f'{func_name}({args_s})'
    decl = target_ty.decorate_var(func_declarator)

    return '\n'.join([
        f'{decl} {{',
        f'  return {target_name};',
        '}',
    ])


def main():
    types_declared = set()

    print('''#include <stdint.h>

// [limitation--no-relocatable] due to a known limitation, we can't process
// relocatable executables (we can't run relocations at all).
// adding main() allows us to compile this to a 'full' executable rather than a .o
int main() {}
''')

    for i, (target_name, target_ty, named_args) in enumerate(gen()):
        func_name = 'func{:03}'.format(i)

        for _, ty in named_args:
            if ty in types_declared:
                continue
            print(ty.decl)
            types_declared.add(ty)

        src = func_code(
            func_name=func_name,
            target_name=target_name,
            target_ty=target_ty,
            named_args=named_args,
        )
        print(src)


if __name__ == "__main__":
    main()
