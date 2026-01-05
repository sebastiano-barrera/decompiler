import sys
import re
from collections import defaultdict

def parse_ssa(filename):
    reg_to_type = {}
    reg_to_op = {}
    reg_to_operands = defaultdict(list)
    upsilon_edges = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '<-' not in line:
                continue
            match = re.match(r'(\s+)(r\d+): ([^<]+) <- (.*)', line)
            if match:
                reg = 'r' + match.group(1)
                type_str = match.group(3).strip()
                op_str = match.group(4).strip()
                reg_to_type[reg] = type_str
                reg_to_op[reg] = op_str
                if 'Upsilon' in op_str:
                    ups_match = re.search(r'value: (r\d+), phi_ref: (r\d+)', op_str)
                    if ups_match:
                        value = ups_match.group(1)
                        phi_ref = ups_match.group(2)
                        upsilon_edges.append((value, phi_ref))
                else:
                    operands = re.findall(r'r\d+', op_str)
                    reg_to_operands[reg] = operands
    # Add upsilon values as operands for phi_ref
    for value, phi_ref in upsilon_edges:
        reg_to_operands[phi_ref].append(value)
    return reg_to_type, reg_to_op, reg_to_operands, upsilon_edges

def get_ancestors(error_regs, reg_to_operands):
    visited = set()
    stack = list(error_regs)
    while stack:
        reg = stack.pop()
        if reg in visited:
            continue
        visited.add(reg)
        for op in reg_to_operands.get(reg, []):
            if op not in visited:
                stack.append(op)
    return visited

def generate_dot(error_regs, reg_to_type, reg_to_op, reg_to_operands, upsilon_edges):
    relevant_regs = get_ancestors(error_regs, reg_to_operands)
    dot = 'digraph {\n'
    dot += '    rankdir=TB;\n'
    dot += '    node [shape=box, fontname="Helvetica", fontsize=9];\n'
    dot += '    edge [fontname="Helvetica", fontsize=8];\n'
    for reg in sorted(relevant_regs):
        type_str = reg_to_type.get(reg, 'Unknown')
        op_str = reg_to_op.get(reg, 'Unknown')
        label = f'{reg}: {type_str}\\n{op_str}'
        if reg in error_regs:
            dot += f'    {reg} [label="{label}", style=filled, fillcolor=red];\n'
        else:
            dot += f'    {reg} [label="{label}"];\n'
    for reg, operands in reg_to_operands.items():
        if reg in relevant_regs:
            for op in operands:
                if op in relevant_regs:
                    dot += f'    {op} -> {reg};\n'
    for value, phi_ref in upsilon_edges:
        if value in relevant_regs and phi_ref in relevant_regs:
            dot += f'    {value} -> {phi_ref} [style=dashed];\n'
    dot += '}\n'
    return dot

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py ssa.txt")
        sys.exit(1)
    filename = sys.argv[1]
    error_regs = {'r1464', 'r587', 'r437', 'r441', 'r482', 'r486', 'r525', 'r529', 'r575'}  # removed r1463 since not in file
    reg_to_type, reg_to_op, reg_to_operands, upsilon_edges = parse_ssa(filename)
    dot = generate_dot(error_regs, reg_to_type, reg_to_op, reg_to_operands, upsilon_edges)
    print(dot)
