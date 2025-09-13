require 'pry'

module Model
  class Type end

  class Scalar < Type
    def initialize(c_repr)
      @c_repr = c_repr
    end
  
    attr_reader :c_repr

    def builtin? = true
    def decorate_var(decl) = "#{@c_repr} #{decl}"
    def each_part
      yield AccessSelf.new(type: self)
    end
  end

  class Struct < Type
    def initialize(name, member_types)
      @name = name
      @members = member_types.each_with_index.map {
        StructMember.new(name: "member#{_2}", type: _1)
      }
    end

    def c_repr
      lines = ["struct #{@name} {"]
      @members.each do |memb|
        lines << "  #{memb};"
      end
      lines << "};"
      lines.join("\n")
    end

    def builtin? = false
    def decorate_var(var) = "struct #{@name} #{var}"

    def each_part
      @members.each do |member|
        member_a = AccessMember.new(member.name)
        member.type.each_part do |part_a|
          yield member_a.and_then(part_a)
        end
      end
    end
  end

  class Array < Type
    def initialize(length, element_type)
      @length = length
      @element_type = element_type
    end

    def decorate_var(var) = @element_type.decorate_var("#{var}[#{@length}]")
    def builtin? = false

    def each_part
      (0..@length).each do |ndx|
        element_a = AccessIndex.new(ndx)
        @element_type.each_part do |part_a|
          yield element_a.and_then(part_a)
        end
      end
    end
  end

  StructMember = ::Struct.new("StructMember", :name, :type) do
    def to_s = self.type.decorate_var(self.name)
  end

  class SSAPattern
    def initialize
      @insn_pats = []
    end
    def add(pat)
      @insn_pats << pat
      Reg.new(@insn_pats.length - 1)
    end
    def rs_test_code
      lines = ["assert_eq!(insns.len(), #{@insn_pats.length});"]
      lines += @insn_pats.each_with_index.map {|pat, ndx| pat.rs_test_code(ndx)}
      lines.join("\n")
    end
  end


  
  Reg = ::Struct.new("Reg", :index) do
    def to_s = "R(#{self.index})"
    def to_rs_expr = "R(#{self.index})"
  end

  class Access
    def and_then(other)
      AccessCompose.new(self, other)
    end
  end
  class AccessSelf < Access
    def initialize(type:)
      @type = type
    end
    attr_reader :type
    def c_expression(start) = start
    def write_pattern_to(ssa, src) = src
  end
  class AccessMember < Access
    def initialize(name)
      @name = name
    end
    def c_expression(start) = "#{start}.#{@name}"
    def write_pattern_to(ssa, src)
      ssa.add(Pattern.new(:StructGetMember, {
        :struct_value => src,
        :name => @name,
      }))
    end
  end
  class AccessIndex < Access
    def initialize(index)
      @index = index
    end
    def c_expression(start) = "#{start}[#{@index}]"
    def write_pattern_to(ssa, src)
      ssa.add(Pattern.new(:ArrayGetElement, {
        :array => src,
        :index => @index,
      }))
    end
  end
  class AccessCompose < Access
    def initialize(first, second)
      @first = first
      @second = second
    end
    def type = @second.type
    def c_expression(start) = @second.c_expression(@first.c_expression(start))
    def write_pattern_to(ssa, src)
      first_ref = @first.write_pattern_to(ssa, src)
      @second.write_pattern_to(ssa, first_ref)
    end
  end
end

class Pattern
  def initialize(opcode, args)
    @opcode = opcode
    @args = args
  end

  def rs_test_code(ndx)
    lines = [
      "assert_matches!(",
      "    &insns[#{ndx}],",
      "    Insn::#{@opcode} { #{@args.keys.join(", ")}, .. },",
      "    {",
    ]

    lines += @args.map do |field_name, value_pat|
      "         assert_eq!(*#{field_name}, #{value_pat.to_rs_expr});"
    end

    lines += [
      "    }",
      ");",
    ]
    lines.join("\n")
  end
end

class Integer
  def to_rs_expr = self.to_s
end

class String
  def to_rs_expr = self.dump
end

class TestCase
  def initialize(name:, param_types:, access_path:)
    @name = name
    @param_types = param_types
    @access_path = access_path
  end

  attr_reader :param_types

  def target_type = @access_path.type

  def arg_name(ndx) = "arg#{ndx}"
  def target_arg_ndx = @param_types.length - 1
  def target_arg_name = arg_name target_arg_ndx
  def c_function_code
    args_s = @param_types
      .each_with_index
      .map {|ty, ndx| ty.decorate_var(arg_name ndx) }
      .join(", ")

    declarator = "#{@name}(#{args_s})"
    func_header = target_type.decorate_var(declarator)

    [
      "#{func_header} {",
      "  return #{@access_path.c_expression(target_arg_name)};",
      "}",
    ].join("\n")
  end

  def rs_test_code
    ssa = Model::SSAPattern.new
    root = ssa.add(Pattern.new(:FuncArgument, { :index => target_arg_ndx }))
    @access_path.write_pattern_to(ssa, root)

    [
      "#[test]",
      "fn #{@name}() {",
      "    let data_flow = compute_data_flow(#{@name.dump});",
      "    let insns = data_flow.as_slice();",
      "",
      "    " + ssa.rs_test_code,
      "}",
    ].join("\n")
  end
end

TY_SCALAR_TYPES = [
    Model::Scalar.new('uint8_t'),
    Model::Scalar.new('uint16_t'),
    Model::Scalar.new('uint32_t'),
    Model::Scalar.new('uint64_t'),
    Model::Scalar.new('void*'),
    Model::Scalar.new('float'),
    Model::Scalar.new('double'),
]

TY_SMALL_STRUCT = Model::Struct.new 'small', [
  Model::Scalar.new('void*'),
  Model::Scalar.new('float'),
  Model::Scalar.new('uint8_t'),
]

TY_SMALL_STRUCT_FLOATS = Model::Struct.new 'small_xmms', [
  Model::Scalar.new('float'),
  Model::Scalar.new('double'),
]

TY_BIG_STRUCT = Model::Struct.new 'big', [
  Model::Scalar.new('float'),
  Model::Scalar.new('double'),
  Model::Scalar.new('void*'),
  Model::Scalar.new('uint8_t'),
  Model::Array.new(3, Model::Scalar.new('uint8_t')),
]

TY_TARGETS = TY_SCALAR_TYPES + [
  TY_SMALL_STRUCT,
  TY_SMALL_STRUCT_FLOATS,
  TY_BIG_STRUCT,
]


def generate_cases(&block)
  counter = 0
  void_star = Model::Scalar.new('void*')
  (1..8).each do |length|
    fillers = [void_star] * (length - 1)
    fillers[-1] = TY_SMALL_STRUCT if length >= 2

    TY_TARGETS.each do |target_type|
      param_types = fillers + [target_type]

      target_type.each_part do |access|
        name = "func" + counter.to_s.rjust(3, '0')
        counter += 1
        block.call(TestCase.new(
          name:,
          param_types: param_types,
          access_path: access,
        ))
      end
    end
  end
end


if $PROGRAM_NAME == __FILE__
  out_c = File.open("test.c", "w")
  out_c.puts <<EOF
  // This file is generated by #{$PROGRAM_NAME} -- do not edit
  #include <stdint.h>

  // [limitation--no-relocatable] due to a known limitation, we can't process
  // relocatable executables (we can't run relocations at all).
  // adding main() allows us to compile this to a 'full' executable rather than a .o
  int main() {}

EOF

  out_rs = File.open("test.rs", "w")
  out_rs.puts <<EOF
  // This file is generated by #{$PROGRAM_NAME} -- do not edit
  mod utils;
  use utils::dataflow::*;
  use decompiler::{Insn, R};

EOF


  declared_types = Set.new

  generate_cases do |testcase|
    # aggregate types can't be assigned directly to a single machine
    # register. we're going to generate test functions that access each
    # part of the aggregate, anyway.
    next unless testcase.target_type.is_a?(Model::Scalar)

    testcase.param_types.each do |ty|
      next if declared_types.include?(ty) or ty.builtin?
      out_c.puts ty.c_repr
      declared_types.add(ty)
    end

    out_c.puts testcase.c_function_code
    out_rs.puts testcase.rs_test_code
  end
end


