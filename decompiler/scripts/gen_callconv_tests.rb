#!/usr/bin/env ruby

require 'shellwords'
require 'erb'

module Model
  class Type
    def access_self = AccessSelf.new(type: self)
    def declaration = ""
  end

  class Void < Type
    def decorate_var(decl) = "void #{decl}"
    def builtin? = true
    def each_part; end
  end

  class Scalar < Type
    def builtin? = true
    def each_part = yield access_self
  end
  class VoidStar < Scalar
    def decorate_var(decl) = "void *#{decl}"
    def c_example_value = "NULL"
    def ssa_example_pattern = Pattern.new(:Const, {value: 0, size: 8})
  end
  class Float < Scalar
    def value = 123.456
    def decorate_var(decl) = "float #{decl}"
    def c_example_value = format('%.3ff', self.value)
    def ssa_example_pattern = Pattern.new(:Const, {value: self.value})
  end
  class Double < Scalar
    def value = 987.321
    def decorate_var(decl) = "double #{decl}"
    def c_example_value = format('%f', self.value)
    def ssa_example_pattern = Pattern.new(:Const, {value: self.value})
  end
  class UInt < Scalar
    def initialize(bytes_size)
      super()
      raise ValueError.new unless [1, 2, 4, 8].include?(bytes_size)
      @bytes_size = bytes_size
    end
    def bit_size = @bytes_size * 8
    def decorate_var(decl) = "uint#{bit_size}_t #{decl}"
    def value = 329875028435 % (2 ** self.bit_size)
    def c_example_value = format('%d', self.value)
    def ssa_example_pattern = Pattern.new(:Const, {value: self.value, size: @bytes_size})
  end

  class Struct < Type
    def initialize(name, member_types)
      super()
      @name = name
      @members = member_types.each_with_index.map {
        StructMember.new(name: "member#{_2}", type: _1)
      }
    end

    def declaration
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
        member_a = AccessMember.new(name: member.name, type: member.type)
        member.type.each_part do |part_a|
          yield member_a.and_then(part_a)
        end
      end
      yield access_self
    end

    def c_example_value
      fields = @members.map { |m| ".#{m.name} = #{m.type.c_example_value}" }
      "(struct #{@name}){ #{fields.join(', ')} }"
    end
  end

  class Array < Type
    def initialize(length, element_type)
      super()
      @length = length
      @element_type = element_type
    end

    def decorate_var(var) = @element_type.decorate_var("#{var}[#{@length}]")
    def builtin? = false

    def indices = (0..@length - 1)
    def each_part
      indices.each do |ndx|
        element_a = AccessIndex.new(index: ndx, type: @element_type)
        @element_type.each_part do |part_a|
          yield element_a.and_then(part_a)
        end
      end
      yield access_self
    end
    def c_example_value
      elm_values = indices.map { @element_type.c_example_value }
      ty_s = @element_type.decorate_var("[#{@length}]")
      "{ #{elm_values.join(', ')} }"
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
    def initialize(type)
      @type = type
    end
    attr_reader :type
    def and_then(other)
      AccessCompose.new(self, other)
    end
  end
  class AccessSelf < Access
    def initialize(type:) = super(type)
    def c_expression(start) = start
    def write_pattern_to(ssa, src) = src
  end
  class AccessMember < Access
    def initialize(name:, type:)
      super(type)
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
    def initialize(index:, type:)
      super(type)
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
      super(second.type)
      @first = first
      @second = second
    end
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
  class << self
    attr_accessor :counter
  end

  def initialize(param_types:, access_path:)
    self.class.counter ||= 0
    @name_prefix = "func" + self.class.counter.to_s.rjust(3, '0')
    self.class.counter += 1

    @param_types = param_types
    @access_path = access_path

    raise InvalidTestCase.new if not @access_path.type.is_a?(Model::Scalar)
  end

  attr_reader :param_types

  def target_type = @access_path.type
  def name_in = "#{@name_prefix}_in"
  def name_out = "#{@name_prefix}_out"

  def arg_name(ndx) = "arg#{ndx}"
  def target_arg_ndx = @param_types.length - 1
  def target_arg_name = arg_name target_arg_ndx
  def c_function_code
    args_s = @param_types
             .each_with_index
             .map { |ty, ndx| ty.decorate_var(arg_name(ndx)) }
             .join(', ')

    example_value = target_type.c_example_value

    in_declarator = target_type.decorate_var("__attribute__ ((noinline)) #{name_in}(#{args_s})")

    tmpl = <<-'EOF'
    <%= in_declarator %> {
      return <%= @access_path.c_expression(target_arg_name) %>;
    }

    <%= target_type.decorate_var(name_out + '()') %> {
      <% @param_types.each_with_index do |ty, ndx| %>
        <%= ty.decorate_var("example_value#{ndx}") %> = <%= ty.c_example_value %>;
      <% end %>
      return <%= @name_prefix %>_in(
        <%= @param_types
          .each_with_index
          .map { |_, ndx| "example_value#{ndx}" }
          .join(', ')
        %>
      );
    }
EOF

    ERB.new(tmpl).result(binding)
  end

  def rs_test_code
    ssa = Model::SSAPattern.new
    root = ssa.add(Pattern.new(:FuncArgument, { :index => target_arg_ndx }))
    @access_path.write_pattern_to(ssa, root)

    [
      "#[test]",
      "fn #{name_in}() {",
      "    let data_flow = compute_data_flow(#{name_in.dump});",
      "    let insns = data_flow.as_slice();",
      "",
      "    " + ssa.rs_test_code,
      "}",
    ].join("\n")
  end
end

class InvalidTestCase < StandardError; end

TY_VOID = Model::Void.new

TY_SCALAR_TYPES = [
    Model::UInt.new(1),
    Model::UInt.new(2),
    Model::UInt.new(4),
    Model::UInt.new(8),
    Model::VoidStar.new,
    Model::Float.new,
    Model::Double.new,
]

TY_SMALL_STRUCT = Model::Struct.new 'small', [
  Model::VoidStar.new,
  Model::Float.new,
  Model::UInt.new(1),
]

TY_SMALL_STRUCT_FLOATS = Model::Struct.new 'small_xmms', [
  Model::Float.new,
  Model::Double.new,
]

TY_BIG_STRUCT = Model::Struct.new 'big', [
  Model::Float.new,
  Model::Double.new,
  Model::VoidStar.new,
  Model::UInt.new(1),
  Model::Array.new(3, Model::UInt.new(1)),
]

TY_TAILS = TY_SCALAR_TYPES + [
  TY_SMALL_STRUCT,
  TY_SMALL_STRUCT_FLOATS,
  TY_BIG_STRUCT,
]


# Generate testcases for "incoming" calls.
#
# "Incoming calls" tests are those where we exercise the decoding of argument
# values and the encoding of return values from within the called function (each
# testcase has a specific function that exercise a specific case, and code path
# in the decompiler).
def generate_cases_incoming(&block)
  void_star = Model::VoidStar.new

  (1..8).each do |length|
    fillers = [void_star] * (length - 1)
    fillers[-1] = TY_SMALL_STRUCT if length >= 2

    TY_TAILS.each do |tail_type|
      param_types = fillers + [tail_type]

      tail_type.each_part do |access|
        # aggregate types can't be assigned directly to a single machine
        # register. we're going to generate test functions that access each
        # part of the aggregate, anyway.
        next unless access.type.is_a?(Model::Scalar)

        begin
          block.call(TestCase.new(
            param_types: param_types,
            access_path: access,
          ))
        rescue InvalidTestCase
          next
        end
      end
    end
  end
end


def run(c_out_filename:, rs_out_filename:)
  out_c = File.open(c_out_filename, "w")
  out_c.puts <<EOF
  // This file is generated by #{$PROGRAM_NAME} -- do not edit
  #include <stdint.h>

  #define NULL ((void*) 0)

  // [limitation--no-relocatable] due to a known limitation, we can't process
  // relocatable executables (we can't run relocations at all).
  // adding main() allows us to compile this to a 'full' executable rather than a .o
  int main() {}

EOF

  out_rs = File.open(rs_out_filename, "w")
  out_rs.puts <<EOF
  // This file is generated by #{$PROGRAM_NAME} -- do not edit
  mod utils;
  use utils::dataflow::*;
  use decompiler::{Insn, R};
  use test_log::test;

EOF


  declared_types = Set.new

  generate_cases_incoming do |testcase|
    # declare any type that was not declared yet
    testcase.param_types.each do |param_ty|
      param_ty.each_part do |access|
        next if declared_types.include?(access.type)
        out_c.puts access.type.declaration
        declared_types.add(access.type)
      end
    end

    out_c.puts testcase.c_function_code
    out_rs.puts testcase.rs_test_code
  end

  out_rs.close
  system(["rustfmt", rs_out_filename].shelljoin)
end

if $PROGRAM_NAME == __FILE__
  require 'optparse'

  options = {}

  opt_parser = OptionParser.new do |opts|
    opts.banner = "Usage: #{$PROGRAM_NAME} --out-c OUT_C_PART.c --out-rs OUT_RUST_PART.rs"

    opts.on(
      "--out-c=FILENAME", "File to write for the C part (to be compiled into a test exe)"
    ) do |filename|
      options[:out_c] = filename
    end

    opts.on(
      "--out-rs=FILENAME", "File to write the Rust integration test cases to"
    ) do |filename|
      options[:out_rs] = filename
    end
  end

  opt_parser.parse!

  if not [:out_rs, :out_c].all?{|k| options.include? k}
    puts "Error: --out-c and --out-rs are both required"
    puts
    puts opt_parser.help
    exit 1
  end

  run(
    c_out_filename: options[:out_c],
    rs_out_filename: options[:out_rs],
  )
end
