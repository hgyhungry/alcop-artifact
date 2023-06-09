# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import
"""Generator for CUTLASS Conv2D kernels."""
from .library import *


class Conv2dOperation:
    """Describes various attributes for instantiating Conv2d kernels."""

    def __init__(
        self,
        conv_kind,
        iterator_algorithm,
        arch,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        stride_support,
        epilogue_functor=EpilogueFunctor.LinearCombination,
        swizzling_functor=SwizzlingFunctor.Identity1,
    ):
        self.operation_kind = OperationKind.Conv2d
        self.arch = arch
        self.tile_description = tile_description
        self.conv_kind = conv_kind
        self.A = A
        self.B = B
        self.C = C
        self.element_epilogue = element_epilogue
        self.epilogue_functor = epilogue_functor
        self.iterator_algorithm = iterator_algorithm
        self.stride_support = stride_support
        self.swizzling_functor = swizzling_functor

    def accumulator_type(self):
        return self.tile_description.math_instruction.element_accumulator

    def core_name(self):
        """ The basic operation kind is prefixed with a letter indicating the accumulation type. """
        intermediate_type = ""

        if self.tile_description.math_instruction.opcode_class == OpcodeClass.TensorOp:
            inst_shape = "%d%d%d" % tuple(self.tile_description.math_instruction.instruction_shape)
            if (
                self.tile_description.math_instruction.element_a != self.A.element
                and self.tile_description.math_instruction.element_a != self.accumulator_type()
            ):
                intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
        else:
            inst_shape = ""

        return "%s%s%s%s_%s" % (
            ShortDataTypeNames[self.accumulator_type()],
            inst_shape,
            intermediate_type,
            ConvKindNames[self.conv_kind],
            IteratorAlgorithmNames[self.iterator_algorithm],
        )

    def extended_name(self):
        """ Append data types if they differ from compute type. """
        if (
            self.C.element != self.tile_description.math_instruction.element_accumulator
            and self.A.element != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif (
            self.C.element == self.tile_description.math_instruction.element_accumulator
            and self.A.element != self.tile_description.math_instruction.element_accumulator
        ):
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = substitute_template(
            extended_name,
            {
                "element_a": DataTypeNames[self.A.element],
                "element_c": DataTypeNames[self.C.element],
                "core_name": self.core_name(),
            },
        )

        return extended_name

    def layout_name(self):
        return "%s" % (ShortLayoutTypeNames[self.A.layout])

    def procedural_name(self):
        """
        The full procedural name indicates architecture, extended name, tile size, and layout.
        """
        opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

        threadblock = "%dx%d_%dx%d" % (
            self.tile_description.threadblock_shape[0],
            self.tile_description.threadblock_shape[1],
            self.tile_description.threadblock_shape[2],
            self.tile_description.stages,
        )

        if self.stride_support == StrideSupport.Unity:
            configuration_name = (
                "cutlass_${opcode_class}_${extended_name}_${threadblock}"
                "_${layout}_align${alignment}_unity_stride"
            )
        else:
            configuration_name = (
                "cutlass_${opcode_class}_${extended_name}_${threadblock}"
                "_${layout}_align${alignment}"
            )

        return substitute_template(
            configuration_name,
            {
                "opcode_class": opcode_class_name,
                "extended_name": self.extended_name(),
                "threadblock": threadblock,
                "layout": self.layout_name(),
                "alignment": "%d" % self.A.alignment,
            },
        )


class EmitConv2dInstance:
    """ Responsible for emitting a CUTLASS template definition."""

    def __init__(self):
        self.template = """
  // Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
  using ${operation_name} =
  typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}<
    ${element_a},
    ${layout_a},
    ${element_b},
    ${layout_b},
    ${element_c},
    ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    ${stages},
    ${math_operator},
    ${iterator_algorithm},
    ${stride_support},
    ${align_a},
    ${align_b}
  >::Kernel;
"""

    def emit(self, operation):
        """Instantiate a Conv2d kernel from given `operation`."""
        warp_shape = [
            int(
                operation.tile_description.threadblock_shape[idx]
                / operation.tile_description.warp_count[idx]
            )
            for idx in range(3)
        ]

        epilogue_vector_length = int(
            min(operation.C.alignment * DataTypeSize[operation.C.element], 128)
            / DataTypeSize[operation.C.element]
        )

        values = {
            "operation_name": operation.procedural_name(),
            "conv_kind": ConvKindTag[operation.conv_kind],
            "conv_kind_name": ConvKindNames[operation.conv_kind].capitalize(),
            "element_a": DataTypeTag[operation.A.element],
            "layout_a": LayoutTag[operation.A.layout],
            "element_b": DataTypeTag[operation.B.element],
            "layout_b": LayoutTag[operation.B.layout],
            "element_c": DataTypeTag[operation.C.element],
            "layout_c": LayoutTag[operation.C.layout],
            "element_accumulator": DataTypeTag[operation.accumulator_type()],
            "opcode_class": OpcodeClassTag[
                operation.tile_description.math_instruction.opcode_class
            ],
            "arch": "cutlass::arch::Sm%d" % operation.arch,
            "threadblock_shape_m": str(operation.tile_description.threadblock_shape[0]),
            "threadblock_shape_n": str(operation.tile_description.threadblock_shape[1]),
            "threadblock_shape_k": str(operation.tile_description.threadblock_shape[2]),
            "warp_shape_m": str(warp_shape[0]),
            "warp_shape_n": str(warp_shape[1]),
            "warp_shape_k": str(warp_shape[2]),
            "instruction_shape_m": str(
                operation.tile_description.math_instruction.instruction_shape[0]
            ),
            "instruction_shape_n": str(
                operation.tile_description.math_instruction.instruction_shape[1]
            ),
            "instruction_shape_k": str(
                operation.tile_description.math_instruction.instruction_shape[2]
            ),
            "epilogue_vector_length": str(epilogue_vector_length),
            "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],
            "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
            "swizzling_functor": SwizzlingFunctorTag[operation.swizzling_functor],
            "stages": str(operation.tile_description.stages),
            "iterator_algorithm": IteratorAlgorithmTag[operation.iterator_algorithm],
            "iterator_algorithm_name": IteratorAlgorithmNames[
                operation.iterator_algorithm
            ].capitalize(),
            "stride_support": StrideSupportTag[operation.stride_support],
            "math_operator": MathOperationTag[
                operation.tile_description.math_instruction.math_operation
            ],
            "align_a": str(operation.A.alignment),
            "align_b": str(operation.B.alignment),
        }

        return substitute_template(self.template, values)
