# mypy: allow-untyped-defs
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Set

import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

__all__ = [
    "get_embedding_operators_config",
    "EmbeddingQuantizer",
]

# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    # TODO: remove, since we can use observer_or_fake_quant_ctr to express this
    is_qat: bool = False

class OperatorConfig(NamedTuple):
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    config: QuantizationConfig
    operators: List[OperatorPatternType]

OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.embedding_quantizer"
)

def get_embedding_operators_config() -> OperatorConfig:
    weight_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=torch.per_channel_affine_float_qparams,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
    )
    quantization_config = QuantizationConfig(None, None, weight_quantization_spec, None)
    ops: List[OperatorPatternType] = [[torch.nn.Embedding]]
    ops.append([F.embedding])
    supported_config_and_operators = OperatorConfig(
        config=quantization_config, operators=ops
    )
    return copy.deepcopy(supported_config_and_operators)


class EmbeddingQuantizer(Quantizer):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = {
            spec for spec, _ in cls.get_supported_operators()
        }
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: QuantizationConfig
    ) -> List[OperatorPatternType]:
        for config, ops in cls.get_supported_operators():
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        self._annotate_embedding_ops(model.graph)
        return model

    def _annotate_embedding_ops(self, graph: torch.fx.Graph) -> None:
        embedding_config: OperatorConfig = get_embedding_operators_config()
        for node in graph.nodes:
            # Keep node parsing based annotations instead of module partitioners
            # just as an example of alternate ways of annotating
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
            ):
                if embedding_config.config.weight is None:
                    raise ValueError(
                        "Embedding config must have a valid weight quantization spec."
                    )
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        node.args[0]: embedding_config.config.weight,
                    }
                )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return [get_embedding_operators_config()]
