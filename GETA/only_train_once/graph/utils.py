import textwrap

import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._globals import GLOBALS


def _is_constant_tensor_list(node):
    if node.kind() != "prim::Constant":
        return False
    output_type = node.output().type()
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True


def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if _is_constant_tensor_list(node):
            inputs = []
            for val in node.output().toIValue():
                input = g.insertConstant(val)
                input.node().moveBefore(node)
                input.node().copyMetadata(node)
                inputs.append(input)

            lc = (
                g.create("prim::ListConstruct", inputs)
                .insertBefore(node)
                .output()
                .setType(_C.ListType.ofTensors())
            )
            lc.node().copyMetadata(node)
            node.output().replaceAllUsesWith(lc)


def _optimize_trace_graph_no_onnx_operator(
    graph: _C.Graph,
    operator_export_type: _C_onnx.OperatorExportTypes,
    _disable_torch_constant_prop: bool = False,
    fixed_batch_size: bool = False,
    params_dict=None,
    dynamic_axes=None,
    input_names=None,
    module=None,
):
    if params_dict is None:
        params_dict = {}

    # Inline everything
    _C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    _C._jit_pass_inline_fork_wait(graph)
    _C._jit_pass_lint(graph)
    _C._jit_pass_onnx_autograd_function_process(graph)
    _C._jit_pass_lower_all_tuples(graph)

    # we now record some ops like ones/zeros
    # into a trace where we previously recorded constants.
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    if _disable_torch_constant_prop is False:
        _C._jit_pass_constant_propagation(graph)

    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    _C._jit_pass_dce(graph)
    _C._jit_pass_lint(graph)

    # CSE should improve perf when Autocast is used with disabled cache
    # Autocast is disabled due to a limitation on tracer as described at https://github.com/pytorch/pytorch/issues/84092
    # Must run before _C._jit_pass_erase_number_types to prevent type substitution
    if _C._jit_pass_cse(graph):
        _C._jit_pass_onnx_lint(graph)

    _C._jit_pass_canonicalize_graph_fuser_ops(graph)
    _C._jit_pass_lint(graph)
    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_fuse_addmm(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_lower_all_tuples(graph)
    # in _jit_pass_onnx, symbolic functions are called for each node for conversion.
    # However, there are nodes that cannot be converted without additional context.
    # For example, the number of outputs from split (and whether it is static or dynamic) is unknown
    # until the point where it is unpacked by listUnpack node.
    # This pass does a preprocess, and prepares the nodes such that enough context can be received
    # by the symbolic function.
    _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
    _C._jit_pass_onnx_preprocess(graph)

    # onnx does not support tuples, so try to remove them
    _C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    _C._jit_pass_prepare_division_for_onnx(graph)

    _C._jit_pass_onnx_remove_print(graph)
    _C._jit_pass_onnx_preprocess_caffe2(graph)

    symbolic_helper._quantized_ops.clear()
    # Unpack quantized weights for conv and linear ops and insert into graph.
    _C._jit_pass_onnx_unpack_quantized_weights(
        graph, params_dict, symbolic_helper.is_caffe2_aten_fallback()
    )
    if symbolic_helper.is_caffe2_aten_fallback():
        # Insert permutes before and after each conv op to ensure correct order.
        _C._jit_pass_onnx_quantization_insert_permutes(graph, params_dict)

        # Find consecutive permutes that are no-ops and remove them.
        _C._jit_pass_custom_pattern_based_rewrite_graph(
            textwrap.dedent(
                """\
                graph(%Pi):
                    %Pq = quantized::nhwc2nchw(%Pi)
                    %Pr = quantized::nchw2nhwc(%Pq)
                    return (%Pr)"""
            ),
            textwrap.dedent(
                """\
                graph(%Ri):
                    return (%Ri)"""
            ),
            graph,
        )

    # onnx only supports tensors, so we turn all out number types into tensors
    _C._jit_pass_erase_number_types(graph)
    if GLOBALS.onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        _C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    _C._jit_pass_onnx_lint(graph)

    graph = _C._jit_pass_onnx(graph, operator_export_type)
    # except:
    #     pass
    _C._jit_pass_onnx_lint(graph)
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_scalar_type_analysis(
        graph, True, GLOBALS.export_onnx_opset_version
    )
    _C._jit_pass_lint(graph)

    _C._jit_pass_onnx_peephole(
        graph, GLOBALS.export_onnx_opset_version, fixed_batch_size
    )
    _C._jit_pass_lint(graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    _C._jit_pass_lint(graph)
    graph = _C._jit_pass_canonicalize(graph)
    _C._jit_pass_lint(graph)
    try:
        if GLOBALS.onnx_shape_inference:
            _C._jit_pass_onnx_graph_shape_type_inference(
                graph, params_dict, GLOBALS.export_onnx_opset_version
            )
    except:
        pass
    return graph


def _get_str_inside_parenthesis(str_to_processed, prefix_strs=None):
    prefix_str = None
    for prefix in prefix_strs:
        if str_to_processed.startswith(prefix):
            prefix_str = prefix
            break
    if prefix_str is None:
        print(
            f"Warning: None of the prefixes {prefix_strs} found in '{str_to_processed}'"
        )
        return None
    # if not str_to_processed.startswith(prefix_str):
    #     return None
    stack = []
    start_idx = len(prefix_str) + 1
    end_idx = -1
    for c in str_to_processed:
        if c == "(":
            stack.append(c)
        elif c == ")":
            stack.pop()
        end_idx += 1
        if len(stack) == 0 and end_idx > len(prefix_str):
            break
    return str_to_processed[start_idx:end_idx]


def _get_tensor_shape(str_to_processed, prefix_strs=["Float", "Long", "Bool"]):
    # Parse output shape given the string of one torch node
    # Should have some better way for completing it
    output_str = _get_str_inside_parenthesis(str_to_processed, prefix_strs=prefix_strs)
    if output_str is None:
        return None
    output_str_splits = output_str.split(",")
    output_shapes = []
    for item in output_str_splits:
        item = item.strip()
        if item.isnumeric():
            output_shapes.append(int(item))
        else:
            break
    return output_shapes


MILLION = 1e6
BILLION = 1e9


def _scale_value(value, in_million=True, in_billion=False):
    if in_million:
        value /= float(MILLION)
    elif in_billion:
        value /= float(BILLION)
    return value


# Find the common parent node and the node with op_name as matmul.
def _find_Qlinear_node_info(
    graph, start_nodes, parent_node_op_name=None, matmul_node_op_name=None
):
    import math

    # Find the node with the smallest id within the "start nodes" list
    temp_min_id = math.inf
    for node in start_nodes:
        if int(node.id.split("-")[-1]) < temp_min_id:
            temp_min_id = int(node.id.split("-")[-1])
            min_id_node = node

    # Find the parent node in the connecting block
    parent_node = _dfs_Qlinear_helper(graph, min_id_node, parent_node_op_name)

    # Find the matmul node in the connecting block
    matmul_node = _dfs_Qlinear_helper(graph, min_id_node, matmul_node_op_name)

    # Collect all node information between start_nodes and parent_node
    node_info = _bfs_Qlinear_helper(graph, start_nodes, parent_node)

    return parent_node, matmul_node, node_info


def _find_closest_node_outgoing(graph, node, target_node_op_name, visited_dict=dict()):
    if node.op_name == target_node_op_name:
        return node
    else:
        for node_out in graph.outgoing(node):
            if node_out.id in visited_dict:
                return visited_dict[node_out.id]
            return _find_closest_node_outgoing(
                graph, node_out, target_node_op_name, visited_dict
            )


def _find_nodes_between_start_end_nodes(graph, start_nodes, end_node):
    from collections import deque

    set_node = set(start_nodes)

    for node in start_nodes:
        # Create a queue
        q = deque()
        q.append(node)
        while q:
            currentNode = q.popleft()
            if (
                currentNode.id == end_node.id
            ):  # Termination when reaching the parent node
                set_node.add(currentNode)
                continue
            set_node.add(currentNode)
            q.extend(graph.outgoing(currentNode))
    return set_node


def _dfs_Qlinear_helper(graph, node_start, node_op_name):
    parent_node = None

    # Create a stack
    next_node_list = []
    next_node_list.append(node_start)

    while len(next_node_list) != 0:
        pop_node = next_node_list.pop()
        if pop_node.op_name == node_op_name:
            parent_node = pop_node
            break
        # Add new nodes to stack data structure
        next_node_list.extend(graph.outgoing(pop_node))

    return parent_node


def _bfs_Qlinear_helper(graph, start_nodes, parent_node):
    from collections import deque

    set_node = set(start_nodes)

    for node in start_nodes:
        # Create a queue
        q = deque()
        q.append(node)
        while q:
            currentNode = q.popleft()
            if currentNode == parent_node:  # Termination when reaching the parent node
                set_node.add(currentNode)
                continue
            set_node.add(currentNode)
            q.extend(graph.outgoing(currentNode))

    return set_node
