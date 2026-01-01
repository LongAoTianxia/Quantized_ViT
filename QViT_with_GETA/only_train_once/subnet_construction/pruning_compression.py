import torch
import os

def automated_pruning_compression(oto_graph, model, merge_lora_to_base, unmerge_lora_to_base, export_huggingface_format, export_float16, \
                          full_group_sparse_model_dir, compressed_model_dir, save_full_group_sparse_model, ckpt_format):
    full_group_spase_model_name = None
    compressed_model_name = None
    model_name_prefix =  (model.name if hasattr(model, 'name') else type(model).__name__)
    if ckpt_format == 'torch':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.pt"
        compressed_model_name = model_name_prefix + "_compressed.pt"
    elif ckpt_format == 'onnx':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.onnx"
        compressed_model_name = model_name_prefix + "_compressed.onnx"
    full_group_sparse_model_path = os.path.join(full_group_sparse_model_dir, full_group_spase_model_name)
    compressed_model_path = os.path.join(compressed_model_dir, compressed_model_name)

    if export_huggingface_format:
        full_group_sparse_model_dir = os.path.join(full_group_sparse_model_dir, 'huggingface_format_full')
        compressed_model_dir = os.path.join(compressed_model_dir, 'huggingface_format_compressed')
        full_group_sparse_model_path = full_group_sparse_model_dir
        compressed_model_path = compressed_model_dir
        
    os.makedirs(full_group_sparse_model_dir, exist_ok=True)
    os.makedirs(compressed_model_dir, exist_ok=True)
    
    if export_float16:
        model.half()
    
    if save_full_group_sparse_model:
        if export_huggingface_format:
            model.save_pretrained(full_group_sparse_model_path)
        elif ckpt_format == 'torch':
            torch.save(model, full_group_sparse_model_path)   
        elif ckpt_format == 'onnx':
            torch.onnx.export(
                model,
                oto_graph.dummy_input,
                full_group_sparse_model_path)
    
    # Set pruning redundant idxes based on the distribution of zero groups
    oto_graph.set_pruning_redundant_idxes()

    # 构建模块到 node_group 的映射，用于后续查找上游剪枝信息
    module_to_node_group = {}
    for ng in oto_graph.node_groups.values():
        for node in ng.nodes.values():
            if hasattr(node, 'op') and hasattr(node.op, 'module'):
                module_to_node_group[node.op.module] = ng

    # First pass conduct out-channel pruning
    # print("Prune along out dim")
    pruned_out_dim_modules = set()
    for node_group in oto_graph.node_groups.values():
        if not node_group.is_prunable and not node_group.is_auxiliary:
            continue
        node_group.prune_out_dim(global_skip_modules=pruned_out_dim_modules)
        pruned_out_dim_modules = pruned_out_dim_modules.union(node_group.get_modules())

    # print("\nModel parameteres after prune out-dims")
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # ========== 特殊处理：ViTAttention 的 proj 层 in_dim 剪枝 ==========
    # 因为 qkv 和 proj 是 ViTAttention 的子模块，不在图的独立节点中
    # 需要在 out_dim 剪枝后手动同步 proj 的 in_dim
    for attn_module, attn_ng in module_to_node_group.items():
        # 检查是否是 ViTAttention 模块
        if not hasattr(attn_module, '__class__') or 'ViTAttention' not in type(attn_module).__name__:
            continue
        
        # 检查是否有剪枝索引
        if attn_ng.pruning_redundant_idxes is None:
            continue
        if len(attn_ng.pruning_redundant_idxes) == 0:
            continue
        
        # 获取 qkv 和 proj 子模块
        qkv_module = getattr(attn_module, 'qkv', None)
        proj_module = getattr(attn_module, 'proj', None)
        
        if qkv_module is None or proj_module is None:
            continue
        
        # 获取 qkv 当前的输出维度（可能已经被剪枝了）
        if hasattr(qkv_module, 'out_features'):
            qkv_out_current = qkv_module.out_features
        elif hasattr(qkv_module, 'weight'):
            qkv_out_current = qkv_module.weight.shape[0]
        else:
            continue
        
        # 获取 proj 当前的输入维度
        if hasattr(proj_module, 'in_features'):
            proj_in_current = proj_module.in_features
        elif hasattr(proj_module, 'weight'):
            proj_in_current = proj_module.weight.shape[1]
        else:
            continue
        
        # 期望的 proj in_features = qkv out_features / 3
        expected_proj_in = qkv_out_current // 3
        
        if proj_in_current != expected_proj_in:
            # 计算需要剪枝的索引
            # qkv 的剪枝索引是按 head 进行的（不是按维度）
            # ViT 有 num_heads 个 heads，每个 head 的维度是 head_dim
            qkv_pruned_head_idxes = attn_ng.pruning_redundant_idxes
            
            # 使用原始的 num_heads 和 head_dim（从 proj 的原始 in_features 计算）
            # ViT-Base: 768 dim, 12 heads, 64 head_dim
            original_num_heads = 12  # ViT-Base 默认 12 heads
            head_dim = proj_in_current // original_num_heads  # 768 / 12 = 64
            
            # 将 head 索引转换为 proj 输入的维度索引
            # 每个被剪的 head 对应 head_dim 个连续的维度
            proj_pruned_idxes = []
            for head_idx in qkv_pruned_head_idxes:
                start_idx = head_idx * head_dim
                for d in range(head_dim):
                    proj_pruned_idxes.append(start_idx + d)
            
            if len(proj_pruned_idxes) > 0:
                # 直接修改 proj 层的权重
                if hasattr(proj_module, 'weight'):
                    weight = proj_module.weight.data
                    # weight shape: [out_features, in_features]
                    keep_idxes = [i for i in range(proj_in_current) if i not in proj_pruned_idxes]
                    proj_module.weight.data = weight[:, keep_idxes].contiguous()
                    if hasattr(proj_module, 'in_features'):
                        proj_module.in_features = len(keep_idxes)

    # Second pass conduct in-channel pruning
    def find_incoming_node_group_stem_node(graph, node, src_ng, visited, incoming_node_groups, incoming_stem_node_ids, verbose=False):
        if verbose:
            print("\tfind_incoming_node_group_stem_node", node.id)
        if src_ng.id not in node.node_group_ids and not src_ng.contain_node(node):
            incoming_node_groups.update(node.node_group_ids)
            return 
        visited[node.id] = True
        for node_in in graph.incoming(node):
            if node_in.is_stem():
                incoming_stem_node_ids.add(node_in)
                return     
            if not visited[node_in.id]:                    
                find_incoming_node_group_stem_node(graph, node_in, src_ng, visited, incoming_node_groups, incoming_stem_node_ids)
    
    # print("\n\nPrune via in-dim")
    pruned_in_dim_modules = set()
    debug_node_ids = []

    verbose = False
    for node_group in oto_graph.node_groups.values():
        # print(node_group.id, pruned_in_dim_modules)
        for node in node_group.nodes.values():
            verbose = True if node.id in debug_node_ids else False
            if verbose:
                print("Node id", node.id, "verbose", node_group.id, node.op)
            
            if node.pruned_status['in_dim']:
                if verbose:
                    print("node.pruned_status['in_dim']", node.pruned_status['in_dim'])
                continue
            
            if node.op.module in pruned_in_dim_modules:
                if verbose:
                    print("node.op.module in pruned_in_dim_modules", node.op.module)
                continue

            if not hasattr(node.op, 'prune_in_dim'):
                if verbose:
                    print("node.op has no prune_in_dim")
                continue

            incoming_node_groups = set()
            incoming_stem_nodes = set()
            if verbose:
                print("find_incoming_node_group_stem_node for ", node.id)
            find_incoming_node_group_stem_node(
                oto_graph,
                node,
                node_group,
                oto_graph.visited_dict(),
                incoming_node_groups,
                incoming_stem_nodes,
                verbose=verbose,
            )
            if verbose:
                print("incoming_node_groups: ", incoming_node_groups)
                print("incoming_stem_nodes: ", incoming_stem_nodes)

            in_dim_pruned_idxes = None
            incoming_ng = None
            if len(incoming_stem_nodes) > 0:
                incoming_stem_node = next(iter(incoming_stem_nodes))
                incoming_ng = oto_graph.node_groups[incoming_stem_node.node_group_ids[0]]
                in_dim_pruned_idxes = incoming_ng.pruning_redundant_idxes
            elif len(incoming_node_groups) > 0:
                incoming_ng_id = None
                for ng_id in incoming_node_groups:
                    ng = oto_graph.node_groups[ng_id]
                    if ng.is_prunable or ng.is_auxiliary:
                        incoming_ng_id = ng_id
                    elif not ng.is_prunable and len(ng.param_names) > 0:
                        incoming_ng_id = None
                        break
                if incoming_ng_id is not None:
                    incoming_ng = oto_graph.node_groups[incoming_ng_id]
                    in_dim_pruned_idxes = incoming_ng.pruning_redundant_idxes
            # fallback：若上面未找到，则尝试直接用上游第一个节点所在的 node_group
            if incoming_ng is None:
                incoming_nodes = oto_graph.incoming(node)
                if len(incoming_nodes) > 0 and len(incoming_nodes[0].node_group_ids) > 0:
                    incoming_ng = oto_graph.node_groups[incoming_nodes[0].node_group_ids[0]]
                    in_dim_pruned_idxes = incoming_ng.pruning_redundant_idxes

            # 特殊处理：对于 MLP 的 fc2 层和 Attention 的 proj 层
            # 因为它们的上游层可能在不同的 node_group 中，需要手动关联
            if hasattr(node.op, 'module'):
                module = node.op.module
                module_name = None
                for name, m in model.named_modules():
                    if m is module:
                        module_name = name
                        break
                
                # 检查是否是 fc2 类型的层（输入来自 MLP 隐藏层）
                if module_name and '.fc2' in module_name:
                    # 找到对应的 fc1
                    fc1_name = module_name.replace('.fc2', '.fc1')
                    fc1_module = None
                    for name, m in model.named_modules():
                        if name == fc1_name:
                            fc1_module = m
                            break
                    if fc1_module is not None and fc1_module in module_to_node_group:
                        fc1_ng = module_to_node_group[fc1_module]
                        if fc1_ng.pruning_redundant_idxes is not None and len(fc1_ng.pruning_redundant_idxes) > 0:
                            # 使用 fc1 的剪枝索引，覆盖之前可能错误的索引
                            in_dim_pruned_idxes = fc1_ng.pruning_redundant_idxes
                            incoming_ng = fc1_ng
                
                # 检查是否是 attn.proj 层（输入来自 attention 的多头输出）
                # proj 的输入维度应该和 qkv 输出的 1/3 保持一致
                elif module_name and '.attn.proj' in module_name:
                    # 找到对应的 qkv 层（直接从模型中查找，不依赖 module_to_node_group）
                    qkv_name = module_name.replace('.attn.proj', '.attn.qkv')
                    qkv_module = None
                    for name, m in model.named_modules():
                        if name == qkv_name:
                            qkv_module = m
                            break
                    
                    # 找到父模块 ViTAttention
                    attn_name = module_name.replace('.proj', '')  # blocks.X.attn
                    attn_module = None
                    for name, m in model.named_modules():
                        if name == attn_name:
                            attn_module = m
                            break
                    
                    # 通过 ViTAttention 模块获取剪枝索引
                    attn_ng = module_to_node_group.get(attn_module, None)
                    
                    if qkv_module is not None and attn_ng is not None:
                        if attn_ng.pruning_redundant_idxes is not None and len(attn_ng.pruning_redundant_idxes) > 0:
                            # qkv 的剪枝索引是针对 3*dim 的输出
                            # proj 的输入是 dim（即 qkv 输出的 1/3）
                            qkv_pruned_idxes = attn_ng.pruning_redundant_idxes
                            
                            # 获取 qkv 的原始输出维度
                            if hasattr(qkv_module, 'out_features'):
                                qkv_out = qkv_module.out_features
                            elif hasattr(qkv_module, 'weight'):
                                qkv_out = qkv_module.weight.shape[0]
                            else:
                                qkv_out = 2304  # 默认值 768*3
                            
                            dim = qkv_out // 3
                            # 只取 v 部分的剪枝索引（索引在 [2*dim, 3*dim) 范围内的）
                            # 并转换为 proj 输入的索引（减去 2*dim）
                            proj_pruned_idxes = []
                            for idx in qkv_pruned_idxes:
                                if 2 * dim <= idx < 3 * dim:
                                    proj_pruned_idxes.append(idx - 2 * dim)
                            
                            print(f"[DEBUG] {module_name}: qkv_out={qkv_out}, dim={dim}, qkv_pruned={len(qkv_pruned_idxes)}, proj_pruned={len(proj_pruned_idxes)}")
                            
                            if len(proj_pruned_idxes) > 0:
                                in_dim_pruned_idxes = proj_pruned_idxes
                                incoming_ng = attn_ng
            
            if in_dim_pruned_idxes is None or len(in_dim_pruned_idxes) == 0:
                continue

            if hasattr(incoming_ng, 'op'):
                num_heads = 1
                head_dim = 1
                if hasattr(incoming_ng.op, 'prune_mode') and incoming_ng.op.prune_mode == 'num_head':
                    if hasattr(incoming_ng.op, 'num_heads'):
                        num_heads = incoming_ng.op.num_heads
                    if hasattr(incoming_ng.op, 'head_dim'):
                        head_dim = incoming_ng.op.head_dim
                    if num_heads > 1 and head_dim > 1:
                        in_dim_pruned_idxes = list()
                        for i in incoming_ng.pruning_redundant_idxes:
                            in_dim_pruned_idxes.extend([h + i * head_dim for h in range(head_dim)])
                else:
                    if hasattr(incoming_ng.op, 'num_heads'):
                        num_heads = incoming_ng.op.num_heads
                    if hasattr(incoming_ng.op, 'head_dim'):
                        head_dim = incoming_ng.op.head_dim
                    if num_heads > 1 and head_dim > 1:
                        in_dim_pruned_idxes = list()
                        for h in range(num_heads):
                            in_dim_pruned_idxes.extend([i + h * head_dim for i in incoming_ng.pruning_redundant_idxes])
                
            # To tackle reshape as flatten operator followed by linear/quantizelinear
            incoming_nodes = oto_graph.incoming(node)
            is_fc2_special_case = False
            if len(incoming_nodes) == 0:
                # 对于 fc2，即使没有上游节点也继续处理（因为我们已经有 fc1 的剪枝索引）
                module_name = None
                if hasattr(node.op, 'module'):
                    for name, m in model.named_modules():
                        if m is node.op.module:
                            module_name = name
                            break
                if module_name and '.fc2' in module_name:
                    is_fc2_special_case = True
                else:
                    continue
            
            # 只有在有上游节点且不是 fc2 特殊情况时才处理 flatten 逻辑
            if len(incoming_nodes) > 0 and not is_fc2_special_case:
                node_in = incoming_nodes[0]
                if node_in.op_name == 'flatten' and node.op_name in ['linear', 'quantizelinear']:
                    if not hasattr(incoming_ng, 'num_groups') or incoming_ng.num_groups == 0:
                        continue
                    if node.op.module.in_features % incoming_ng.num_groups != 0:
                        continue
                    expand_time = node.op.module.in_features // incoming_ng.num_groups
                    in_dim_pruned_idxes_refined = list()
                    for idx in in_dim_pruned_idxes:
                        in_dim_pruned_idxes_refined.extend([i + idx * expand_time for i in range(expand_time)])
                    in_dim_pruned_idxes = in_dim_pruned_idxes_refined

            if not node.pruned_status['in_dim']:
                if verbose:
                    print(type(node.op), node.param_names)
                    print(node.op.module)
                    print(node.param_names)
                    print(node.id)
                    print(len(in_dim_pruned_idxes))
                # 类型检查
                if not hasattr(node.op.module, 'in_features'):
                    # 跳过没有 in_features 属性的模块（如 ViTAttention 等复合模块）
                    continue
                # 越界保护
                if len(in_dim_pruned_idxes) > 0 and max(in_dim_pruned_idxes) >= node.op.module.in_features:
                    continue
                node.op.prune_in_dim(pruned_idxes=in_dim_pruned_idxes, param_names=node.param_names, verbose=verbose)
                node.pruned_status['in_dim'] = True
                # Skip composed node group since such groups may contain multiple nodes correspond to the same module 
                if node.op.is_basic and not node_group.contain_lora():
                    pruned_in_dim_modules.add(node.op.module)
                if verbose:
                    print(node.op.module)
    
    # print("\nModel parameteres after prune in channels")
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    if merge_lora_to_base:
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()

    if unmerge_lora_to_base:
        if hasattr(model, 'unmerge_and_unload'):
            model = model.unmerge_and_unload()
            
    if export_huggingface_format:
        model.save_pretrained(compressed_model_path)
    elif ckpt_format == 'torch':
        torch.save(model, compressed_model_path)
    elif ckpt_format == 'onnx':
        torch.onnx.export(
            model,
            oto_graph.dummy_input,
            compressed_model_path)
    return compressed_model_path, full_group_sparse_model_path 
