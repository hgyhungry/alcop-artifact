import tvm
from tvm import tir
from collections import defaultdict
import pdb
DEBUG=False

def InjectPipelinedBuffer():
    class PipelinedBufferInfo:
        def __init__(self, buffer_var, num_stage, scope, pipelined_for=None, 
            realization_node=None, stride=-1, predecessor=None, first_consumer_node=None,
            last_consumer_node=None, prologue_inject_node=None) -> None:
            self.buffer = buffer_var
            self.num_stage = num_stage
            self.scope = str(scope)
            self.stride = stride
            self.pipelined_for = pipelined_for
            self.predecessor = predecessor
            self.realization_node = realization_node
            self.first_consumer_node = first_consumer_node
            self.last_consumer_node = last_consumer_node
            self.prologue_inject_node = prologue_inject_node
            self.realization_body = None

        def __str__(self) -> str:
            return f"""buffer: {self.buffer}; stage: {self.num_stage}; scope: {self.scope}
            stride: {self.stride}; pipelined_for: {self.pipelined_for}; 
            predecessor: {self.predecessor}; realization_node: {self.realization_node}
            first_consum: {self.first_consumer_node}; last_consum: {self.last_consumer_node}
            prologue_inject: {self.prologue_inject_node}\n"""

    pipelined_buffers = set()
    pipelined_buffer_name2node = dict()
    buffer_info = dict()
    
    def extract_pipelined_buffers(op):
        if isinstance(op, tir.AttrStmt):
            if op.attr_key == "pipelined_buffer_scope":
                buffer_var = op.node.data
                num_stage  = op.value
                scope      = str(op.node.scope())
                pipelined_buffers.add(buffer_var)
                pipelined_buffer_name2node[buffer_var.name] = buffer_var
                buffer_info[buffer_var] = PipelinedBufferInfo(
                    buffer_var=buffer_var, 
                    num_stage=num_stage, 
                    scope=scope)
    
    ############## Analysis ##############
    
    scope_to_buffers = defaultdict(list)
    node_to_injected_prologues = defaultdict(list)
    for_loops = list()
    streaming_loop_extension = defaultdict(list)

    def _analyze_buffer_store(buffer, predecessor_buffer):
        if buffer not in buffer_info:
            buffer = pipelined_buffer_name2node[buffer.name]
        try:
            predecessor_buffer = pipelined_buffer_name2node[predecessor_buffer.name]
        except:
            pass
        
        if DEBUG:
            pdb.set_trace()
        info = buffer_info[buffer]
        try:
            block_node = for_loops[for_loops.index(info.pipelined_for) +1]
        except:
            # raise RuntimeError("pipeline buffer load not inside a spatial loop")
            # DO NOTHING
            return
        buffer_info[buffer].realization_node = block_node.loop_var
        buffer_info[buffer].realization_body = block_node
        
        buffer_info[buffer].predecessor = predecessor_buffer

        this, pred = buffer, predecessor_buffer
        while pred.name in pipelined_buffer_name2node:
            this, pred = pred, buffer_info[pred].predecessor
        prologue_inject_node = buffer_info[this].pipelined_for
        buffer_info[buffer].prologue_inject_node = prologue_inject_node.loop_var
        node_to_injected_prologues[prologue_inject_node.loop_var].append(buffer)

    def _analyze_buffer_load(buffer):
        if buffer not in buffer_info:
            buffer = pipelined_buffer_name2node[buffer.name]
            
        if DEBUG:
            pdb.set_trace()
        info = buffer_info[buffer]

        try:
            block_node = for_loops[for_loops.index(info.pipelined_for) +1]
        except:
            # raise RuntimeError("pipeline buffer load not inside a spatial loop")
            # DO NOTHING
            return

        if info.first_consumer_node is None:
            info.first_consumer_node = block_node.loop_var
        info.last_consumer_node = block_node.loop_var
        buffer_info[buffer] = info

    def analysis_preorder(op):
        if DEBUG:
            print(type(op))

        if isinstance(op, tir.For):
            # if DEBUG:
            #     pdb.set_trace()
            for_loops.append(op) # maintain for node stack

        elif isinstance(op, tir.AttrStmt):
            if op.attr_key == "pipelined_buffer_scope":
                # record pipelined for
                buffer = op.node.data
                # sanity recursion TODO (guyue) check if necessary
                buffer = pipelined_buffer_name2node[buffer.name]

                streaming_loop = for_loops[-1]
                buffer_info[buffer].pipelined_for = streaming_loop
                # boundary check
                if streaming_loop.extent < buffer_info[buffer].num_stage:
                    buffer_info[buffer].num_stage = streaming_loop.extent
                    if buffer_info[buffer].num_stage == 1:
                        # Stage = 1 is not a pipelined buffer; 
                        # This is for working-around the bug when
                        # shared memory is pipelined stage=1 and reg is pipelined with >1 stages
                        pipelined_buffers.remove(buffer)
                        del pipelined_buffer_name2node[buffer.name]
                # assert streaming_loop.extent >= buffer_info[buffer].num_stage, \
                #     "pipeline number of stage cannot be larger than the extent of streaming loop"
                # if for_loops.index(streaming_loop) > 0:
                #     assert streaming_loop.extent % buffer_info[buffer].num_stage == 0, \
                #         "inner-loop pipelined buffer range must be divisible by num_stage"

                scope = buffer_info[buffer].scope
                scope_to_buffers[scope].append(buffer)

        elif isinstance(op, tir.Allocate):
            if op.buffer_var.name in pipelined_buffer_name2node:
                # update the scope because a weird bug: attr buffer and actual buffer is not same
                buffer = pipelined_buffer_name2node[op.buffer_var.name]
                buffer_info[buffer].scope = op.buffer_var.type_annotation.storage_scope
                # record total buffer size
                stride = 1
                for extent in op.extents:
                    stride *= extent
                buffer_info[buffer].stride = stride
                
        elif isinstance(op, tir.Store) :
            if DEBUG:
                pdb.set_trace()
            if op.buffer_var.name in pipelined_buffer_name2node:
                """update predecessor, prologue_inject_node, """
                if DEBUG:
                    pdb.set_trace()
                buffer = op.buffer_var
                predecessor_buffer = None
                if isinstance(op.value, tir.Load):
                    predecessor_buffer = op.value.buffer_var
                elif isinstance(op.value, tir.Call) and \
                    op.value.op.name=='tir.if_then_else':
                    # Handle predicated case
                    cond, then_, else_ = op.value.args
                    if ((isinstance(else_, tir.Broadcast) and else_.value.value==0) or\
                       ((isinstance(else_, tir.IntImm) or isinstance(else_, tir.FloatImm)) \
                            and else_.value==0 )) and \
                        isinstance(then_, tir.Load): # Handle vectorized case
                        predecessor_buffer = then_.buffer_var
                if predecessor_buffer is None:
                    assert False, f"value type {type(op.value)} for pipeline buffer load is not supported"
                
                _analyze_buffer_store(buffer, predecessor_buffer)
                
        elif isinstance(op, tir.Load):
            if op.buffer_var.name in pipelined_buffer_name2node:
                if DEBUG:
                    pdb.set_trace()
                """update consumer node"""
                _analyze_buffer_load(op.buffer_var)
                    
        ##### handle mma intrinsic that imply 'store' and 'load'
        elif isinstance(op, tir.Call):
            _op, _args = op.op, op.args
            if _op.name in ["tir.tvm_load_matrix_sync", "tir.tvm_asm_ldmatrix"]:
                if DEBUG:
                    pdb.set_trace()
                store_buffer, wmma_m, wmma_n, wmma_k, _, load_address, _, _ = _args[:8]
                _, load_buffer, _, _, _ = load_address.args

                if store_buffer.name in pipelined_buffer_name2node:
                    _analyze_buffer_store(store_buffer, load_buffer)

                    # wmma buffers are indexed based on tensorized shape, so change stride
                    store_tensorized_stride = 1
                    if "wmma.matrix_a" in store_buffer.type_annotation.storage_scope:
                        store_tensorized_stride = int(wmma_m) * int(wmma_k)
                    elif "wmma.matrix_b" in store_buffer.type_annotation.storage_scope:
                        store_tensorized_stride = int(wmma_n) * int(wmma_k)
                    buffer_info[pipelined_buffer_name2node[store_buffer.name]].stride //= store_tensorized_stride
                    # print(f"re-set {store_buffer} stride to {buffer_info[store_buffer].stride}")
                    
                if load_buffer.name in pipelined_buffer_name2node:
                    _analyze_buffer_load(load_buffer)
            elif _op.name == "tir.tvm_mma_sync":
                _, _, load_buffer_a, _, load_buffer_b, _, _, _ = _args
                if load_buffer_a.name in pipelined_buffer_name2node:
                    _analyze_buffer_load(load_buffer_a)
                if load_buffer_b.name in pipelined_buffer_name2node:
                    _analyze_buffer_load(load_buffer_b)
            elif _op.name == "tir.tvm_load_shared":
                load_address, store_address, _, _, _ = _args[:5]
                _, load_buffer, _, _, _ = load_address.args
                _, store_buffer, _, _, _ = store_address.args

                if store_buffer.name in pipelined_buffer_name2node:
                    _analyze_buffer_store(store_buffer, load_buffer)
                if load_buffer.name in pipelined_buffer_name2node:
                    _analyze_buffer_load(load_buffer)

    def analysis_postorder(op):
        if isinstance(op, tir.For):
            for_loops.pop() # maintain for node stack

            # extension for multi streaming loop
            if isinstance(op.body, tir.For):
                inner_loop = op.body
                while isinstance(inner_loop.body, tir.For):
                    inner_loop = inner_loop.body
                for info in buffer_info.values():
                    if info.pipelined_for == inner_loop:
                        # print(info.scope, op.loop_var, inner_loop.loop_var)
                        assert info.scope in ['shared.dyn', 'shared'], \
                            "cannot handle multiple streaming loop for register level pipelining"
                        streaming_loop_extension[inner_loop.loop_var].append(op.loop_var)
        
    class ScopePipelineSynchInfo:
        def __init__(self, scope, first_produced_buffer, last_produced_buffer, fusible):
            self.scope = scope
            self.first_produced_buffer = first_produced_buffer
            self.last_produced_buffer = last_produced_buffer
            self.fusible = fusible
        def __str__(self) -> str:
            return f"""scope: {self.scope}; first_producer_buffer: {self.first_produced_buffer};
            last_produced_buffer: {self.last_produced_buffer}; 
            fusible: {self.fusible}\n
            """

    scope_info = dict()
    node_to_synch_injection = defaultdict(list)

    intrin_pip_decl = "tir.tvm_pipeline_decl"
    intrin_prod_acq = "tir.tvm_pipeline_producer_acquire"
    intrin_prod_com = "tir.tvm_pipeline_producer_commit"
    intrin_cons_wai = "tir.tvm_pipeline_consumer_wait"
    intrin_cons_rel = "tir.tvm_pipeline_consumer_release"
    intrin_memcpy_async = "tir.tvm_pipeline_memcpy_async"
    intrin_flush = "tir.tvm_pipeline_flush"
    def _get_pipeline_uname(fusible, buffer=None, scope=None):
        if fusible:
            return f"_pipeline_scope_{scope}"
        return f"_pipeline_{buffer.name}"

    def _post_analysis():
        for scope, buffer_list in scope_to_buffers.items():
            first_produced_buffer = buffer_list[0]
            last_produced_buffer = buffer_list[-1]
            
            #### decide if synchronization can be fused
            fusible = True
            first_buffer_info = buffer_info[first_produced_buffer]
            for buffer in buffer_list[1:]:
                info = buffer_info[buffer]
                if info.num_stage != first_buffer_info.num_stage or \
                    info.pipelined_for != first_buffer_info.pipelined_for or \
                    info.first_consumer_node != first_buffer_info.first_consumer_node or \
                    info.last_consumer_node != first_buffer_info.last_consumer_node:
                    fusible = False
                    break
            scope_info[scope] = ScopePipelineSynchInfo(scope=scope,
                                    first_produced_buffer=first_produced_buffer,
                                    last_produced_buffer=last_produced_buffer,
                                    fusible=fusible)
            #### inject synch point
            if fusible:
                first_buffer_info = buffer_info[buffer_list[0]]
                pa_before = first_buffer_info.realization_node
                pc_after  = buffer_info[buffer_list[-1]].realization_node
                cw_before = first_buffer_info.first_consumer_node
                cr_after  = first_buffer_info.last_consumer_node
                pipeline_uname = _get_pipeline_uname(fusible=True, scope=scope)
                
                args = [pipeline_uname, scope, first_buffer_info.num_stage]
                node_to_synch_injection[pa_before].append((intrin_prod_acq, args, buffer_list[0]))
                node_to_synch_injection[pc_after].append((intrin_prod_com, args, buffer_list[-1]))    
                node_to_synch_injection[cw_before].append((intrin_cons_wai, args, buffer_list[0]))
                node_to_synch_injection[cr_after].append((intrin_cons_rel, args, buffer_list[0]))
            else:
                for buffer in buffer_list:
                    info = buffer_info[buffer]
                    pipeline_uname = _get_pipeline_uname(fusible=False, buffer=buffer)
                    
                    args = [pipeline_uname, scope, info.num_stage]
                    node_to_synch_injection[info.realization_node].append((intrin_prod_acq, args, buffer))
                    node_to_synch_injection[info.realization_node].append((intrin_prod_com, args, buffer))    
                    node_to_synch_injection[info.first_consumer_node].append((intrin_cons_wai, args, buffer))
                    node_to_synch_injection[info.last_consumer_node].append((intrin_cons_rel, args, buffer))


    ############## Transformation ##############

    def _transform_store(store_buffer, load_buffer, store_index, store_tensorized_stride=1):
        """ This transformation has 3 functionalities: if k is the pipelined for var
        1. change the store index by offsetting (k + num_stage -1) % num_stage
        2. substitute the k in load index (k + num_stage -1) % k.extent
        3. if the producer itself is pipelined buffer, when k goes over num_stage and is wrapped down,
            the stage iterator in producer stage should +1 """
        if store_buffer not in buffer_info:
            store_buffer = pipelined_buffer_name2node[store_buffer.name]
        try:
            load_buffer = pipelined_buffer_name2node[load_buffer.name]
        except:
            pass
        store_info = buffer_info[store_buffer]
        ### step 1
        global_index = store_info.pipelined_for.loop_var
        extent_prod = 1
        for i in range(for_loops.index(store_info.pipelined_for)-1, -1, -1):
            extent_prod *= for_loops[i+1].extent
            global_index = tir.Add(tir.Mul(for_loops[i].loop_var, extent_prod), global_index)
        offset = tir.Mul(
                    a=tir.FloorMod(
                        a=tir.Add(global_index, store_info.num_stage-1),
                        b=store_info.num_stage
                    ),
                    b=store_info.stride if int(store_tensorized_stride)==1 else \
                      tir.FloorDiv(
                        a=store_info.stride,
                        b=store_tensorized_stride
                    )
        )
        if isinstance(store_index, tir.Ramp):
            new_store_index = tir.Ramp(base=tir.Add(offset, store_index.base), 
                stride=store_index.stride, lanes=store_index.lanes)
        else:
            new_store_index = tir.Add(offset, store_index)
        
        ### step 2:
        store_stage_var = store_info.pipelined_for.loop_var
        replacement1 = tir.FloorMod(
            a=tir.Add(
                a=store_stage_var,
                b=store_info.num_stage-1
            ),
            b=store_info.pipelined_for.extent
        )
        replace_dict = {store_stage_var:replacement1}
        
        # step 3
        sum = tir.Add(store_stage_var, store_info.num_stage-1)
        for i in range(for_loops.index(store_info.pipelined_for)-1, -1, -1):
            carry = tir.FloorDiv(sum, for_loops[i+1].extent)
            sum = tir.Add(for_loops[i].loop_var, carry)
            replace_dict[for_loops[i].loop_var] = tir.FloorMod(sum, for_loops[i].extent)

        return new_store_index, replace_dict
    
    def _transform_shared_memcpy_async(op):
        if isinstance(op, tir.Store):
            args = []
            
            args.append(op.buffer_var)                              # 0
            
            store_index, lanes = op.index, 1
            if isinstance(store_index, tir.Ramp):
                lanes = store_index.lanes
                store_index = store_index.base * store_index.stride
                # TODO: cannot support element-wise predication
                # info in op.predicate
            args.append(store_index)                                # 1
            
            # if cannot satisfy alignment should drop this optimziation
            type_name = op.buffer_var.type_annotation.element_type.dtype
            if '16' in type_name and lanes not in [2,4,8]:
                return op

            value = op.value
            if isinstance(value, tir.Call) and value.op.name=="tir.if_then_else":
                value = value.args[1]
            args.append(value.buffer_var)                           # 2
            load_index = value.index
            if isinstance(value.index, tir.Ramp):
                load_index = load_index.base * load_index.stride
            args.append(load_index)                                 # 3
            
            args.append(lanes)                                      # 4

            scope = buffer_info[pipelined_buffer_name2node[op.buffer_var.name]].scope
            pipeline_uname = _get_pipeline_uname(fusible=scope_info[scope].fusible,
                buffer=op.buffer_var, scope=scope)
            args.append(pipeline_uname)                             # 5

            # import pdb; pdb.set_trace()
            args.append((op.buffer_var.type_annotation.element_type.dtype)) # 6

            # if isinstance(op.value, tir.IfThenElse):
            #     args.append(op.value.condition)                     # 7
            if isinstance(op.value, tir.Call) and op.value.op.name=="tir.if_then_else":
                args.append(op.value.args[0])                       # 7

            return tir.Evaluate(tir.Call("handle", intrin_memcpy_async, args))

        elif isinstance(op, tir.Call) and op.op.name == "tir.tvm_load_shared":
            new_args = op.args[:5]
            new_args.append("async")
            if len(new_args) > 5:
                new_args = new_args + op.args[5:]
            return tir.Call(op.dtype, op.op, new_args) # just add async copy
        
        else:
            assert False, f"unsupported op type {type(op)} passed to async copy transformation"

    def _transform_load(buffer, index, tensorized_stride=1):
        if buffer not in buffer_info:
            buffer = pipelined_buffer_name2node[buffer.name]
            
        info = buffer_info[buffer]
        if info.pipelined_for not in for_loops:
            return index # DO NOTHING
        stage_var = info.pipelined_for.loop_var
        global_index = stage_var
        extent_prod = 1
        for i in range(for_loops.index(info.pipelined_for)-1, -1, -1):
            extent_prod *= for_loops[i+1].extent
            global_index = tir.Add(tir.Mul(for_loops[i].loop_var, extent_prod), global_index)
        offset = tir.Mul(
            a=tir.FloorMod(
                a=global_index,
                b=info.num_stage
            ),
            b=info.stride if int(tensorized_stride)==1 else \
              tir.FloorDiv(
                a=info.stride,
                b=tensorized_stride
            )
        )
        if isinstance(index, tir.Ramp):
            new_index = tir.Ramp(base=tir.Add(offset, index.base), 
                stride=index.stride, lanes=index.lanes)
        else:
            new_index = tir.Add(offset, index)
        return new_index
        
    def transform_preorder(op):
        if isinstance(op, tir.For):
            for_loops.append(op)
    
    relocated_prologue_epilogue = defaultdict(list)
    def transform_postorder(op):
        if isinstance(op, tir.For):
            for_loops.pop()
            """This part does 3 jobs:
            1. inject synchronization primitives,  
            2. if it is producer, update realization body
                the third step cannot be done before this point, 
                because we want all index shifting in prologue
            3. inject prologues"""

            body = op

            #### inject synchronization
            if op.loop_var in node_to_synch_injection:
                # sandwich
                before, inside, after = [], [], []
                for key, args, _ in node_to_synch_injection[op.loop_var]:
                    if key == intrin_prod_acq:
                        before.append(tir.Evaluate(tir.Call("handle", key, args)))
                    elif key == intrin_prod_com:
                        after.append(tir.Evaluate(tir.Call("handle", key, args)))
                    elif key == intrin_cons_rel:
                        after.append(tir.Evaluate(tir.Call("handle", key, args)))
                    elif key == intrin_cons_wai:
                        # need to handle if successor buffer is also pipelined
                        consumer_max_stage = 1
                        for info in buffer_info.values():
                            if info.pipelined_for.loop_var == op.loop_var:
                                consumer_max_stage = max(consumer_max_stage, info.num_stage)
                        if consumer_max_stage == 1:
                            # not consumed by pipelined buffer
                            before.append(tir.Evaluate(tir.Call("handle", key, args)))
                        else:
                            # consumed by some pipelined buffer, sync should be pre-posed
                            condition = tir.EQ(
                                tir.FloorMod(
                                    tir.Add(op.loop_var, consumer_max_stage-1),
                                    op.extent
                                ),
                                0
                            )
                            inside.append(tir.IfThenElse(
                                condition=condition,
                                then_case=tir.Evaluate(tir.Call("handle", key, args)),
                                else_case=None
                            ))
                if len(inside):
                    nest = tir.SeqStmt(inside + [op.body])
                    body = tir.For(body.loop_var, body.min, body.extent, body.kind, nest, body.thread_binding, body.annotations)
                if len(after) or len(before):
                    body = tir.SeqStmt(before + [body] + after)
                # update realization node for related buffers produced in this node
                # this should be done before injecting prologue
                for key, _, buffer in node_to_synch_injection[op.loop_var]:
                    if key in [intrin_prod_acq, intrin_prod_com]:
                        buffer_info[buffer].realization_body = body

            #### inject prologue
            if op.loop_var in node_to_injected_prologues:
                buffer_list = node_to_injected_prologues[op.loop_var]
                prologue_nodes = []
                epilogue_nodes = []

                initial_wait_injected = defaultdict(lambda : False)
                last_var_, last_extent_ = None, None
                
                for buffer in buffer_list:
                    info = buffer_info[buffer]
                    body_copy = info.realization_body
                    
                    if info.predecessor.name in pipelined_buffer_name2node:
                        predecessor_info = buffer_info[info.predecessor]
                        if scope_info[predecessor_info.scope].fusible :
                            if not initial_wait_injected[predecessor_info.scope]:
                                pipeline_uname = _get_pipeline_uname(fusible=True, scope=predecessor_info.scope)
                                args = [pipeline_uname, predecessor_info.scope, predecessor_info.num_stage]
                                prologue_nodes.append(tir.Evaluate(tir.Call("handle", intrin_cons_wai, args)))
                                initial_wait_injected[predecessor_info.scope] = True
                        else:
                            if not initial_wait_injected[info.predecessor]:
                                pipeline_uname = _get_pipeline_uname(fusible=False, buffer=info.predecessor)
                                args = [pipeline_uname, predecessor_info.scope, predecessor_info.num_stage]
                                prologue_nodes.append(tir.Evaluate(tir.Call("handle", intrin_cons_wai, args)))
                                initial_wait_injected[info.predecessor] = True
                        # pred_var_shift = -1 * (predecessor_info.num_stage -1)
                        body_copy = tir.stmt_functor.substitute(body_copy, 
                            {predecessor_info.pipelined_for.loop_var: tir.IntImm('int', 0)}) # TODO
                    
                    var_, extent_ = info.pipelined_for.loop_var, info.num_stage-1
                    if len(prologue_nodes) and isinstance(prologue_nodes[-1], tir.For) and \
                        (last_var_ == var_ and last_extent_ == extent_): 
                        
                        # can be merged
                        last_node = prologue_nodes.pop()
                        body_copy = tir.stmt_functor.substitute(body_copy, {var_: tir.Sub(last_node.loop_var, extent_)})
                        new_body = tir.SeqStmt([last_node.body, body_copy])
                        last_node = tir.For(last_node.loop_var, last_node.min, last_node.extent, last_node.kind,            
                                            body=new_body)
                        prologue_nodes.append(last_node)
                    else:
                        var_copy = tir.Var("prologue."+var_.name, var_.dtype)
                        body_copy = tir.stmt_functor.substitute(body_copy, {var_: tir.Sub(var_copy, extent_)})
                        new_node = tir.For(var_copy, 0, extent_, info.pipelined_for.kind, body=body_copy)
                        prologue_nodes.append(new_node)
                        last_var_, last_extent_ = var_, extent_
                        if scope_info[info.scope].fusible:
                            pipeline_uname = _get_pipeline_uname(fusible=True, scope=info.scope)
                        else:
                            pipeline_uname = _get_pipeline_uname(fusible=False, buffer=buffer)
                        args = [pipeline_uname, info.scope, info.num_stage]
                        epilogue_nodes = [tir.Evaluate(tir.Call("handle", intrin_flush, args))] + epilogue_nodes
                # fuse pipeline that is wrapped inside another sequential loop
                if body.loop_var in streaming_loop_extension:
                    substitute = dict([(var, 0) for var in streaming_loop_extension[body.loop_var]])
                    relocated_prologue_nodes = [tir.stmt_functor.substitute(n, substitute) for n in prologue_nodes]
                    relocated_epilogue_nodes = [tir.stmt_functor.substitute(n, substitute) for n in epilogue_nodes]
                    relocated_loop_var = streaming_loop_extension[body.loop_var][-1]
                    relocated_prologue_epilogue[relocated_loop_var] = [relocated_prologue_nodes, relocated_epilogue_nodes]
                else:    
                    body = tir.SeqStmt(prologue_nodes + [body] + epilogue_nodes)
            if op.loop_var in relocated_prologue_epilogue:
                p, e = relocated_prologue_epilogue[op.loop_var]
                body = tir.SeqStmt(p+[body]+e)
            return body

        elif isinstance(op, tir.AttrStmt):
            if op.attr_key == "pipelined_buffer_scope":
                # omit this attribute
                return op.body

        elif isinstance(op, tir.Allocate):
            if op.buffer_var.name in pipelined_buffer_name2node:
                buffer = op.buffer_var
                if buffer not in buffer_info:
                    buffer = pipelined_buffer_name2node[buffer.name]
                info = buffer_info[buffer]
                body = op.body
                #### inject pipeline declaration if necessary
                if not scope_info[info.scope].fusible or \
                    scope_info[info.scope].first_produced_buffer == buffer:
                    pipeline_uname = _get_pipeline_uname(fusible=scope_info[info.scope].fusible, buffer=buffer, scope=info.scope)
                    intrin_args = [pipeline_uname, info.scope, info.num_stage]
                    call_node = tir.Evaluate(tir.Call("handle", intrin_pip_decl, intrin_args))
                    body = tir.SeqStmt([call_node, body])
                #### multiply buffer size by num_stage
                new_extents = [tir.const(int(info.num_stage), op.extents[0].dtype)]
                for extent in op.extents:
                    new_extents.append(extent)
                body = tir.Allocate(op.buffer_var, op.dtype,  
                                    new_extents, op.condition, body, op.annotations, op.span)

                return body

        elif isinstance(op, tir.Store):
            if op.buffer_var.name in pipelined_buffer_name2node:
                buffer, index = op.buffer_var, op.index
                
                from_buffer = None
                if isinstance(op.value, tir.Load):
                    from_buffer = op.value.buffer_var
                elif isinstance(op.value, tir.Call) and \
                    op.value.op.name=='tir.if_then_else':
                    # Handle predicated case
                    cond, then_, else_ = op.value.args
                # elif isinstance(op.value, tir.IfThenElse):
                #     # Handle predicated case
                #     then_, else_ = op.value.then_case, op.value.else_case
                    if ((isinstance(else_, tir.IntImm)  or isinstance(else_, tir.FloatImm))and else_.value==0 or \
                        isinstance(else_, tir.Broadcast) and else_.value.value==0 ) and \
                        isinstance(then_, tir.Load): # Handle vectorized case
                        from_buffer = then_.buffer_var
                if from_buffer is None:
                    assert False, f"value type {type(op.value)} for pipeline buffer load is not supported"

                new_index, value_substi_map = _transform_store(buffer, from_buffer, index)
                
                node = tir.Store(op.buffer_var, 
                            value=tir.stmt_functor.substitute(op.value, value_substi_map),
                            index=new_index, 
                            predicate=op.predicate)
                
                ### inject memcpy async intrinsic
                if buffer_info[pipelined_buffer_name2node[op.buffer_var.name]].scope \
                    in [ "shared", "shared.dyn"]:
                    node = _transform_shared_memcpy_async(node)
                
                return node
                
        elif isinstance(op, tir.Load):
            if op.buffer_var.name in pipelined_buffer_name2node:
                new_index = _transform_load(op.buffer_var, op.index)
                return tir.Load(op.dtype, op.buffer_var, new_index, op.predicate, op.span)

        ##### handle mma intrinsic that imply 'store' and 'load'
        elif isinstance(op, tir.Call):
            _op, _args = op.op, op.args
            if _op.name in ["tir.tvm_load_matrix_sync", "tir.tvm_asm_ldmatrix"]:
                store_buffer, _, _, _, index, load_address, _, _ = _args[:8]
                _, load_buffer, load_index, _, _ = load_address.args
                
                
                if load_buffer.name in pipelined_buffer_name2node:
                    load_index = _transform_load(load_buffer, load_index)
                
                if store_buffer.name in pipelined_buffer_name2node:
                    index, value_substi_map = _transform_store(store_buffer, load_buffer, index)
                    load_index = tir.stmt_functor.substitute(load_index, value_substi_map)
                
                if store_buffer.name in pipelined_buffer_name2node or \
                        load_buffer.name in pipelined_buffer_name2node:
                    _addr_args = load_address.args[:2] + [load_index] + load_address.args[3:]
                    new_load_address = tir.Call(load_address.dtype, load_address.op, _addr_args)
                    _new_args = _args[:4] + [index, new_load_address] + _args[6:]
                    return tir.Call(op.dtype, op.op, _new_args)
            
            elif _op.name == "tir.tvm_mma_sync":
                _, _, load_buffer_a, index_a, load_buffer_b, index_b, _, _ = _args
                if load_buffer_a.name in pipelined_buffer_name2node:
                    index_a = _transform_load(load_buffer_a, index_a)
                if load_buffer_b.name in pipelined_buffer_name2node:
                    index_b = _transform_load(load_buffer_b, index_b)
                if load_buffer_a.name in pipelined_buffer_name2node or load_buffer_b.name in pipelined_buffer_name2node:
                    _new_args = _args[:3] + [index_a, _args[4], index_b] + _args[6:]
                    return tir.Call(op.dtype, op.op, _new_args)
            
            elif _op.name == "tir.tvm_load_shared":
                load_address, store_address, _, _, _ = _args[:5]
                _, load_buffer, load_index, _, _ = load_address.args
                _, store_buffer, store_index, _, _ = store_address.args

                if load_buffer.name in pipelined_buffer_name2node:
                    load_index = _transform_load(load_buffer, load_index)
                
                if store_buffer.name in pipelined_buffer_name2node:
                    store_index, value_substi_map = _transform_store(store_buffer, load_buffer, store_index)
                    load_index = tir.stmt_functor.substitute(load_index, value_substi_map)
                
                if store_buffer.name in pipelined_buffer_name2node or \
                        load_buffer.name in pipelined_buffer_name2node:
                    _load_addr_args = load_address.args[:2] + [load_index] + load_address.args[3:]
                    new_load_address = tir.Call(load_address.dtype, load_address.op, _load_addr_args)
                    _store_addr_args = store_address.args[:2] + [store_index] + store_address.args[3:]
                    new_store_address = tir.Call(store_address.dtype, store_address.op, _store_addr_args)
                    _new_args = [new_load_address, new_store_address] + _args[2:]
                
                    node = tir.Call(op.dtype, op.op, _new_args)
                    if buffer_info[pipelined_buffer_name2node[store_buffer.name]].scope \
                        in [ "shared", "shared.dyn"]:
                        return _transform_shared_memcpy_async(node)
                    return node

    def _ftransform(f):
        pipelined_buffers.clear()
        buffer_info.clear()
        print("[Info] calling pipeline buffer transformation")


        tvm.tir.stmt_functor.post_order_visit(f.body, extract_pipelined_buffers)
        if len(pipelined_buffers)==0:
            return f
        f = f.with_body(tvm.tir.stmt_functor.ir_transform(
            f.body, analysis_preorder, analysis_postorder,
            ["tir.For", "tir.AttrStmt", "tir.Load", "tir.Store", "tir.Allocate",
            "tir.Call"]))
        
        _post_analysis()
        
        return f.with_body(tvm.tir.stmt_functor.ir_transform(
            f.body, transform_preorder, transform_postorder, 
            ["tir.For", "tir.AttrStmt", "tir.Load", "tir.Store", "tir.Allocate",
            "tir.Call"]
        ))
    return _ftransform

def InjectSharedMemSwizzle():
    swizzled_buffers = set()
    produced = set()
    reused = set()
    def find_buffers(op):
        if isinstance(op, tir.AttrStmt) and op.attr_key == "swizzled_buffer_scope":
            buffer_var = op.node.data
            swizzled_buffers.add(buffer_var.name)
    
    def _swizzle(index):
        if isinstance(index, tir.Ramp):
            # assert index.lanes % 8 == 0
            return tir.Ramp(_swizzle(index.base), index.stride, index.lanes)
        return tir.bitwise_xor(
            a=index,
            b=tir.Mul(
                a=tir.FloorMod(
                    a=tir.FloorDiv(
                        a=index,
                        b=64
                    ),
                    b=8
                ),
                b=8
            )
        )

    def swizzle_buffer(op):
        if isinstance(op, tir.Store):
            if op.buffer_var.name in swizzled_buffers:
                new_index = _swizzle(op.index)
                if op.buffer_var.name not in produced:
                    produced.add(op.buffer_var.name)
                return tir.Store(op.buffer_var, op.value, index=new_index, predicate=op.predicate)
        elif isinstance(op, tir.Load):
            if op.buffer_var.name in swizzled_buffers:
                if op.buffer_var.name in produced and (
                    op.buffer_var.name not in reused 
                ):
                    new_index = _swizzle(op.index)
                    return tir.Load(op.dtype, op.buffer_var, new_index, op.predicate)
        elif isinstance(op, tir.Call) :
            if op.op.name == "tir.tvm_store_matrix_sync":
                access_ptr = op.args[5]
                buffer_name = access_ptr.args[1].name
                if buffer_name in swizzled_buffers:
                    # always cancel because we cannot implement shared buffer swizzle
                    if buffer_name not in produced:
                        swizzled_buffers.remove(access_ptr.args[1])
                    else:
                        reused.add(buffer_name)
            elif op.op.name == "tir.tvm_access_ptr":
                _, buffer, index, = op.args[:3]
                if buffer.name in swizzled_buffers and (buffer.name in produced) \
                    and (buffer.name not in reused):
                    new_index = _swizzle(index)
                    new_args = op.args[:2]
                    new_args.append(new_index)
                    for x in op.args[3:]:
                        new_args.append(x)
                    return tir.Call(op.dtype, op.op, new_args)
            elif op.op.name=="tir.tvm_pipeline_memcpy_async":
                dst_buffer, dst_index = op.args[:2]
                if dst_buffer.name in swizzled_buffers:
                    produced.add(dst_buffer.name)
                    new_index = _swizzle(dst_index)
                    new_args = op.args[:1]
                    new_args.append(new_index)
                    for x in op.args[2:]:
                        new_args.append(x)
                    return tir.Call(op.dtype, op.op, new_args)
    def cancel_attribute(op):
        if isinstance(op, tir.AttrStmt) and op.attr_key == "swizzled_buffer_scope":
            return op.body # omit this annotation

    def ftransform(f):
        swizzled_buffers.clear()
        print("[Info] calling swizzle buffer transformation")
        # import pdb; pdb.set_trace()
        tvm.tir.stmt_functor.post_order_visit(f.body, find_buffers)
        if len(swizzled_buffers) == 0:
            return f
        return f.with_body(tvm.tir.stmt_functor.ir_transform(
            f.body, swizzle_buffer, cancel_attribute, ['tir.Store', 'tir.Load', 'tir.Call', 'tir.AttrStmt']
        ))
    return ftransform


