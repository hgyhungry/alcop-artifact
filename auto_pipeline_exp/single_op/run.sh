mkdir -p result
python3 dense_tensorcore_autotvm.py 2>&1 | tee result/run_dense_tensorcore_autotvm.log
# python3 conv2d_nhwc_tensorcore_autotvm.py 2>&1 | tee result/run_conv2d_nhwc_tensorcore_autotvm.log
# python3 batch_matmul_autotvm.py 2>&1 | tee result/run_batch_matmul_autotvm.log
# python3 conv3d_ndhwc_tensorcore_autotvm.py 2>&1 | tee result/run_conv3d_ndhwc_tensorcore_autotvm.log
