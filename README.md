# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py




## Assignment Report

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jatinkulkarni/Cornell-Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py (163)
--------------------------------------------------------------------------|loop #ID
    def _map(                                                             |
        out: Storage,                                                     |
        out_shape: Shape,                                                 |
        out_strides: Strides,                                             |
        in_storage: Storage,                                              |
        in_shape: Shape,                                                  |
        in_strides: Strides,                                              |
    ) -> None:                                                            |
        # TODO: Implement for Task 3.1.                                   |
        # raise NotImplementedError("Need to implement for Task 3.1")     |
                                                                          |
        if np.array_equal(out_strides, in_strides) and np.array_equal(    |
            out_shape, in_shape                                           |
        ):                                                                |
            for i in prange(len(out)):------------------------------------| #0
                out[i] = fn(in_storage[i])                                |
            return                                                        |
                                                                          |
        for i in prange(len(out)):----------------------------------------| #1
            out_index: Index = np.empty(MAX_DIMS, np.int16)               |
            in_index: Index = np.empty(MAX_DIMS, np.int16)                |
            to_index(i, out_shape, out_index)                             |
            broadcast_index(out_index, out_shape, in_shape, in_index)     |
            o = index_to_position(out_index, out_strides)                 |
            j = index_to_position(in_index, in_strides)                   |
            out[o] = fn(in_storage[j])                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(182) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int16)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(183) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, np.int16)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(217)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jatinkulkarni/Cornell-Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py (217)
-------------------------------------------------------------------------|loop #ID
    def _zip(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        b_storage: Storage,                                              |
        b_shape: Shape,                                                  |
        b_strides: Strides,                                              |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
        if (                                                             |
            np.array_equal(out_strides, a_strides)                       |
            and np.array_equal(out_strides, b_strides)                   |
            and np.array_equal(out_shape, a_shape)                       |
            and np.array_equal(out_shape, b_shape)                       |
        ):                                                               |
            for i in prange(len(out)):-----------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                  |
            return                                                       |
                                                                         |
                                                                         |
                                                                         |
        for i in prange(len(out)):---------------------------------------| #3
            out_index = np.empty(len(out_shape), np.int32)               |
            a_index = np.empty(len(a_shape), np.int32)                   |
            b_index = np.empty(len(b_shape), np.int32)                   |
            to_index(i, out_shape, out_index)                            |
            o = index_to_position(out_index, out_strides)                |
            broadcast_index(out_index, out_shape, a_shape, a_index)      |
            j = index_to_position(a_index, a_strides)                    |
            broadcast_index(out_index, out_shape, b_shape, b_index)      |
            k = index_to_position(b_index, b_strides)                    |
            out[o] = fn(a_storage[j], b_storage[k])                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(243) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(244) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(245) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(279)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jatinkulkarni/Cornell-Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py (279)
-------------------------------------------------------------------------|loop #ID
    def _reduce(                                                         |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        a_storage: Storage,                                              |
        a_shape: Shape,                                                  |
        a_strides: Strides,                                              |
        reduce_dim: int,                                                 |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # raise NotImplementedError("Need to implement for Task 3.1")    |
        for i in prange(len(out)):---------------------------------------| #4
            # out_index: Index = np.zeros(MAX_DIMS, np.int32)            |
            out_index = np.empty(len(out_shape), np.int32)               |
            reduce_size = a_shape[reduce_dim]                            |
            to_index(i, out_shape, out_index)                            |
            o = index_to_position(out_index, out_strides)                |
            for s in range(reduce_size):                                 |
                out_index[reduce_dim] = s                                |
                j = index_to_position(out_index, a_strides)              |
                out[o] = fn(out[o], a_storage[j])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(292) is hoisted out of the parallel loop labelled #4 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/jatinkulkarni/Cornell-
Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py
(304)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jatinkulkarni/Cornell-Tech/Fall-2024/cs5781-MLE/workspace/mod3-jatinkulkarni/minitorch/fast_ops.py (304)
------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                    |
    out: Storage,                                                                               |
    out_shape: Shape,                                                                           |
    out_strides: Strides,                                                                       |
    a_storage: Storage,                                                                         |
    a_shape: Shape,                                                                             |
    a_strides: Strides,                                                                         |
    b_storage: Storage,                                                                         |
    b_shape: Shape,                                                                             |
    b_strides: Strides,                                                                         |
) -> None:                                                                                      |
    """NUMBA tensor matrix multiply function.                                                   |
                                                                                                |
    Should work for any tensor shapes that broadcast as long as                                 |
                                                                                                |
    ```                                                                                         |
    assert a_shape[-1] == b_shape[-2]                                                           |
    ```                                                                                         |
                                                                                                |
    Optimizations:                                                                              |
                                                                                                |
    * Outer loop in parallel                                                                    |
    * No index buffers or function calls                                                        |
    * Inner loop should have no global writes, 1 multiply.                                      |
                                                                                                |
                                                                                                |
    Args:                                                                                       |
    ----                                                                                        |
        out (Storage): storage for `out` tensor                                                 |
        out_shape (Shape): shape for `out` tensor                                               |
        out_strides (Strides): strides for `out` tensor                                         |
        a_storage (Storage): storage for `a` tensor                                             |
        a_shape (Shape): shape for `a` tensor                                                   |
        a_strides (Strides): strides for `a` tensor                                             |
        b_storage (Storage): storage for `b` tensor                                             |
        b_shape (Shape): shape for `b` tensor                                                   |
        b_strides (Strides): strides for `b` tensor                                             |
                                                                                                |
    Returns:                                                                                    |
    -------                                                                                     |
        None : Fills in `out`                                                                   |
                                                                                                |
    """                                                                                         |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                      |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                      |
                                                                                                |
    # TODO: Implement for Task 3.2.                                                             |
    # raise NotImplementedError("Need to implement for Task 3.2")                               |
    """                                                                                         |
    read in all of a in shared memory                                                           |
    read in all of b in shared memory                                                           |
    do matrix multiply in shared memory                                                         |
    ~~~~~~~~~~~                                                                                 |
    use len(out) threads                                                                        |
    nested loop                                                                                 |
    outer loop for al                                                                           |
                                                                                                |
                                                                                                |
    """                                                                                         |
    assert a_shape[-1] == b_shape[-2]                                                           |
                                                                                                |
    out_batch_stride = out_strides[0] if len(out_shape) > 2 else 0                              |
                                                                                                |
    common_dim = a_shape[-1]                                                                    |
                                                                                                |
    for batch in prange(out_shape[0]):----------------------------------------------------------| #5
        for row in range(out_shape[1]):                                                         |
            for col in range(out_shape[2]):                                                     |
                out_offset = (                                                                  |
                    batch * out_batch_stride + row * out_strides[-2] + col * out_strides[-1]    |
                )                                                                               |
                                                                                                |
                result = 0.0                                                                    |
                                                                                                |
                for k in range(common_dim):                                                     |
                    a_offset = (                                                                |
                        batch * a_batch_stride                                                  |
                        + row * a_strides[-2]                                                   |
                        + k * a_strides[-1]                                                     |
                    )                                                                           |
                    b_offset = (                                                                |
                        batch * b_batch_stride                                                  |
                        + k * b_strides[-2]                                                     |
                        + col * b_strides[-1]                                                   |
                    )                                                                           |
                    result += a_storage[a_offset] * b_storage[b_offset]                         |
                                                                                                |
                    out[out_offset] = result                                                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

```



### GPU Training

#### Simple
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s) 
------------------------------------------------------------
    0      |    9.667915     |     27     |      6.116     
    10     |    69.077523    |     27     |      2.068     
    20     |    82.893025    |     27     |      1.864     
    30     |    69.077511    |     27     |      1.814     
    40     |    82.893020    |     27     |      1.794     
    50     |    69.077504    |     27     |      1.770     
    60     |    69.077518    |     27     |      1.757     
    70     |    69.077501    |     27     |      1.742     
    80     |    27.631004    |     27     |      1.739     
    90     |    55.262013    |     27     |      1.730     
   100     |   110.524006    |     27     |      1.722     
   110     |    55.261999    |     27     |      1.725     
   120     |    55.262001    |     27     |      1.720     
   130     |    69.077515    |     27     |      1.714     
   140     |    55.262007    |     27     |      1.715     
   150     |    69.077490    |     27     |      1.710     
   160     |    41.446493    |     27     |      1.706     
   170     |    96.708514    |     27     |      1.707     
   180     |    82.893005    |     27     |      1.708     
   190     |    41.446498    |     27     |      1.705     
   200     |    55.262009    |     27     |      1.706     
   210     |    27.630991    |     27     |      1.702     
   220     |    82.892994    |     27     |      1.698     
   230     |    69.077502    |     27     |      1.698     
   240     |    55.262008    |     27     |      1.695     
   250     |    41.446494    |     27     |      1.693     
   260     |    69.077494    |     27     |      1.693     
   270     |    82.892988    |     27     |      1.691     
   280     |    41.446485    |     27     |      1.688     
   290     |    55.262002    |     27     |      1.688     
   300     |    82.893000    |     27     |      1.686     
   310     |    69.077482    |     27     |      1.684     
   320     |    69.077496    |     27     |      1.688     
   330     |    82.892991    |     27     |      1.686     
   340     |    69.077491    |     27     |      1.684     
   350     |    41.446484    |     27     |      1.684     
   360     |    13.815490    |     27     |      1.683     
   370     |    41.446490    |     27     |      1.682     
   380     |    55.261986    |     27     |      1.683     
   390     |    41.446475    |     27     |      1.681     
   400     |    69.077474    |     27     |      1.679     
   410     |    27.630983    |     27     |      1.680     
   420     |    96.708464    |     27     |      1.678     
   430     |    69.077492    |     27     |      1.677     
   440     |    55.261974    |     27     |      1.677     
   450     |    82.892980    |     27     |      1.678     
   460     |    41.446479    |     27     |      1.677     
   470     |    69.077485    |     27     |      1.677     
   480     |    55.261922    |     27     |      1.676     
   490     |    55.261961    |     27     |      1.675 
```

#### Split
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s) 
------------------------------------------------------------
    0      |    55.262036    |     24     |      5.308     
    10     |    96.708571    |     24     |      2.043     
    20     |    69.077548    |     24     |      1.835     
    30     |   124.339594    |     24     |      1.767     
    40     |    82.893059    |     24     |      1.750     
    50     |    69.077548    |     24     |      1.721     
    60     |    55.262036    |     24     |      1.704     
    70     |    82.893059    |     24     |      1.699     
    80     |    96.708571    |     24     |      1.704     
    90     |    69.077548    |     24     |      1.693     
   100     |    82.893059    |     24     |      1.687     
   110     |    96.708571    |     24     |      1.685     
   120     |    55.262036    |     24     |      1.679     
   130     |    55.262036    |     24     |      1.673     
   140     |    69.077548    |     24     |      1.675     
   150     |    82.893059    |     24     |      1.670     
   160     |    55.262036    |     24     |      1.667     
   170     |    82.893059    |     24     |      1.669     
   180     |    82.893059    |     24     |      1.666     
   190     |    96.708571    |     24     |      1.664     
   200     |    69.077548    |     24     |      1.665     
   210     |    69.077548    |     24     |      1.662     
   220     |    55.262036    |     24     |      1.663     
   230     |    55.262036    |     24     |      1.663     
   240     |    96.708571    |     24     |      1.661     
   250     |    69.077548    |     24     |      1.658     
   260     |    55.262036    |     24     |      1.659     
   270     |    41.446525    |     24     |      1.656     
   280     |    82.893059    |     24     |      1.654     
   290     |    41.446525    |     24     |      1.653     
   300     |    69.077548    |     24     |      1.652     
   310     |    27.631013    |     24     |      1.650     
   320     |    82.893059    |     24     |      1.649     
   330     |    82.893059    |     24     |      1.650     
   340     |    55.262036    |     24     |      1.648     
   350     |    55.262036    |     24     |      1.647     
   360     |    82.893059    |     24     |      1.650     
   370     |    55.262036    |     24     |      1.648     
   380     |    55.262036    |     24     |      1.647     
   390     |    82.893059    |     24     |      1.647     
   400     |    82.893059    |     24     |      1.646     
   410     |    55.262036    |     24     |      1.645     
   420     |    55.262036    |     24     |      1.645     
   430     |    96.708571    |     24     |      1.644     
   440     |    82.893059    |     24     |      1.643     
   450     |    82.893059    |     24     |      1.643     
   460     |    82.893059    |     24     |      1.643     
   470     |   124.339594    |     24     |      1.642     
   480     |    41.446525    |     24     |      1.641     
   490     |    41.446525    |     24     |      1.643   
```

#### Xor
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s) 
------------------------------------------------------------
    0      |    6.968083     |     28     |      4.492     
    10     |    55.261683    |     28     |      1.951     
    20     |    27.630837    |     28     |      1.790     
    30     |    82.892534    |     28     |      1.732     
    40     |    55.261687    |     28     |      1.726     
    50     |    82.892538    |     28     |      1.702     
    60     |    55.261690    |     28     |      1.687     
    70     |    69.077116    |     28     |      1.687     
    80     |    69.077118    |     28     |      1.679     
    90     |    69.077119    |     28     |      1.670     
   100     |    41.446268    |     28     |      1.672     
   110     |    55.261695    |     28     |      1.668     
   120     |    55.261696    |     28     |      1.671     
   130     |    27.630843    |     28     |      1.676     
   140     |    69.077124    |     28     |      1.671     
   150     |    55.261698    |     28     |      1.668     
   160     |    41.446271    |     28     |      1.669     
   170     |    82.892553    |     28     |      1.666     
   180     |    82.892554    |     28     |      1.664     
   190     |    55.261700    |     28     |      1.665     
   200     |    82.892555    |     28     |      1.662     
   210     |    41.446273    |     28     |      1.659     
   220     |    41.446273    |     28     |      1.662     
   230     |    69.077128    |     28     |      1.660     
   240     |    69.077128    |     28     |      1.658     
   250     |    82.892556    |     28     |      1.663     
   260     |    69.077128    |     28     |      1.661     
   270     |    27.630845    |     28     |      1.659     
   280     |    55.261700    |     28     |      1.660     
   290     |    69.077127    |     28     |      1.659     
   300     |    55.261699    |     28     |      1.657     
   310     |    82.892553    |     28     |      1.658     
   320     |    41.446271    |     28     |      1.656     
   330     |    69.077125    |     28     |      1.655     
   340     |    55.261697    |     28     |      1.656     
   350     |    41.446270    |     28     |      1.654     
   360     |    69.077122    |     28     |      1.653     
   370     |    69.077121    |     28     |      1.655     
   380     |    41.446268    |     28     |      1.653     
   390     |    69.077119    |     28     |      1.654     
   400     |    41.446267    |     28     |      1.655     
   410     |    69.077116    |     28     |      1.654     
   420     |    55.261690    |     28     |      1.653     
   430     |    41.446264    |     28     |      1.653     
   440     |    69.077112    |     28     |      1.652     
   450     |    55.261686    |     28     |      1.651     
   460     |    55.261684    |     28     |      1.652     
   470     |    55.261683    |     28     |      1.651     
   480     |    82.892526    |     28     |      1.650     
   490     |    69.077101    |     28     |      1.651 g
```


### CPU Training (Small)

#### Simple
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    4.321593     |     47     |      5.125
    10     |    0.973634     |     49     |      0.525
    20     |    0.293593     |     48     |      0.306
    30     |    0.353236     |     50     |      0.229
    40     |    0.277722     |     50     |      0.189
    50     |    1.140204     |     50     |      0.165
    60     |    0.337648     |     50     |      0.148
    70     |    0.807764     |     50     |      0.137
    80     |    0.213452     |     50     |      0.128
    90     |    0.080068     |     50     |      0.121
   100     |    0.166397     |     50     |      0.116
   110     |    0.304809     |     50     |      0.111
   120     |    0.551010     |     50     |      0.107
   130     |    0.652005     |     50     |      0.104
   140     |    0.291901     |     50     |      0.101
   150     |    0.422547     |     50     |      0.099
   160     |    0.139145     |     50     |      0.097
   170     |    0.255600     |     50     |      0.095
   180     |    0.203475     |     50     |      0.093
   190     |    0.566889     |     50     |      0.092
   200     |    0.081024     |     50     |      0.090
   210     |    0.058725     |     50     |      0.089
   220     |    0.152583     |     50     |      0.088
   230     |    0.978971     |     50     |      0.087
   240     |    0.243989     |     50     |      0.086
   250     |    0.018535     |     50     |      0.085
   260     |    0.142276     |     50     |      0.084
   270     |    0.028799     |     50     |      0.084
   280     |    0.078979     |     50     |      0.083
   290     |    0.505404     |     50     |      0.083
   300     |    0.001214     |     50     |      0.082
   310     |    0.052206     |     50     |      0.081
   320     |    0.825689     |     50     |      0.081
   330     |    0.115916     |     50     |      0.080
   340     |    0.010401     |     50     |      0.080
   350     |    0.008008     |     50     |      0.080
   360     |    0.031683     |     50     |      0.079
   370     |    0.634547     |     50     |      0.079
   380     |    0.027744     |     50     |      0.078
   390     |    0.114179     |     50     |      0.078
   400     |    0.679461     |     50     |      0.078
   410     |    0.007086     |     50     |      0.078
   420     |    0.088755     |     50     |      0.077
   430     |    0.602461     |     50     |      0.077
   440     |    0.009009     |     50     |      0.077
   450     |    0.407495     |     50     |      0.077
   460     |    0.416314     |     50     |      0.076
   470     |    0.737183     |     50     |      0.076
   480     |    0.036965     |     50     |      0.076
   490     |    0.529706     |     50     |      0.076
```

#### Split
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    7.969444     |     35     |      4.876
    10     |    3.664924     |     45     |      0.503
    20     |    3.029066     |     47     |      0.296
    30     |    2.367748     |     47     |      0.222
    40     |    2.040020     |     47     |      0.184
    50     |    6.464387     |     46     |      0.160
    60     |    1.249883     |     49     |      0.145
    70     |    1.519580     |     50     |      0.133
    80     |    1.750928     |     50     |      0.125
    90     |    0.834470     |     50     |      0.118
   100     |    0.782870     |     50     |      0.113
   110     |    1.276038     |     50     |      0.108
   120     |    1.013367     |     50     |      0.105
   130     |    0.582056     |     50     |      0.102
   140     |    0.443012     |     50     |      0.099
   150     |    0.392535     |     50     |      0.097
   160     |    0.520750     |     50     |      0.095
   170     |    0.636304     |     50     |      0.093
   180     |    0.431087     |     50     |      0.091
   190     |    0.209432     |     50     |      0.090
   200     |    0.416176     |     50     |      0.089
   210     |    0.307698     |     50     |      0.088
   220     |    0.221215     |     50     |      0.087
   230     |    0.582870     |     50     |      0.086
   240     |    0.104072     |     50     |      0.085
   250     |    0.188017     |     50     |      0.084
   260     |    0.070556     |     50     |      0.083
   270     |    0.340862     |     50     |      0.082
   280     |    0.228703     |     50     |      0.082
   290     |    0.067883     |     50     |      0.081
   300     |    0.161916     |     50     |      0.081
   310     |    0.233183     |     50     |      0.080
   320     |    0.279531     |     50     |      0.080
   330     |    0.094036     |     50     |      0.079
   340     |    0.310538     |     50     |      0.079
   350     |    0.185708     |     50     |      0.078
   360     |    0.043693     |     50     |      0.078
   370     |    0.051174     |     50     |      0.078
   380     |    0.232759     |     50     |      0.077
   390     |    0.116429     |     50     |      0.077
   400     |    0.080737     |     50     |      0.077
   410     |    0.106737     |     50     |      0.076
   420     |    0.076475     |     50     |      0.076
   430     |    0.048040     |     50     |      0.076
   440     |    0.141016     |     50     |      0.075
   450     |    0.077705     |     50     |      0.075
   460     |    0.059062     |     50     |      0.075
   470     |    0.116877     |     50     |      0.075
   480     |    0.042547     |     50     |      0.074
   490     |    0.019340     |     50     |      0.074
```

#### Xor
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    13.142271    |     19     |      5.019
    10     |    4.201931     |     48     |      0.515
    20     |    3.813869     |     49     |      0.301
    30     |    2.412477     |     49     |      0.225
    40     |    2.033525     |     48     |      0.186
    50     |    2.850396     |     46     |      0.162
    60     |    2.084010     |     49     |      0.146
    70     |    2.419934     |     49     |      0.135
    80     |    2.785664     |     49     |      0.126
    90     |    0.618354     |     49     |      0.119
   100     |    0.665290     |     49     |      0.114
   110     |    0.666716     |     49     |      0.109
   120     |    1.778911     |     49     |      0.106
   130     |    0.786174     |     49     |      0.103
   140     |    0.729832     |     49     |      0.100
   150     |    0.377062     |     49     |      0.097
   160     |    2.042351     |     50     |      0.095
   170     |    0.522404     |     49     |      0.094
   180     |    0.917360     |     49     |      0.092
   190     |    0.663861     |     49     |      0.091
   200     |    0.764100     |     49     |      0.089
   210     |    1.119398     |     49     |      0.088
   220     |    0.170402     |     50     |      0.087
   230     |    0.472717     |     49     |      0.086
   240     |    0.483503     |     50     |      0.085
   250     |    0.428218     |     50     |      0.084
   260     |    0.716786     |     49     |      0.084
   270     |    0.618214     |     49     |      0.083
   280     |    1.128587     |     50     |      0.082
   290     |    0.886317     |     50     |      0.082
   300     |    0.347370     |     50     |      0.081
   310     |    0.240118     |     50     |      0.080
   320     |    0.395539     |     50     |      0.080
   330     |    0.236191     |     49     |      0.080
   340     |    0.136026     |     50     |      0.079
   350     |    0.276664     |     50     |      0.079
   360     |    0.127519     |     50     |      0.078
   370     |    0.055230     |     50     |      0.078
   380     |    0.886187     |     50     |      0.078
   390     |    0.177921     |     50     |      0.077
   400     |    0.043963     |     50     |      0.077
   410     |    0.028364     |     50     |      0.077
   420     |    0.070087     |     50     |      0.076
   430     |    0.538233     |     50     |      0.076
   440     |    0.052936     |     50     |      0.076
   450     |    0.099965     |     50     |      0.076
   460     |    0.061978     |     50     |      0.075
   470     |    0.203915     |     50     |      0.075
   480     |    0.251491     |     50     |      0.075
   490     |    0.134882     |     50     |      0.075
```

### CPU Training (Large 500 Hidden Layers)

#### Simple
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    69.077037    |     25     |      5.866
    10     |    82.893042    |     25     |      0.817
    20     |    55.261151    |     25     |      0.587
    30     |    96.708566    |     25     |      0.499
    40     |    41.446513    |     25     |      0.454
    50     |    4.241975     |     48     |      0.426
    60     |    0.001678     |     42     |      0.407
    70     |    1.759028     |     46     |      0.394
    80     |    1.557292     |     46     |      0.384
    90     |    0.090743     |     47     |      0.376
   100     |    3.811720     |     49     |      0.369
   110     |    2.224538     |     45     |      0.364
   120     |    0.942384     |     47     |      0.360
   130     |    0.782875     |     49     |      0.356
   140     |    4.019699     |     47     |      0.353
   150     |    0.139344     |     48     |      0.350
   160     |    0.000594     |     49     |      0.348
   170     |    1.569970     |     46     |      0.346
   180     |    2.315313     |     48     |      0.344
   190     |    0.368562     |     49     |      0.342
   200     |    2.669944     |     44     |      0.341
   210     |    0.551691     |     49     |      0.340
   220     |    0.038035     |     46     |      0.340
   230     |    1.001901     |     49     |      0.339
   240     |    0.170736     |     48     |      0.338
   250     |    0.466221     |     49     |      0.337
   260     |    1.164698     |     49     |      0.336
   270     |    1.031748     |     49     |      0.336
   280     |    0.169334     |     49     |      0.336
   290     |    0.089730     |     49     |      0.335
   300     |    0.059089     |     49     |      0.335
   310     |    0.096175     |     49     |      0.334
   320     |    0.001572     |     49     |      0.334
   330     |    0.573440     |     49     |      0.333
   340     |    0.054499     |     48     |      0.333
   350     |    0.096041     |     49     |      0.332
   360     |    0.521870     |     49     |      0.332
   370     |    0.681863     |     49     |      0.332
   380     |    0.001441     |     50     |      0.331
   390     |    1.919371     |     49     |      0.331
   400     |    0.746558     |     49     |      0.331
   410     |    0.018960     |     49     |      0.331
   420     |    0.001173     |     49     |      0.331
   430     |    1.739875     |     48     |      0.331
   440     |    1.006647     |     49     |      0.331
   450     |    0.078578     |     49     |      0.331
   460     |    0.821369     |     49     |      0.331
   470     |    0.001747     |     49     |      0.331
   480     |    0.000610     |     49     |      0.331
   490     |    0.699229     |     49     |      0.331
```

#### Split
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    41.446296    |     33     |      5.145
    10     |    55.261292    |     33     |      0.749
    20     |    27.629668    |     33     |      0.541
    30     |    4.048429     |     41     |      0.468
    40     |    5.058938     |     17     |      0.429
    50     |    0.741054     |     48     |      0.406
    60     |    2.359847     |     47     |      0.391
    70     |    0.560993     |     48     |      0.380
    80     |    0.623482     |     50     |      0.371
    90     |    0.055092     |     49     |      0.367
   100     |    3.863313     |     43     |      0.364
   110     |    0.892341     |     48     |      0.360
   120     |    1.764296     |     47     |      0.356
   130     |    0.785506     |     48     |      0.353
   140     |    0.082961     |     48     |      0.350
   150     |    0.154547     |     49     |      0.348
   160     |    0.646360     |     49     |      0.346
   170     |    1.029811     |     48     |      0.344
   180     |    0.204578     |     50     |      0.343
   190     |    0.292586     |     50     |      0.341
   200     |    0.164903     |     49     |      0.340
   210     |    0.077655     |     50     |      0.339
   220     |    0.336666     |     50     |      0.338
   230     |    0.934426     |     50     |      0.338
   240     |    0.350862     |     49     |      0.337
   250     |    0.254250     |     50     |      0.336
   260     |    1.832926     |     48     |      0.336
   270     |    0.200732     |     50     |      0.335
   280     |    0.474163     |     50     |      0.335
   290     |    0.276197     |     50     |      0.334
   300     |    0.131189     |     50     |      0.334
   310     |    0.632426     |     49     |      0.334
   320     |    0.089418     |     50     |      0.334
   330     |    0.788246     |     50     |      0.334
   340     |    0.101722     |     50     |      0.333
   350     |    0.124107     |     50     |      0.333
   360     |    0.640346     |     50     |      0.332
   370     |    0.149920     |     50     |      0.332
   380     |    0.320602     |     50     |      0.332
   390     |    0.076336     |     50     |      0.331
   400     |    0.211169     |     50     |      0.331
   410     |    0.058917     |     50     |      0.331
   420     |    0.065329     |     50     |      0.331
   430     |    0.087441     |     50     |      0.331
   440     |    0.141414     |     50     |      0.331
   450     |    0.109429     |     50     |      0.330
   460     |    0.321920     |     50     |      0.330
   470     |    0.352498     |     50     |      0.330
   480     |    0.050292     |     50     |      0.330
   490     |    0.169243     |     50     |      0.330
```

#### Xor
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    27.631013    |     21     |      5.319
    10     |    69.029603    |     29     |      0.772
    20     |    5.446326     |     42     |      0.556
    30     |    1.946265     |     44     |      0.480
    40     |    0.650846     |     48     |      0.441
    50     |    5.051083     |     43     |      0.417
    60     |    1.167654     |     49     |      0.401
    70     |    0.087685     |     49     |      0.393
    80     |    1.101058     |     48     |      0.385
    90     |    1.789636     |     48     |      0.378
   100     |    1.545789     |     49     |      0.373
   110     |    0.342160     |     50     |      0.369
   120     |    1.394998     |     48     |      0.366
   130     |    1.867970     |     49     |      0.363
   140     |    0.232466     |     50     |      0.360
   150     |    1.337952     |     50     |      0.358
   160     |    2.088572     |     46     |      0.356
   170     |    0.581937     |     50     |      0.354
   180     |    1.768969     |     48     |      0.353
   190     |    0.017679     |     49     |      0.351
   200     |    0.265280     |     50     |      0.350
   210     |    1.019389     |     49     |      0.349
   220     |    0.104460     |     50     |      0.347
   230     |    0.060904     |     49     |      0.347
   240     |    1.133786     |     50     |      0.347
   250     |    0.234019     |     50     |      0.346
   260     |    0.088378     |     50     |      0.345
   270     |    0.094987     |     50     |      0.345
   280     |    1.015469     |     50     |      0.344
   290     |    0.227841     |     50     |      0.344
   300     |    1.108461     |     49     |      0.344
   310     |    0.052871     |     50     |      0.343
   320     |    0.151181     |     50     |      0.342
   330     |    0.199239     |     50     |      0.342
   340     |    0.452222     |     50     |      0.341
   350     |    0.065964     |     50     |      0.341
   360     |    0.104178     |     50     |      0.340
   370     |    0.406162     |     50     |      0.340
   380     |    0.028362     |     50     |      0.339
   390     |    0.741891     |     50     |      0.339
   400     |    0.331205     |     50     |      0.338
   410     |    0.107241     |     50     |      0.338
   420     |    0.497920     |     50     |      0.338
   430     |    0.433704     |     50     |      0.337
   440     |    0.459275     |     50     |      0.337
   450     |    0.010535     |     50     |      0.337
   460     |    0.390516     |     50     |      0.336
   470     |    0.045393     |     50     |      0.336
   480     |    0.347218     |     50     |      0.336
   490     |    0.392284     |     50     |      0.335
```