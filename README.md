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
    0      |    7.019863     |     21     |      4.010
    10     |    6.653527     |     29     |      1.899
    20     |    6.579413     |     29     |      1.742
    30     |    6.755929     |     29     |      1.692
    40     |    6.194727     |     29     |      1.671
    50     |    7.368235     |     29     |      1.662
    60     |    6.740683     |     29     |      1.648
    70     |    7.060211     |     29     |      1.638
    80     |    6.125310     |     29     |      1.639
    90     |    6.425041     |     29     |      1.633
   100     |    7.064367     |     29     |      1.627
   110     |    7.064443     |     29     |      1.627
   120     |    6.738624     |     29     |      1.623
   130     |    7.064295     |     29     |      1.619
   140     |    7.399387     |     29     |      1.616
   150     |    6.738526     |     29     |      1.618
   160     |    7.401441     |     29     |      1.621
   170     |    6.421522     |     29     |      1.618
   180     |    7.750266     |     29     |      1.619
   190     |    6.421604     |     29     |      1.618
   200     |    6.117144     |     29     |      1.615
   210     |    6.422441     |     29     |      1.613
   220     |    6.738629     |     29     |      1.615
   230     |    6.738546     |     29     |      1.613
   240     |    7.063825     |     29     |      1.612
   250     |    6.424389     |     29     |      1.613
   260     |    6.424052     |     29     |      1.613
   270     |    6.738589     |     29     |      1.612
   280     |    6.422871     |     29     |      1.612
   290     |    7.064770     |     29     |      1.612
   300     |    6.113961     |     29     |      1.611
   310     |    6.420474     |     29     |      1.610
   320     |    7.065159     |     29     |      1.611
   330     |    7.065775     |     29     |      1.610
   340     |    6.738476     |     29     |      1.609
   350     |    7.746068     |     29     |      1.610
   360     |    8.097722     |     29     |      1.609
   370     |    6.738556     |     29     |      1.608
   380     |    7.743318     |     29     |      1.609
   390     |    6.422246     |     29     |      1.610
   400     |    6.738246     |     29     |      1.609
   410     |    7.399447     |     29     |      1.608
   420     |    6.738541     |     29     |      1.609
   430     |    6.421815     |     29     |      1.608
   440     |    6.424024     |     29     |      1.607
   450     |    6.738618     |     29     |      1.607
   460     |    6.738568     |     29     |      1.607
   470     |    8.100354     |     29     |      1.606
   480     |    6.738601     |     29     |      1.605
   490     |    6.422434     |     29     |      1.606
```

#### Split
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    6.939494     |     22     |      3.985
    10     |    6.870722     |     28     |      1.864
    20     |    6.545202     |     28     |      1.757
    30     |    6.977786     |     28     |      1.702
    40     |    6.775253     |     28     |      1.684
    50     |    6.997294     |     28     |      1.676
    60     |    6.766343     |     28     |      1.665
    70     |    7.513277     |     28     |      1.654
    80     |    6.764445     |     28     |      1.657
    90     |    7.005744     |     28     |      1.659
   100     |    7.258986     |     28     |      1.653
   110     |    6.763891     |     28     |      1.654
   120     |    7.255891     |     28     |      1.649
   130     |    7.005509     |     28     |      1.644
   140     |    6.532009     |     28     |      1.646
   150     |    6.530803     |     28     |      1.643
   160     |    6.763679     |     28     |      1.640
   170     |    7.258589     |     28     |      1.642
   180     |    6.763708     |     28     |      1.640
   190     |    7.521286     |     28     |      1.637
   200     |    7.258335     |     28     |      1.639
   210     |    6.763970     |     28     |      1.638
   220     |    7.005371     |     28     |      1.635
   230     |    6.310585     |     28     |      1.636
   240     |    7.256332     |     28     |      1.636
   250     |    6.310100     |     28     |      1.635
   260     |    7.259304     |     28     |      1.634
   270     |    6.531600     |     28     |      1.636
   280     |    7.005907     |     28     |      1.635
   290     |    6.531937     |     28     |      1.634
   300     |    6.763463     |     28     |      1.635
   310     |    6.763607     |     28     |      1.636
   320     |    7.006265     |     28     |      1.636
   330     |    6.308890     |     28     |      1.638
   340     |    7.005766     |     28     |      1.637
   350     |    6.531729     |     28     |      1.636
   360     |    6.763452     |     28     |      1.637
   370     |    6.310246     |     28     |      1.636
   380     |    7.005090     |     28     |      1.635
   390     |    7.256786     |     28     |      1.636
   400     |    7.005755     |     28     |      1.634
   410     |    6.763805     |     28     |      1.634
   420     |    7.005970     |     28     |      1.635
   430     |    6.763599     |     28     |      1.634
   440     |    7.257612     |     28     |      1.633
   450     |    6.763937     |     28     |      1.633
   460     |    7.005338     |     28     |      1.633
   470     |    6.762854     |     28     |      1.632
   480     |    7.006214     |     28     |      1.632
   490     |    7.521718     |     28     |      1.632
```

#### Xor
`python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    6.855161     |     24     |      4.311
    10     |    6.931590     |     24     |      1.895
    20     |    6.969560     |     26     |      1.792
    30     |    6.843286     |     26     |      1.724
    40     |    6.820671     |     26     |      1.709
    50     |    6.938374     |     26     |      1.682
    60     |    6.938840     |     26     |      1.666
    70     |    6.799274     |     26     |      1.657
    80     |    6.795948     |     26     |      1.653
    90     |    7.024935     |     26     |      1.645
   100     |    7.122723     |     26     |      1.638
   110     |    7.027134     |     26     |      1.640
   120     |    6.861675     |     26     |      1.636
   130     |    6.939686     |     26     |      1.632
   140     |    6.939485     |     26     |      1.633
   150     |    7.226131     |     26     |      1.630
   160     |    7.121407     |     26     |      1.627
   170     |    7.025775     |     26     |      1.626
   180     |    6.693356     |     26     |      1.626
   190     |    7.121361     |     26     |      1.624
   200     |    6.797760     |     26     |      1.622
   210     |    7.225378     |     26     |      1.624
   220     |    6.939541     |     26     |      1.622
   230     |    6.939560     |     26     |      1.625
   240     |    6.796739     |     26     |      1.626
   250     |    6.939821     |     26     |      1.625
   260     |    6.939914     |     26     |      1.623
   270     |    6.861414     |     26     |      1.625
   280     |    7.029447     |     26     |      1.624
   290     |    7.126582     |     26     |      1.623
   300     |    7.027911     |     26     |      1.623
   310     |    6.939751     |     26     |      1.623
   320     |    6.939336     |     26     |      1.622
   330     |    6.863537     |     26     |      1.621
   340     |    6.939613     |     26     |      1.622
   350     |    7.122031     |     26     |      1.620
   360     |    7.123999     |     26     |      1.619
   370     |    6.862646     |     26     |      1.620
   380     |    6.939795     |     26     |      1.619
   390     |    6.940058     |     26     |      1.618
   400     |    7.027976     |     26     |      1.619
   410     |    6.792175     |     26     |      1.619
   420     |    6.940038     |     26     |      1.618
   430     |    6.685261     |     26     |      1.617
   440     |    6.793401     |     26     |      1.618
   450     |    7.026941     |     26     |      1.619
   460     |    6.794223     |     26     |      1.618
   470     |    6.939828     |     26     |      1.619
   480     |    7.026553     |     26     |      1.619
   490     |    7.123152     |     26     |      1.618
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