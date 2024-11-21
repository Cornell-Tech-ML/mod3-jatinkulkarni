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
    0      |    6.709112     |     27     |      4.984
    10     |    2.642705     |     50     |      0.512
    20     |    1.460630     |     49     |      0.299
    30     |    0.953672     |     49     |      0.223
    40     |    0.889769     |     49     |      0.185
    50     |    1.391341     |     49     |      0.163
    60     |    0.910175     |     50     |      0.147
    70     |    1.023222     |     50     |      0.136
    80     |    1.350462     |     49     |      0.127
    90     |    0.849162     |     49     |      0.120
   100     |    0.040089     |     50     |      0.115
   110     |    0.447061     |     50     |      0.110
   120     |    0.437563     |     49     |      0.107
   130     |    0.722896     |     49     |      0.103
   140     |    0.908522     |     50     |      0.101
   150     |    0.750874     |     49     |      0.098
   160     |    0.020228     |     49     |      0.096
   170     |    0.159237     |     49     |      0.094
   180     |    0.067238     |     50     |      0.093
   190     |    0.598599     |     50     |      0.091
   200     |    0.226713     |     50     |      0.090
   210     |    0.130101     |     49     |      0.088
   220     |    0.610936     |     50     |      0.087
   230     |    0.533039     |     49     |      0.086
   240     |    0.025644     |     50     |      0.085
   250     |    0.802695     |     50     |      0.085
   260     |    0.823380     |     50     |      0.084
   270     |    0.741782     |     50     |      0.083
   280     |    0.158822     |     50     |      0.083
   290     |    0.087564     |     50     |      0.082
   300     |    0.016196     |     49     |      0.081
   310     |    0.952635     |     50     |      0.081
   320     |    0.033224     |     50     |      0.080
   330     |    1.109884     |     49     |      0.080
   340     |    0.107726     |     50     |      0.079
   350     |    0.004772     |     50     |      0.079
   360     |    0.001006     |     50     |      0.079
   370     |    0.113641     |     50     |      0.078
   380     |    0.696721     |     50     |      0.078
   390     |    0.058926     |     49     |      0.078
   400     |    0.057373     |     50     |      0.077
   410     |    1.082760     |     49     |      0.077
   420     |    0.019655     |     50     |      0.077
   430     |    0.221261     |     50     |      0.076
   440     |    0.001280     |     50     |      0.076
   450     |    0.215568     |     50     |      0.076
   460     |    0.140168     |     50     |      0.076
   470     |    0.025617     |     50     |      0.075
   480     |    0.036962     |     49     |      0.075
   490     |    0.795004     |     50     |      0.075
```

#### Split
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    7.250871     |     29     |      4.917
    10     |    5.378345     |     33     |      0.507
    20     |    4.898483     |     39     |      0.296
    30     |    4.841932     |     42     |      0.222
    40     |    4.176460     |     46     |      0.184
    50     |    2.737897     |     46     |      0.160
    60     |    2.463372     |     47     |      0.145
    70     |    4.341203     |     46     |      0.134
    80     |    2.339579     |     48     |      0.125
    90     |    2.190074     |     49     |      0.119
   100     |    2.600476     |     46     |      0.113
   110     |    0.831321     |     47     |      0.109
   120     |    0.704093     |     48     |      0.105
   130     |    1.467579     |     49     |      0.102
   140     |    2.250899     |     50     |      0.100
   150     |    1.753938     |     49     |      0.097
   160     |    0.367748     |     50     |      0.095
   170     |    1.090259     |     50     |      0.094
   180     |    1.026484     |     50     |      0.092
   190     |    0.753287     |     49     |      0.091
   200     |    1.679017     |     50     |      0.089
   210     |    0.860224     |     49     |      0.088
   220     |    0.252121     |     50     |      0.087
   230     |    0.926881     |     50     |      0.086
   240     |    0.625522     |     49     |      0.085
   250     |    1.167677     |     49     |      0.085
   260     |    2.789128     |     48     |      0.084
   270     |    0.696156     |     49     |      0.083
   280     |    1.028318     |     49     |      0.083
   290     |    0.797489     |     49     |      0.082
   300     |    1.633768     |     49     |      0.082
   310     |    0.337953     |     49     |      0.081
   320     |    1.395790     |     47     |      0.081
   330     |    0.562888     |     49     |      0.080
   340     |    0.789504     |     47     |      0.080
   350     |    0.112770     |     49     |      0.080
   360     |    0.038702     |     48     |      0.079
   370     |    1.293749     |     49     |      0.079
   380     |    0.207796     |     48     |      0.079
   390     |    0.550874     |     49     |      0.078
   400     |    0.787646     |     49     |      0.078
   410     |    0.801121     |     49     |      0.078
   420     |    1.353745     |     50     |      0.077
   430     |    0.078667     |     50     |      0.077
   440     |    0.381829     |     49     |      0.077
   450     |    0.211953     |     50     |      0.077
   460     |    1.358688     |     49     |      0.076
   470     |    0.035782     |     49     |      0.076
   480     |    0.918248     |     47     |      0.076
   490     |    1.523212     |     50     |      0.076
```

#### Xor
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    6.177413     |     27     |      5.002
    10     |    4.397927     |     43     |      0.514
    20     |    3.751676     |     43     |      0.300
    30     |    4.466069     |     43     |      0.224
    40     |    2.323740     |     44     |      0.186
    50     |    2.592670     |     46     |      0.163
    60     |    3.629491     |     46     |      0.148
    70     |    3.918475     |     47     |      0.136
    80     |    2.600581     |     46     |      0.127
    90     |    1.420566     |     47     |      0.120
   100     |    2.289930     |     45     |      0.115
   110     |    2.340029     |     47     |      0.111
   120     |    1.205823     |     47     |      0.107
   130     |    1.381746     |     46     |      0.104
   140     |    1.013375     |     48     |      0.101
   150     |    2.946499     |     47     |      0.099
   160     |    1.962696     |     49     |      0.097
   170     |    0.764778     |     48     |      0.095
   180     |    1.367499     |     49     |      0.093
   190     |    0.946778     |     49     |      0.092
   200     |    1.559051     |     47     |      0.091
   210     |    1.015126     |     47     |      0.090
   220     |    1.377736     |     49     |      0.089
   230     |    0.823279     |     47     |      0.088
   240     |    0.348366     |     48     |      0.087
   250     |    0.954122     |     49     |      0.086
   260     |    0.867169     |     50     |      0.085
   270     |    2.567996     |     47     |      0.085
   280     |    0.281616     |     49     |      0.084
   290     |    1.256531     |     50     |      0.083
   300     |    1.211439     |     48     |      0.083
   310     |    0.277061     |     49     |      0.082
   320     |    1.601814     |     47     |      0.082
   330     |    1.150868     |     48     |      0.081
   340     |    1.456056     |     49     |      0.081
   350     |    1.816096     |     50     |      0.080
   360     |    1.167485     |     50     |      0.080
   370     |    1.077910     |     50     |      0.080
   380     |    0.345689     |     50     |      0.079
   390     |    0.964457     |     48     |      0.079
   400     |    1.461720     |     48     |      0.079
   410     |    0.671247     |     49     |      0.079
   420     |    1.198880     |     50     |      0.079
   430     |    0.904236     |     49     |      0.079
   440     |    0.450883     |     50     |      0.078
   450     |    1.669559     |     50     |      0.078
   460     |    0.125037     |     50     |      0.078
   470     |    0.197107     |     50     |      0.078
   480     |    1.171861     |     48     |      0.077
   490     |    1.556688     |     49     |      0.077
```

### CPU Training (Large 500 Hidden Layers)

#### Simple
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    55.262036    |     23     |      5.593
    10     |    41.446524    |     23     |      0.816
    20     |    82.893059    |     23     |      0.588
    30     |    82.893059    |     23     |      0.505
    40     |    41.446525    |     23     |      0.465
    50     |    96.708570    |     23     |      0.442
    60     |    96.708566    |     23     |      0.427
    70     |    69.077548    |     23     |      0.414
    80     |    69.077548    |     23     |      0.409
    90     |    82.893059    |     23     |      0.402
   100     |    69.077528    |     23     |      0.397
   110     |    55.262036    |     23     |      0.393
   120     |    82.893056    |     23     |      0.389
   130     |    69.077514    |     23     |      0.385
   140     |    55.262028    |     23     |      0.381
   150     |    82.893042    |     23     |      0.378
   160     |    41.446508    |     23     |      0.376
   170     |    69.077547    |     23     |      0.374
   180     |    82.893037    |     23     |      0.372
   190     |    69.077548    |     23     |      0.370
   200     |    69.077548    |     23     |      0.369
   210     |    41.446524    |     23     |      0.368
   220     |    96.708562    |     23     |      0.367
   230     |    69.077530    |     23     |      0.365
   240     |    69.077544    |     23     |      0.364
   250     |    41.446521    |     23     |      0.362
   260     |    55.262000    |     23     |      0.361
   270     |    69.077544    |     23     |      0.360
   280     |    55.262032    |     23     |      0.359
   290     |    82.893054    |     23     |      0.359
   300     |    69.077533    |     23     |      0.358
   310     |    82.893044    |     23     |      0.357
   320     |    96.708571    |     23     |      0.356
   330     |    82.893039    |     23     |      0.355
   340     |    96.708567    |     23     |      0.354
   350     |    69.077523    |     23     |      0.354
   360     |    96.708546    |     23     |      0.353
   370     |    82.893044    |     23     |      0.353
   380     |    82.893044    |     23     |      0.352
   390     |    69.077548    |     23     |      0.352
   400     |    96.708566    |     23     |      0.352
   410     |    55.262036    |     23     |      0.351
   420     |    96.708567    |     23     |      0.351
   430     |    69.077548    |     23     |      0.350
   440     |   110.524057    |     23     |      0.350
   450     |    69.077544    |     23     |      0.350
   460     |    96.708565    |     23     |      0.349
   470     |    69.077528    |     23     |      0.349
   480     |    55.262032    |     23     |      0.348
   490     |    96.708571    |     23     |      0.348
```

#### Split
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET split --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    55.261942    |     25     |      5.686
    10     |    96.708571    |     25     |      0.813
    20     |    55.262036    |     25     |      0.585
    30     |    55.262036    |     25     |      0.503
    40     |    41.446525    |     25     |      0.461
    50     |    96.708571    |     25     |      0.437
    60     |    55.262036    |     25     |      0.420
    70     |    41.446525    |     25     |      0.408
    80     |    55.261938    |     25     |      0.398
    90     |    41.446525    |     25     |      0.391
   100     |    82.893059    |     25     |      0.384
   110     |    27.631013    |     25     |      0.379
   120     |    69.077447    |     25     |      0.375
   130     |    69.077446    |     25     |      0.372
   140     |    55.262036    |     25     |      0.368
   150     |    55.262036    |     25     |      0.367
   160     |    96.708571    |     25     |      0.365
   170     |    82.893059    |     25     |      0.363
   180     |    41.446525    |     25     |      0.362
   190     |    82.893059    |     25     |      0.362
   200     |    55.262036    |     25     |      0.361
   210     |    69.077548    |     25     |      0.359
   220     |    82.893059    |     25     |      0.358
   230     |    96.708463    |     25     |      0.358
   240     |    69.077548    |     25     |      0.357
   250     |    69.077548    |     25     |      0.356
   260     |    55.262036    |     25     |      0.355
   270     |    96.708571    |     25     |      0.355
   280     |    82.893059    |     25     |      0.354
   290     |    82.893059    |     25     |      0.353
   300     |    69.077548    |     25     |      0.353
   310     |    69.077548    |     25     |      0.353
   320     |    96.708571    |     25     |      0.352
   330     |    69.077548    |     25     |      0.351
   340     |    55.262036    |     25     |      0.351
   350     |    69.077548    |     25     |      0.350
   360     |    69.077548    |     25     |      0.351
   370     |    96.708453    |     25     |      0.350
   380     |    69.077548    |     25     |      0.351
   390     |    82.893059    |     25     |      0.351
   400     |    55.262036    |     25     |      0.350
   410     |    82.892938    |     25     |      0.350
   420     |    82.893059    |     25     |      0.350
   430     |    96.708571    |     25     |      0.350
   440     |    96.708447    |     25     |      0.350
   450     |    96.708571    |     25     |      0.349
   460     |    96.708571    |     25     |      0.349
   470     |    82.892932    |     25     |      0.349
   480     |    55.262036    |     25     |      0.348
   490     |    96.708571    |     25     |      0.348
```

#### Xor
`python run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET xor --RATE 0.05`
```
  Epoch    |      Loss       |  Correct   | Time/Epoch (s)
------------------------------------------------------------
    0      |    82.716010    |     27     |      5.667
    10     |    4.562357     |     40     |      0.825
    20     |    3.736857     |     47     |      0.591
    30     |    0.677195     |     47     |      0.510
    40     |    1.851649     |     49     |      0.470
    50     |    0.944383     |     48     |      0.445
    60     |    2.549966     |     46     |      0.430
    70     |    1.133278     |     47     |      0.420
    80     |    2.259532     |     48     |      0.415
    90     |    1.142636     |     49     |      0.409
   100     |    1.547122     |     49     |      0.405
   110     |    0.380594     |     49     |      0.399
   120     |    1.353234     |     48     |      0.395
   130     |    1.905288     |     49     |      0.391
   140     |    2.182314     |     49     |      0.388
   150     |    1.817168     |     48     |      0.385
   160     |    0.019251     |     49     |      0.383
   170     |    0.752528     |     49     |      0.380
   180     |    1.497544     |     49     |      0.378
   190     |    0.561492     |     48     |      0.376
   200     |    0.142793     |     50     |      0.375
   210     |    0.675918     |     49     |      0.375
   220     |    0.713492     |     48     |      0.373
   230     |    0.025114     |     49     |      0.372
   240     |    0.879688     |     49     |      0.372
   250     |    0.056706     |     49     |      0.371
   260     |    0.101785     |     49     |      0.372
   270     |    0.005790     |     50     |      0.371
   280     |    0.984024     |     50     |      0.370
   290     |    0.175457     |     50     |      0.370
   300     |    0.190439     |     50     |      0.369
   310     |    0.011746     |     50     |      0.369
   320     |    0.043167     |     50     |      0.368
   330     |    0.122722     |     49     |      0.367
   340     |    0.120953     |     50     |      0.366
   350     |    0.732873     |     50     |      0.366
   360     |    0.055286     |     50     |      0.365
   370     |    1.643561     |     49     |      0.365
   380     |    0.866613     |     50     |      0.364
   390     |    0.137840     |     50     |      0.364
   400     |    0.846631     |     49     |      0.363
   410     |    0.571994     |     50     |      0.364
   420     |    0.011317     |     50     |      0.363
   430     |    1.068338     |     50     |      0.363
   440     |    0.507275     |     50     |      0.363
   450     |    0.483212     |     50     |      0.362
   460     |    0.088297     |     50     |      0.362
   470     |    0.107582     |     50     |      0.361
   480     |    0.504044     |     50     |      0.361
   490     |    0.348940     |     50     |      0.360
```