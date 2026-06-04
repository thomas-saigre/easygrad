# Conclusion

Forward mode (dual numbers, JVP) pushes a tangent forward and gets one Jacobian column per pass.
Reverse mode (the tape, VJP) pulls a cotangent backward and gets one Jacobian row per pass.
They are transpose of one another.
To chose between the two is: if you have few inputs, then you should use the forwar moded.
Otherwise, you have few outputs, and you should use reverse mode
In ML, a scalar loss over millions/billions of parameters, hence, we use the reverse mode.

Many topics were not covered in this short intro (maybe a future version of this course?):

- higher-order derivatives: nesting `grad`/`jvp`
- checkpointing / rematerialization: trading compute for memory in large computational graph
- batching (`vmap`): another IR transformation.
- JIT / XLA / fusion / kernels
- control-flow primitives (`cond`, `scan`)
- GPU, dtypes, performance tuning

