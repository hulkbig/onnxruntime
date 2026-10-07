# WIP handoff: CPU LayerNormalization with an empty normalized extent

Prepared 2026-10-07. This branch preserves an existing candidate for independent
review. It is **unvalidated WIP**, with no completed C++ RED/GREEN result, no final
formatting pass, and no pull request for this candidate.

## Exact starting point and scope

- Repository: `hulkbig/onnxruntime`, forked from `microsoft/onnxruntime`.
- Base: `aac791ef2408a266d8ee59dd5fa0ce11d1a5a0bb`. This is the saved source base,
  not a claim that the branch tracks the latest upstream main.
- The four candidate files are preserved byte-for-byte. The original candidate
  diff SHA-256 is `8ce91a03624483875e2af5eba6d7a071460e691244a839b43ab8be654227f645`.
- Production files: `onnxruntime/core/providers/cpu/nn/layer_norm_helper.h` and
  `onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc`.
- Tests: `onnxruntime/test/contrib_ops/layer_norm_op_test.cc` and
  `onnxruntime/test/optimizer/graph_transform_test_layernorm.cc`.
- `HANDOFF.md` is handoff documentation only and should be excluded from an
  eventual upstream code submission.

The candidate permits an empty Y for ordinary CPU LayerNormalization only when
the normalized extent is zero and neither Mean nor InvStdDev is requested.
Axis, scale/bias rank, and broadcast checks remain. The helper opt-in defaults
to false. Dynamic fusion is retained; the fusion implementation is unchanged.
SimplifiedLayerNormalization, other execution providers, training statistics,
and empty optional-statistics semantics remain outside this change.

## Evidence and limitations

Previously recorded executions used the official ONNX Runtime 1.30.0 wheel
(build `f2c39fe`), CPUExecutionProvider, Python 3.12, ONNX 1.23.2, and NumPy 2.5.3.
For symbolic X `[2,H]` and scale/bias `[H]`, running with X `[2,0]` and parameters
`[0]` returned empty Y with optimization disabled. BASIC/EXTENDED fused the graph
and raised `Size of X.shape[axis:] must be at least 1, got 0`.

The archived evidence contains 27 executions across nine cases. It was not
rerun for this branch, and the archived logs/models are not included here.
One historical static-zero divergence concerns the older wheel; the source base
already includes the static guard from upstream PR
[33079](https://github.com/microsoft/onnxruntime/pull/33079).
Current-base reproduction is supported by source inspection only, not a
completed executable run. Empty ReduceMean semantics are undefined, so the
claim is an optimization-dependent behavior difference rather than a complete
specification for all empty reductions.

On 2026-10-06 at 17:19 UTC, continuation of the original build/test work was
blocked by platform risk control. No clearance of that restriction is established.
This publication does not resume that work or authorize a bypass or automatic
retry. The last saved RED-only build log ended at `1253/1801`; it provides no
successful build or test result. Historical formatting diagnostics also have no
subsequent passing result. The branch publication only preserves source and this
document; no build, test, or code modification was performed for it.

## Take over from the branch

```sh
git fetch https://github.com/hulkbig/onnxruntime.git handoff/layernorm-cpu-empty-y-wip-20261007
git switch -c review/layernorm-cpu-empty-y FETCH_HEAD
git diff aac791ef2408a266d8ee59dd5fa0ce11d1a5a0bb...HEAD -- onnxruntime
```

Keep unrelated local work safe and use a fresh review branch if that local name
already exists. First review the four-file change and the repository's AGENTS.md
and applicable guidance. Any subsequent execution requires legitimate approval
and available access, including resolution of applicable restrictions.

Once execution is permitted, establish RED on unchanged production code with
compatible regression tests and GREEN on this full candidate, using separate
source/build states. Do not describe a prepared RED-only tree as an observed
failing test. Follow the repository's build/test instructions and record actual
test counts, commands, exit codes, and logs; a zero-test filter is not a pass.

Pending acceptance:

- Reproduce the dynamic-empty failure on the exact source base.
- Compile and execute `LayerNormTest.EmptyNormalizedDimension*` and
  `GraphTransformationTests.LayerNormFusionDynamicEmptyNormalizedDimension`.
- Cover float/double/float16/bfloat16, runtime/initializer parameters, bias
  presence, invalid ranks/broadcasts, optional statistics rejection, and helper
  strict-default behavior.
- Verify actual LayerNormalization fusion and one-session H=4 -> 0 -> 4 behavior.
- Run LayerNorm and related fusion regressions, then final formatting/lint checks.
- Recheck upstream state before rebasing or proposing a PR. No GPU, performance,
  or training-configuration validation is claimed.
