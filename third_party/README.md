# Third-party checkouts

Not committed - see .gitignore. Each one is pinned here instead, so the repo
stays small (a 3.7 GB directory swept into an early commit is what blocked the
push on 28 Sep and cost 27 commits) while the exact version stays recoverable.

## accelerated_features (XFeat)

    git clone https://github.com/verlab/accelerated_features.git
    cd accelerated_features && git checkout e92685f57f8318b18725c5c8c0bd28c7fe188d9a

- commit: `e92685f57f8318b18725c5c8c0bd28c7fe188d9a` (2025-01-15 13:22:03 -0300)
- licence: Apache-2.0, read from the LICENSE file in the checkout itself
- why: replaced the gradient-correlation map matcher on 4 Oct. Map matching
  went 45% -> 75% on the swap. AKAZE/SIFT scored 0% on this imagery.
- runtime: PyTorch only. Upstream ships NO ONNX or TensorRT export, so
  map_db.py's comment about "XFeat exported to ONNX" describes a route that
  does not exist. Runs on CPU; this Jetson has no CUDA OpenCV and no CUDA torch.
- weights used: `weights/xfeat.pt` (6.2 MB), loaded by vps_vio/map_match_xfeat.py
- override the location with `XFEAT_ROOT`

### Pinned versions that matter

- `torch==2.3.1` (CPU, from https://download.pytorch.org/whl/cpu). NOT newer.
  torch 2.14.1+cpu installs fine and then fails to import on this Jetson:
  `libtorch_cpu.so: undefined symbol: sbgemm_`, a BLAS symbol this Ubuntu
  22.04 / JetPack 6 OpenBLAS does not export. 2.3.1 imports and coexists with
  the system numpy 1.21.5 and cv2 4.8.0, which must not be disturbed - the
  whole flow and map pipeline is built on them.
- Do NOT `pip install torch` without the CPU index: on aarch64 it pulls
  nvidia-cusparselt (211 MB) and nvidia-nccl (206 MB), datacenter CUDA
  libraries that are useless here, since CUDA comes from JetPack.
- `tqdm` is an undeclared runtime dependency of XFeat's modules.
