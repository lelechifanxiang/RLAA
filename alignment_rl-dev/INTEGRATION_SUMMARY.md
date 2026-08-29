# Double Gauss optics_core Integration

The alignment environment uses one optical model end to end: the six-element
Double Gauss prescription in:

```text
optics_core-dev/tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX
```

`env/lens_env.py` loads the prescription with `zemax_utils`, represents
alignment with paired Coordinate Breaks, applies radius/thickness domain
randomization through `ParameterVectorBatch`, and computes polychromatic MTF
with `optics_core`.

`env/batch_lens_env.py` expands the same prescription into a design batch. All
logical environments share one `MultiOpticalSystem`, CUDA context, material
cache, and prepared topology.

The training environment has no secondary lens object and does not translate
state between different optical prescriptions.
