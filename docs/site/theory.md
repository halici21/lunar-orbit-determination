# Theory and model contracts

The state is a six-component position/velocity vector in Moon-centered inertial
coordinates aligned with J2000 axes. Physical units are metres, seconds and m/s
unless a particular interface explicitly says otherwise.

The library combines lunar and third-body dynamics, variational propagation,
station geometry, observables and batch/sequential estimation. Each optional
model has its own scope; selecting a richer observable is not itself evidence
of higher navigation accuracy.

## Read the actual contracts

- [Frame transformations](../theory/frame_transformations.md)
- [Frame architecture and known inconsistencies](../theory/frame_architecture_contracts_and_inconsistencies.md)
- [One-way light-time Jacobians](../theory/one_way_light_time_jacobians.md)
- [Two-way counted Doppler](../theory/two_way_counted_doppler.md)
- [Converged two-way range](../theory/two_way_range.md)
- [Research methodology](../research/research-guide.md)

## Read results at the correct truth level

Same-model synthetic truth checks internal consistency. Richer internal truth
tests selected model mismatch. SPICE-backed checks constrain ephemeris/frame
geometry. Independent tool or operational references require explicit alignment
of frames, epochs, force models and observable conventions. They cannot be
substituted for each other.

The [support matrix](../validation/support-matrix.md) records the important
estimator/measurement restrictions. Current study conclusions and defect status
are linked from [Research studies](studies.md).
