# Lunar OD Ölçüm Modelleri: Fizik, Türevler, İzlenebilirlik ve Eksikler

## 1. Yönetici özeti

Bu doküman hedeflenen mimariyi değil, `eb92461` HEAD'inde gerçekten çalışan
ölçüm zincirini denetler. Kapsam; sentetik ölçüm üretimi, computed observable,
residual, Jacobian, estimator adapter'ları, observability, metadata, testler ve
birincil kaynak izlenebilirliğidir.

Repository'de üç production measurement record türü vardır:

1. `position`: one-way range, azimuth ve elevation.
2. `range_rate`: companion range/azimuth/elevation ile geometric instantaneous
   range-rate veya simplified two-way counted Doppler.
3. `two_way_range`: M3 converged two-way half-round-trip range; raw ve sabit
   transponder-delay-calibrated convention seçenekleri.

En güçlü doğrulama zincirleri M2 implicit one-way CN/CN+S initial-state
Jacobian'ları ile M3 two-way range event/Jacobian zinciridir. Bunlarda full-state
finite difference, double-STM mutation, BLS-LM/SRIF/observability shared-row ve
SPICE operator/frame ya da independent simultaneous-root testleri vardır.
[PROJECT-DERIVED]

Başlıca açık riskler:

- **CRITICAL, ilgili config için:** `run_lunar_ukf`, `position` sigma-point
  ölçümünü her durumda instantaneous geometric hesaplıyor. Config CN veya CN+S
  üretimini UKF ile birlikte reddetmiyor. Böylece apparent truth measurement ile
  geometric estimator measurement function sessizce eşleşebiliyor. Default
  geometric profil etkilenmiyor. [MISSING-PHYSICS]
- **HIGH:** one-way observable ve legacy counted-Doppler observable, light-time
  solve converge olmazsa son iterasyonu kullanabiliyor; sensitivity helper'ları
  daha katı. [NUMERICAL-APPROXIMATION]
- **HIGH, nonzero delay için:** legacy counted Doppler iki leg için tek
  spacecraft bounce state kullanıyor; `transponder_delay_s` yalnız event
  bookkeeping'e giriyor. M3 ayrı `t2u`/`t2d` state kullanıyor.
  [LEGACY-COMPATIBILITY]
- **HIGH, kaba grid için:** counted Doppler 6x6 `sxform` matrislerini lineer
  interpolate ediyor. Kaydedilmiş 60 s tanısında 11.5 m station-position,
  1.4 m two-way-range ve 5.5e-4 m/s equivalent range-rate etkisi görüldü. M3
  exact event-epoch `sxform` seçti. [NUMERICAL-APPROXIMATION]
- **MEDIUM:** M3, one-way `station.sigma_range_m` değerini yeniden kullanıyor;
  dedicated two-way sigma, bias/clock solve-for ve UKF desteği yok.
  [ENGINEERING-CHOICE]
- **MEDIUM/HIGH-fidelity sınırı:** media, relativistic, antenna phase-center,
  station displacement ve EOP solve-for düzeltmeleri yok. Modeller faydalı
  sentetik observable'lardır; operational DSN accuracy modeli değildir.
  [MISSING-PHYSICS]

Testlerin geçmesi implementation consistency gösterir; operational radiometric
accuracy göstermez. M3 range, project-defined geometric half-round-trip
observable'dır; Moyer'daki tam coded DSN range data type değildir. Counted
Doppler da constant-frequency kontrollü bir modeldir; tam ramped DSN phase/frequency
modeli değildir. [PROJECT-DERIVED]

## 2. Repository ve audit baseline

| Alan | Denetlenen değer |
|---|---|
| Audit tarihi | 2026-07-12 (ilk ölçüm audit'i 2026-07-11; final evidence review 2026-07-12) |
| Repository | `C:/Users/erayh/Documents/Python/Grad/python_port` |
| Branch | `feature/lunar-j2-force-models` |
| HEAD | `eb92461f781c3fccd39012e3a02cd6ace64d893b` |
| Authoritative M3 commit | `eb92461` - `Add converged two-way range observable (M3)` |
| Working tree | `examples/phase13g_gravity_orbit_effects.py` ve `tests/test_phase13g_gravity_orbit_effects.py` modified; ikisi de paralel gravity işi ve audit dışı |
| Python | 3.13.12 |
| spiceypy | 8.1.0 |
| Kernel durumu | `%USERPROFILE%/Documents/mice/kernels` mevcut |
| Görülen ilgili kernel'ler | `de421.bsp`, `naif0012.tls.txt`, `earth_assoc_itrf93.tf.txt`, Earth binary PCK'leri, `pck00010.tpc.txt`, lunar frame/orientation kernel'leri |
| Authoritative regression baseline | Isolated M3 tree: 515 passed, 20 skipped, 0 failed, 0 deselected, 1 warning, yaklaşık 22 s |

Bu documentation-only audit sırasında full suite yeniden koşulmadı; canlı tree'de
ilişkisiz gravity değişiklikleri var. Aşağıdaki hiçbir sayı yeni test koşusu gibi
sunulmaz. [ENGINEERING-CHOICE]

### 2.1 İncelenen dosyalar

Production/config: `lunar_od/measurements.py`, `radiometrics.py`,
`two_way_range.py`, `accelerated.py`, `geometry.py`, `config.py`,
`scenario_config.py`, `scenarios.py`, `estimators.py`, `observability.py`,
`filters.py`, `visibility.py`, `ephemeris.py`, `lunar_frames.py`, `orbit.py`,
`dynamics.py`, `noise_models.py`, `measurement_ingestion.py`, `reporting.py`,
`lunar_od/__init__.py`.

Test/example: `tests/test_measurements.py`, `test_stellar_aberration_vv.py`,
`test_two_way_range.py`, `test_two_way_range_integration.py`,
`test_estimators.py`, `test_observability.py`, `test_frame_transformations.py`,
`test_frame_spice_validation.py`, `test_scenario_config.py` ve görevde belirtilen
SPICE/scenario example'ları.

Doküman: `README.md`,
`LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md`,
`docs/one_way_light_time_jacobians.md`, `docs/two_way_range.md`,
`docs/two_way_counted_doppler.md`, `docs/frame_transformations.md`,
`docs/spice_cross_validation.md`, `docs/stellar_aberration_od_impact.md`.

## 3. Ölçüm modeli envanteri

| Ölçüm/model | Production | Boyut | Time tag | Spacecraft epoch | Station epoch | Frame zinciri | Jacobian | Estimator desteği | Ana kaynak |
|---|---|---:|---|---|---|---|---|---|---|
| Geometric one-way range | evet, default `position` row 1 | 1 | receive/sample | receive | receive | MCI/J2000 -> Earth-centered J2000 -> ITRF93 -> SEZ | exact local analytic + receive STM | BLS-LM, SRIF, UKF | Montenbruck-Gill Ch. 7; project convention |
| Geometric azimuth/elevation | evet, default rows 2-3 | 2 | receive/sample | receive | receive | aynı, SEZ `[S,E,Z]` | local analytic; legacy zenith fallback | BLS-LM, SRIF, UKF | Moyer Ch. 9; Montenbruck-Gill Ch. 7 |
| Converged one-way range CN | opt-in | 1 | receive `t_r` | transmit `t_t` | receive `t_r` | inertial CN LOS | implicit initial-state analytic | BLS-LM, SRIF, observability; **UKF mismatch** | NAIF `abcorr` reception Eq. (1) |
| Converged CN az/el | opt-in | 2 | receive | transmit | receive | CN J2000 LOS -> receive ITRF93 -> SEZ | implicit chain-rule initial-state | BLS-LM, SRIF, observability; **UKF mismatch** | NAIF; Moyer Ch. 8/9 |
| CN+S `local_mci` | opt-in | 2 | receive | transmit for CN | receive | Moon-relative observer velocity ile +S | analytic CN + tangent FD + analytic angle | BLS-LM, SRIF; **UKF mismatch** | NAIF + project approximation |
| CN+S `spice_ssb` | opt-in, SPICE | 2 | receive | transmit for CN | receive | SSB observer velocity, J2000 +S -> SEZ | aynı hybrid chain | BLS-LM, SRIF; **UKF mismatch** | NAIF `abcorr`/`spkezr` |
| Geometric range-rate | evet, default `range_rate` row 2 | 1 m/s | sample | sample | sample | full J2000->ITRF93 `sxform` | local analytic 4-row + receive STM | BLS-LM, SRIF, UKF | Montenbruck-Gill; project derivative |
| Two-way counted Doppler | opt-in `range_rate` row 2 | 1 m/s veya Hz | receive midpoint | endpoint bounce event | endpoint tx/rx | legacy interpolated station/`sxform` | endpoint implicit initial-state | BLS-LM, SRIF, UKF, observability | Moyer Ch. 13; Thornton-Border Ch. 3 |
| M3 raw two-way range | evet | 1 m | receive `t3` | `t2u`,`t2d` | exact `t1`,`t3` | exact event `sxform`, MCI translation | four-event / three-equation implicit initial-state | BLS-LM, SRIF, observability; UKF rejected | Moyer event yapısı + project observable |
| M3 calibrated two-way range | evet, default | 1 m | receive `t3` | `t2u`,`t2d` | exact `t1`,`t3` | aynı | fixed delay için aynı row | BLS-LM, SRIF, observability; UKF rejected | aynı |
| Noise/weighting | evet | record başına 3, 4 veya 1 sigma | n/a | n/a | station seçilir | diagonal covariance | n/a | desteklenen tüm yollar | Tapley et al.; project station değerleri |
| Bias states | position/range-rate evet; M3 yok | global/per-station additive | n/a | n/a | station-scoped | observable coordinates | identity columns | özellikle SRIF/UKF; M3 rejected | Tapley et al.; project design |
| Media/relativity/antenna | **MISSING** | n/a | n/a | n/a | n/a | metadata `none` | yok | yok | Moyer Ch. 10/11 |

### 3.1 Record layout

| Type | Shape | Kolonlar | Birimler | Station col | Residual helper |
|---|---:|---|---|---:|---|
| `position` | `(N,6)` veya `(N,7)` | `[t, range, az, el, station_id, time_index, (arc_id)]` | s,m,rad,rad,id | 4 | `compute_position_residuals_analytic` |
| `range_rate` | `(N,7)` veya `(N,8)` | `[t, range, rate_or_counted, az, el, station_id, time_index, (arc_id)]` | s,m,m/s,rad,rad,id | 5 | `compute_range_rate_residuals_analytic` + counted row replacement |
| `two_way_range` | `(N,4)` veya `(N,5)` | `[t3, two_way_range, station_id, time_index, (arc_id)]` | s,m,id | 2 | `compute_two_way_range_residuals` |

Counted-Doppler `range_rate` record'ındaki `range`, M3 two-way range değildir;
`companion_geometry` ile seçilen one-way companion range'dir.
[ENGINEERING-CHOICE]

## 4. Ortak frame, origin ve time convention'ları

### 4.1 State origin ve axes

- Spacecraft propagation state: Moon-centered inertial (`MCI`), J2000-aligned,
  SI units. Earth/Sun state'leri Moon-relative.
- Station: WGS84 geodetic -> Cartesian; rigid ITRF93 kabulü
  (`config.Station`, lines 13-35).
- Kod değişkenlerindeki `ECEF`, production'da ITRF93 anlamına gelir; bütün
  terrestrial corrections'ın uygulandığı anlamına gelmez.
- Local frame SEZ'dir. Azimuth north'tan clockwise,
  `atan2(E,-S)` ve `[0,2*pi)`; elevation `asin(Z/range)` veya eşdeğer
  `atan2(Z,sqrt(S^2+E^2))` (`geometry.ecef2razel_sez`, line 66).

### 4.2 Rotation ve translation ayrımı

\[
\mathbf r_B=C_{B\leftarrow A}(t)\mathbf r_A,
\qquad
\mathbf x_B=X_{B\leftarrow A}(t)\mathbf x_A.
\]

Full state için

\[
\mathbf v_B=\dot C\mathbf r_A+C\mathbf v_A.
\]

`spice.pxform(A,B,et)` 3x3 position rotation, `spice.sxform(A,B,et)` 6x6
state transform sağlar; NAIF bu ayrımı doğrudan tanımlar. [SOURCE-DIRECT]

Origin translation ayrıdır. Kod Earth MCI center state'ini çıkarır, sonra
Earth-centered J2000 vektörü ITRF93'e rotate eder. Rotation tek başına
Moon-centered origin'i Earth-centered origin yapmaz. [PROJECT-DERIVED]

### 4.3 Epoch sözleşmesi

- Geometric: spacecraft/station/frame sample-receive epoch.
- One-way CN: spacecraft transmit; station/frame receive.
- CN+S: CN'den sonra receive-epoch observer velocity.
- M3: station `t1`,`t3`; spacecraft `t2u`,`t2d`; differentiation'da `t3` fixed.
- Counted Doppler: receive count interval start/end'de ayrı round-trip solve.

SPICE çağrıları `et0_s + relative_seconds`; SPICE ET, J2000 TDB'den geçen
saniyedir. Measurement array'leri relative seconds taşır. [SOURCE-DIRECT]

### 4.4 Station model kapsamı

Station rigid WGS84-as-ITRF93'tür. Plate motion, solid Earth tide, ocean
loading, pole tide, antenna phase center, station-coordinate ve EOP solve-for
sensitivity yoktur. IERS Ch. 4/5/7 daha geniş terrestrial frame, Earth
orientation ve displacement bağlamını verir. [MISSING-PHYSICS]

## 5. Ortak state, STM ve residual convention'ları

Solve-for dynamic state:

\[
\mathbf x_0=[\mathbf r_0^T,\mathbf v_0^T]^T.
\]

Augmented propagation `[x(6); Phi(36)]`, STM column-major'dır. Local mapping:

\[
H_{x_0}=\widetilde H_{x(t)}\Phi(t,t_0).
\]

`accelerated.apply_stm_to_jacobian` (line 436) local geometric block'lara bu
mapping'i uygular. Initial-state derivative döndüren helper'a tekrar uygulanmaz.

Residual convention observed-minus-computed'dır. Angle residual:

\[
\operatorname{wrap}(\Delta)=\operatorname{atan2}(\sin\Delta,\cos\Delta).
\]

Position/range-rate küçük residual suppression eşikleri sırasıyla `1e-7 m`,
`1e-10 m/s`, `1e-14 rad` mertebesindedir; physics değil numerical/reporting
tercihidir. [ENGINEERING-CHOICE]

### 5.1 STM ownership tablosu

| Model | Helper output | STM helper içinde? | Caller STM? | Koruma |
|---|---|---:|---:|---|
| Geometric position | local `(3N,6)` | hayır | receive'de evet | `apply_stm_to_jacobian` testleri |
| First-order LT position | approximate local `(3N,6)` | hayır | receive'de evet | legacy regression |
| Implicit CN/CN+S | initial `(3N,6)` | transmit STM evet | hayır | no-double-STM mutation |
| Geometric range-rate | local `(4N,6)` | hayır | receive'de evet | estimator/observability tests |
| Counted Doppler row | initial `(1,6)` | bounce STM evet | replacement row'a hayır | analytic-FD/shared-row |
| M3 range | initial `(N,6)` | `t2u/t2d` STM evet | hayır | mutation/shared-row |

## 6. Geometric range ve angles

### 6.1 Ne ölçülüyor?

`position`, instantaneous station-spacecraft `[range_m, azimuth_rad,
elevation_rad]` üretir. One-way, geometric ve receive/sample-time tagged'dır.
Default profildir. [LEGACY-COMPATIBILITY]

### 6.2 Neden kullanılıyor?

Range LOS position'ı güçlü, kısa arc transverse directions'ı zayıf sınırlar.
Az/el transverse geometry ekler; zenith yakınında kötü koşullu, station/frame
fidelity'ye duyarlıdır. Multi-station ve değişen geometry rank'i iyileştirir.
[SOURCE-DERIVED]

### 6.3 Ne zaman devreye giriyor?

`measurement_type="position"`,
`measurement_model_profile="geometric_instantaneous"`,
`jacobian_model="analytic_exact_geometric"`. Legacy light-time boolean'ları
default off. BLS-LM, SRIF ve UKF geometric modelde tutarlıdır.

### 6.4 Event timeline

```text
spacecraft state t_r ---- instantaneous geometry ---- station state t_r
```

### 6.5 Frame/origin/epoch

`position_observables` (`accelerated.py:372`) Earth MCI state'ini çıkarır,
J2000->ITRF93 rotation uygular, station ITRF93 position çıkarır ve SEZ'e geçer.

### 6.6 Observable denklemi

\[
\rho=r_{sc}(t_r)-r_{st}(t_r),\quad R=\|\rho\|,\quad u=\rho/R,
\]

\[
A=\operatorname{atan2}(E,-S),\qquad
e=\operatorname{atan2}(Z,\sqrt{S^2+E^2}).
\]

### 6.7 Numerical solution

Closed form. Exact zenith observable `azimuth=0` döndürür; azimuth fiziksel
olarak tanımsız olduğundan bu display convention'dır. [ENGINEERING-CHOICE]

### 6.8 Residual

Observed-minus-computed; range m, wrapped angles rad.

### 6.9 Noise/covariance/weighting

Diagonal `diag([sigma_R^2,sigma_A^2,sigma_e^2])`. Built-in ITU: 94 m ve
18 arcsec; diğer predefined station'lar: 5 m ve 3.6 arcsec
(`config.py:38-47`). Correlation/elevation dependence yoktur.
[ENGINEERING-CHOICE]

### 6.10 Jacobian/türev zinciri

\[
\frac{\partial R}{\partial r}=u^T,\qquad
\partial u/\partial\rho=(I-uu^T)/R.
\]

`h=sqrt(S^2+E^2)` ile

\[
\partial A/\partial[S,E,Z]=[E/h^2,-S/h^2,0],
\]

\[
\partial e/\partial[S,E,Z]=[-ZS/(u^2h),-ZE/(u^2h),h/u^2].
\]

`compute_position_residuals_analytic` (`measurements.py:1679`) local `(3N,6)`
döndürür; local velocity columns zero'dur; caller receive STM'yi bir kez uygular.
Legacy geometric path çok küçük horizontal norm'da zero fallback kullanır;
implicit M2 path `1e-6` altında error verir. [LEGACY-COMPATIBILITY]

Range row m/state-unit, angle row rad/state-unit'tir.

### 6.11 Estimator entegrasyonu

`compute_position_residuals_analytic` ->
`position_initial_state_jacobian_from_augmented_history` -> whitening ->
BLS-LM/SRIF -> posterior. Observability aynı mapper'ı kullanır. UKF
`_position_measurement_from_state` (`filters.py:1480`) ile aynı geometric
nonlinear observable'ı hesaplar.

### 6.12 Validation/test

- `test_frame_transformations.py`: SEZ handedness/cardinals, transpose,
  direction, double-SEZ, zenith, FD chain.
- `test_frame_spice_validation.py`: real `sxform`, epoch mutation, norm,
  Jacobian FD; kaydedilmiş max relative mismatch `2.24e-10`.
- `test_measurements.py`: generation/residual closure ve angle wrap.

Real tracking-data validation yoktur. [MISSING-VALIDATION]

### 6.13 Literatür eşlemesi

Montenbruck-Gill Ch. 7 (publisher TOC'da pp. 193-232) tracking/observation
bağlamıdır. Moyer Ch. 9 DSN angles kaynağıdır. Project SEZ convention ve exact
formül project-specific'tir. Vallado exact section/page bu audit'te doğrulanmadı.

### 6.14 Eksikler

- **MEDIUM / MEDIUM:** refraction ve fiziksel boresight/calibration modeli yok.
- **MEDIUM / MEDIUM:** observable ve implicit Jacobian zenith policy farklı.
- **MEDIUM / HARD:** station displacement/EOP sensitivities yok.
- **LOW / EASY:** API/metadata'da `ECEF` yerine actual ITRF93 açıklığı.

## 7. One-way converged light-time CN

### 7.1 Ne ölçülüyor?

Range receive-station ile transmit-spacecraft arasındaki normdur; az/el bu CN
LOS'un receive-epoch SEZ yönüdür. Receive-time-tagged, one-way Newtonian apparent
geometry modelidir. [SOURCE-DIRECT]

### 7.2 Neden kullanılıyor?

Lunar distance'ta spacecraft'i receive yerine transmit epoch'ta değerlendirmek
coherent geometry farkını giderir. Systematic range/angle farkının state'e bias
olarak taşınmasını azaltır.

### 7.3 Ne zaman devreye giriyor?

`measurement_model_profile="one_way_light_time"` veya legacy
`apply_light_time=True`; M2 sensitivity için
`jacobian_model="implicit_light_time"`. `analytic_first_order_light_time`
eski local approximation'dır.

### 7.4 Event timeline

```text
spacecraft transmit t_t=t_r-tau  --->  station receive/time tag t_r
```

### 7.5 Frame/origin/epoch

Spacecraft MCI/J2000 state ve `Phi_r` transmit'te; station MCI ve
J2000->ITRF93 frame fixed receive'de. Current solve-for yalnız spacecraft
`x0` olduğu için frame matrix differentiate edilmez. Station/EOP/clock veya
state-dependent receive tag eklenirse bu varsayım değişir. [PROJECT-DERIVED]

### 7.6 Observable denklemi

\[
t_t=t_r-\tau,\qquad
\tau=\frac{\|r_{sc}(t_t,x_0)-r_{st}(t_r)\|}{c}.
\]

Bu NAIF reception-case CN equation'dır; relativistic bending/delay içermez.
[SOURCE-DIRECT]

### 7.7 Numerical solution

`solve_one_way_light_time` (`measurements.py:330`) fixed-point; default
`1e-12 s`, max 10. Spacecraft state grid içinde cubic Hermite, dışında linear
extrapolation. [NUMERICAL-APPROXIMATION]

Sensitivity helper `converged=False` ise error verir; observable path son
iterasyonu kullanabilir. Independent equation-residual threshold yoktur.
[NUMERICAL-APPROXIMATION]

### 7.8 Residual

Observed-minus-computed, wrapped angles. Truth/estimator mismatch yalnız açık
model-mismatch deneyinde kullanılmalıdır.

### 7.9 Noise/covariance/weighting

Geometric position ile aynı station sigmaları. Light-time açılması covariance
modelini değiştirmez; bu error-budget doğruluğu iddiası değil weighting
tercihidir. [ENGINEERING-CHOICE]

### 7.10 Jacobian/türev zinciri

`rho=r_sc(t_t)-r_st(t_r)`, `u=rho/R`:

\[
\frac{\partial\tau}{\partial x_0}=
\frac{u^T\Phi_r(t_t,t_0)}{c+u^Tv_{sc}(t_t)},
\]

\[
\frac{\partial R}{\partial x_0}=c\frac{\partial\tau}{\partial x_0},
\]

\[
J_{\rho,x_0}=\Phi_r(t_t,t_0)-v_{sc}(t_t)\frac{\partial\tau}{\partial x_0},
\]

\[
J_{u,x_0}=\frac1R(I-uu^T)J_{\rho,x_0}.
\]

`one_way_light_time_initial_state_sensitivity` (`measurements.py:505`)
`d_tau_dx0`, `phi_r_tx`, `j_los_dx0`, `j_unit_los_dx0` üretir. M2.1 range
kernel reuse edilir. Receive frame/SEZ chain sonunda ordered `(3,6)` block
oluşur; `position_initial_state_jacobian_from_augmented_history` (line 794)
ikinci STM uygulamaz. [PROJECT-DERIVED]

Implicit angle path horizontal unit norm `<1e-6` ise error verir; nominal
`1/h` amplification yaklaşık `1e6` ile sınırlanır. [ENGINEERING-CHOICE]

### 7.11 Estimator entegrasyonu

BLS-LM, SRIF, posterior ve observability aynı initial block'u kullanır. UKF
kullanmaz: `_position_measurement_from_state` daima instantaneous geometry'dir.
Bu canlı en büyük physics parity açığıdır. [MISSING-PHYSICS]

### 7.12 Validation/test

Static/linear light-time, local range FD, transmit-epoch STM mutation, unit-LOS
tangency/step sweep, M2.1 range-row equality, wrap-aware azimuth full-state FD,
double-STM mutation ve BLS/SRIF/observability shared-block testleri vardır.

Synthetic orbiter SPK body olmadığı için complete CN M2.3 fixture'da SPICE
tarafından bağımsız çözülmedi. Reproducible full SPICE CN validation eksiktir.
[MISSING-VALIDATION]

### 7.13 Literatür eşlemesi

NAIF `abcorr` Reception Eq. (1), target `t-lt`/observer `t` convention'ını
doğrudan destekler. Moyer Ch. 8 ve 12 event ve precision-light-time partials
bağlamıdır. Yukarıdaki six-state derivative repository solve-for contract'ına
ait project implicit-function derivation'dır; doğrudan “Moyer formula” değildir.

### 7.14 Eksikler

- **HIGH / EASY-MEDIUM:** observable nonconvergence fail/metadata policy.
- **HIGH / MEDIUM:** M3 benzeri pre-roll/domain policy; silent extrapolation yok.
- **CRITICAL / HARD:** UKF CN/CN+S implement veya config reject.
- **MEDIUM / HARD:** clock/station/EOP solve-for derivatives.
- **MEDIUM / HARD:** spacecraft SPK veya independent ephemeris ile end-to-end CN.

## 8. Stellar aberration CN+S

### 8.1 Ne ölçülüyor?

CN range korunur; CN LOS, Newtonian reception-case apparent direction'a rotate
edilir. Yalnız az/el değişir. [SOURCE-DIRECT]

### 8.2 Neden kullanılıyor?

Observer motion apparent direction'ı değiştirir. Kontrollü project arc'ında
`spice_ssb` correction'ı kapatmak median 11.22 arcsec bias ve 5.34 km/2.43 m/s
state shift üretti. Bu weak-arc project sonucu, universal metric değildir.
[PROJECT-DERIVED]

### 8.3 Ne zaman devreye giriyor?

`one_way_light_time_aberrated_local_mci` veya
`one_way_light_time_aberrated_spice_ssb`. Stellar yalnız light-time üstüne
izinlidir. `local_mci` approximate; `spice_ssb` NAIF SSB observer velocity
contract'ına uygundur.

### 8.4 Event timeline

```text
CN solve t_t -> t_r; receive epoch'ta v_obs(t_r) ile CN LOS rotate edilir
```

### 8.5 Frame/origin/epoch

Input LOS J2000. `spice_ssb`: Earth SSB velocity + station inertial rotational
velocity, receive epoch. `local_mci`: Earth Moon-relative velocity + station.
İkincisi Moon barycentric velocity'yi atlar. [NUMERICAL-APPROXIMATION]

### 8.6 Observable denklemi

NAIF:

\[
\sin\phi=(v/c)\sin w,
\]

ve reception için CN vector `rho x v` axis etrafında observer velocity'ye doğru
rotate edilir. `apply_stellar_aberration` (`measurements.py:876`) Rodrigues
rotation'dır, normu korur. Relativistic term yoktur. [SOURCE-DIRECT]

### 8.7 Numerical solution

Operator algebraic. Parallel/anti-parallel geometry zero rotation verir.
`apply_stellar_aberration` observable operator'u `sin(phi)` değerini
`[-1,1]` aralığına clip eder ve `|v|>=c` için ayrı bir exception üretmez.
Yalnız derivative helper `_stellar_aberration_local_jacobian`, finite/positive
`c` ve `|v|<c` koşullarını açıkça enforce eder. Bu iki failure policy aynı
değildir; production observable için superluminal input rejection iddiası
yapılmamalıdır.

### 8.8 Residual

CN ile aynı observed-minus-computed/wrapped angle. Yalnız +S değişiyorsa range
residual floating-point düzeyinde değişmez.

### 8.9 Noise/covariance/weighting

Position covariance aynıdır. Correction seçimi/atlamasının systematic etkisi
random sigma'ya gömülmez.

### 8.10 Jacobian/türev zinciri

Hybrid chain:

\[
J_{app,x_0}=J_S^{tan}J_{CN,x_0}.
\]

Deterministic tangent basis `B=[b1,b2]`:

\[
u_{i,\pm}=\frac{u\pm hb_i}{\|u\pm hb_i\|},\quad
d_i=\frac{f_S(u_{i,+})-f_S(u_{i,-})}{2h},\quad J_S^{tan}=DB^T.
\]

`_stellar_aberration_local_jacobian`: `h=1e-5`; `1e-3...1e-8` sweep'te
`3e-5...3e-6` stable region. `v_obs`, spacecraft `x0`'a göre fixed; future
station/EOP/clock solve-for için eksik derivative eklenmelidir.
[NUMERICAL-APPROXIMATION]

Metadata yöntemi doğru biçimde `local_central_finite_difference` ve
`hybrid_apparent_chain_rule` diye raporlar; tamamen analitik değildir.

### 8.11 Estimator entegrasyonu

BLS-LM/SRIF/posterior/observability hybrid `(3,6)` block kullanır. UKF position
instantaneous kalır.

### 8.12 Validation/test

Direct `spice.stelab` operator:

- max angular difference `3.494e-16 arcsec`;
- unit-LOS difference `1.694e-21`;
- vector difference `2.274e-13 m`;
- tangent action max `2.484e-16` absolute, `1.756e-16` relative.

Full six-state chain FD:

- apparent LOS absolute Frobenius `1.674e-13`;
- scaled Frobenius `5.413e-8`;
- max absolute component `1.065e-13`;
- azimuth relative `8.214e-9`, elevation `6.171e-8`;
- CN/CN+S range-row difference exactly zero;
- worst range-row FD difference `4.499e-6 s`, `v_z` column,
  `1e-3 m/s` perturbation.

Direct `stelab`, yalnız +S algebraic operator'ı doğrular. Realistic fixture
internal CN kullanır; independent end-to-end SPICE light-time değildir.
[MISSING-VALIDATION]

### 8.13 Literatür eşlemesi

NAIF `abcorr` Stellar Aberration/Reception Case, Newtonian SSB observer
velocity ve light-time'dan sonra correction sırasını doğrudan verir. `spkezr`
`CN+S` ve relativistic exclusion'ı açıklar. Tangent derivative project-specific.

### 8.14 Eksikler

- **CRITICAL / HARD:** UKF parity/rejection.
- **MEDIUM / HARD:** tamamen analitik aberration derivative yok; hybrid iyi testli.
- **MEDIUM / HARD:** relativistic aberration/light bending yok.
- **MEDIUM / HARD:** future observer-state derivative yok.
- **LOW / EASY:** high-fidelity kullanımda `spice_ssb` guidance belirginleşmeli.

## 9. Geometric range-rate

### 9.1 Ne ölçülüyor?

Instantaneous geometric LOS rate:

\[
\dot R=u^T\dot\rho.
\]

One-way kinematic m/s observable'dır; counted coherent radio Doppler değildir.
Record companion range/az/el de taşır.

### 9.2 Neden kullanılıyor?

LOS velocity bilgisi sağlar; range'i tamamlar. Transverse directions zayıf,
station rotation ve frame-rate terms önemlidir.

### 9.3 Ne zaman devreye giriyor?

`measurement_type="range_rate"`,
`range_rate_physics="geometric_instantaneous"` default. `apparent_one_way`
yalnız companion range/angles'i değiştirir, rate'i değil.

### 9.4 Event timeline

Spacecraft ve station aynı `t` sample epoch'ta.

### 9.5 Frame/origin/epoch

Earth center çıkarılır. Full 6x6 `X_ITRF93<-J2000(t)` ile
`v_F=Cdot r_I+C v_I`. Station fixed-frame velocity zero olsa da inertial
station velocity zero değildir.

### 9.6 Observable denklemi

\[
\rho_F=Cr_I-r_{st,F},\quad
\dot\rho_F=\dot C r_I+C v_I,\quad
\dot R=\rho_F^T\dot\rho_F/R.
\]

### 9.7 Numerical solution

Closed form; rate'in kendisine light-time uygulanmaz. [LEGACY-COMPATIBILITY]

### 9.8 Residual

Observed-minus-computed, `[range,rate,az,el]`; angles wrapped.

### 9.9 Noise/covariance/weighting

Diagonal `[sigma_range,sigma_range_rate,sigma_angle,sigma_angle]`. ITU rate
`1e-3 m/s`, diğer predefined `1e-4 m/s`.

### 9.10 Jacobian/türev zinciri

`v_perp=(I-uu^T)v_rel`:

\[
\frac{\partial\dot R}{\partial r_I}=\frac{v_\perp^T}{R}C+u^T\dot C,
\qquad
\frac{\partial\dot R}{\partial v_I}=u^TC.
\]

`compute_range_rate_residuals_analytic` (`measurements.py:1816`) local
`(4N,6)` döndürür; receive STM bir kez uygulanır. [PROJECT-DERIVED]

### 9.11 Estimator entegrasyonu

BLS-LM/SRIF/posterior/observability aynı local block'u kullanır. UKF aynı
geometric formula'yı sigma point başına hesaplar. Additive bias columns vardır.

### 9.12 Validation/test

Generation/residual closure; real-frame station rotation contribution
`+307.703899 m/s`, production-versus-inertial agreement `1.364e-12 m/s`;
frame Jacobian FD ve sign mutation testleri.

### 9.13 Literatür eşlemesi

Montenbruck-Gill Ch. 7 geometry bağlamı. Rotating-frame partial project-derived.
Bu model DSN counted Doppler diye kaynaklandırılmamalıdır.

### 9.14 Eksikler

- **MEDIUM / MEDIUM:** one-way light-time range derivative observable yok.
- **MEDIUM / EASY:** UI/report etiketi açıkça geometric olmalı.
- **MEDIUM / HARD:** media/relativity/proper-time/oscillator effects yok.

## 10. Two-way counted Doppler

### 10.1 Ne ölçülüyor?

Receive count interval uçları arasındaki round-trip light-time farkı. m/s
half-round-trip equivalent veya simplified Hz. Interval-averaged'dır,
instantaneous `rho_dot` değildir. [SOURCE-DIRECT]

### 10.2 Neden kullanılıyor?

Coherent Doppler hassas LOS velocity/phase-change bilgisi taşır. Count integration
noise davranışını değiştirir, endpoint/event ve frequency-standard convention
gerektirir.

### 10.3 Ne zaman devreye giriyor?

`measurement_type="range_rate"`,
`range_rate_physics="two_way_counted_doppler"`. Default `Tc=60 s`,
`f_u=7.2e9 Hz`, `k=880/749`, output m/s, local model `ode`, clock/delay zero
(`RangeRatePhysicsConfig`, `radiometrics.py:20`).

### 10.4 Event timeline

```text
t_start=t_mid-Tc/2: t1_start -> t2_start -> t3_start
t_end  =t_mid+Tc/2: t1_end   -> t2_end   -> t3_end
```

### 10.5 Frame/origin/epoch

Station `t3` ve `t1` event'lerinde; Earth states ve 6x6 transforms pass grid'den
lineer interpolate. Spacecraft state grid içinde cubic Hermite'dir; eski docs'un
“linear state interpolation” ifadesi güncel kodla uyuşmaz.

### 10.6 Observable denklemi

\[
\tau_{RT}(t_3)=t_3-t_1,
\]

\[
y_{m/s}=\frac{c}{2T_c}[\tau_{RT}(t_{end})-\tau_{RT}(t_{start})],
\]

\[
y_{Hz}=\frac{k f_u}{T_c}[\tau_{RT}(t_{end})-\tau_{RT}(t_{start})].
\]

Sign repository convention'dır ve receding testle korunur. Hz biçimi full DSN
ramp/phase convention değildir. [PROJECT-DERIVED]

### 10.7 Numerical solution

`radiometrics.solve_two_way_light_time` (`line 287`) downlink `t2`, sonra
uplink `t1`; default `1e-10 s`, max 20. `converged` döndürür fakat
`two_way_counted_doppler_observable` (`line 111`) false'u reject etmez. Grid
dışında extrapolation olabilir. [NUMERICAL-APPROXIMATION]

Nonzero delay'de aynı `r_sc(t2)` iki leg için; delay yalnız epoch bookkeeping.
[LEGACY-COMPATIBILITY]

### 10.8 Residual

Observed-minus-computed, configured output unit. Normal scenario output m/s.
Companion range one-way'dir.

### 10.9 Noise/covariance/weighting

Rate row `sigma_range_rate_mps` kullanır. Scenario Hz output expose etmediği için
normal config'de hidden Hz/mps weighting mismatch yoktur. Clock offset/drift ve
delay fixed model input olabilir, solve-for değildir.

### 10.10 Jacobian/türev zinciri

Endpoint başına:

\[
\frac{\partial t_2}{\partial x_0}=
-\frac{u_d^T\Phi_r(t_2,t_0)}{c+u_d^Tv_2}.
\]

`dt1/dx0` uplink spacecraft ve station transmit velocity terms içerir.
`round_trip_light_time_initial_state_jacobian=-dt1/dx0`; endpoint rows
`two_way_counted_doppler_initial_state_jacobian` (`radiometrics.py:164`) içinde
difference ve scale edilir. Row initial-state'tir; geometric 4-row block'un
yalnız ikinci row'unu değiştirir. [PROJECT-DERIVED]

### 10.11 Estimator entegrasyonu

BLS/SRIF `_range_rate_two_way_analytic_initial_jacobian`; observability
`_build_two_way_range_rate_initial_state_jacobian` (`observability.py:370`).
UKF sigma point başına local short history kurar ve nonlinear counted observable
hesaplar; legacy transform interpolation aynıdır.

### 10.12 Validation/test

Static zero, receding sign, acceleration/grid stability, Hz/turnaround/clock,
delay bookkeeping, residual closure; analytic-vs-full-state FD; BLS/SRIF ve
observability row tests; M3 zero-delay endpoint consistency. External
GMAT/Orekit/Tudat Doppler karşılaştırması yoktur. [MISSING-VALIDATION]

### 10.13 Literatür eşlemesi

Thornton-Border Ch. 3 range/Doppler ve count-time accumulated phase kavramını
verir. Moyer Ch. 13 §13.3.1.1 Eq. 13-31, unramped two/three-way Doppler'ı
count interval boyunca effective transmit-minus-receive frequency average'i
olarak tanımlar. Repository'nin endpoint RTLT-difference denklemi Eq. 13-31'in
doğrudan implementasyonu değildir; constant coherent frequency altında türetilmiş
project simplification'dır ve ramp/media/proper-time terimlerini içermez.

### 10.14 Eksikler

- **HIGH / EASY-MEDIUM:** nonconverged endpoint ve unsupported extrapolation fail.
- **HIGH / MEDIUM:** nonzero delay reject veya ayrı `t2u/t2d` upgrade.
- **HIGH / MEDIUM-HARD:** exact event station provider/error-budgeted scheme.
- **HIGH / HARD:** ramp, oscillator, proper-time, relativity, media.
- **HIGH / HARD:** external observable validation.
- **MEDIUM / EASY:** stale interpolation docs ve output-unit ownership.

## 11. Converged two-way range M3

### 11.1 Ne ölçülüyor?

\[
R_{raw}=\frac c2(t_3-t_1),\qquad
R_{cal}=\frac c2[(t_3-t_1)-\delta_{tr}].
\]

Metre, receive-time tagged. Default calibrated. Project geometric observable;
tam DSN coded range değildir. [PROJECT-DERIVED]

### 11.2 Neden kullanılıyor?

İki leg, iki station event ve delay boyunca iki spacecraft state'i açık temsil
eder. Future two-way observable'lar için ortak geometry temeli sağlar.

### 11.3 Ne zaman devreye giriyor?

`measurement_type="two_way_range"`, selected convention ve shared
`transponder_delay_s`. BLS-LM/SRIF allowed; UKF ve bias mode
`scenario_config._validate_cross_field_rules` (`lines 612-623`) ve UKF içinde
explicit rejected.

### 11.4 Event timeline

```text
station transmit t1
 -> spacecraft uplink receive t2u
 -> fixed coordinate-time delta_tr
 -> spacecraft downlink transmit t2d
 -> station receive/time tag t3
```

`t1<t2u<=t2d<t3` enforced.

### 11.5 Frame/origin/epoch

`make_exact_sxform_station_state_provider` (`two_way_range.py:134`) exact
event-epoch J2000->ITRF93 `sxform`, inverse solve ile station J2000 state
`t1/t3`. Earth-center MCI translation cubic Hermite. Spacecraft state/STM
`t2u` ve `t2d` için ayrı. [ENGINEERING-CHOICE]

### 11.6 Event/observable equations

\[
G_u=t_{2u}-t_1-\rho_u/c=0,
\]

\[
G_d=t_3-t_{2d}-\rho_d/c=0,
\]

\[
G_{tr}=t_{2d}-t_{2u}-\delta_{tr}=0.
\]

`1/2`, round-trip time'ı one-way-equivalent distance'a çevirir. Calibration
delay'i bir kez çıkarır. [PROJECT-DERIVED]

### 11.7 Numerical solution

`solve_two_way_range_events` (`two_way_range.py:265`) causal nested fixed-point.
Default update `1e-12 s`, equation residual `1e-11 s` (~3 mm), max 25. Update
ve equation residual, finite state, positive LT, ordering ve spacecraft history
support kontrol edilir. Failure controlled exception; last iterate observable
olmaz. Spacecraft extrapolation yok.

Generation unsupported early tags'i drop eder ve metadata kaydeder; residual
existing record'ı sessiz skip etmez. Earth-center pre-grid translation uplink
pre-roll boyunca extrapolate olabilir; documented estimate ~1 cm/2.7 s, her
senaryo için universal bound değildir. [NUMERICAL-APPROXIMATION]

### 11.8 Residual

`compute_two_way_range_residuals` (`line 713`): observed-minus-computed metre.
Generation/residual convention aynı olmalıdır.

### 11.9 Noise/covariance/weighting

M3 `station.sigma_range_m` scalar variance kullanır. Metadata açıkça
`two_way_range_noise_source=station_sigma_range_m` der; one-way/two-way physical
error budget eşitliği iddiası değildir. [ENGINEERING-CHOICE]

### 11.10 Jacobian/türev zinciri

`y=[t1,t2u,t2d]`:

\[
G_y\frac{\partial y}{\partial x_0}+G_x=0,
\qquad
\frac{\partial y}{\partial x_0}=\operatorname{solve}(G_y,-G_x).
\]

`two_way_range_event_sensitivity` (`line 432`) `np.linalg.solve` kullanır,
explicit inverse değil. `G_x` ayrı `u_u^T Phi_r(t2u)` ve
`u_d^T Phi_r(t2d)`; `G_y` spacecraft/station event velocity terms içerir.
Uniform seconds sayesinde nominal condition ~2.

\[
H_R=-\frac c2\frac{\partial t_1}{\partial x_0}.
\]

Fixed delay için raw/cal Jacobian aynı. `(N,6)` output initial-state'tir, ikinci
STM yok (`two_way_range_nominal_and_initial_jacobian`, line 746).

### 11.11 Estimator entegrasyonu

Shared helper -> scalar residual -> `_two_way_range_weight_diagonal` ->
`estimate_two_way_range_bls_lm`/`estimate_two_way_range_srif` -> posterior.
Observability aynı helper'ı `observability.py:342-348` doğrudan çağırır.

Controlled short arc: BLS final cost `1.389e-7`, RMS `4.167e-4 m`, position
error `30.22 m`; SRIF cost `1.231e-9`, RMS `3.923e-5 m`, position error
`22.77 m`. Düşük residual, güçlü truth recovery/observability kanıtı değildir.
[PROJECT-DERIVED]

### 11.12 Validation/test

`test_two_way_range.py`: 27 test; factor-2, delay convention, motion, distinct
event states, history/convergence, event FD, state-step sweep, static analytic,
condition, double STM, station velocity, independent `scipy.optimize.root`.

`test_two_way_range_integration.py`: 16 test; drop metadata, truth closure,
magnitude, BLS/SRIF shared rows, UKF reject, exact/grid transforms, Earth
interpolation, wrong epoch/reverse transform mutation, legacy zero-delay,
counted endpoint consistency, config.

Exact-vs-linear-transform range effect: 10 s `0.086 m`, 30 s `0.321 m`,
60 s `0.674 m`. Earth Hermite midpoint: `1.2e-7 m`, `5e-9 m/s`.

### 11.13 Literatür eşlemesi

Moyer Ch. 8/12 multi-event LT ve partials bağlamını destekler; Ch. 13 DSN range
terminology verir. `R=c(t3-t1)/2`, fixed-delay calibration, uniform 3x3 event
matrix ve initial-state row project adaptations'tır. [SOURCE-DERIVED]

### 11.14 Eksikler

- **HIGH / HARD:** full-SPICE spacecraft light-time comparison yok; layers ayrı.
- **HIGH / HARD:** leg media/clock/relativity/proper-time/hardware/antenna yok.
- **MEDIUM / MEDIUM:** cross-station three-way link yok.
- **MEDIUM / MEDIUM:** delay solve-for ve dedicated sigma yok.
- **MEDIUM / HARD:** UKF yok.
- **MEDIUM / EASY:** drop metadata broader exception'ı yalnız pre-roll diye
  sınıflandırabiliyor.

## 12. Noise, covariance, weighting ve biases

### 12.1 Standard generation

Position/range-rate generators independent Gaussian component'lar ve optional
caller RNG kullanır. Bias `Station.bias` veya explicit range-rate arguments'tan
gelebilir. Mandatory scenario `noise_seed` yok; reproducibility caller'a bağlı.
[ENGINEERING-CHOICE]

`noise_models.generate_measurement_noise` white/correlated Gaussian, AR(1) ve
Student-t destekler; standard generators ile tek unified production path değildir.

### 12.2 Whitening/covariance

`measurement_sigma_vector` (`measurements.py:1937`) stacked sigma döndürür.

\[
H_w=R^{-1/2}H,\qquad F=H_w^TH_w.
\]

Covariance diagonal; standard likelihood'ta pass/elevation/time correlation yok.

### 12.3 Robust editing

BLS/SRIF estimator-side robust reweighting; UKF NIS/component gates,
Student-t/Huber inflation ve adaptive measurement noise destekler. Standard
record persistent quality flags taşımaz; M3 drop metadata istisnadır. Synthetic
dropout/burst-gap first-class corruption model değildir. [MISSING-PHYSICS]

### 12.4 Bias ve clock

Position/range-rate global, station-angle, station-full additive bias states.
Bias derivatives identity. Counted Doppler fixed clock offset/drift alabilir ama
estimate etmez. M3 bias reject ve clock yok. Clock mapping observable-specific
türetilmelidir. [MISSING-PHYSICS]

### 12.5 Interface riski

`observability.build_measurement_bias_jacobian` (`lines 224-262`) tüm
non-position measurement'ları block 4/station col 5 varsayar. Direct M3 call
yanlış olur; config M3 bias'i block etse de **MEDIUM / EASY API risk** vardır.

## 13. Estimator ve observability entegrasyonu

### 13.1 BLS-LM

State+STM propagation, nominal/H, observed-minus-computed stack, whitening,
prior, `_lm_step` (`estimators.py:1375`), candidate repropagation ve cost-decrease
acceptance. Position step cap 20 km; lambda `/5` veya `*10`. Stop: convergence,
cost stability, singularity, max iter. [ENGINEERING-CHOICE]

### 13.2 SRIF

Whitened measurement rows + prior square-root rows, QR, triangular solve.
Candidate acceptance/termination iterative batch yapısındadır; square-root
information algebra kullanır.

### 13.3 Posterior information/covariance

Model-specific aynı initial Jacobian tekrar hesaplanır, measurement+prior
information oluşturulur. M2/M3 shared-helper tests row drift riskini azaltır.

### 13.4 Observability/Fisher

`analyze_initial_state_observability` (`observability.py:265`) STM propagate,
shared H, whitening, Fisher, singular values, rank, condition, weakest
eigenvector. Local linear information, nonlinear truth-recovery garantisi değildir.

### 13.5 UKF support matrisi

| Profil | Durum | Physics parity |
|---|---|---|
| geometric `position` | supported | consistent |
| CN/CN+S `position` | config accepts, sigma function geometric | **CRITICAL inconsistent** |
| geometric `range_rate` | supported | consistent |
| counted `range_rate` | local nonlinear history | conceptually same; legacy limits var |
| apparent companion | supported | implemented |
| M3 `two_way_range` | explicit error | güvenli unsupported policy |

### 13.6 Fit ve truth recovery ayrımı

Residual RMS/final cost seçilen model ve weights altında data fit'tir. Truth
state closeness, credible covariance veya observability kanıtı değildir. Synthetic
comparison state error, covariance/NEES, rank/condition, residual structure ve
measurement physics'i birlikte raporlamalıdır.

## 14. Tam derivative/Jacobian haritası

| Observable | Symbol | Independent variable | Yöntem | Shape/obs | Units | STM |
|---|---|---|---|---:|---|---|
| geometric range | `compute_position_residuals_analytic` | local receive state | analytic | `(1,6)` | pos dimensionless; local vel zero | caller receive |
| geometric az/el | aynı | local receive | analytic SEZ | `(2,6)` | rad/m | caller receive |
| CN range | `one_way_light_time_initial_state_sensitivity` | arc `x0` | implicit analytic | `(1,6)` | pos dimensionless; vel s | helper transmit |
| CN az/el | `one_way_light_time_position_initial_state_jacobian` | arc `x0` | implicit LOS + analytic SEZ | `(2,6)` | rad/m, rad/(m/s) | helper transmit |
| CN+S az/el | aynı + `_stellar_aberration_local_jacobian` | arc `x0` | hybrid | `(2,6)` | aynı | helper transmit |
| geometric rate | `compute_range_rate_residuals_analytic` | local receive | analytic | `(1,6)` | pos 1/s; vel dimensionless | caller receive |
| counted Doppler | `two_way_counted_doppler_initial_state_jacobian` | arc `x0` | endpoint implicit | `(1,6)` | m/s veya Hz per state | helper event |
| M3 range | `two_way_range_event_sensitivity` | arc `x0` | four-event / three-equation implicit solve | `(1,6)` | pos dimensionless; vel s | helper t2u/t2d |
| additive bias | estimator bias helpers | bias state | identity | profile | observable/bias unit | yok |

`u^T J_u=0` tangent property testlidir. Frame matrices current spacecraft-only
solve-for'a göre fixed. Initial-state helper'lara ikinci STM uygulanmaması
load-bearing regression gate'tir.

## 15. Kaynak ve literatür izlenebilirlik matrisi

| ID | Model/denklem | Repository symbol | Kaynak | Exact section/page | Sınıf | Durum |
|---|---|---|---|---|---|---|
| S01 | reception one-way LT | `solve_one_way_light_time` | NAIF Aberration Required Reading | Reception case Eq. (1), web lines 194-208 | [SOURCE-DIRECT] | verified |
| S02 | `NONE/LT/CN/CN+S` | profiles/SPICE validation | NAIF `spkezr_c` | Detailed Input, lines 75-162 | [SOURCE-DIRECT] | verified |
| S03 | stellar order/operator | `apply_stellar_aberration` | NAIF `abcorr` | Stellar/Reception, lines 80-85, 231-242 | [SOURCE-DIRECT] | verified |
| S04 | `pxform`/`sxform` | frame paths | NAIF Frames | Frame Transformation Functions, lines 190-201 | [SOURCE-DIRECT] | verified |
| S05 | ITRF93 | station/frame | NAIF Frames | Frames Supported, lines 250-255 | [SOURCE-DIRECT] | verified |
| S06 | DSN event structure | M3/counted | Moyer JPL 00-7 | Ch. 8 | [SOURCE-DIRECT] | chapter verified; project adaptation ayrı |
| S07 | LT partials | M2/M3 | Moyer | Ch. 12 | [SOURCE-DERIVED] | project `x0` derivative direct copy değil |
| S08 | counted interval Doppler concept | counted | Moyer | Ch. 13 §13.3.1.1 Eq. 13-31, printed 13-26 | [SOURCE-DERIVED] | official PDF verified; project endpoint-RTLT equation is not a direct copy |
| S09 | range/Doppler concepts | rationale | Thornton-Border | Ch. 3, starts p. 9 | [SOURCE-DIRECT] | subsection pages not reverified |
| S10 | media/antenna | gap list | Moyer | Ch. 10; Ch. 11 | [SOURCE-DIRECT] | chapter titles verified |
| S11 | terrestrial/celestial | station frame | IERS 2010 | Ch. 5 | [SOURCE-DIRECT] | official link verified |
| S12 | station displacement | gaps | IERS 2010 | Ch. 7 | [SOURCE-DIRECT] | title verified |
| S13 | tracking/linearization/OD | common chain | Montenbruck-Gill | Ch. 7 pp.193-232; Ch.8 pp.233-256; Ch.9 pp.257-291 | [SOURCE-DIRECT] | publisher TOC verified |
| S14 | batch/SRIF/info | estimators | Tapley-Schutz-Born | section/page not verified | [SOURCE-DERIVED] | bibliographic mapping |
| S15 | topocentric support | geometric | Vallado 5e | section/page not verified | [SOURCE-DERIVED] | publisher citation |
| S16 | M2 six-state rows | M2 helpers | source events + project chain | code symbols | [PROJECT-DERIVED] | FD validated |
| S17 | M3 uniform event system | M3 helper | Moyer event concept + project | code symbols | [PROJECT-DERIVED] | root+FD validated |

## 16. Validation ve test coverage matrisi

| Model | Closed form | Full-state FD | Independent root | SPICE observable | SPICE frame/state | Mutation | Estimator integration | Real data | En zayıf katman |
|---|---|---|---|---|---|---|---|---|---|
| geometric range/angles | evet | evet | n/a | yok | evet | frame/SEZ | BLS/SRIF/UKF | yok | real/external observable |
| CN range | static/linear | evet | yalnız self fixed-point | full SPK yok | station/frame evet | tx STM/double STM | BLS/SRIF/obs | yok | independent end-to-end CN |
| CN angles | analytic chain | wrap-aware FD | yok | full synthetic SC yok | evet | zenith/frame | shared rows | yok | external angles |
| CN+S | operator cases | evet | n/a | direct `stelab` operator | observer/frame fixture | parallel branches | BLS/SRIF impact | yok | full external CN+S |
| geometric rate | analytic | frame/local FD | n/a | yok | evet | sign/epoch | all | yok | external observable |
| counted Doppler | static/receding | Jacobian FD | yok | yok | interpolation diagnostics | endpoint/row | all | yok | external + nonzero delay |
| M3 range | factor-2 | step-sweep FD | `scipy.root` | no spacecraft SPK | exact station/frame | factor2/epoch/STM | BLS/SRIF/obs | yok | full external two-way |
| noise/bias | empirical synthetic | n/a | n/a | n/a | n/a | config/rank | recovery tests | yok | calibrated real error model |

Authoritative M3 isolated baseline 515 passed, 20 skipped, 0 failed'dır.

## 17. Eksik fizik ve numerical limitation'lar

| Eksik | Modeller | Mevcut | Büyüklük | Risk | Öncelik | Zorluk | Kaynak | Önerilen test |
|---|---|---|---|---|---|---|---|---|
| UKF position parity | CN/CN+S | geometric sigma prediction | project +S örneği ~11 arcsec | CRITICAL | P0 | HARD | NAIF+UKF design | identical sigma-point observable |
| nonconvergence policy | CN/counted | last iterate mümkün | not quantified | HIGH | P0 | EASY-MEDIUM | numerical contract | forced failure mutation |
| silent extrapolation | CN/counted | linear outside support | not quantified | HIGH | P0 | MEDIUM | M3 policy | pre-roll sweep |
| single-bounce delay | counted | same t2 both legs | default delay zero; else not quantified | HIGH | P0/P1 | MEDIUM | Moyer | nonzero-delay root |
| transform interpolation | counted | linear 6x6 | fixture max 1.4 m, 5.5e-4 m/s | HIGH | P1 | MEDIUM-HARD | NAIF Frames | exact-event cadence A/B |
| troposphere | Earth links | none | not quantified | HIGH low elevation | P1 | HARD | Moyer Ch.10/IERS | mapping benchmark |
| ionosphere/plasma | range/Doppler | none | not quantified | MEDIUM-HIGH | P2 | HARD | Moyer Ch.10 | frequency sweep |
| Shapiro/relativistic LT | CN/two-way | none | not quantified | MEDIUM-HIGH | P2 | HARD | Moyer Ch.11 | reference cases |
| relativistic Doppler/proper time | counted | none | not quantified | HIGH for DSN claim | P2 | HARD | Moyer Ch.13 | external comparison |
| station displacement/EOP | all | rigid/kernel only | not quantified | MEDIUM | P2 | HARD | IERS | station-state campaign |
| antenna/hardware | radiometric | fixed delay dışında none | not quantified | MEDIUM | P2 | MEDIUM-HARD | Moyer Ch.10 | injected offset |
| dedicated M3 sigma | M3 | one-way sigma reuse | not quantified | MEDIUM | P1 | EASY | error model | weight regression |
| clock solve-for | counted/M3 | fixed/none | not quantified | HIGH operational | P1 | HARD | Moyer/Tapley | recovery test |
| ramp tables | counted | constant frequency | not quantified | HIGH operational | P2 | HARD | Moyer Ch.13 | phase integral oracle |
| cross-station | M3/counted | same station | not quantified | MEDIUM | P2 | MEDIUM | Moyer | three-way cases |
| elevation/correlated noise | all | fixed independent | not quantified | MEDIUM | P1 | MEDIUM | tracking literature | seeded MC whiteness |
| quality flags/dropout | all | limited M3 metadata | not quantified | MEDIUM | P1 | MEDIUM | data processing | burst-gap campaign |
| real data ingest | all | CSV position/RR, weak provenance | n/a | HIGH for real-data claims | P2 | HARD | mission ICD | manifest round trip |

## 18. Code, metadata ve documentation tutarsızlıkları

| Bulgu | Sınıf | Severity | Eylem |
|---|---|---|---|
| UKF non-geometric position profili geometric function kullanıyor | physics/API bug | CRITICAL | implement veya reject |
| one-way/counting observable `converged` ignore | numerical risk | HIGH | strict common contract |
| counted docs linear state derken code cubic Hermite | docs risk | MEDIUM | docs update |
| counted metadata single-bounce delay'i fazla güçlü gösterebilir | metadata risk | MEDIUM | legacy event contract açık |
| summary CSV M2 fields var, M3 convention/delay/event metadata yok | traceability | MEDIUM | M3 columns/manifest |
| `PassGeometry` M3 metadata `ScenarioResult` summary'ye tam taşınmıyor | traceability | MEDIUM | preserve manifest |
| bias Jacobian tüm non-position'ı 4-row varsayıyor | API risk | MEDIUM | M3 explicit reject/dispatch |
| M3 private interpolation imports | technical debt | LOW-MEDIUM | behavior freeze sonrası shared module |
| `range` adı farklı physics taşıyor | API/docs | MEDIUM | explicit schema/metadata |
| standard generators ve `noise_models.py` ayrı | architecture debt | MEDIUM | unified corruption contract |
| M3 drop reason broad error'ı pre-roll diye kaydedebilir | metadata | LOW-MEDIUM | exception code/details |
| CN'de M3 dual convergence criterion yok | numerical | MEDIUM-HIGH | update+equation residual |
| M3 exact station, counted interpolated station | intentional split | bug değil | upgrade'e kadar açık tut |
| `local_mci` SSB hareketini atlar | intentional approximation | MEDIUM if mislabeled | profile guidance |

Physics change ile formula refactor aynı patch'e konmamalı. M3 counted
Doppler'ı bilinçli olarak değiştirmedi; ortaklaştırma önce regression fixture
ister.

## 19. Önceliklendirilmiş roadmap

### P0 - Safety ve physics parity

1. **UKF position parity veya hard rejection - HARD.** CN/CN+S sigma point'leri
   bounded local history ile aynı observable'dan geçir veya config reject et.
   Default geometric bitwise davranış korunmalı.
2. **Unified light-time failure/domain policy - MEDIUM.** CN ve counted için
   convergence + equation residual, explicit pre-roll, no silent extrapolation,
   persistent failure metadata.
3. **Counted nonzero-delay kararı - MEDIUM.** Reject veya M3 separate `t2u/t2d`;
   zero-delay baseline sessiz değişmemeli.

### P1 - M4 noise, bias, clock ve editing

- `sigma_two_way_range_m`, elevation/station covariance, explicit seed,
  truth-noise/estimator-weight split, quality flags, dropout/outlier ve common
  robust report - **MEDIUM**.
- M3 range bias ve clock/delay solve-for yalnız observability/prior/rank policy
  tasarlandıktan sonra - **HARD**.
- Event/convention metadata result CSV/JSON manifest'e - **EASY-MEDIUM**.

### P2 - Counted Doppler ve corrections

- Counted için exact event station provider, cadence budget, sonra mümkünse M3
  common event geometry - **HARD**.
- Frequency ramps, coherent phase convention, clock/proper-time, relativistic
  frequency - **HARD**.
- Önce troposphere, sonra ionosphere/plasma, hardware/antenna; correction delay
  geometric event time'dan ayrı raporlanmalı - **HARD**.

### P3 - External validation ve real data

- SPK target ile reproducible SPICE CN/CN+S.
- Frozen frame/time/correction contract ile GMAT + Orekit veya Tudat range,
  rate, two-way comparison.
- Time scale, station, frame, units, quality, provenance validasyonlu real-data
  ingestion. Bu katmandan önce operational accuracy claim yapılmamalı.

## 20. Kısa “ne, nasıl, niçin, ne zaman?” özeti

### 20.1 Ne kullanılıyor?

| Model | Observable | Birim | Fidelity |
|---|---|---|---|
| geometric position | one-way range + az/el | m,rad | instantaneous baseline |
| CN | transmit-state range + apparent angles | m,rad | converged Newtonian LT |
| CN+S | CN range + Newtonian apparent direction | m,rad | `spice_ssb` SPICE-like; `local_mci` approximate |
| geometric rate | instantaneous LOS rate | m/s | kinematic |
| counted Doppler | count-interval RTLT difference | m/s equiv. veya Hz | simplified constant-frequency |
| M3 range | raw/cal half round trip | m | converged explicit events |

### 20.2 Nasıl hesaplanıyor?

| Model | Denklem | Numerical method | Frame/event |
|---|---|---|---|
| geometric | `R=norm(rho)`, SEZ angles | closed | receive J2000->ITRF93->SEZ |
| CN | `tau=norm(r_sc(t_r-tau)-r_st(t_r))/c` | fixed point | SC transmit, station/frame receive |
| CN+S | CN LOS -> observer velocity | Rodrigues + tangent FD H | J2000 CN -> +S -> SEZ |
| geometric rate | `u dot rho_dot` | closed | full receive `sxform` |
| counted | endpoint `tau_RT` difference | endpoint legacy nested solves | interpolated event station/frame |
| M3 | three equations / four events | nested solve + implicit H | exact t1/t3, separate t2u/t2d |

### 20.3 Niçin kullanılıyor?

| Model | OD bilgisi | Güçlü yön | Zayıf yön |
|---|---|---|---|
| range | LOS distance | radial position | short-arc transverse |
| az/el | direction | transverse geometry | zenith/calibration/media |
| rate/Doppler | LOS velocity/phase | LOS velocity | transverse/clock/model bias |
| two-way range | two-leg distance | LOS/event geometry | transverse/clock-delay correlation |

### 20.4 Ne zaman kullanılıyor?

| Config | Measurement | Estimator | Jacobian | Limit |
|---|---|---|---|---|
| default geometric `position` | range/az/el | all | local analytic + STM | no finite LT |
| one-way LT + implicit | CN | BLS/SRIF | transmit-event initial | UKF mismatch |
| aberrated + implicit | CN+S | BLS/SRIF | hybrid | no relativistic +S; UKF mismatch |
| default `range_rate` | geometric 4-component | all | local analytic + STM | not radiometric Doppler |
| counted `range_rate` | companion + counted row | all | endpoint initial | legacy delay/interpolation |
| `two_way_range` | scalar M3 | BLS/SRIF | four-event / three-equation implicit | no UKF/bias/dedicated sigma |

## 21. Bibliyografya ve doğrulanmış bağlantılar

1. T. D. Moyer, *Formulation for Observed and Computed Values of Deep Space
   Network Data Types for Navigation*, JPL Publication 00-7, DESCANSO Monograph
   2, 2000. [Section index](https://descanso.jpl.nasa.gov/monograph/series2_section.html),
   [full PDF](https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_all.pdf).
   İlgili: Ch. 5, 8-13.
2. C. L. Thornton, J. S. Border, *Radiometric Tracking Techniques for
   Deep-Space Navigation*, DESCANSO Monograph 1.
   [Section index](https://descanso.jpl.nasa.gov/monograph/series1_section.html),
   [full PDF](https://descanso.jpl.nasa.gov/monograph/series1/Descanso1_all.pdf).
3. NAIF/JPL, [Aberration Corrections Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html).
4. NAIF/JPL, [`spkezr_c`](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html).
5. NAIF/JPL, [Frames Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html).
6. NAIF/JPL, [SPK Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html).
7. IERS, *IERS Conventions (2010), TN36*:
   [official material](https://iers-conventions.obspm.fr/conventions_material.php),
   [full PDF](https://iers-conventions.obspm.fr/content/tn36.pdf),
   [Chapter 5](https://iers-conventions.obspm.fr/content/chapter5/icc5.pdf).
8. B. D. Tapley, B. E. Schutz, G. H. Born, *Statistical Orbit Determination*,
   Elsevier, 2004, [publisher](https://www.sciencedirect.com/book/monograph/9780126836301/statistical-orbit-determination).
   Exact chapter/page bu audit'te doğrulanmadı.
9. O. Montenbruck, E. Gill, *Satellite Orbits: Models, Methods and
   Applications*, Springer, 2000,
   [publisher](https://link.springer.com/book/10.1007/978-3-642-58351-3).
   Publisher TOC: tracking Ch. 7, linearization Ch. 8, OD Ch. 9.
10. D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 5th ed.,
    Microcosm Press, 2022, [publisher](https://microcosmpress.com/vallado/).
    Exact section/page bu audit'te doğrulanmadı.

---

**Audit sonucu:** M2 ve M3 event/initial-state derivative zincirleri güçlü
internal verification'a sahiptir. Bir sonraki bilimsel öncelik yeni nominal
observable eklemekten önce estimator-physics parity ve strict light-time
failure/domain contract'tır; sonra traceable noise/clock/media ve independent
external validation gelmelidir. Bu audit production source code değiştirmedi.

---

## 22. Final evidence-review correction register

Bu bölüm 2026-07-12 tarihinde, HEAD `eb92461f781c3fccd39012e3a02cd6ace64d893b`
üzerinde yapılan ikinci, read-only source/test/literature incelemesinin
sonucudur. Aşağıdaki kayıtlar önceki bölümlerin nasıl okunacağını belirler.

| ID | İlk dokümandaki durum | Source ile doğrulanan durum | Sonuç |
|---|---|---|---|
| D-01 | Geometric range ve az/el tek model bölümüydü | Aynı `position` record'unda üretilseler de farklı observable, birim, singularity ve bilgi içeriğine sahipler | §23.1 ve §23.2'de ayrıldı |
| D-02 | CN range ve CN az/el tek bölümdeydi | Aynı converged solution'ı paylaşırlar; range ve angle chain rule/Jacobian farklıdır | §23.3 ve §23.4'te ayrıldı |
| D-03 | M3 raw/calibrated tek bölümdeydi | İki convention config ile ayrı seçilebilir; bağımsız solver değildir, aynı event solution ve fixed-delay Jacobian'ı paylaşır | §23.8 ve §23.9'da ayrıldı |
| D-04 | Stellar operator için superluminal rejection genellenmişti | Observable operator clip uygular; yalnız derivative helper `speed<c` enforce eder | §8.7 ve §23.5 düzeltildi |
| D-05 | M3 sensitivity “3-event” diye özetlenmişti | Dört event (`t1,t2u,t2d,t3`), üç constraint equation vardır | Terminoloji düzeltildi |
| D-06 | Bir range derivative satırında `\qquad` kaçışı bozuktu | Production formül `dR/dr=u^T` | LaTeX düzeltildi |
| D-07 | UKF mismatch kısa bir tabloydu | CN ve CN+S config tarafından kabul edilip geometric position operator'una düşüyor; counted Doppler uygulanıyor; M3 açık reddediliyor | §27'de issue record'a dönüştürüldü |
| D-08 | Range-rate apparent companion + implicit config anlamı açık değildi | Config kombinasyonu kabul eder; production companion rows first-order receive-frame chain kullanır, exact implicit initial-state companion block yoktur | GAP-CFG-01 eklendi |
| D-09 | One-way observable convergence davranışı helper ile karışabiliyordu | Sensitivity helper fail eder; observable path `converged` flag'ini enforce etmez | §28'de ayrıldı |
| D-10 | Counted state interpolation “linear” diye eski docs'ta geçiyordu | Spacecraft state grid içinde cubic Hermite, dışarıda linear extrapolation; station/transform arrays lineer interpolate edilir | §23.7 ve §28'de düzeltildi |
| D-11 | M3 dropped reason tüm failures için pre-roll gibi okunabiliyordu | Generator yalnız `TwoWayEventHistoryError` yakalar; exception sınıfı history/nonfinite/order alanlarını kapsar, metadata tek generic reason yazar | GAP-META-02 eklendi |
| D-12 | Regression sayısı güncel çalışma ağacı sonucu gibi okunabilirdi | `515 passed / 20 skipped` yalnız isolated M3 checkpoint kaydıdır; bu audit dirty çalışma ağacında suite çalıştırmadı | Sayı provenance ile sınırlandı |
| D-13 | Generic LT metadata solver policy'si gibi okunabiliyordu | Position CN nominal solver `1e-12 s / 10 iter`; metadata ise `RangeRatePhysicsConfig` default'u `1e-10 s / 20 iter` yazar | GAP-META-03 eklendi |
| D-14 | Apparent RR companion Jacobian etiketi gerçek derivative sanılabiliyordu | Scenario default `analytic_exact_geometric` kalabilir; explicit `implicit_light_time` da kabul edilir; production companion H first-order receive-frame satırdır | GAP-CFG-01 genişletildi |

Major ambiguity bulunmadı. Aşağıdaki model dosyaları ve matrisler source
symbol'larını authoritative kabul eder; line number yalnız kolaylık içindir.

## 23. Dokuz production measurement model dosyası

### 23.1 Geometric instantaneous range

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Receive/sample epoch'ta station-spacecraft Euclidean distance, metre |
| Why it exists | Default, ucuz ve backward-compatible one-way geometric baseline; güçlü LOS-position bilgisi |
| Selection | `measurement_type="position"`, profile `geometric_instantaneous`; ayrıca `range_rate` record'undaki companion range |
| Configuration path | `ScenarioConfig` -> `prepare_measurement_arcs` -> `generate_position_measurements` |
| Runtime call path | `generate_position_measurements` -> `accelerated.position_observables`; prediction: `compute_position_residuals[_analytic]`; UKF: `_position_measurement_from_state` |
| Event epochs | Measurement tag, spacecraft, station ve frame: `t_r` |
| Frames/origins | Spacecraft Moon-centered J2000; Earth-center translation çıkarılır; `J2000 -> ITRF93`; station rigid WGS84/ITRF93 |
| Observable | `R(t_r)=norm(r_sc(t_r)-r_st(t_r))` [m] |
| Residual | `r_R=R_obs-R_comp` [m] |
| Noise | `Station.sigma_range_m`; independent zero-mean Gaussian when enabled; constant per station |
| Bias | Position bias modes include additive range component; no physical hardware-delay mapping |
| Jacobian | Local analytic `dR/dr=u^T`, then receive-epoch STM once; units `[1,1,1,s,s,s]` by state column |
| Estimator consumers | BLS-LM, SRIF, posterior information, UKF |
| Observability | `build_initial_state_jacobian` -> whitening -> Fisher/SVD |
| UKF | Supported and physics-consistent for geometric profile |
| Metadata | `measurement_model_profile`, receive epochs, `range_jacobian_model=analytic_exact_geometric`, sigma fields |
| Tests | `test_measurements.py`, `test_frame_transformations.py`, `test_frame_spice_validation.py`, estimator/filter/observability suites |
| Approximations | Euclidean instantaneous range; rigid station; no finite light time |
| Missing physics | Media, relativity/Shapiro, station displacement, phase center, clock/hardware delay |

### 23.2 Geometric azimuth/elevation

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Instantaneous LOS direction in station SEZ: azimuth/elevation [rad] |
| Why it exists | Range'in zayıf transverse geometry'sini tamamlar; station/network geometry bilgisini taşır |
| Selection | Geometric `position`; ayrıca `range_rate` için `companion_geometry="instantaneous"` |
| Configuration path | `ScenarioConfig` -> generation profile normalization |
| Runtime call path | `position_observables`/`_razel_scalar` -> `geometry.ecef2razel_sez`; UKF `_position_measurement_from_state` |
| Event epochs | LOS, station ve transform `t_r` |
| Frames/origins | Earth-centered J2000 difference -> ITRF93 -> right-handed SEZ using geodetic latitude |
| Observable | `A=atan2(E,-S)` wrapped to `[0,2pi)` [rad]; `e=atan2(Z,sqrt(S^2+E^2))` [rad] |
| Residual | `wrap(A_obs-A_comp)` and `wrap(e_obs-e_comp)` [rad] |
| Noise | `Station.sigma_angle_rad` independently on az/el; constant per station |
| Bias | Global/per-station angle bias modes are additive radians |
| Jacobian | Analytic SEZ chain; local `(2,6)`, local velocity columns zero, receive STM once |
| Estimator consumers | BLS-LM, SRIF, posterior, UKF |
| Observability | Same initial-state mapper as geometric range |
| UKF | Supported and physics-consistent for geometric profile |
| Metadata | `angle_jacobian_model=analytic_exact_geometric`; receive frame/epoch |
| Tests | Cardinal/handedness, transpose/direction mutation, zenith policy, wrap-aware residual, FD chain |
| Approximations | Exact zenith observable reports azimuth 0 as display convention; geometric local Jacobian has legacy fallback |
| Missing physics | Refraction, boresight, antenna calibration, station/EOP sensitivities |

Production angle rows for `q=[S,E,Z]^T`, `h=sqrt(S^2+E^2)`, `R=||q||`:

\[
\frac{\partial A}{\partial q}=
\begin{bmatrix}E/h^2&-S/h^2&0\end{bmatrix}\quad[\mathrm{rad/m}],
\]

\[
\frac{\partial e}{\partial q}=
\begin{bmatrix}-ZS/(R^2h)&-ZE/(R^2h)&h/R^2\end{bmatrix}
\quad[\mathrm{rad/m}].
\]

### 23.3 One-way CN light-time range

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Receive station `t_r` ile transmit spacecraft `t_t=t_r-tau` arasındaki norm [m] |
| Why it exists | Lunar-distance finite propagation time geometry'sini temsil etmek |
| Selection | `measurement_model_profile="one_way_light_time"` veya legacy `apply_light_time=True` |
| Configuration path | `ScenarioConfig` -> `generate_position_measurements`; implicit derivative ayrıca `jacobian_model="implicit_light_time"` |
| Runtime call path | `_apparent_position_observable` -> `solve_one_way_light_time`; derivative `one_way_light_time_initial_state_sensitivity` |
| Event epochs | Tag/station/frame `t_r`; spacecraft state/STM `t_t` |
| Frames/origins | CN solve Moon-centered J2000 LOS; receive transform only angle pathında |
| Observable | `tau=norm(r_sc(t_r-tau,x0)-r_st(t_r))/c` [s], `R=c tau` [m] |
| Residual | `R_obs-R_comp` [m] |
| Noise | Geometric position ile aynı `sigma_range_m` |
| Bias | Aynı additive range bias; clock/light-time bias olarak fiziksel parametreleştirilmemiş |
| Jacobian | Implicit initial-state analytic row when selected; first-order local legacy alternative remains |
| Estimator consumers | BLS-LM, SRIF, posterior |
| Observability | Same implicit initial row via shared mapper |
| UKF | Config accepts but runtime position measurement remains geometric: silent physics downgrade, CRITICAL |
| Metadata | Transmit spacecraft epoch; `range_jacobian_model=implicit_light_time` when selected |
| Tests | Static/linear LT, local/initial FD, transmit STM mutation, M2.1 row reuse, no-double-STM |
| Approximations | Fixed receive time; fixed station solve-for; Newtonian Euclidean LT; cubic Hermite in-grid, linear extrapolation outside |
| Missing physics | Strict observable convergence/equation residual policy, relativity, media, clock/station parameter derivatives, independent end-to-end external CN |

Current spacecraft-only solve-for vector gives

\[
\frac{\partial\tau}{\partial x_0}=
\frac{\hat\rho^T\Phi_r(t_t,t_0)}{c+\hat\rho^Tv_{sc}(t_t)}
\quad[\mathrm{s/state\ unit}],
\]

because `t_r` and station parameters are fixed, while
`dt_t/dx_0=-d tau/dx_0`. The equation changes if receive time, station state,
clock, EOP or observer coordinates become solve-for parameters.

### 23.4 One-way CN azimuth/elevation

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | CN LOS direction at receive station in SEZ [rad] |
| Why it exists | Finite-light-time direction geometry'sini range ile tutarlı yapmak |
| Selection | One-way LT profile; exact chain requires `jacobian_model="implicit_light_time"` |
| Configuration path | Same position profile path |
| Runtime call path | `_apparent_position_observable`; derivative `one_way_light_time_position_initial_state_jacobian` |
| Event epochs | Spacecraft `t_t`; station/frame and tag `t_r`; STM `t_t` |
| Frames/origins | Unit CN LOS J2000 -> receive `C_ITRF93<-J2000` -> station SEZ |
| Observable | Same az/el equations as §23.2, but LOS uses `r_sc(t_t)-r_st(t_r)` |
| Residual | Wrapped observed-minus-computed [rad] |
| Noise | Same fixed station angle sigmas |
| Bias | Same additive angle bias modes |
| Jacobian | Initial-state implicit LOS normalization + analytic frame/SEZ chain, `(2,6)` |
| Estimator consumers | BLS-LM, SRIF, posterior |
| Observability | Shared `(3,6)` position block |
| UKF | Config accepted, geometric measurement operator used: inconsistent |
| Metadata | `line_of_sight_jacobian_model=implicit_light_time_chain_rule`; singularity threshold `1e-6` |
| Tests | Unit-LOS tangency, step-sweep FD, wrap-aware azimuth, near-zenith raise, shared-row/mutation tests |
| Approximations | Receive frame treated fixed with respect to spacecraft `x0`; named conditioning threshold |
| Missing physics | Refraction, station/EOP/clock derivatives, external end-to-end apparent-angle validation |

\[
J_{\rho,x_0}=\Phi_r(t_t,t_0)-v_{sc}(t_t)\frac{\partial\tau}{\partial x_0}
\quad[\mathrm{m/state\ unit}],
\]

\[
J_{\hat\rho,x_0}=\frac{I-\hat\rho\hat\rho^T}{R}J_{\rho,x_0},\qquad
J_{SEZ,x_0}=C_{SEZ\leftarrow ITRF93}C_{ITRF93\leftarrow J2000}(t_r)
J_{\hat\rho,x_0}.
\]

### 23.5 One-way CN+S apparent azimuth/elevation

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | CN range korunarak receive-case Newtonian stellar-aberrated direction [rad] |
| Why it exists | Observer velocity'nin apparent direction üzerindeki `v/c` etkisini modellemek |
| Selection | `one_way_light_time_aberrated_local_mci` veya `..._spice_ssb` |
| Configuration path | Profile normalization forces LT+stellar; `stellar_aberration_model` profile'dan türetilir |
| Runtime call path | CN solve -> `_observer_velocity_j2000_at_receive_epoch` -> `apply_stellar_aberration` -> SEZ |
| Event epochs | CN spacecraft `t_t`; observer velocity, station and frame `t_r` |
| Frames/origins | LOS J2000; `spice_ssb` observer velocity SSB/J2000; `local_mci` Moon-relative approximation |
| Observable | `sin(phi)=(norm(v_obs)/c) sin(w)` and Rodrigues rotation toward observer velocity; range unchanged |
| Residual | Wrapped az/el observed-minus-computed [rad] |
| Noise | Same angle sigmas; omitted correction is systematic, not random noise |
| Bias | Additive angle bias may absorb some effect but is not a physical aberration model |
| Jacobian | Hybrid: analytic CN initial chain times local tangent-space central-FD aberration derivative |
| Estimator consumers | BLS-LM, SRIF, posterior |
| Observability | Same hybrid shared position block |
| UKF | Config accepted, geometric operator used: CN and +S both omitted |
| Metadata | `hybrid_apparent_chain_rule`, `local_central_finite_difference`, step `1e-5`, full residual physics match true for current chain |
| Tests | Direct `spice.stelab`, tangent action, step plateau, parallel/antiparallel, six-state FD, estimator shared rows |
| Approximations | Newtonian reception `+S`, not relativistic; observer velocity fixed w.r.t. spacecraft `x0`; `local_mci` omits Moon barycentric motion |
| Missing physics | Analytic +S derivative, observer-state derivatives, relativistic bending/aberration, full external CN+S spacecraft solution |

For deterministic tangent basis `B=[b1,b2]` and `h=1e-5`:

\[
u_{i,\pm}=\frac{u\pm hb_i}{||u\pm hb_i||},\qquad
J_S^{tan}=\begin{bmatrix}d_1&d_2\end{bmatrix}B^T,\quad
d_i=\frac{f_S(u_{i,+})-f_S(u_{i,-})}{2h},
\]

\[
J_{app,x_0}=J_S^{tan}J_{CN,x_0}.
\]

`_stellar_aberration_local_jacobian` nonfinite velocity, invalid `c` and
`||v||>=c` inputsını reddeder. `apply_stellar_aberration` observable operator'u
aynı rejection contract'ına sahip değildir; bu ayrım test ve metadata
tasarımında korunmalıdır.

### 23.6 Geometric instantaneous range-rate

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Instantaneous LOS projection of relative velocity [m/s] |
| Why it exists | Baseline LOS-velocity observable; counted Doppler'dan bağımsız kinematic comparison |
| Selection | `measurement_type="range_rate"`, `range_rate_physics="geometric_instantaneous"` |
| Configuration path | `ScenarioConfig` -> `RangeRatePhysicsConfig` -> generator/residual/filter |
| Runtime call path | `accelerated.geometric_range_rate_observables`; analytic `compute_range_rate_residuals_analytic`; UKF `_range_rate_measurement_from_state` |
| Event epochs | Spacecraft, station, frame and tag `t_r` |
| Frames/origins | Full relative state is transformed by 6x6 `sxform` into ITRF93; LOS/rate evaluated there |
| Observable | `Rdot=u^T v_rel` [m/s] |
| Residual | `Rdot_obs-Rdot_comp` [m/s] |
| Noise | `Station.sigma_range_rate_mps`, constant/independent; companion range/angles retain own sigmas |
| Bias | Additive rate bias through range-rate bias modes |
| Jacobian | Local analytic position/velocity row; includes lower-left `X[3:6,0:3]`, receive STM once |
| Estimator consumers | BLS-LM, SRIF, posterior, UKF |
| Observability | Four-row block weighted by range/rate/angle sigmas |
| UKF | Supported and physics-consistent |
| Metadata | `range_rate_physics=geometric_instantaneous`; receive epochs |
| Tests | Closed form, production-vs-inertial station velocity, local/full-state FD, estimator/filter/observability recovery |
| Approximations | Instantaneous; not phase/count interval observable |
| Missing physics | One-way LT range derivative observable, media/clock/frequency effects |

With `rho_F=C rho_I`, `v_F=D rho_I+C v_I`, where
`D=X[3:6,0:3]`, the local inertial derivatives are

\[
\frac{\partial\dot R}{\partial r_I}=
\left(\frac{v_F-\hat\rho\dot R}{R}\right)^TC+\hat\rho^TD
\quad[\mathrm{s^{-1}}],
\qquad
\frac{\partial\dot R}{\partial v_I}=\hat\rho^TC
\quad[1].
\]

### 23.7 Two-way counted Doppler

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Count interval endpoint'lerindeki round-trip LT farkı; equivalent m/s veya Hz |
| Why it exists | Coherent interval-averaged radiometric velocity/phase-change surrogate |
| Selection | `range_rate_physics="two_way_counted_doppler"` |
| Configuration path | `ScenarioConfig` -> `scenario_range_rate_physics_config` -> `RangeRatePhysicsConfig` |
| Runtime call path | `two_way_counted_doppler_observable` -> `solve_two_way_light_time` at `t_-`,`t_+`; derivative endpoint `two_way_counted_doppler_initial_state_jacobian` |
| Event epochs | Each endpoint has station receive `t3`, one spacecraft bounce `t2`, station transmit `t1`; endpoint tags around midpoint |
| Frames/origins | Station/event states Moon-centered J2000 reconstructed from linearly interpolated Earth state and 6x6 transform grid |
| Observable | `y_mps=c[ tau_RT(t_+)-tau_RT(t_-) ]/(2Tc)` [m/s]; `y_Hz=k f_u Delta tau_RT/Tc` [Hz] |
| Residual | Counted row observed-minus-computed in configured output unit; companion rows own units |
| Noise | Reuses `sigma_range_rate_mps` even for simplified Hz path unless caller treats units consistently; no phase/count noise model |
| Bias | Fixed station clock offset/drift and fixed transponder delay inputs; additive estimator rate bias possible; no clock solve-for |
| Jacobian | Initial-state implicit endpoint partial difference; companion rows local+STM; counted row replaces geometric row |
| Estimator consumers | BLS-LM, SRIF, posterior, UKF |
| Observability | Same endpoint initial row through `_build_two_way_range_rate_initial_state_jacobian` |
| UKF | Accepted and nonlinear counted operator is called on locally propagated histories |
| Metadata | Count interval, frequency, ratio, clock/delay, local state model, LT tolerances; legacy single-bounce limitation is not a dedicated enum |
| Tests | Static/receding, grid density, clock/frequency, residual closure, FD Jacobian, estimator/UKF/observability, frame interpolation diagnostics |
| Approximations | Constant uplink frequency; same `t2` state for both legs with nonzero delay; spacecraft cubic Hermite in-grid/linear extrapolation outside; station/transform linear interpolation; convergence flag not enforced by observable |
| Missing physics | Separate `t2u/t2d`, exact event frame policy, strict failure/domain policy, ramps, media, relativistic/proper time, hardware/phase-center model, external Doppler validation |

Legacy nonzero-delay counted Doppler, M3'ün four-event modeline fiziksel olarak
eşdeğer değildir. Bu fark default zero-delay regression'ı sessiz değiştirmeden
ayrı bir fazda ele alınmalıdır.

### 23.8 M3 raw half-round-trip two-way range

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Full station transmit-to-receive elapsed time'ın yarısı: raw two-way range [m] |
| Why it exists | Explicit uplink/downlink event geometry ve nonzero transponder delay bookkeeping'i |
| Selection | `measurement_type="two_way_range"`, `two_way_range_convention="raw_half_round_trip"` |
| Configuration path | `ScenarioConfig` -> `scenario_two_way_range_config` -> `prepare_measurement_arcs` |
| Public/selectable status | Public config enum ile seçilebilir; calibrated modelden ayrı solver değildir |
| Runtime call path | `generate_two_way_range_measurements` -> `solve_two_way_range_events` -> `two_way_range_from_solution` |
| Event epochs | `t1 < t2u <= t2d < t3`; tag `t3`; `t2d-t2u=delta_tr` |
| Frames/origins | Exact event-epoch station `sxform` at `t1`,`t3`; spacecraft state/STM separately at `t2u`,`t2d`; Moon-centered J2000 |
| Observable | `R_raw=c(t3-t1)/2` [m] |
| Residual | `R_obs-R_raw` [m] |
| Noise | `Station.sigma_range_m`; no dedicated two-way sigma |
| Bias | M3 bias solve-for rejected; delay is fixed config, not estimated |
| Jacobian | Scaled implicit three-equation event system, initial-state `(1,6)`; fixed-delay raw/cal rows identical |
| Estimator consumers | BLS-LM, SRIF, posterior |
| Observability | Shared `two_way_range_nominal_and_initial_jacobian` row |
| UKF | Rejected in config, scenario runner and `run_lunar_ukf` |
| Metadata | Convention, delay inclusion, four epochs, exact station policy, interpolation types, conditioning, tolerances, drops |
| Tests | Static/factor-two, nonzero delay, event order, failure, independent `scipy.root`, FD step sweep, no-double-STM, shared estimator/observability rows |
| Approximations | Newtonian Euclidean legs; Earth-center cubic Hermite; no corrections |
| Missing physics | Dedicated sigma, bias/clock solve-for, media, relativity, cross-station/three-way, external full spacecraft validation |

### 23.9 M3 delay-calibrated half-round-trip two-way range

| Alan | Doğrulanmış sözleşme |
|---|---|
| What it measures | Raw elapsed time'dan configured fixed transponder delay çıkarılmış half-round-trip range [m] |
| Why it exists | Known fixed turnaround delay'i geometric range convention'ından ayırmak |
| Selection | `measurement_type="two_way_range"`, default `two_way_range_convention="delay_calibrated_half_round_trip"` |
| Configuration path | `ScenarioConfig` -> `scenario_two_way_range_config` -> `prepare_measurement_arcs` |
| Public/selectable status | Public config enum ile seçilebilir; raw ile aynı event solver ve same fixed-delay sensitivity |
| Runtime call path | Raw path + `two_way_range_from_solution` convention branch |
| Event epochs | Raw modelle aynı four-event chain |
| Frames/origins | Raw modelle aynı |
| Observable | `R_cal=c[(t3-t1)-delta_tr]/2` [m] |
| Residual | `R_obs-R_cal` [m] |
| Noise | `Station.sigma_range_m` |
| Bias | Fixed delay removed by convention; uncertainty/drift/solve-for yok |
| Jacobian | `delta_tr` fixed olduğundan `dR_cal/dx0=dR_raw/dx0` |
| Estimator consumers | BLS-LM, SRIF, posterior |
| Observability | Raw ile shared row |
| UKF | Rejected |
| Metadata | Convention, `transponder_delay_model=fixed_coordinate_time` ve delay değeri yazılır; delay'in calibrated observable'dan çıkarıldığını söyleyen ayrı boolean field yoktur |
| Tests | Static nonzero-delay raw/cal offset `c delta/2`, convention selection, Jacobian equality |
| Approximations | Delay exactly known and constant kabul edilir |
| Missing physics | Delay uncertainty/drift and clock/hardware calibration states |

M3 event equations, uniform seconds:

\[
G_u=t_{2u}-t_1-\rho_u/c=0\ [\mathrm{s}],\quad
G_d=t_3-t_{2d}-\rho_d/c=0\ [\mathrm{s}],\quad
G_{tr}=t_{2d}-t_{2u}-\delta_{tr}=0\ [\mathrm{s}].
\]

For `y=[t1,t2u,t2d]^T`, fixed `t3`:

\[
\frac{\partial y}{\partial x_0}=-G_y^{-1}G_x
\quad[\mathrm{s/state\ unit}],\qquad
H_R=-\frac c2\frac{\partial t_1}{\partial x_0}
\quad[\mathrm{m/state\ unit}].
\]

## 24. Sembol düzeyinde code traceability

Line numbers review kolaylığı içindir; symbol adı authoritative referanstır.

| File | Symbol | Responsibility | Relevant equation/operation | Input shape | Output shape | Units |
|---|---|---|---|---|---|---|
| `lunar_od/measurements.py` | `generate_position_measurements` | Position truth/noise records and `PassGeometry` | profile-selected geometric/CN/CN+S | state `(Nt,6)`, visibility `(Nt,Ns)` | obs `(No,6/7)` | s,m,rad,index |
| same | `_apparent_position_observable` | One CN/CN+S computed observable | `R,Az,El` | one time/station + state history | `(3,)` | m,rad,rad |
| same | `compute_position_residuals` | Position O-C residual | wrapped angles | state `(Nt,6)`, obs `(No,>=6)` | residual `(3No,)`, computed `(No,3)` | mixed m/rad |
| same | `compute_position_residuals_analytic` | Residual + local/legacy Jacobian contract | geometric or first-order local rows | same | residual `(3No,)`, computed `(No,3)`, `Htilde (3No,6)` | per local state |
| same | `position_initial_state_jacobian_from_augmented_history` | Local-to-initial mapping or implicit block replacement | `H=Htilde Phi` or shared event helper | augmented `(Nt,42)` | `(3No,6)` | observable/state |
| same | `one_way_light_time_position_initial_state_jacobian` | Shared CN/CN+S initial block | implicit LT + LOS + SEZ (+ tangent +S) | one obs + augmented history | `(3,6)` | m/rad per state |
| same | `generate_range_rate_measurements` | Four-component range-rate records | companion + selected rate physics | state `(Nt,6)` | obs `(No,7/8)` | s,m,m/s,rad,index |
| same | `compute_range_rate_residuals[_analytic]` | Four-row O-C and local analytic H | `R,Rdot,Az,El` | state `(Nt,6)` | residual `(4No,)`, Htilde `(4No,6)` | mixed |
| same | `measurement_sigma_vector` | Stacked standard deviations | station component sigma lookup | obs + pass geometry | `(3No,)`, `(4No,)` or `(No,)` | observable units |
| same | `measurement_covariance_matrix` | Diagonal covariance | `diag(sigma^2)` | sigma `(M,)` | `(M,M)` | observable² |
| `lunar_od/estimators.py` | `_position_weight_diagonal`, `_range_rate_weight_diagonal`, `_two_way_range_weight_diagonal` | Batch whitening weights | `w_i=1/sigma_i^2`, equivalent `H_w=H/sigma` | obs + pass geometry | diagonal `(M,)` | inverse observable² |
| `lunar_od/accelerated.py` | `position_observables` | Batched geometric range/angles | J2000 translation/rotation + SEZ | states `(N,6)`, transforms `(N,6,6)` | `(N,3)` | m,rad |
| same | `geometric_range_rate_observables` | Batched geometric four-row observable | full `sxform` state | same | `(N,4)` | m,m/s,rad |
| same | `apply_stm_to_jacobian` | Local H mapping | `Htilde Phi` | `(M,6)`, augmented `(Nt,42)` | `(M,6)` | observable/x0 |
| `lunar_od/radiometrics.py` | `two_way_counted_doppler_observable` | Counted nominal | endpoint RTLT difference | midpoint + histories/config | scalar | m/s or Hz |
| same | `two_way_counted_doppler_initial_state_jacobian` | Counted initial row | endpoint partial difference | augmented `(Nt,42)` | `(6,)` | output/state |
| same | `solve_two_way_light_time` | Legacy single-bounce RTLT | nested fixed point | receive time + histories | dataclass | s,m,epochs |
| `lunar_od/two_way_range.py` | `solve_two_way_range_events` | M3 four-event solve | `Gu,Gd,Gtr` | `t3`, provider, history `(Nt,6)` | `TwoWayRangeEventSolution` | s,m |
| same | `two_way_range_event_sensitivity` | M3 implicit event derivative | `-Gy^-1 Gx` | solution + augmented `(Nt,42)` | sensitivity + `(1,6)` | s/state,m/state |
| same | `two_way_range_nominal_and_initial_jacobian` | Shared nominal/H consumer API | one solve/obs, no second STM | obs `(No,4/5)`, augmented `(Nt,42)` | nominal `(No,)`, H `(No,6)` | m,m/state |
| `lunar_od/estimators.py` | `estimate_position_bls_lm`, `estimate_range_rate_bls_lm`, `estimate_two_way_range_bls_lm` | Iterative batch LM | whitened O-C, prior, damped normal step | arc data + x0 | estimate/stats | SI/mixed |
| same | `estimate_position_srif`, `estimate_range_rate_srif`, `estimate_two_way_range_srif` | Square-root information batch solve | whitened rows + QR | arc data + x0 | estimate/stats | SI/mixed |
| same | `_position_posterior_information`, `_range_rate_posterior_information`, `_two_way_range_posterior_information` | Posterior information/covariance | `H_w^T H_w + P0^-1` | initial H | `(Nstate,Nstate)` | inverse state² |
| `lunar_od/observability.py` | `build_initial_state_jacobian` | Model-dispatched initial H | shared position/counted/M3 helpers | arc | `(M,6)` | observable/state |
| same | `analyze_initial_state_observability` | Whitening/Fisher/SVD | `F=H_w^T H_w` | H/sigma | result dataclass | scaled |
| `lunar_od/filters.py` | `run_lunar_ukf` | Sequential sigma-point estimator | nonlinear measurement operator | obs, state/cov | `LunarUKFResult` | state SI |
| same | `_position_measurement_from_state` | UKF position operator | geometric-only `R,Az,El` | sigma state `(>=6,)` | `(3,)` | m,rad |
| same | `_range_rate_measurement_from_state` | UKF geometric/counted RR operator | companion + RR physics | sigma state | `(4,)` | m,m/s or Hz,rad |
| `lunar_od/scenario_config.py` | `ScenarioConfig`, `_validate_cross_field_rules` | JSON schema and combination validation | enums/cross-field rules | mapping | normalized config | declared per field |
| `lunar_od/scenarios.py` | `prepare_measurement_arcs`, `run_batch_arc_sequence` | Generation and estimator dispatch | model/profile routing | scenario arrays | `PreparedArc`/`ScenarioResult` | mixed |
| `lunar_od/reporting.py` | `write_scenario_summary_csv` | Scenario summary traceability | selected M2 metadata columns | scenarios | CSV | mixed |

## 25. Derivative taxonomy ve ownership

### 25.1 Tanımlar

\[
\widetilde H_k=\frac{\partial h_k}{\partial x_k},\qquad
H_k=\frac{\partial h_k}{\partial x_0},\qquad
H_k=\widetilde H_k\Phi(t_k,t_0).
\]

`Htilde` local receive-epoch derivative, `H` arc-initial derivative'dir.
Implicit event helpers doğrudan `H` üretir; bunlara ikinci STM uygulanmaz.
Test-only full-state FD (`estimators._range_rate_numerical_initial_jacobian`
ve test-local perturbation loops) production estimator pathı değildir.

| Measurement model | Production Jacobian type | State reference | STM applied where | Count | Analytic components | Numerical components | Independent FD evidence |
|---|---|---|---|---:|---|---|---|
| Geometric range | local analytic | receive state -> x0 | mapper at `t_r` | 1 | range, frame | none | local/full FD |
| Geometric az/el | local analytic | receive -> x0 | mapper at `t_r` | 1 | SEZ/angles | none | wrap-aware/full FD |
| CN range implicit | implicit event | x0 | inside `Phi_r(t_t,t0)` | 1 | LT/range | none | state step sweep |
| CN az/el implicit | implicit event chain | x0 | inside transmit helper | 1 | LT, unit LOS, frame, angles | none | wrap-aware six-state FD |
| CN+S az/el | hybrid implicit | x0 | inside transmit helper | 1 | LT/LOS/frame/angles | local +S tangent FD | SPICE tangent + full-state FD |
| Geometric range-rate | local analytic | receive -> x0 | mapper at `t_r` | 1 | kinematic + full sxform | none | local/full FD |
| Apparent RR companion | first-order local | receive -> x0 | receive STM | 1 | companion receive-frame rows | none | residual closure; exact implicit FD gap |
| Counted Doppler row | implicit endpoint | x0 | event STM in helper | 1 | RTLT endpoint partials | none | full initial FD |
| M3 raw/cal range | scaled implicit event | x0 | `t2u/t2d` STMs in helper | 1 | event matrix | none | independent root + step sweep |

## 26. Comprehensive frame and epoch matrix

| Measurement | Tag | Station epoch | Spacecraft epoch | Frame transform epoch | LOS/evaluation frame | Station fixed frame | State reference center | Observer velocity epoch | STM epoch |
|---|---|---|---|---|---|---|---|---|---|
| Geometric range | `t_r` | `t_r` | `t_r` | `t_r` | range norm invariant; implementation ITRF93 | ITRF93 | spacecraft/earth states Moon-centered before translation | n/a | `t_r` |
| Geometric az/el | `t_r` | `t_r` | `t_r` | `t_r` | J2000 -> ITRF93 -> SEZ | ITRF93 | Moon-centered inputs, Earth-centered LOS | n/a | `t_r` |
| CN range | `t_r` | `t_r` | `t_t=t_r-tau` | n/a for norm | MCI/J2000 | ITRF93 station reconstructed in J2000 | Moon | n/a | `t_t` |
| CN az/el | `t_r` | `t_r` | `t_t` | `t_r` | CN J2000 -> ITRF93 -> SEZ | ITRF93 | Moon | n/a | `t_t` |
| CN+S az/el local | `t_r` | `t_r` | `t_t` | `t_r` | CN/+S J2000 -> ITRF93 -> SEZ | ITRF93 | Moon | `t_r`, Moon-relative | `t_t` |
| CN+S az/el SSB | `t_r` | `t_r` | `t_t` | `t_r` | CN/+S J2000 -> ITRF93 -> SEZ | ITRF93 | Moon for LOS; SSB for observer velocity | `t_r`, SSB/J2000 | `t_t` |
| Geometric range-rate | `t_r` | `t_r` | `t_r` | `t_r` | full state ITRF93 | ITRF93 | Moon inputs then Earth translation | n/a | `t_r` |
| Counted Doppler | midpoint; endpoints `t_-/t_+` | each `t1,t3` | single `t2` per endpoint | interpolated event epochs | Moon-centered J2000 event geometry | ITRF93 | Moon | n/a | each `t2` |
| M3 raw range | `t3` | `t1,t3` | `t2u,t2d` | exact `t1,t3` | Moon-centered J2000 event geometry | ITRF93 | Moon | n/a | `t2u,t2d` |
| M3 calibrated range | `t3` | `t1,t3` | `t2u,t2d` | exact `t1,t3` | same as raw | ITRF93 | Moon | n/a | `t2u,t2d` |

## 27. Critical issue record: UKF position physics parity

| Field | Record |
|---|---|
| Issue ID | `ISSUE-UKF-001` |
| Severity | **CRITICAL** physics/configuration inconsistency |
| Affected configuration | `estimator_type="ukf"`, `measurement_type="position"`, profile `one_way_light_time` or either aberrated profile |
| Accepted configuration path | `scenario_config_from_mapping` -> `_validate_cross_field_rules`; no UKF rejection for non-geometric position profile |
| Actual UKF observable physics | `_measurement_context` -> `_position_measurement_from_state`: receive-epoch geometric range/az/el only |
| Expected physics | Sigma-point prediction must use the selected CN or CN+S profile used for measurement generation, or config must reject it |
| Files/symbols | `scenario_config._validate_cross_field_rules`; `scenarios.run_batch_arc_sequence`; `filters.run_lunar_ukf`, `_measurement_context`, `_position_measurement_from_state` |
| Why tests did not catch it | UKF tests use geometric position; M2/M2.3 tests cover BLS/SRIF/observability shared Jacobians, not UKF profile parity; config tests validate acceptance, not runtime operator identity |
| Potential estimator effect | Systematic innovations, biased state/covariance, misleading NIS and apparent filter weakness; CN+S can omit both LT and angular correction |
| Minimal reproduction | Build a UKF `position` scenario with `measurement_model_profile="one_way_light_time_aberrated_spice_ssb"`; compare one truth-state row from `compute_position_residuals` with `_position_measurement_from_state` at same epoch; difference is nonzero although state is truth |
| Recommended fix | P0: either hard-reject non-geometric UKF position profiles or provide bounded history/profile-aware sigma-point measurement operator; rejection is smaller backward-compatibility surface |
| Required validation | Config rejection/acceptance tests; truth-state closure per profile; geometric bitwise regression; CN/CN+S UKF seeded tests; NIS/covariance consistency; performance/caching audit |
| Backward compatibility risk | Hard rejection breaks previously accepted but scientifically inconsistent configs; implementation may alter runtime/caching and requires history domain policy |

UKF branch status, ayrı ayrı:

| Physics | Config | Runtime | Status |
|---|---|---|---|
| Geometric position | accepted | geometric operator | supported |
| CN position | accepted | geometric operator | silently downgraded |
| CN+S local/SSB position | accepted | geometric operator | silently downgraded |
| Geometric range-rate | accepted | geometric rate | supported |
| Apparent one-way companion | accepted | apparent companion on local history | supported nominally; exact implicit Jacobian irrelevant to UKF |
| Two-way counted Doppler | accepted | counted operator on local histories | supported with legacy solver limits |
| M3 raw/cal range | rejected by config, runner and filter | not run | safely unsupported |

## 28. Solver failure, domain and extrapolation policy

| Solver | Convergence criterion | Equation residual criterion | Max iter | Failure behavior | History-bound behavior | Extrapolation | Metadata |
|---|---|---|---:|---|---|---|---|
| One-way CN `solve_one_way_light_time` | `abs(tau_new-tau) <= 1e-12 s` default | none separate | 10 | returns `converged=False`; sensitivity helper raises, nominal observable does not enforce | no explicit pre-roll guard | spacecraft cubic Hermite in-grid, linear outside | generic position metadata incorrectly inherits RR defaults `1e-10 s / 20`; convergence result not persisted |
| Legacy `solve_two_way_light_time` | downlink/uplink fixed-point update tolerance `1e-10 s` default | none separate | 20 | returns `converged`; counted observable ignores flag | no strict spacecraft support guard | spacecraft cubic Hermite in-grid/linear outside; Earth/station/transform linearly interpolate/extrapolate | config parameters present; convergence/failure not persisted |
| M3 `solve_two_way_range_events` | event update `1e-12 s` default | `max(abs(Gu),abs(Gd),abs(Gtr)) <= 1e-11 s` | 25 | controlled `TwoWayEventConvergenceError`; no last-iterate use | spacecraft event epochs explicitly guarded; generator drops `TwoWayEventHistoryError` rows | spacecraft no extrapolation; Earth-center provider cubic Hermite with provider behavior; exact station `sxform` | tolerances, policy, dropped count/reason attached |

M3 failure handling bu audit'te açık bir correctness gap değildir. Açık kalan
konular one-way/legacy counted policy ve M3 drop-reason sınıflandırmasının
granularity'sidir.

## 29. Noise, bias, clock ve data-editing audit'i

| Observable | Sigma source/unit | Geometry dependence | Stochastic form | Bias support | Bias solve-for | Clock support | Media | Editing/robustness |
|---|---|---|---|---|---|---|---|---|
| Geometric/CN range | `Station.sigma_range_m` [m] | constant per station | generator independent Gaussian | additive range component | BLS/SRIF/UKF mode-dependent | none | none | batch hard reweight; UKF NIS/component/robust gates |
| Geometric/CN/CN+S az/el | `Station.sigma_angle_rad` [rad] | constant per station; no elevation mapping | independent Gaussian | additive angle components | global/per-station modes | none | none | same estimator-side methods |
| Geometric range-rate | `Station.sigma_range_rate_mps` [m/s] | constant per station | independent Gaussian | additive rate component | supported in range-rate bias modes | none | none | same |
| Counted Doppler m/s | same `sigma_range_rate_mps` [m/s] | constant | independent Gaussian on generated scalar | additive rate bias can be fitted | supported as generic range-rate bias | fixed offset/drift enter nominal; not solved | none | same |
| Counted Doppler Hz | no dedicated Hz sigma field | constant | caller must ensure unit consistency | generic row bias risks unit ambiguity | generic mechanism exists | fixed offset/drift | none | same |
| M3 raw/cal range | reuses `Station.sigma_range_m` [m] | constant | independent Gaussian | none in M3 | explicitly rejected | none | none | history-domain rows may be dropped; no persistent per-record quality flags |

`noise_models.generate_measurement_noise` ayrıca white/correlated Gaussian,
AR(1) ve Student-t üretir; standard measurement generators ile tek bir
truth-corruption contract'ına bağlı değildir. BLS/SRIF robust reweighting ve
UKF innovation gating fiziksel data editing ile aynı şey değildir: record
quality flag, outlier provenance, dropout reason ve pre/post weight manifest'i
genel üretim formatında bulunmaz.

## 30. Model-by-model missing-physics matrix

Kısaltmalar: **I** implemented, **P** partially implemented, **N** not
implemented, **A** not applicable, **U** unknown/not demonstrated. Modeller:
`GR` geometric range, `GA` geometric angles, `CR` CN range, `CA` CN angles,
`CS` CN+S angles, `RR` geometric range-rate, `CD` counted Doppler, `TR` M3 raw,
`TC` M3 calibrated.

| Physics/correction | GR | GA | CR | CA | CS | RR | CD | TR | TC | Impact | Priority |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| Troposphere | N | N | N | N | N | N | N | N | N | Low-elevation range/angle/rate systematic | P1/P2 |
| Ionosphere | N | N | N | N | N | N | N | N | N | Frequency-dependent Earth link delay/rate | P2 |
| Solar plasma | N | A | N | A | A | N | N | N | N | Sun-proximity range/Doppler bias | P2 |
| Station clock offset | N | N | N | N | N | N | P | N | N | Time/range bias; CD fixed input only | P1 |
| Station clock drift | N | N | N | N | N | N | P | N | N | Doppler trend; CD fixed input only | P1 |
| Spacecraft clock | N | N | N | N | N | N | N | N | N | One-way/operational timing | P2 |
| Transponder delay | A | A | A | A | A | A | P | I | I | CD single-bounce limitation; M3 fixed delay | P0/P1 |
| Transponder delay drift | A | A | A | A | A | A | N | N | N | Time-varying range/Doppler bias | P2 |
| Relativistic light time | A | A | N | N | N | A | N | N | N | Precision event-time bias | P2 |
| Shapiro delay | A | A | N | N | N | A | N | N | N | Solar/body gravitational delay | P2 |
| Station displacement | N | N | N | N | N | N | N | N | N | Position/angle/range bias | P2 |
| Solid Earth tides | N | N | N | N | N | N | N | N | N | Station displacement component | P2 |
| Ocean loading | N | N | N | N | N | N | N | N | N | Site-dependent displacement | P2 |
| Pole tide | N | N | N | N | N | N | N | N | N | Site displacement | P2 |
| Plate motion | N | N | N | N | N | N | N | N | N | Long-term station coordinate drift | P2 |
| EOP solve-for | N | N | N | N | N | N | N | N | N | Frame uncertainty not estimated | P2 |
| Antenna phase center | N | N | N | N | N | N | N | N | N | Range/angle/Doppler site offset | P2 |
| Hardware delay | N | N | N | N | N | N | P | P | P | Only fixed transponder delay represented | P1/P2 |
| Range bias | P | A | P | A | A | A | A | N | N | Generic additive state, not calibrated hardware model | P1 |
| Angle bias | A | P | A | P | P | A | A | A | A | Generic additive state | P1 |
| Doppler/rate bias | A | A | A | A | A | P | P | A | A | Generic additive row | P1 |
| Correlated noise | P | P | P | P | P | P | P | P | P | Utility exists but standard generators not unified | P1 |
| Light-time media iteration | A | A | N | N | N | A | N | N | N | Corrections not coupled to event solve | P2 |
| Frequency ramps | A | A | A | A | A | A | N | A | A | Operational Doppler mismatch | P2 |
| Ramp tables | A | A | A | A | A | A | N | A | A | No time-varying phase/frequency integration | P2 |
| Multi-station same-epoch records | I | I | I | I | I | I | I | I | I | Network observations exist | existing |
| Cross-station/three-way links | A | A | A | A | A | A | N | N | N | Uplink/downlink station separation absent | P2 |
| Proper-time conversion | A | A | A | A | A | A | N | N | N | Operational frequency/time standard mismatch | P2 |

`P` correlated noise yalnız utility-level capability anlamındadır; her
production generator'da aktif ve traceable olduğu anlamına gelmez.

## 31. Literature-source audit and hierarchy

### 31.1 Source relationship matrix

| ID / class | Author; title; edition/year; institution | Section/equation | Stable URL/DOI | Repository claim supported | Relationship |
|---|---|---|---|---|---|
| SRC-01 Primary authoritative | T. D. Moyer, *Formulation for Observed and Computed Values of DSN Data Types for Navigation*, JPL Pub. 00-7, 2000 | Ch. 8 LT; Ch. 10 media/antenna; Ch. 11 precision LT; Ch. 12 partials; Ch. 13 observables; §13.3.1.1 Eq. 13-31 | https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_all.pdf | Multi-event radiometrics, count-interval concept, correction gaps | Event/correction authority; repository M2/M3 derivatives are project adaptations and endpoint-RTLT Doppler is only a simplified derived relationship, not Eq. 13-31 verbatim |
| SRC-02 Primary authoritative | C. L. Thornton, J. S. Border, *Radiometric Tracking Techniques for Deep-Space Navigation*, DESCANSO Monograph 1, 2003 | Ch. 3 radiometric tracking | https://descanso.jpl.nasa.gov/monograph/series1/Descanso1_all.pdf | Why range/Doppler exist and operational context | Conceptual; repository is not a full DSN implementation |
| SRC-03 Implementation documentation | NAIF/JPL, *Aberration Corrections Required Reading* | Reception LT Eq. (1); stellar reception discussion | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html | CN event epochs, `NONE/LT/CN/CN+S`, Newtonian +S ordering | Exact convention reference; internal synthetic spacecraft solve is independently coded |
| SRC-04 Implementation documentation | NAIF/JPL, `spkezr_c` | Detailed input/particulars | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html | Observer-time tag, aberration flags, state frame | Exact API semantics; not production measurement call for synthetic orbiter |
| SRC-05 Independent validation source | NAIF/JPL, `stelab_c` | Procedure/particulars | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/stelab_c.html | Newtonian reception stellar-aberration operator | Exact operator oracle only; not end-to-end CN validation |
| SRC-06 Implementation documentation | NAIF/JPL, *Frames Required Reading* | Frame transformation functions and frame centers | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html | Rotation vs state transform, ITRF93/J2000 semantics | Exact SPICE frame contract |
| SRC-07 Implementation documentation | NAIF/JPL, `sxform_c` | Detailed output | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sxform_c.html | 6x6 state transform and velocity coupling | Exact API semantics |
| SRC-08 Implementation documentation | NAIF/JPL, `pxform_c` | Detailed output | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pxform_c.html | 3x3 position rotation distinction | Exact API semantics |
| SRC-09 Implementation documentation | NAIF/JPL, *Time Required Reading* | ET/TDB and time systems | https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html | `et0 + relative seconds`, SPICE ET meaning | Exact SPICE time-system context; project relative-time origin remains project metadata |
| SRC-10 Primary authoritative | IERS Conventions 2010, Gérard Petit and Brian Luzum (eds.), IERS TN36 | Ch. 5 celestial/terrestrial transform; Ch. 7 station displacement | https://iers-conventions.obspm.fr/content/tn36.pdf | Missing EOP/station displacement physics | Exact scope reference; repository delegates orientation to SPICE and omits solve-for/displacements |
| SRC-11 Primary registry | IOGP/EPSG Geodetic Parameter Dataset, Ellipsoid 7030 `WGS 84` | `a=6378137 m`, `1/f=298.257223563`; source DMA TM 8350.2-B | https://epsg.org/ellipsoid_7030/WGS-84.html | Exact ellipsoid constants used by `geometry.py` | Exact constants; repository's rigid WGS84-as-ITRF93 station policy remains a project approximation |
| SRC-12 Secondary textbook | B. D. Tapley, B. E. Schutz, G. H. Born, *Statistical Orbit Determination*, 2004, Elsevier Academic Press | Ch. 3 observations; Ch. 4 fundamentals; Ch. 5 square-root methods | https://www.sciencedirect.com/book/monograph/9780126836301/statistical-orbit-determination | O-C linearization, weighted batch, information/square-root context | Conceptual/derived; exact project update and bias modes are code-defined |
| SRC-13 Secondary textbook | O. Montenbruck, E. Gill, *Satellite Orbits*, 1st ed., 2000, Springer | Ch. 7 pp. 193-232; Ch. 8 pp. 233-256; Ch. 9 pp. 257-291 | https://doi.org/10.1007/978-3-642-58351-3 | Tracking, linearization, parameter estimation context | Conceptual/secondary |
| SRC-14 Secondary textbook | D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 5th ed., 2022, Microcosm | Coordinate systems/OD discussion; exact section not verified | https://microcosmpress.com/vallado/ | Topocentric/frame and differential-correction background | Background only; not equation authority in this audit |
| SRC-15 Secondary/estimation | G. J. Bierman, *Factorization Methods for Discrete Sequential Estimation*, 1977, Academic Press | Factorization/square-root estimation; exact project QR section not mapped | https://shop.elsevier.com/books/factorization-methods-for-discrete-sequential-estimation/bierman/978-0-12-097350-7 | Motivation for square-root numerical methods | Conceptual; repository SRIF QR details are code-defined |
| SRC-16 Primary numerical method | K. Levenberg, “A Method for the Solution of Certain Non-Linear Problems in Least Squares,” 1944 | Whole article | https://doi.org/10.1090/qam/10666 | Damped least-squares lineage | Conceptual; repository lambda update/step cap are engineering choices |
| SRC-17 Primary numerical method | D. W. Marquardt, “An Algorithm for Least-Squares Estimation of Nonlinear Parameters,” 1963 | Whole article | https://doi.org/10.1137/0111030 | LM lineage | Conceptual; not exact implementation claim |
| SRC-18 Primary statistical | R. A. Fisher, “On the Mathematical Foundations of Theoretical Statistics,” 1922 | Information concept | https://doi.org/10.1098/rsta.1922.0009 | Fisher-information terminology | Conceptual; local `H_w^T H_w` is Gaussian linearized project construction |
| SRC-19 Primary observability | R. Hermann, A. J. Krener, “Nonlinear Controllability and Observability,” 1977 | Local nonlinear observability rank concepts | https://doi.org/10.1109/TAC.1977.1101601 | Observability terminology | Conceptual only; repository reports finite-arc sensitivity/Fisher rank, not a nonlinear observability theorem proof |

Kaynak hiyerarşisi: physical/event equations için SRC-01/03/10/11; SPICE API
semantics için SRC-04..09; estimator yöntemi için SRC-12/15..19; general
background için SRC-13/14. Blog ve tutorial sayfaları equation authority
olarak kullanılmamıştır.

## 32. Verification evidence matrix

| Model | Unit test | Analytic reference | FD reference | SPICE reference | Independent solver | Estimator integration | Observability | Real data | Remaining gap |
|---|---|---|---|---|---|---|---|---|---|
| Geometric range | yes | closed form | yes | frame/state only | n/a | BLS/SRIF/UKF | yes | no | external observable/real data |
| Geometric az/el | yes | SEZ closed form | wrap-aware | frame chain | n/a | BLS/SRIF/UKF | yes | no | calibrated angles/refraction |
| CN range | yes | static/constant velocity | step sweep | CN convention, not full target solve | no independent full solver | BLS/SRIF | yes | no | end-to-end external CN; UKF |
| CN az/el | yes | chain rule | wrap-aware six-state | frame/state only | no | BLS/SRIF | yes | no | external apparent angles; UKF |
| CN+S az/el | yes | zero/parallel/orthogonal cases | tangent + full-state | direct `stelab` operator | SPICE operator only | BLS/SRIF | yes | no | full CN+S target solution; UKF |
| Geometric RR | yes | closed form | local/full | station velocity/frame | n/a | BLS/SRIF/UKF | yes | no | external RR |
| Counted Doppler | yes | static/receding | initial-state | frame interpolation diagnostic | no independent endpoint solver | BLS/SRIF/UKF | yes | no | nonzero-delay/external DSN |
| M3 raw | yes | static/factor-two | step sweep | exact station/frame | `scipy.optimize.root` | BLS/SRIF | yes | no | external full two-way/real data |
| M3 calibrated | yes | `c delta/2` relation | same row | same | same | BLS/SRIF | yes | no | delay calibration/clock states |

No row in this matrix constitutes real tracking-data validation. SPICE frame
validation is not full spacecraft light-time validation.

## 33. Prioritized gap records

| Gap ID / category | Severity / difficulty | Affected models | Scientific impact | Estimator impact | User-visible impact | Current mitigation | Recommended change | Required tests | Phase / dependencies |
|---|---|---|---|---|---|---|---|---|---|
| GAP-PHY-01 physics | CRITICAL / HARD | CN, CN+S with UKF | Truth and filter use different observation physics | Bias, NIS/covariance invalidation | Config appears valid but result can be wrong | Use BLS/SRIF or geometric UKF | Implement profile-aware UKF or reject | profile closure, seeded UKF, geometric regression | P0, history-domain design |
| GAP-NUM-01 numerical | HIGH / MEDIUM | CN, counted | Last iterate may be used without verified equation closure | Discontinuous/biased residuals | Silent bad observable possible | Nominal cases converge in tests | Common strict update+equation residual policy | forced nonconvergence, boundary, metadata | P0, backward-compatible defaults |
| GAP-NUM-02 numerical | HIGH / MEDIUM | CN, counted | Out-of-domain extrapolation can hide insufficient pre-roll | Jacobian/nominal mismatch risk | No explicit failure | Dense local histories in UKF counted path | Explicit support contract; no silent spacecraft extrapolation | pre-roll sweep/mutation | P0, shared interpolation policy |
| GAP-PHY-02 physics | HIGH / MEDIUM | counted with nonzero delay | Single bounce is not four-event transponder physics | Systematic Doppler derivative/model error | Delay option overpromises fidelity | Default delay zero; docs disclose | Reject nonzero delay or adopt separate `t2u/t2d` | independent root, zero-delay regression | P0/P1, M3 event reuse decision |
| GAP-PHY-03 physics | HIGH / HARD | counted | Linear transform interpolation yields measured frame error | Can exceed Doppler noise budget | Cadence-dependent output | Diagnostics quantify 10/30/60 s | Exact-event station provider or justified cadence budget | exact/interpolated A/B, performance | P1, common provider API |
| GAP-CFG-01 configuration | HIGH / MEDIUM | range-rate apparent companion | Metadata/config can imply geometric-exact or implicit companion rows not produced | BLS/SRIF linearization mismatch | Accepted setting is misleading | Direct generator with `jacobian_model=None` normalizes to first-order, but `ScenarioConfig` supplies a string default | Constrain config to first-order until an exact implicit block exists, or implement that block | default/explicit config branches, FD rows, metadata | P0/P1, M2 helper adaptation |
| GAP-META-03 metadata | HIGH / EASY-MEDIUM | one-way CN/CN+S position | Artifact records wrong solver tolerance/iteration source | Reproduction/failure diagnosis unreliable | Metadata appears precise but is false for this path | Solver defaults are stable in code | Store one-way solver policy on `PassGeometry` or emit actual constants from one-way path | metadata-vs-call contract, reporting round trip | P0, strict solver policy |
| GAP-META-01 metadata | MEDIUM / EASY | M3 | Summary CSV omits convention/delay/event policy | Result reproduction weakened | Artifact cannot show raw/cal directly | `PassGeometry.measurement_metadata` retains it | Add M3 manifest/CSV columns | reporting round trip | P1, schema decision |
| GAP-META-02 metadata | LOW-MEDIUM / EASY | M3 generation | Generic dropped reason can misclassify non-pre-roll history errors | Failure diagnosis obscured | Misleading reason string | Controlled exception type | Structured error code/counts | injected history/nonfinite cases | P1 |
| GAP-STO-01 physics/statistical | HIGH / MEDIUM | all | Fixed independent sigma can misrepresent geometry/station errors | Weighting/covariance consistency suffers | Overconfident results | Per-station constants; noise utility | Truth-noise vs estimator-covariance profiles; elevation/station models | seed, empirical sigma, whiteness, MC | M4/P1 |
| GAP-STO-02 architecture | MEDIUM / MEDIUM | all | Outlier/dropout provenance not first-class | Robustness claims hard to audit | No record-level quality status | Estimator-side gates/reweighting | Quality flags and corruption manifest | ratios, clean regression, rejection metrics | M4/P1 |
| GAP-BIAS-01 physics | HIGH / HARD | counted, M3 | Clock/delay uncertainty cannot be estimated | Dynamics may absorb systematic trend | Fixed inputs only | Generic additive bias on RR; M3 rejects bias | Observable-specific clock/delay states with priors | recovery, rank/correlation, multi-station | M4/P1, observability design |
| GAP-PHY-04 physics | HIGH / HARD | Earth links | No media corrections | Low-elevation systematic range/angle/Doppler error | Operational fidelity limited | Explicit metadata `none` | Modular correction layer, troposphere first | standard cases, zero-correction regression | P2, environmental inputs |
| GAP-PHY-05 physics | MEDIUM-HIGH / HARD | CN, counted, M3 | No relativistic/Shapiro/proper-time terms | Precision LT/frequency bias | Cannot claim DSN accuracy | Limitations documented | Add correction terms after frame/clock contract | Moyer/reference cases, external comparison | P2 |
| GAP-ARC-01 architecture | MEDIUM / MEDIUM | measurement stack | Duplicate interpolation/provider logic | Behavior drift and inconsistent domains | Maintenance cost | Regression tests | Shared behavior-frozen interpolation/event provider | parity and mutation tests | P1/P2 after policies fixed |
| GAP-VAL-01 validation | HIGH / HARD | all high-fidelity models | Internal consistency can share bugs | Scientific confidence limited | No external acceptance report | SPICE operator/frame and independent M3 root | SPICE target + GMAT/Orekit/Tudat contract; real data later | frozen config/artifact comparison | P3, source data/tool versions |
| GAP-DOC-01 documentation | LOW / EASY | counted docs | Legacy state interpolation wording stale | None directly | Reader confusion | This audit records truth | Update focused model docs in separate approved task | doc grep/link check | Documentation phase |

## 34. Evidence-driven roadmap

1. **P0: UKF measurement-physics consistency.** CRITICAL çünkü accepted config
   truth/filter physics'ini sessiz ayırıyor. En küçük güvenli fix hard rejection;
   full implementation bounded history ve caching tasarımı ister.
2. **P0: strict legacy light-time policy.** One-way ve counted nominal yollarına
   update + equation residual, explicit history domain ve persisted failure
   metadata getirilmeli. M3 davranışı regression oracle olabilir.
3. **P0/P1: counted nonzero-delay contract.** Default zero-delay bitwise
   korunarak nonzero delay reject edilmeli veya M3 separate-event geometry'sine
   taşınmalı.
4. **P1: range-rate companion config/Jacobian doğruluğu.** Exact implicit
   companion row yokken config/metadata bunu söylememeli.
5. **P1: exact-event counted frame policy.** 10/30/60 s measured error budget'i
   kullanılarak exact provider veya cadence threshold seçilmeli.
6. **P1 M4: noise, bias and data editing.** Dedicated
   `sigma_two_way_range_m`, elevation/station covariance, explicit seed,
   truth/estimator split, quality flags, outlier/dropout ve clock/bias state'leri.
7. **P1/P2: shared interpolation helper refactor.** Yalnız davranış ve failure
   policy dondurulduktan sonra; physics change ile aynı patch'te değil.
8. **P2: media corrections.** Troposphere ilk; sonra ionosphere/solar plasma,
   hardware/antenna. Geometric LT ile correction delay ayrı metadata olmalı.
9. **P2: relativistic LT, proper time and ramps.** Operational DSN accuracy
   iddiasından önce Moyer-aligned conventions ve external oracle gerekir.
10. **P3: external and real-data validation.** SPICE SPK target, ardından frozen
    GMAT/Orekit/Tudat contract, son olarak provenance'lı real tracking ingest.
11. **Continuous: CI/reproducibility.** Focused V&V + natural full regression,
    optional-kernel skips explicit, artifacts config/source/tool version taşır.

M4 otomatik olarak ilk sırada değildir; UKF physics parity ve legacy solver
failure policy scientific correctness açısından önce gelir.

## Appendix A - Symbol glossary

| Symbol | Meaning | Units/frame |
|---|---|---|
| `t0` | arc initial epoch | relative s |
| `tr` | one-way receive/tag epoch | relative s |
| `tt` | one-way transmit epoch `tr-tau` | relative s |
| `t1` | two-way station uplink transmit | relative s |
| `t2` | legacy counted single spacecraft bounce | relative s |
| `t2u` | M3 spacecraft uplink receive | relative s |
| `t2d` | M3 spacecraft downlink transmit | relative s |
| `t3` | station downlink receive/tag | relative s |
| `Tc` | count interval | s |
| `tau` | one-way light time | s |
| `tau_RT` | round-trip light time | s |
| `delta_tr` | fixed transponder delay | s |
| `c` | light speed | m/s |
| `r_sc`, `v_sc` | spacecraft state | m, m/s; Moon-centered J2000 |
| `r_st`, `v_st` | station state | m, m/s; Moon-centered J2000 unless fixed-frame subscript |
| `rho` | station-to-spacecraft LOS vector | m |
| `R` | LOS norm/range | m |
| `u` | unit LOS | dimensionless |
| `S,E,Z` | south/east/zenith LOS components | m or dimensionless, context-dependent |
| `A,e` | azimuth/elevation | rad |
| `Rdot` | instantaneous geometric range-rate | m/s |
| `fu` | uplink frequency | Hz |
| `k` | turnaround ratio | dimensionless |
| `C` | 3x3 position rotation | dimensionless |
| `X` | 6x6 state transform | velocity blocks include 1/s |
| `Phi` | state transition matrix `dx(t)/dx0` | mixed state units |
| `Htilde` | local measurement Jacobian | observable/current-state unit |
| `H` | initial-state measurement Jacobian | observable/initial-state unit |
| `R_meas` | measurement covariance, not range | observable-unit squared |
| `F` | local Fisher/information matrix | inverse scaled-state squared |

## Appendix B - Array-shape glossary

| Array/object | Shape | Layout |
|---|---:|---|
| State history | `(Nt,6)` | `[rx,ry,rz,vx,vy,vz]` SI |
| Augmented history | `(Nt,42)` | state + column-major flattened `Phi(6,6)` |
| Transform grid | `(Nt,6,6)` | `X_ITRF93<-J2000` |
| Position observation | `(No,6)` or `(No,7)` | `[t,R,Az,El,station_id,time_index,(arc_id)]` |
| Range-rate observation | `(No,7)` or `(No,8)` | `[t,R,Rdot,Az,El,station_id,time_index,(arc_id)]` |
| M3 observation | `(No,4)` or `(No,5)` | `[t3,R2w,station_id,time_index,(arc_id)]` |
| Position computed | `(No,3)` | `[R,Az,El]` |
| Range-rate computed | `(No,4)` | `[R,rate,Az,El]` |
| Local position H | `(3No,6)` | ordered range/az/el rows |
| Local range-rate H | `(4No,6)` | ordered range/rate/az/el rows |
| M3 initial H | `(No,6)` | one row/measurement |
| Sigma vector | `(3No,)`, `(4No,)`, `(No,)` | same stacking as residual |
| Covariance | `(M,M)` | diagonal in standard path |
| Weighted H | `(M,Nstate)` | `H/sigma[:,None]` |
| Fisher/information | `(Nstate,Nstate)` | symmetric |

## Appendix C - Configuration matrix

| Combination | Valid? | Effective behavior/reason |
|---|:---:|---|
| `position + geometric + analytic_exact_geometric` | yes | default |
| `position + one_way_light_time + implicit_light_time` | yes | BLS/SRIF exact implicit; UKF mismatch |
| `position + CN+S + implicit_light_time` | yes | BLS/SRIF hybrid; UKF mismatch |
| stellar true without LT in legacy booleans | no | validation error |
| non-geometric profile with non-position type | no | use companion geometry for RR |
| `range_rate + geometric + instantaneous companion` | yes | default four-row model |
| `range_rate + counted + instantaneous companion` | yes | legacy counted row |
| `range_rate + counted + apparent_one_way companion` | yes | counted row + apparent range/angles |
| `range_rate + apparent companion + implicit_light_time` | accepted | exact implicit companion H not implemented; GAP-CFG-01 |
| non-instant companion with non-range-rate type | no | validation error |
| `two_way_range + raw + BLS/SRIF` | yes | public convention |
| `two_way_range + calibrated + BLS/SRIF` | yes | default convention |
| `two_way_range + UKF` | no | explicit M3 rejection |
| `two_way_range + bias_mode` | no | explicit M3 rejection |
| raw convention on non-two-way type | no | validation error |
| `finite_difference_reference` | enum accepted | reference label; production path must not be assumed to switch all estimators to FD without symbol trace |

## Appendix D - Metadata field dictionary

| Field | Meaning / values |
|---|---|
| `measurement_type` | `position`, `range_rate`, `two_way_range` |
| `measurement_model_profile` | geometric/CN/CN+S or M3 `two_way_light_time` |
| `range_rate_physics` | geometric or counted |
| `companion_geometry` | instantaneous/apparent one-way |
| `jacobian_model` | selected high-level model label |
| `range_jacobian_model` | actual range derivative label |
| `line_of_sight_jacobian_model` | geometric/implicit chain label |
| `angle_jacobian_model` | geometric, implicit or hybrid apparent chain |
| `aberration_jacobian_model` | not applied or local central FD |
| `angle_jacobian_matches_full_residual_physics` | metadata assertion for current BLS/SRIF chain; not UKF parity |
| `measurement_tag_epoch` | receive/tag convention |
| `spacecraft_position_epoch`, `spacecraft_velocity_epoch` | receive/transmit/event label |
| `station_position_epoch`, `station_velocity_epoch` | current station epoch label |
| `frame_transformation_epoch` | epoch at which orientation/state transform is evaluated |
| `observer_velocity_*` | +S velocity epoch/frame/reference center |
| `light_time_tolerance_s`, `light_time_max_iter` | Accurate for RR/legacy counted config; currently incorrect for position CN, whose solver defaults are `1e-12 s / 10` (GAP-META-03) |
| `count_interval_s`, `uplink_frequency_hz`, `turnaround_ratio` | counted settings |
| `station_clock_offset_s`, `station_clock_drift` | fixed counted clock inputs |
| `transponder_delay_s` | fixed delay; semantics depend on model/convention |
| `two_way_range_convention` | raw or delay-calibrated half RTLT |
| `station_state_evaluation` | interpolated legacy or exact event `sxform` M3 |
| `spacecraft_state_interpolation`, `stm_interpolation` | interpolation policy labels |
| `event_update_tolerance_s`, `event_equation_tolerance_s`, `event_max_iterations` | M3 strict solver policy |
| `dropped_measurements`, `dropped_measurement_reason` | aggregate M3 generation loss; reason currently generic |
| `noise_enabled`, `noise_seed`, `station_sigmas` | stochastic provenance; seed may be unavailable |
| `troposphere_model`, `ionosphere_model`, `solar_plasma_model` | currently `none` in generic metadata |

`ScenarioResult`/summary CSV currently transports a selected subset of this
dictionary. Full M3 metadata retention at artifact level is not guaranteed.

## Appendix E - Source bibliography

1. Moyer, Theodore D. *Formulation for Observed and Computed Values of Deep
   Space Network Data Types for Navigation*. JPL Publication 00-7, DESCANSO
   Monograph 2, 2000. https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_all.pdf
2. Thornton, Catherine L.; Border, James S. *Radiometric Tracking Techniques
   for Deep-Space Navigation*. DESCANSO Monograph 1, 2003.
   https://descanso.jpl.nasa.gov/monograph/series1/Descanso1_all.pdf
3. NAIF/JPL. *Aberration Corrections Required Reading*.
   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/abcorr.html
4. NAIF/JPL. *Frames Required Reading*.
   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html
5. NAIF/JPL. *Time Required Reading*.
   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html
6. NAIF/JPL. `spkezr_c`, `stelab_c`, `sxform_c`, `pxform_c` API headers.
   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/index.html
7. Petit, Gérard; Luzum, Brian, eds. *IERS Conventions (2010)*, IERS Technical
   Note 36. https://iers-conventions.obspm.fr/content/tn36.pdf
8. IOGP/EPSG Geodetic Parameter Dataset. `WGS 84`, Ellipsoid EPSG:7030;
   information source DMA Technical Manual 8350.2-B.
   https://epsg.org/ellipsoid_7030/WGS-84.html
9. Tapley, Byron D.; Schutz, Bob E.; Born, George H. *Statistical Orbit
   Determination*. Elsevier Academic Press, 2004, ISBN 978-0-12-683630-1.
   https://www.sciencedirect.com/book/monograph/9780126836301/statistical-orbit-determination
10. Montenbruck, Oliver; Gill, Eberhard. *Satellite Orbits: Models, Methods and
    Applications*. Springer, 2000. https://doi.org/10.1007/978-3-642-58351-3
11. Vallado, David A. *Fundamentals of Astrodynamics and Applications*, 5th
    ed. Microcosm Press, 2022. https://microcosmpress.com/vallado/
12. Bierman, Gerald J. *Factorization Methods for Discrete Sequential
    Estimation*. Academic Press, 1977.
    https://shop.elsevier.com/books/factorization-methods-for-discrete-sequential-estimation/bierman/978-0-12-097350-7
13. Levenberg, Kenneth. “A Method for the Solution of Certain Non-Linear
    Problems in Least Squares.” 1944. https://doi.org/10.1090/qam/10666
14. Marquardt, Donald W. “An Algorithm for Least-Squares Estimation of
    Nonlinear Parameters.” 1963. https://doi.org/10.1137/0111030
15. Fisher, Ronald A. “On the Mathematical Foundations of Theoretical
    Statistics.” 1922. https://doi.org/10.1098/rsta.1922.0009
16. Hermann, Robert; Krener, Arthur J. “Nonlinear Controllability and
    Observability.” 1977. https://doi.org/10.1109/TAC.1977.1101601

## Appendix F - Equation-to-code cross-reference

| Eq ID | Equation | Source | Production symbol | Test symbol/file | Status |
|---|---|---|---|---|---|
| E-GR-01 | `R=norm(rho)` | project + standard geometry | `position_observables` | measurement/frame tests | verified |
| E-GA-01 | `A=atan2(E,-S)` | project convention | `ecef2razel_sez`, `_razel_scalar` | cardinal/wrap tests | verified |
| E-GA-02 | angle derivative rows §23.2 | project derivation | `_range_az_el_partials_sez`, `_position_measurement_jacobian_from_unit_los` | FD/mutation tests | verified |
| E-CN-01 | `tau=norm(r_sc(tr-tau)-r_st(tr))/c` | NAIF reception convention | `solve_one_way_light_time` | static/linear tests | internally verified |
| E-CN-02 | `d tau/dx0` denominator form | project implicit derivation, Moyer context | `one_way_light_time_initial_state_sensitivity` | full-state FD | verified |
| E-CN-03 | unit LOS projector | project chain rule | `_unit_line_of_sight_sensitivity` | tangency/FD | verified |
| E-CS-01 | Newtonian reception +S | NAIF | `apply_stellar_aberration` | direct `stelab` | operator verified |
| E-CS-02 | tangent central FD | project | `_stellar_aberration_local_jacobian` | plateau/tangent tests | verified hybrid |
| E-RR-01 | `Rdot=u^T v_rel` | standard/project | `instantaneous_geometric_range_rate` | analytic/FD | verified |
| E-RR-02 | full sxform local derivative | project | `compute_range_rate_residuals_analytic` | frame FD | verified |
| E-CD-01 | endpoint RTLT m/s form | Moyer concept/project scaling | `two_way_counted_doppler_observable` | static/receding/FD | internally verified |
| E-CD-02 | Hz endpoint form | project simplified constant-frequency | same | frequency/clock test | limited, not DSN-complete |
| E-M3-01 | `Gu,Gd,Gtr` | Moyer event concept/project | `solve_two_way_range_events` | independent root | verified |
| E-M3-02 | raw/cal observables | project convention | `two_way_range_from_solution` | factor-two/delay tests | verified |
| E-M3-03 | `dy/dx0=-Gy^-1Gx` | implicit function theorem/project | `two_way_range_event_sensitivity` | step-sweep FD | verified |
| E-EST-01 | `Hw=R^-1/2 H`, `F=Hw^T Hw` | Gaussian linearization/Tapley/Fisher context | observability/estimator helpers | observability tests | verified implementation |

## Appendix G - Open questions

1. UKF için en doğru kısa-vadeli policy hard rejection mı, yoksa bounded local
   history kullanan profile-aware sigma measurement implementation'ı mı?
2. Range-rate `apparent_one_way + implicit_light_time` config'i hemen
   reddedilmeli mi, yoksa M2 helper'ı dört-row companion block'a genişletilmeli
   mi?
3. Counted Doppler nonzero delay desteklenmeye devam edecekse M3 event solver
   ortaklaştırılmalı mı, yoksa backward-compatible ayrı legacy profile mı
   tanımlanmalı?
4. Counted exact-event `sxform` maliyeti uzun UKF kampanyalarında kabul edilebilir
   mi; cadence-based error budget hangi sigma profiline göre kapatılmalı?
5. M3 için dedicated two-way range sigma station-level mı, link/profile-level mı
   olmalı?
6. İlk external two-way oracle GMAT, Orekit veya Tudat'tan hangisi olacak ve aynı
   time/frame/correction contract'ı hangisinde en az belirsizlikle kurulabilir?
7. Real-data validation için erişilebilir tracking dataset ve measurement ICD
   hangisidir? Mevcut repository'de doğrulanmış real tracking dataset yoktur.

Bu soruların hiçbiri mevcut production sonucuyla cevaplanmış gibi sunulmamalıdır.
