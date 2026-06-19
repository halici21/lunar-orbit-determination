# Lunar OD Python Port - Arayüz Tasarım README

Bu doküman, Lunar OD Python portu için geliştirilecek bir kullanıcı arayüzünün
hangi proje bileşenlerini göstermesi, hangi ayarları kullanıcıya açması ve hangi
çıktıları üretip görselleştirmesi gerektiğini tarif eder.

Ana hedef, kullanıcıya tek tek Python scriptlerini ezberletmeden şu işleri
yaptırmaktır:

- Yörünge, görünürlük, ölçüm üretimi ve OD senaryolarını seçmek
- İstasyon ağı, ölçüm tipi, estimator ve start mode ayarlarını değiştirmek
- Geometric range-rate ve two-way counted Doppler modellerini karşılaştırmak
- Q/R noise modeli, bias, Monte Carlo ve diagnostics sonuçlarını görmek
- PNG/CSV raporlarını tek panelden üretmek ve incelemek

Bu arayüz bir mission operasyon sistemi değil; mevcut kod tabanına uygun olarak
research-grade synthetic OD sandbox arayüzü olmalıdır.

## 1. Proje Omurgası

Python tarafındaki ana paket:

```text
python_port/lunar_od/
```

Ana modüller:

| Modül | Arayüzdeki karşılığı |
|---|---|
| `config.py` | İstasyon kataloğu, sigma/R ayarları |
| `orbit.py` | Başlangıç yörünge/state dönüşümleri |
| `dynamics.py` | Ay merkezli dinamik model ve propagasyon |
| `ephemeris.py` | SPICE Earth/Sun ephemeris interpolasyonu |
| `visibility.py` | İstasyon görünürlüğü, yay çıkarımı |
| `measurements.py` | Position ve range-rate ölçüm üretimi/residual |
| `radiometrics.py` | Geometric RR ve two-way counted Doppler fiziği |
| `estimators.py` | BLS/LM ve SRIF/QR estimatorları |
| `scenarios.py` | Arc-by-arc OD runner |
| `reporting.py` | PNG/CSV scenario ve visibility raporları |
| `diagnostics.py` | Residual, chi-square, Mahalanobis, NIS/NEES, boundary-jump |
| `observability.py` | Fisher bilgi, rank, condition number |
| `q_tuning.py` | Process-noise Q grid/sweep |
| `monte_carlo.py` | Seed kontrollü trial ve aggregation |
| `scenario_config.py` | JSON scenario config schema/validator |
| `measurement_ingestion.py` | Internal ObsData CSV okuma |
| `thesis_matrix.py` | Sabit 16-case tez factorial matrix |

Arayüz için en önemli fikir: kullanıcı doğrudan bu modülleri görmez; ekranlar
bu modüllerin üstünde sade iş akışları sunar.

## 2. Önerilen Arayüz Sayfaları

### 2.1 Dashboard

Amaç: Projenin son durumunu ve üretilmiş raporları hızlı göstermek.

Göstergeler:

- Son test durumu: `105 tests OK`
- SPICE kernel durumu: kernel dizini bulundu mu?
- Son üretilen raporlar
- Son 4 günlük campaign özeti
- Son 28 günlük ITU visibility/OD özeti
- Son range-rate physics karşılaştırma özeti

Okunacak dosyalar:

```text
python_port/results/*.csv
python_port/results/*.png
python_port/results/run_all_experiments_summary.csv
python_port/results/run_all_experiments_summary.json
```

Önerilen kartlar:

- Test Suite
- SPICE Status
- Visibility Reports
- OD Reports
- Doppler Physics
- Q/R Model
- Diagnostics

### 2.2 Scenario Builder

Amaç: Kullanıcının bir OD senaryosu tanımlaması.

Mevcut JSON config alanları:

| Alan | Seçenek / Tip |
|---|---|
| `name` | string |
| `measurement_type` | `position`, `range_rate` |
| `estimator_type` | `bls_lm`, `srif` |
| `start_mode` | `cold`, `hot`, `formal`, `sqrt_formal` |
| `network` | `single`, `multi` |
| `duration_h` | pozitif sayı |
| `sample_step_s` | pozitif sayı |
| `max_iter` | pozitif integer |
| `rtol` | pozitif sayı |
| `atol` | pozitif sayı |
| `noise` | boolean |
| `bias_mode` | `none`, `global`, `station_angles`, `station_full` |
| `output_dir` | path |

Cross-field kuralları:

- `sqrt_formal` sadece `srif` ile kullanılmalı
- bias solve-for şimdilik sadece `range_rate` + `srif` için kullanılmalı
- `position` ölçümünde `bias_mode` kapalı olmalı

CLI karşılığı:

```powershell
python python_port/examples/scenario_config_cli.py --schema
python python_port/examples/scenario_config_cli.py scenario.json --write-normalized python_port/results/scenario.normalized.json
```

Not: Mevcut `scenario_config.py` config doğruluyor; henüz config dosyasından
tam propagation + visibility + measurement + estimator zincirini başlatan full
runner değildir. Arayüz yazılırken bu eksik bağlanacak ana workflow olabilir.

### 2.3 Station & Network Panel

Amaç: İstasyonları ve network seçimlerini göstermek.

Mevcut station setleri:

Position-only ve range-rate station listeleri `config.py` içinde tanımlıdır.

Önemli istasyonlar:

```text
ITU Ayazaga
Goldstone DSN
Madrid DSN
Canberra DSN
Daejeon KGS
Dongara KGS
Chuuk KGS
Svalbard KGS
Malargue ESA
Cebreros ESA
New Norcia ESA
Evpatoria RUS
Ussuriisk RUS
Bear Lakes RUS
Byalalu ISRO
```

Canonical thesis networks:

| Network | İstasyonlar |
|---|---|
| `single` | Canberra DSN |
| `multi` | Goldstone DSN, Madrid DSN, Canberra DSN, ITU Ayazaga |

Station noise varsayımları:

| Station family | Range sigma | Range-rate sigma | Açı sigma |
|---|---:|---:|---:|
| ITU Ayazaga | `94 m` | `1e-3 m/s` | `0.005 deg` |
| DSN/diger | `5 m` | `1e-4 m/s` | `0.001 deg` |

Arayüzde gösterilecek alanlar:

- İstasyon adı
- Enlem, boylam, yükseklik
- Range sigma
- Range-rate sigma
- Açı sigma
- Ölçüm tipi desteği
- Network üyeliği

### 2.4 Dynamics & Propagation Panel

Amaç: Truth trajectory ve estimator propagation ayarlarını göstermek.

Mevcut ana dinamik model:

```text
Moon point-mass + Earth third-body + Sun third-body
```

State:

```text
[x, y, z, vx, vy, vz]
```

Birimler:

```text
position: m
velocity: m/s
time: s
```

Propagator:

```text
scipy.integrate.solve_ivp(method="DOP853")
```

Ayarlar:

| Ayar | Anlam |
|---|---|
| `duration_h` | campaign/senaryo süresi |
| `sample_step_s` | ölçüm/rapor zaman adımı |
| `ephemeris_step_s` | SPICE Earth/Sun sampling adımı |
| `rtol` | propagasyon toleransı |
| `atol` | propagasyon toleransı |

Mevcut yardımcı fizik:

- J2 acceleration helper var
- numerical J2 gravity-gradient helper var
- ana propagator hâlâ point-mass + third-body kullanıyor

Arayüzde bu açık ayrılmalı:

```text
Aktif force model: point-mass + Earth/Sun third-body
Mevcut ama ana senaryoya bağlı olmayan helper: J2
```

### 2.5 Visibility Panel

Amaç: Dünya istasyonlarından Ay yörüngesindeki uzay aracının görünürlüğünü
göstermek.

Mevcut visibility modelleri:

| Model | Açıklama |
|---|---|
| GST-based | Yaklaşık Earth rotation/GST tabanlı |
| SPICE-transform | `J2000 -> ITRF93` transform tabanlı |

Visibility ayarları:

| Ayar | Örnek |
|---|---:|
| Minimum yükselim açısı | `5 deg` |
| Maximum gap bridge | `1200 s`, `1800 s` |
| Moon occultation | aktif |
| Station mask | tek veya çoklu istasyon |

Görselleştirilecek çıktılar:

- Zaman çizgisi üzerinde görünür/görünmez maskesi
- İstasyon bazlı visibility
- Network visibility
- Yay başlangıç/bitiş saatleri
- Yay süresi histogramı
- OD-ready yay sayısı
- Toplam görünürlük saati
- Görünürlük oranı

İlgili raporlar:

```text
visibility_report_from_fixture.py
long_visibility_report.py
compare_visibility_models.py
campaign_4day_visibility_rr_report.py
campaign_28day_itu_report.py
campaign_diagnostic_plots.py
```

Örnek çıktı dosyaları:

```text
python_port/results/campaign_28day_itu_visibility_analysis.png
python_port/results/campaign_28day_itu_visibility_summary.csv
python_port/results/campaign_28day_itu_all_visibility_arcs.png
python_port/results/campaign_28day_itu_all_visibility_arcs.csv
```

### 2.6 Measurement Model Panel

Amaç: Ölçüm tiplerini, noise/bias ayarlarını ve residual sırasını göstermek.

Measurement type seçenekleri:

| Tip | Bileşenler |
|---|---|
| `position` | `[range, az, el]` |
| `range_rate` | `[range, range_rate, az, el]` |

Residual tanımı:

```text
residual = observed - computed = O - C
```

R/sigma sırası:

```text
position:
[range, az, el]

range_rate:
[range, range_rate, az, el]
```

Range-rate physics seçenekleri:

| Mode | Açıklama |
|---|---|
| `geometric_instantaneous` | Anlık geometrik görüş doğrultusu menzil hızı |
| `two_way_counted_doppler` | Sabit uplink + turnaround ratio + count interval ile two-way Doppler m/s eşdeğeri |

Two-way Doppler ayarları:

| Ayar | Varsayılan |
|---|---:|
| `count_interval_s` | `1`, `60`, `300` karşılaştırma raporunda |
| `uplink_frequency_hz` | `7.2e9` |
| `turnaround_ratio` | `880/749` |
| `output_unit` | `mps_equivalent` |
| `bias_rr_mps` | kullanıcı girişi, örn. `0.01` |

Önemli not:

Bu model tam operasyonel DSN Doppler değildir. Şunlar henüz yoktur:

- ramped frequency
- troposfer/iyonosfer
- station clock
- transponder delay
- relativistik düzeltmeler
- gerçek DSN metric parser

Arayüzde bu uyarı mutlaka görünmelidir.

### 2.7 Estimator Panel

Amaç: OD çözüm yöntemini seçmek ve sonuçlarını göstermek.

Estimator seçenekleri:

| Estimator | Açıklama |
|---|---|
| `bls_lm` | Batch least squares / Levenberg-Marquardt |
| `srif` | Square-root information / QR estimator |

Measurement destekleri:

- position-only
- range-rate

Start mode seçenekleri:

| Start mode | Anlam |
|---|---|
| `cold` | Her arc bağımsız/perturbed başlangıç |
| `hot` | Önceki arc sonucunu sonraki arc'a taşıma |
| `formal` | Formal covariance/information handoff |
| `sqrt_formal` | SRIF posterior R faktörüyle square-root handoff |

Bias mode seçenekleri:

| Bias mode | Çözülen bias |
|---|---|
| `none` | bias yok |
| `global` | global `[range, range_rate, az, el]` |
| `station_angles` | istasyon bazlı açı bias |
| `station_full` | istasyon bazlı `[range, range_rate, az, el]` |

Arayüzde gösterilecek estimator metrikleri:

- final position error
- final velocity error
- iteration count
- stop reason
- final cost
- cost reduction
- step norm
- condition number
- rank
- rejected outlier count
- active weight fraction
- posterior covariance/information varsa summary
- bias estimate ve recovered bias error

### 2.8 Q/R Noise Panel

Amaç: Kullanıcının ölçüm güveni ve model güvenini anlaması.

R modeli:

```text
R = diag(sigma^2)
```

Sıra:

```text
position:    [range, az, el]
range_rate: [range, range_rate, az, el]
```

Q modeli:

```text
[x, y, z, vx, vy, vz]
```

Canonical Q grid:

| Case | Position sigma | Velocity sigma |
|---|---:|---:|
| `Q0` | none | none |
| `Qsmall` | `0.1 m` | `1e-5 m/s` |
| `Qmedium` | `1.0 m` | `1e-4 m/s` |
| `Qlarge` | `10.0 m` | `1e-3 m/s` |

Arayüzde yapılacaklar:

- R değerlerini tablo olarak göster
- Station seçilince sigma değerlerini göster
- Q case seçimi için segmented control/dropdown kullan
- Q tuning sonuçlarını CSV/plot üzerinden göster
- “Bu değerler kalibrasyon değil, sentetik senaryo varsayımıdır” uyarısını göster

### 2.9 Diagnostics Panel

Amaç: Çözümün sadece final error ile değil, residual ve covariance kalitesiyle
de değerlendirilmesi.

Mevcut diagnostics:

| Diagnostic | Açıklama |
|---|---|
| raw RMS | residual RMS |
| whitened RMS | sigma ile normalize residual RMS |
| chi-square | ağırlıklı residual kalite metriği |
| reduced chi-square | degree-of-freedom normalize hali |
| Mahalanobis | covariance tabanlı uzaklık |
| lag-1 autocorrelation | residual zamansal korelasyon |
| boundary jump | arc boundary geçiş hatası |
| state-bias correlation | state ile bias birbirini yiyor mu; scenario CSV prior/posterior max correlation yazar |
| NIS | innovation ve S kovaryansi R/P ile uyumlu mu? |
| NEES | state hatasi ve P kovaryansi truth ile uyumlu mu? |

Arayüzde önerilen grafikler:

- residual vs time
- whitened residual histogram
- chi-square bar chart
- arc boundary jump vs arc id
- condition number vs arc id
- final error vs condition number
- bias estimate vs true bias
- state-bias correlation heatmap
- NIS/NEES vs time veya arc id

### 2.10 Experiment Runner Panel

Amaç: Mevcut example scriptlerini tek panelden koşturmak.

Mevcut orchestrator:

```powershell
python python_port/examples/run_all_experiments.py --quick
python python_port/examples/run_all_experiments.py --full
python python_port/examples/run_all_experiments.py --full --list
python python_port/examples/run_all_experiments.py --full --dry-run
```

Orchestrator özellikleri:

- quick/full mode
- only/skip filter
- dry-run
- list
- timeout
- stop-on-fail
- CSV/JSON run summary

Arayüz kontrolleri:

- Run mode: `quick`, `full`
- Experiment list checkbox
- Requires SPICE badge
- Long-running badge
- Dry-run toggle
- Timeout input
- Stop on fail toggle
- Run progress/status
- stdout/stderr tail viewer

Çıktılar:

```text
python_port/results/run_all_experiments_summary.csv
python_port/results/run_all_experiments_summary.json
```

## 3. Önemli Example Raporları

### 3.1 Kısa synthetic cold/hot-start raporu

Komut:

```powershell
python python_port/examples/synthetic_hot_start_report.py
```

Çıktılar:

```text
python_port/results/synthetic_hot_start_comparison.png
python_port/results/synthetic_hot_start_summary.csv
```

Arayüzde:

- cold vs hot final error
- arc bazlı error
- estimator sanity

### 3.2 Fixture visibility raporu

Komut:

```powershell
python python_port/examples/visibility_report_from_fixture.py
```

Çıktılar:

```text
python_port/results/visibility_single_analysis.png
python_port/results/visibility_single_summary.csv
python_port/results/visibility_multi_analysis.png
python_port/results/visibility_multi_summary.csv
```

### 3.3 Two-way Doppler karşılaştırması

Komut:

```powershell
python python_port/examples/compare_range_rate_physics.py
```

Bias örneği:

```powershell
python python_port/examples/compare_range_rate_physics.py --bias-rr-mps 0.01 --output-prefix range_rate_physics_bias_rr_0p01
```

Çıktılar:

```text
python_port/results/range_rate_physics_comparison.png
python_port/results/range_rate_physics_comparison.csv
python_port/results/range_rate_physics_residual_mismatch.csv
python_port/results/range_rate_physics_comparison_summary.csv
python_port/results/range_rate_physics_bias_rr_0p01_comparison.png
python_port/results/range_rate_physics_bias_rr_0p01_comparison_summary.csv
```

Arayüzde:

- geometric RR eğrisi
- two-way Doppler m/s eşdeğeri eğrisi
- `two-way - geometric` farkı
- `Tc = 1/60/300 s` karşılaştırması
- wrong-model residual sigma
- bias residual

### 3.4 4 günlük campaign

Komut:

```powershell
python python_port/examples/campaign_4day_visibility_rr_report.py
```

Başlıca çıktılar:

```text
python_port/results/campaign_4day_summary.csv
python_port/results/campaign_4day_rr_od_summary.csv
python_port/results/campaign_4day_rr_od_comparison.png
python_port/results/campaign_4day_multi_dsn_itu_visibility_analysis.png
python_port/results/campaign_4day_multi_dsn_itu_visibility_summary.csv
```

Son kayıtlı önemli metrikler:

```text
Single Canberra visibility: 31.17 h, fraction 0.324, 29 arc
Multi DSN+ITU visibility: 59.83 h, fraction 0.622, 50 arc
Multi extended visibility: 60.00 h, fraction 0.624, 50 arc
```

### 3.5 28 günlük ITU-only campaign

Komut:

```powershell
python python_port/examples/campaign_28day_itu_report.py
```

Çıktılar:

```text
python_port/results/campaign_28day_itu_summary.csv
python_port/results/campaign_28day_itu_selected_arcs.csv
python_port/results/campaign_28day_itu_rr_od_summary.csv
python_port/results/campaign_28day_itu_rr_od_comparison.png
python_port/results/campaign_28day_itu_visibility_analysis.png
python_port/results/campaign_28day_itu_visibility_summary.csv
```

Son kayıtlı metrikler:

```text
Visible sample time: 216.17 h
Visibility fraction: 0.322
Visibility arcs: 114
OD-ready arcs: 105
Median arc span: 1.00 h
Max arc span: 13.00 h
```

### 3.6 Tüm ITU OD-ready yaylar

Komut:

```powershell
python python_port/examples/campaign_28day_itu_all_arc_hot_report.py --max-iter 20 --rtol 1e-10 --atol 1e-11
```

Çıktılar:

```text
python_port/results/campaign_28day_itu_all_arc_hot_maxiter20_aggregate.csv
python_port/results/campaign_28day_itu_all_arc_hot_maxiter20_errors.csv
python_port/results/campaign_28day_itu_all_arc_hot_maxiter20_rr_od_summary.csv
python_port/results/campaign_28day_itu_all_arc_hot_maxiter20_errors.png
```

Son kayıtlı metrikler:

```text
OD-ready arc count: 105
Median final position error: 1.32e-2 m
P90 final position error: 5.05e-2 m
P95 final position error: 7.94e-2 m
Max final position error: 1.26e-1 m
Median condition number: 6.44e4
P90 condition number: 1.16e5
```

### 3.7 Thesis factorial matrix

Komut:

```powershell
python python_port/examples/thesis_factorial_report.py
```

Çıktılar:

```text
python_port/results/thesis_factorial_summary.png
python_port/results/thesis_factorial_aggregate.csv
python_port/results/thesis_factorial_detail.csv
```

Canonical matrix:

```text
network: single, multi
measurement: position, range_rate
estimator: bls_lm, srif
start: cold, hot
case count: 16
```

### 3.8 Formal handoff ve process-noise raporları

Komutlar:

```powershell
python python_port/examples/formal_handoff_report.py
python python_port/examples/formal_handoff_process_noise_report.py
python python_port/examples/formal_bias_handoff_report.py
```

Çıktılar:

```text
python_port/results/formal_handoff_comparison.png
python_port/results/formal_handoff_covariance.png
python_port/results/formal_handoff_summary.csv
python_port/results/formal_handoff_process_noise_comparison.png
python_port/results/formal_handoff_process_noise_covariance.png
python_port/results/formal_handoff_process_noise_summary.csv
python_port/results/formal_bias_handoff_comparison.png
python_port/results/formal_bias_handoff_bias_recovery.png
python_port/results/formal_bias_handoff_summary.csv
python_port/results/formal_bias_handoff_bias_recovery.csv
```

Arayüzde:

- hot/formal/sqrt_formal karşılaştırması
- covariance büyümesi
- Q sweep sonuçları
- bias recovery plot

## 4. Arayüz Input Modeli

Arayüzde temel input grupları şöyle olabilir.

### 4.1 Campaign ayarları

| Input | Tip | Örnek |
|---|---|---|
| Campaign name | string | `itu_28day_hot` |
| Duration | number | `672 h` |
| Truth sample step | number | `600 s` |
| Ephemeris sample step | number | `3600 s` |
| Start epoch | string | fixture epoch |
| Output directory | path | `python_port/results` |

### 4.2 Visibility ayarları

| Input | Tip | Örnek |
|---|---|---|
| Station set | multi-select | ITU, DSN |
| Visibility model | select | GST, SPICE transform |
| Minimum elevation | number | `5 deg` |
| Max gap bridge | number | `1800 s` |
| Moon occultation | toggle | on |

### 4.3 Measurement ayarları

| Input | Tip | Örnek |
|---|---|---|
| Measurement type | select | `position`, `range_rate` |
| Range-rate physics | select | `geometric`, `two_way_counted_doppler` |
| Count interval | number/list | `60 s`, `[1, 60, 300]` |
| Noise | toggle | on/off |
| Range bias | number | `m` |
| RR bias | number | `m/s` |
| Az/el bias | number | `rad` veya `deg` UI input |

### 4.4 Estimator ayarları

| Input | Tip | Örnek |
|---|---|---|
| Estimator | select | `srif` |
| Start mode | select | `hot` |
| Bias mode | select | `station_full` |
| Max iteration | number | `20` |
| R tolerance | number | `1e-10` |
| A tolerance | number | `1e-11` |
| Outlier rejection | toggle/number | scalar sigma clip |

### 4.5 Q/R ayarları

| Input | Tip | Örnek |
|---|---|---|
| R station preset | select | ITU, DSN |
| Range sigma | number | `94 m` |
| RR sigma | number | `1e-3 m/s` |
| Angle sigma | number | `0.005 deg` |
| Q case | select | `Q0`, `Qsmall`, `Qmedium`, `Qlarge` |
| Custom Q position sigma | number | `1 m` |
| Custom Q velocity sigma | number | `1e-4 m/s` |

## 5. Arayüz Output Modeli

### 5.1 Ortak run metadata

Her arayüz koşusu bir run objesi üretmeli:

```json
{
  "run_id": "itu_28day_hot_20260608_120000",
  "scenario_name": "itu_28day_hot",
  "started_at": "2026-06-08T12:00:00+03:00",
  "ended_at": "2026-06-08T12:04:20+03:00",
  "status": "success",
  "command": "python python_port/examples/...",
  "output_dir": "python_port/results"
}
```

### 5.2 Visibility outputs

| Output | Tip |
|---|---|
| visibility timeline | PNG |
| station/network summary | CSV |
| arc table | CSV |
| duration histogram | plot |
| OD-ready count | scalar |

### 5.3 OD outputs

| Output | Tip |
|---|---|
| final position error per arc | CSV/plot |
| final velocity error per arc | CSV/plot |
| condition number per arc | CSV/plot |
| rank per arc | CSV |
| stop reason per arc | CSV |
| covariance summary | CSV/plot |
| bias recovery | CSV/plot |

### 5.4 Diagnostics outputs

| Output | Tip |
|---|---|
| residual RMS | scalar/table |
| whitened RMS | scalar/table |
| chi-square | scalar/table |
| Mahalanobis | scalar/table |
| lag-1 autocorrelation | scalar/table |
| boundary jump | table/plot |
| state-bias correlation | CSV/table/heatmap |
| convergence reason flags | CSV/table/bar chart |
| NIS / normalized NIS | scalar/table/plot |
| NEES / normalized NEES | scalar/table/plot |

## 6. Arayüz İçin Önerilen İlk MVP

İlk arayüz sürümünde her şeyi full custom runner yapmak yerine mevcut scriptleri
çağıran bir panel daha hızlı sonuç verir.

MVP ekranları:

1. Dashboard
2. Experiment Runner
3. Results Browser
4. Doppler Physics Comparison
5. 28-Day ITU Campaign Viewer
6. Q/R Noise Model Viewer

MVP'de yapılacaklar:

- `run_all_experiments.py --quick` ve `--full --dry-run` çalıştır
- `compare_range_rate_physics.py` için UI parametreleri aç
- `campaign_28day_itu_report.py` sonuçlarını oku ve göster
- `campaign_diagnostic_plots.py` çıktısı olan PNG/CSV dosyalarını göster
- README'deki Q/R tablolarını UI içinde statik bilgi paneli olarak göster

MVP'de ertelenebilecekler:

- full custom scenario runner
- gerçek zamanlı progress parsing
- web üzerinden SPICE kernel yönetimi
- real measurement ingestion workflow
- UKF/EKF ekranları

## 7. İkinci Aşama Arayüz

İkinci aşamada script çağırma yerine modülleri doğrudan kullanan backend
eklenebilir.

Önerilen backend endpointleri:

| Endpoint | İş |
|---|---|
| `GET /api/stations` | station listesi |
| `GET /api/scenario-schema` | JSON schema |
| `POST /api/visibility/run` | visibility analizi |
| `POST /api/measurements/generate` | sentetik ölçüm üretimi |
| `POST /api/od/run` | estimator koşusu |
| `POST /api/radiometrics/compare` | geometric vs two-way Doppler |
| `POST /api/q-sweep/run` | Q sweep |
| `GET /api/results` | sonuç dosyaları |
| `GET /api/results/{id}` | tek run çıktıları |

Önerilen run klasör yapısı:

```text
python_port/results/ui_runs/
  run_20260608_120000_itu_28day/
    config.json
    run_summary.json
    visibility_summary.csv
    od_summary.csv
    diagnostics.csv
    plots/
      visibility.png
      od_errors.png
      residuals.png
```

## 8. Arayüzde Mutlaka Yazılması Gereken Uyarılar

Kullanıcı yanlış yorumlamasın diye bu uyarılar görünür olmalı:

1. Clean/noise-free geometric RR sonuçları fiziksel DSN gerçekçiliği değildir.
2. Two-way counted Doppler modeli sade modeldir; full DSN Doppler değildir.
3. R sigma değerleri saha kalibrasyonu değil, senaryo varsayımıdır.
4. Q değerleri otomatik kalibre edilmemiş canonical grid değerleridir.
5. Formal covariance local linear/noise model sonucudur; tek başına gerçek hata
   garantisi değildir.
6. Tek istasyon/kısa yaylarda rank ve condition number kritik hale gelir.
7. Bias solve-for ile initial state korelasyonu izlenmelidir.
8. `MaxIter` stop reason her zaman başarısızlık anlamına gelmez; cost ve step
   norm ile birlikte yorumlanmalıdır.

## 9. Test ve Doğrulama

Arayüz geliştirmeden önce temel test:

```powershell
python -m unittest discover -s python_port/tests -t python_port
```

Son kayıtlı sonuç:

```text
105 tests OK
```

Dar testler:

```powershell
python -m unittest discover -s python_port/tests -t python_port -p test_measurements.py
python -m unittest discover -s python_port/tests -t python_port -p test_od_contracts.py
python -m unittest discover -s python_port/tests -t python_port -p test_q_tuning.py
python -m unittest discover -s python_port/tests -t python_port -p test_observability.py
python -m unittest discover -s python_port/tests -t python_port -p test_diagnostics.py
```

Arayüz backend'i bu testleri bozmamalıdır.

## 10. UI İçin Öncelikli Geliştirme Sırası

Önerilen sıra:

1. Results Browser: mevcut PNG/CSV dosyalarını göster
2. Experiment Runner: mevcut scriptleri butondan çalıştır
3. Doppler Physics Comparison: `Tc` ve bias ayarlarıyla grafik üret
4. Visibility Viewer: 4 gün ve 28 gün raporlarını görselleştir
5. OD Summary Viewer: final error, condition, rank, stop reason tabloları
6. Q/R Panel: R sigma ve Q grid gösterimi
7. Scenario Builder: JSON config üretme/doğrulama
8. Full Scenario Runner: config'ten propagation + visibility + OD zinciri
9. Diagnostics Dashboard: residual, chi-square, NIS/NEES, boundary jump
10. UKF/EKF ekranları

Bu sırayla gidilirse arayüz önce mevcut çalışan raporları görünür hale getirir,
sonra yavaş yavaş yeni hesap zincirlerini doğrudan kontrol etmeye başlar.

