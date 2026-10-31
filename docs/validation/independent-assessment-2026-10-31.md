# Bağımsız teknik değerlendirme — 2026-10-31

**Kapsam:** `main` @ `549093d` (PR #64 sonrası), ayrıca birleşmemiş `phase-f1-lro-residuals`
(F2r/F2s) ve `feature/f3-lro-ekf` (F3) dalları. Hesaplar, fizik, hesaplar arası tutarlılık,
NASA/ESA standartlarına uygunluk ve risk.

**Yöntem:** dokuz salt-okunur inceleme (dinamik/çerçeve/zaman; ölçüm ve tahminci tutarlılığı;
V&V ve standartlar; kuvvet modelleri, ölçüm geometrisi ve tahminciler satır satır; gerçek LRO veri
hattı; test kehaneti kalitesi; konfigürasyon, masaüstü ve tekrarlanabilirlik). En yüksek etkili
iddialar kodda ayrıca okundu, sekizi de çalıştırılabilir kanıt testine dönüştürüldü
([`tests/test_assessment_evidence.py`](../../tests/test_assessment_evidence.py)).

**İşaretler:** **✔** kodda okunarak doğrulandı; **T** kanıt testiyle ölçüldü (aşağıda); işaretsiz
satırlar inceleme bulgusudur ve ilgili iş paketinde önce başarısız olan bir testle kanıtlanmadan
düzeltilmez. Önem: **K** kritik, **B** büyük, **O** orta, **D** düşük.

**Kullanım amacı** ve kabul ölçütleri: [intended-use](../assurance/intended-use.md). Program:
[V&V programı](../assurance/vv-program.md). Risk kaydı: [risk-register](../assurance/risk-register.md).

## 0. Kanıt testleri (hesaplandığı gibi)

Her bulgu testi `xfail(strict=True)`; bulgu geçerliyken XFAIL, düzeltilince XPASS olur ve işaret
düzeltmeyle birlikte kaldırılır. Her birinin yanında kehaneti kolay durumda doğrulayan işaretsiz
bir kontrol testi vardır.

| Bulgu | Test | Sonuç | Ölçülen büyüklük |
|---|---|---|---|
| MEAS-03 | uzun yayda sayılan Doppler yakınsaması | XFAIL | `converged` oranı yay başlangıç epoch'una göre: 0 s → 200/200; 1.4e5 s → 140; 3e5 s → 66; 1.1e6 s → 16; 2.5e6 s → 12 |
| MEAS-04 | UKF yerel J2000→ITRF93 dönüşümü ortonormal ve mm doğru | PASS (WP1.1 kapattı; gerçek sxform karşısında 60/240 s ızgarada < 1 mm) — önceki XFAIL | ızgara ortasında ‖RRᵀ−I‖ = 6.8e-6 / 1.5e-5 / 1.1e-4; istasyon konum hatası 15 / 34 / 244 m (60 / 90 / 240 s ızgara) |
| DYN-05 | Ay yönelimi ızgara ortasında sürekli | XFAIL | 60 s ızgarada sıçrama 1.6e-4 rad (= ω·cadence) |
| EST-09 | DMC Q uzun boşlukta sonlu, PSD ve doğru | XFAIL | alt-adımlı referansa göre bağıl hata: dt/τ=10 → 1.2e-12; 30 → 8.7e-5; 144 → 7.8e42 (en küçük özdeğer −3.1e37) |
| EST-19 | UKF P0 enjekte edilen soğuk başlangıç σ'sı ile aynı | XFAIL | P0 100 m / 0.1 m/s; enjekte 180 m / 0.03 m/s |
| EST-01 | 6-durum ve 42-durum yayılım yolları estimator toleransında < 1 mm | XFAIL | 4 saatlik 100 km yörüngede en büyük konum farkı: 0.041 m (rtol 1e-11), 0.214 m (1e-10), 2.95 m (1e-8) |
| DYN-10 | Kuvvet kimliği efemeris kernel'i ve interpolasyonunu içerir | XFAIL | `ForceModelContract` ve `ForceExecutionSpec` alanlarında kernel/SPK/interpolasyon yok |

**Not (MEAS-03):** ilk sürüm testi istisna bekliyordu ve XPASS verdi; çözücü istisna atmıyor,
`converged=False` döndürüyor ve üretim çağıranları `RoundTripLightTimeConvergenceError` atıyor. Test
bayrağı sayacak biçimde düzeltildi. Ölçüm, inceleme ajanının öngörüsüyle (2^17 s'de ~%31, 2^18 s'de
~%65 başarısızlık) uyumlu.

**Not (MEAS-04):** istasyon konum hatası ızgara ortasındaki dönüşümün kendisidir. Sayılan Doppler
gözlemine etkisi sayım uçlarının ızgara hücresi içindeki konumuna bağlıdır ve WP1.1'deki UKF–batch
parite testiyle ölçülecektir; burada Doppler büyüklüğü iddia edilmez.

### 0.1 Tam regresyon (kernel'lı, `origin/main` 549093d + bu belge commit'i)

2205 test: 2146 geçti, 29 atlandı, 10 XFAIL (yukarıdaki kanıt testleri), **10 FAIL, 10 ERROR**.
Bu commit yalnız belge ve test dosyası ekler (`src/`, `examples/` değişmedi); 20 sorunun tamamı
`origin/main`'de vardır ve üç kök nedene iner (TST-09…TST-11):

| Kök neden | Etkilenen testler | Kanıt |
|---|---|---|
| TST-09: `dynamics.py` korumalı kimliği kaydı | 10 ERROR (R2 kampanya fixture'ı: `R2 compatibility gate failed: ['lunar_od/dynamics.py', …, 'accepted_r1_zero_j2_vector']`), `test_r2_current_tree_protection_holds…`, `test_f1_refreeze_is_additive…`, `test_a_later_bridge_edit_still_fails…` | 17 korumalı satırdan yalnız `dynamics.py` FAIL: beklenen `ea6f6322…` (71e088f'nin CRLF baytları, sayısal çekirdek re-freeze'i), gerçek `b19c2d80…` (a8f1bef, PR #61 "Phase F2 G4e-R", 2026-10-27). PR #61 eklemeli re-freeze satırı eklemedi. |
| TST-10: D19 eski bayt testleri | `test_d19_accepted_edge_requests_return_the_legacy_bytes[×5]`, `test_d19_explicit_opt_in_extends_the_bound…` | `_interp_state` kenar devamı 4ce66da'da (sayısal çekirdek, 2026-10-26) bilerek Hermite'e geçti; test hâlâ eski doğrusal baytları bekliyor. |
| TST-11: köprü provenans testi | `test_refreeze_keeps_the_historical_record_and_protects_approved_bytes` | Sayısal çekirdek re-freeze'i `examples/run_scenario_config.py` beklenen kimliğini `1bf731…` yaptı; test hâlâ `APPROVED_COMMIT` a9f9e56 baytlarını (`fd427a…`) bekliyor. |

Ek kırılganlık: koruma hash'leri CRLF çalışma ağacı baytlarıdır; sonuç `core.autocrlf=true`
olmasına bağlıdır (LF checkout'ta `dynamics.py` = `28e29ce8…`, tüm satırlar FAIL olur).
**Durum (2026-10-31, sahip onayıyla "re-freeze"):** TST-09, TST-10 ve TST-11 testleri kapandı;
CRLF bağımlılığı açık (WP0.5'in kalan kısmı). Re-freeze sonrası tam regresyon: 2205 test, 2166 geçti,
29 atlandı, 10 XFAIL (kanıt testleri), 0 FAIL, 0 ERROR.
- `examples/r2_measurement_fidelity_validation.py`: eklemeli `PHASE_F2_G4E_R_REFROZEN_CURRENT_TREE_SHA256`
  (`dynamics.py` `ea6f63 → b19c2d`). PR #61'in değişikliği yalnız `_shadow_contact_check` koruyucusudur
  (ön-geçiş gölge kesiminin son yörüngedeki temasa ≤ 1e-3 s uzaklıkta olması ya da ≤ 1e-9 rad);
  entegrasyon, kuvvetler ve STM değişmedi. Önceki tablolar tarihsel kayıt olarak aynen duruyor.
- `tests/test_harden_d18_d19_interpolation_bounds.py`: `_interp_vector` hâlâ eski baytlarla bit-bit;
  `_interp_state` ve `_interp_state_transition_position` kenar devamı SciPy `CubicHermiteSpline`
  kehanetine karşı (atol 1e-12 × ölçek), uç noktalar saklanan örneğe bit-bit.
- `tests/test_desktop_bridge_provenance.py`: onaylı a9f9e56 baytları tarihsel tabloda, güncel beklenen
  kimlik 71e088f baytları (sayısal çekirdek) olarak doğrulanıyor.
- `tests/test_f1_srp_variational_gradient.py`: re-freeze zinciri `ea6f6322 → b19c2d80` ile uzatıldı.

## 1. Genel hüküm

| Eksen | Not | Gerekçe |
|---|---|---|
| Temel formüller | **İyi** | J2, 3. cisim, SRP + penumbra gradyanı, Pines (N=300 kararlı), van Loan cebiri, RTS kazancı, Joseph formu, sigma ağırlıkları, aberasyon (stelab ile aynı), c sabiti, Doppler işareti — hepsi literatürle uyumlu bulundu. |
| Sayısal doğrulama | **İyi** | Analitik Jacobian + sonlu fark, ön-kayıtlı kapılar, hata bütçesi, HARDEN kusur defteri, Orekit'e karşı mm uyumu. |
| Üretim fizik modeli | **Orta** | Doğru modeller (SSB ışık zamanı, Shapiro, medya, ITRF2020, gelgit) yalnız çalışma yolunda; üretim tahmincileri basitleştirilmiş. Eksik kuvvetler: Ay k2 gelgiti, albedo/IR, Dünya gölgesi, göreli dinamik. |
| Hesaplar arası tutarlılık | **Zayıf** | 15+ tutarsızlık (§3). Sentetik testlerde gerçek değer ve tahminci aynı hatayı paylaştığı için gizleniyor. |
| Tahminci sağlamlığı | **Orta–Zayıf** | SRIF kabul testi yanlış (iki ayrı neden), erken yakınsama gevşek toleransla, birim bağımlı durma/rank kriterleri, DMC Q taşması, UKF jitter/regülarizasyon hataları. |
| Gerçek veri validasyonu | **Başlangıç + metodoloji riski** | Bağımsız hold-out gün kalmadı; F3 aynı günde ayarlıyor ve puanlıyor (✔); bias önselleri puanlama referansından. |
| Test kehaneti kalitesi | **Orta** | ~%22 bağımsız referans, ~%25 aynı-model gidiş-dönüş, ~%24 duman testi. Bilinen-yanlış davranışı sabitleyen 5 test. CI fizik testlerinin neredeyse hiçbirini koşmuyor. |
| Tekrarlanabilirlik | **Karışık** | Benchmark/quickstart örnek düzeyde iyi; CLI ve masaüstü koşuları tohum/kernel/commit kaydetmiyor; masaüstü gizlice farklı bilim ayarları kullanıyor (✔). |
| Standartlar | **Zayıf** | Gereksinim/izlenebilirlik/ICD/kapsam/statik analiz/CCSDS yok. |

---

## 2. Bulgu kataloğu (kimlikli)

Önem: **K** kritik (sonucu yanlış yapar), **B** büyük, **O** orta, **D** düşük.

### 2.1 Dinamik ve kuvvet modelleri (DYN)

| ID | Ö | Bulgu | Konum | Büyüklük |
|---|---|---|---|---|
| DYN-01 ✔ | B | GM'ler DE431, efemeris DE440; yorum "not yet wired" eskimiş | `constants.py:32-37` | Ay GM 1e-8 bağıl → ~3 m/gün |
| DYN-02 ✔ | B | Senaryo gravite çerçevesi varsayılanı `MOON_PA_DE421`; doğrulama gezegen efemerisini kontrol etmiyor | `scenario_config.py:183,1160` | ~1.65 m yüzeyde |
| DYN-03 ✔ | O | Ay J2 sabit IAU ortalama kutbu; yorum "0.04°" diyor, gerçek 18.6 yıllık salınım ~1.5° | `dynamics.py:57-82` | J2 açıkken |
| DYN-04 ✔ | O | J2 = 2.0346e-4 şekil yarıçapı 1737.4 km ile; GRAIL çifti (2.032e-4, 1738.0 km) | `constants.py:44,57` | ~2.7e-7 m/s² @100 km |
| DYN-05 ✔T | B | Ay yönelimi 60 s ızgarada **en yakın örnek** (basamak); kuvvet süreksiz, adım denetimini bozuyor | `lunar_frames.py:188`, `dynamics.py:971,1127,1565` | ≤8e-5 rad; 1e-8–1e-6 m/s² |
| DYN-06 | B | Ay k2 katı gelgit kuvveti yok; yükleyici `tide_system` alanını yok sayıyor | `gravity_model_loader.py:237-307` | ~3e-7 m/s² (değişken kısım 5e-8–1e-7) > SRP |
| DYN-07 | B | Ay albedo/kızılötesi basıncı yok | `srp.py` | ~4e-8 m/s², SRP mertebesinde |
| DYN-08 | O | Dünya gölgesi (Ay tutulması) yok; gölge olayları opt-in | `srp.py:518`, `dynamics.py:937` | tutulma günlerinde büyük |
| DYN-09 | O | Göreli dinamik (1PN Schwarzschild) yok | `src/` | ~1e-9–1e-8 m/s² |
| DYN-10 ✔T | B | Kuvvet parmak izi efemeris kimliğini (SPK, ızgara, interpolasyon türü, pertürbasyon) ve dönüş modunu hash'lemiyor → farklı fizik aynı hash | `force_contract.py:444,517` | parite iddiası geçersiz olabilir |
| DYN-11 | D | `earth_j2_mode` J2 kapalıyken de hash'leniyor (sahte FAIL); `bind_spec_to_runtime` Earth J2/harmonikleri bağlamıyor | `force_contract.py:504,551-571` | — |
| DYN-12 | O | Yükleyici `frame="MOON_PA"` etiketliyor, STM yolu reddediyor; scenario_config Pines gradyanı varken harmonikleri "gradient yok" diye engelliyor | `gravity_model_loader.py:443`, `dynamics.py:1101`, `scenario_config.py:~1173` | yüksek derece gravite senaryoda kullanılamıyor |
| DYN-13 | D | Harmonikler dosya GM'i, merkez terim DE GM'i | `gravity_model_loader.py:330`, `dynamics.py:262` | 1.6e-8 m/s² |
| DYN-14 | O | Entegratör: tez rtol 1e-10, kütüphane 1e-11; tek atol m/m/s/STM için; truth DOP853, BLS Adams, UKF RK4+doğrusal efemeris; sayısal çekirdek yalnız 8 481 s yayları sertifikalıyor | `dynamics.py:437,1338,1363`, `thesis_matrix.py:43` | mm–m |
| DYN-15 | D | Numba Pines `fastmath=True` truth yolunda; 42-durum yolu saf Python → farklı uygulamalar, N≥300 parite yok | `accelerated.py:1033,1323` | — |
| DYN-16 | D | `propagation_contract`: SRP açık + NO_SHADOW'da `shadow_events=True` raporluyor | `propagation_contract.py:17,53` | rapor hatası |
| DYN-17 | D | SRP/gölge için Güneş geometrik (`NONE`), ışık zamanı yok | `ephemeris.py:230` | gölge temas ~0.1 s |
| DYN-18 ✔ | D | `orbit.lunar_initial_state_mci` sxform kullanıyor → elemanlar dönen MOON_PA'da; belgelenmemiş | `orbit.py:94` | eğimde ~0.17° |
| DYN-19 | D | Efemeris 10 s ekstrapolasyon izni, history-domain politikası (2 ULP) ile çelişiyor; Dünya J2 çerçevesi birim matris; `et0+t` tek float64 | `ephemeris.py:32`, `dynamics.py:93,585` | küçük |

### 2.2 Ölçüm modelleri, çerçeveler, zaman (MEAS)

| ID | Ö | Bulgu | Konum | Büyüklük |
|---|---|---|---|---|
| MEAS-01 ✔ | K | Üretim ışık zamanı Ay merkezli Newton, Shapiro yok. SSB yolu yalnız `two_way_light_time_ssb` profilli iki yönlü menzilde ve bu profil `scenario_config`'de izinli değil | `relativistic_light_time.py`, `scenario_config.py:79-84`, `estimators.py:2171` | 2YM +1.9–2.5 m; Doppler −68…+87 µm/s; 1YM 14–19 km; ΔDOR 97 µrad; Güneş Shapiro 7.3–7.8 m |
| MEAS-02 ✔ | K | `spice_ssb` aberasyon: Ay merkezli ışık zamanı + SSB gözlemci hızı → çerçeve karışımı; belgedeki "~11″ SSB bias" muhtemelen bu artefakt | `measurements.py:1503-1561,1404,1354`; `docs/theory/stellar_aberration_od_impact.md` | ~11–20″ açı |
| MEAS-03 ✔T | B | Sayılan Doppler tek-sekme çözücüsü yakınsamayı epoch farkıyla (`abs(new_t2 - t2)`, `(receive_time_s - t2) - ρ/c`) test ediyor → t ≳ 2^17 s'de ulp sınırı; `converged=False`, üretimde `RoundTripLightTimeConvergenceError` | `radiometrics.py:1074,1109,1122-1145` | 2.5e6 s'de 200 noktanın 12'si yakınsıyor (§0) |
| MEAS-04 ✔T (kapandı, WP1.1) | B | SR-UKF/EKF sayılan Doppler: Dünya konum/hız ve 6×6 J2000→ITRF93 **doğrusal** interpolasyon (ortonormal değil); batch Hermite. Test bunu bit-bit sabitliyor | `filters.py:1991-1993,2118`; `tests/test_filters.py:50-80` | D23'e göre ~1.7e-3 m/s; dönüşümde ek |
| MEAS-05 (açık; WP1.1 dokunmadı) | O | UKF hızlı sigma yayıcısı Dünya/Güneş'i doğrusal interpolasyonla alıyor | `accelerated.make_lerp_vec3`, `dynamics.py:1363` | ~120 m (600 s), ~4 km (3600 s) Dünya konumu |
| MEAS-06 | B | Medya, topocentric istasyon zamanı, ITRF2020+katı gelgit yalnız SSB yolunda; IERS EOP yok, predict BPC; levha hareketi, kutup gelgiti, okyanus yüklemesi yok | `media.py`, `station_frames.py`, `spice_loader.py:64` | troposfer 2–12 m; gelgit 0.23 m; ITRF 0.33 m; predict 112 m/2027 |
| MEAS-07 | O | Az/El'de troposfer kırılması yok | `geometry.py:86`, `measurements.py:1365` | 5′ @10°, 1′ @45° |
| MEAS-08 | O | Rampalı uplink yok; iki yönlü menzilde bias solve-for yok; SSB yolu transponder gecikmesini reddediyor | `radiometrics.py:704`, `estimators.py:~2160`, `relativistic_light_time.py:~808` | DSN verisi işlenemez |
| MEAS-09 | O | Varsayılan range-rate anlık geometrik (sayılan Doppler değil); Doppler modunda eşlik eden menzil/açı anlık | `radiometrics.py:98` | model karışımı |
| MEAS-10 | O | Konum gözlemlerinde ışık zamanı Jacobian'ı varsayılan birinci derece | `measurements.py:223,2029` | yakınsama/kovaryans |
| MEAS-11 | D | İki-döngü geri dönüşü 1e-11 s'yi sessizce kabul ediyor | `relativistic_light_time.py:244` | ~25 µm/s |
| MEAS-12 | D | Shapiro'da cisim konumu iki uç için de alış epoch'unda (Moyer 8-38 her ucu kendi epoch'unda ister) | `relativistic_light_time.py:179-206` | ~30 µm |
| MEAS-13 | O (gizli) | Landmark optik: nadir tutum Jacobian'ı eksik, MOON_PA/ME karışımı, yarım piksel; ΔDOR kuasar terimi statik taban çizgisi | `lunar_landmark_optical.py`, `delta_dor.py:337` | kullanılmıyor ama yanlış |
| MEAS-14 | D | Görünürlük: geometrik maske, küresel Ay (arazi yok), düzgün ızgara varsayımı; elevasyon `arcsin` (belge `atan2`), az 2π sarma | `visibility.py`, `geometry.py:86-94` | küçük |

### 2.3 Tahminciler ve istatistik (EST)

| ID | Ö | Bulgu | Konum |
|---|---|---|---|
| EST-01 ✔T | K | SRIF'te mevcut maliyet 42-durum, aday maliyeti 6-durum yayılımıyla → iyi adımlar reddediliyor (düzeltme yalnız iki yönlü BLS-LM'e taşınmış) | `estimators.py:157 vs 220`, `:377 vs 460`, `:3115 vs 3214` |
| EST-02 | K | İki yönlü SRIF MAP adımı atıyor, yalnız veri maliyetiyle puanlıyor | `estimators.py:3144,3167,3222` |
| EST-03 | K | SRIF'te sönümleme/line-search yok → MaxIter (benchmark'ta yakınsama %25–44) | `estimators.py:244,484` |
| EST-04 | B | Yakınsama, `_adaptive_tol` rtol 1e-8 iken ilan edilebiliyor; sonuç istenen toleranstan 1000× gevşek | `estimators.py:45-54,237,477,680,904,3233` |
| EST-05 | B | Durma testleri birim bağımlı (m, m/s, rad, K karışık); J-Stab ağır sönümlü adımda tetikleniyor | `estimators.py:240,683,907,2841` |
| EST-06 | B | Posterior kovaryans fiziksel birimde eigen-taban ile kırpılıyor (açı bias bilgisi max'ı belirliyor → aşırı güvenli); rank eşikleri cond 1e14–1e15'e izin | `estimators.py:1561-1572,1661`, `bounded_least_squares.py:45` |
| EST-07 | O | BLS ve SRIF farklı tekillik eşikleri (1e15 normal vs 1e14 R → ~1e28) → karşılaştırma yanlı | `estimators.py:1697,206,446,3181` |
| EST-08 | O | LM λ güncellemesi kazanç oranını yok sayıyor; D = diag(AᵀA) önseli dışlıyor | `estimators.py:678,688,1701` |
| EST-09 ✔T | B | DMC van Loan: −A bloğu +β → e^{dt/τ} büyüme; dt/τ=30'da 8.7e-5 bağıl hata, dt/τ=144'te negatif tanımlı (§0) | `dmc.py:61-68` |
| EST-10 | B | SR-UKF zaman güncellemesi negatif ağırlık downdate geri dönüşsüz → yakalanmamış `LinAlgError` | `filters.py:1389` |
| EST-11 | O | UKF varsayılan α=1e-3 (Wc0≈−1e6), ortalama mutlak formda → ~1 m² kovaryans hatası; senaryo α=0.35 | `filters.py:34,1341` |
| EST-12 | O | Sabit 1e-12 jitter birimden bağımsız; açı ölçümünde P'yi ~%100 fazla küçültüyor | `filters.py:1428,673,1439` |
| EST-13 | O | UKF regülarizasyon pseudo-ölçümü her güncellemede tekrar (önsel N kez); dondurulmuş durumlar Schmidt-Kalman değil | `filters.py:1217-1230` |
| EST-14 | O | "Discrete" süreç gürültüsü dt'yi yok sayıyor; Q taraması fiziksel değil | `filters.py:1539`, `q_tuning.py:37` |
| EST-15 | O | Korelasyonlu/AR(1) gürültü üretiliyor ama hiç beyazlatılmıyor; AR(1) satır indeksinde (zaman/istasyon karışık) | `noise_models.py:51-66` |
| EST-16 | O | ANEES/ANIS χ² aralıkları yok; adaptif R'de NIS aşağı yanlı; artık lag-1 iç içe [ρ,az,el] vektöründe; adaptif ayarda truth başlangıç durumuna yayınlanıyor; ortalama NIS reddedilenleri içeriyor | `diagnostics.py:202`, `adaptive_tuning.py:316,387`, `filters.py:594` |
| EST-17 | O | Batch ayıklama: a priori σ, sert 3σ, %30 "ya hep ya hiç" | `estimators.py:1876-1900` |
| EST-18 | O | Gözlemlenebilirlik koşul/rank/zayıf yön ölçeklenmemiş Jacobian'dan; bias dondurma kararı birime bağlı | `observability.py:114,467` |
| EST-19 ✔T | B | UKF P0 (100 m / 0.1 m/s) enjekte edilen soğuk başlangıç hatasıyla (180 m / 0.03 m/s) uyumsuz → NEES ve BLS–UKF karşılaştırması bozuk | `scenarios.py:1434`, `thesis_matrix.py:40-41` |
| EST-20 | D | Monte Carlo tohumları durum başına farklı (ortak rastgele sayı yok), kampanyalar arası çakışma | `monte_carlo.py:79,312` |
| EST-21 | D | DMC yalnız EKF'de; UKF ışık zamanlı/iki yönlü gözlemleri reddediyor → karşılaştırma yalnız geometrik gözlemlerde | `filters.py:733-768` |

### 2.4 Gerçek veri (LRO) hattı ve metodoloji (RD)

| ID | Ö | Bulgu |
|---|---|---|
| RD-01 ✔ | K | **F3**: σ*, gün 263'ün veriye karşı χ²'si ile seçiliyor; H2 aynı gün puanlanıyor → H2 geçmeye yanlı. **Sweep koşmadan önce** düzeltilmeli. (`phase_f3_lro_dmc_smoother.py:~237-240`, ön-kayıt `selection_rule`) |
| RD-02 | K | Dokunulmamış hold-out gün kalmadı (262 model seçti, 263 F2j–F2r–F3'te defalarca kullanıldı). Manevrasız günler 264–268 mevcut. |
| RD-03 | B | İstasyon bias önselleri (489/477/475 m) F1'in LOLA'ya göre artıklarından → puanlama referansıyla döngüsel (`phase_f2_lro_od.py:805`) |
| RD-04 | O | Eşik/kapı değişiklikleri sonuç görüldükten sonra (F2i, F2j-A1/A2, F2m, F2r, F2s-1) — her biri açıklanmış ama birlikte "forking paths" |
| RD-05 | O | LOLA'nın ~10 m doğruluğunun altındaki farklarla çözüm sıralanıyor (F3 eşikleri 5–8 m) |
| RD-06 | O | Doppler artıkları lag-1 0.74–0.81 korelasyonlu ama beyaz ağırlıklı → formal σ 2.5–3× iyimser |
| RD-07 | B | Zaman etiketi ofseti hipotezi erken kapatıldı (tüm günler aynı işaret, ~−3 ms) |
| RD-08 | O | Açıklanamayan −8 mm/s Doppler bias'ı (S-band koherent iki yönlüde ~0 beklenir) |
| RD-09 | O | İstasyon koordinatları WGS-84 tablosundan, epoch/levha/anten referansı bilinmiyor; istasyon başına tek bias 3–4 gün boyunca |
| RD-10 | B | LOLA'nın gravite kurulumu (GRGM900C N=600, kendi çerçevesi) test edilmedi; E3 alanı yanlış çerçevede denedi; k2 gelgiti ve panel SRP test edilmedi |
| RD-11 | D | UTDF: geçerlilik bayrağı donmuş sayaçta güvenilmez; `pad_xmit` kontrol edilmiyor (üç yönlü kayıt riski); artık saniye geçişi; sabit kodlu frekanslar |
| RD-12 | O | F3 DMC τ=1800 s, eksen ataletsel, σ taraması F2s artığından 2 mertebe küçük |

### 2.5 Konfigürasyon, masaüstü, tekrarlanabilirlik (CFG)

| ID | Ö | Bulgu |
|---|---|---|
| CFG-01 ✔ | B | Masaüstü gizli varsayılanlar: rtol 1e-8/atol 1e-10 (tezin 100× gevşeği), efemeris adımı 3600 s, λ0 1e-3 (`desktop_app/models/scenario_model.py:14,42-43`) |
| CFG-02 ✔ | B | Senaryo şemasında tohum alanı yok; CLI `measurement_seed=None` → tohumsuz gürültü, kaydedilmiyor (`examples/run_scenario_config.py:86,238`) |
| CFG-03 | B | Elevasyon maskesi, boşluk eşiği, efemeris adımı, yay modu config alanı değil (`desktop_options`); JSON koşuyu tam tanımlamıyor |
| CFG-04 | B | Üç efemeris nesli karışık (DE440 Dünya/Güneş; DE421 başlangıç durumu/Ay yönelimi; DE431 GM); kernel listesi çıktılara yazılmıyor |
| CFG-05 | O | CLI JSON'daki `bls_lambda0`, `tol_cost_stability` yok sayılıyor; bilinmeyen anahtarlar sessizce yok sayılıyor; şema eksik |
| CFG-06 | O | Geçersiz kombinasyonlar kabul ediliyor (aberasyon profili/model, sayım > örnek adımı, UKF alanları BLS'de) |
| CFG-07 | O | `enable_lunar_harmonics`, `enable_earth_j2` her zaman hata veriyor ama şemada |
| CFG-08 | O | İstasyon koordinatları 0.01° (~1 km) yuvarlanmış, yükseklikler 100 m yer tutucu; σ'lar isimde "ITU" geçmesine göre; iki istasyon tablosu |
| CFG-09 | O | Masaüstü tek tohumu gürültü ve soğuk başlangıç için kullanıyor; "prescribed" yay modu görünürlük segmentasyonunu değiştiriyor; ön ayar açıklamaları yanlış |
| CFG-10 | O | CLI çıktıları commit/config hash/tohum/kernel/paket sürümü kaydetmiyor (benchmark kaydediyor) |
| CFG-11 | O | Üç sayılan Doppler uygulaması, iki efemeris interpolatörü, altı `C_LIGHT` tanımı — yetkili olan belgelenmemiş |
| CFG-12 | D | Depoda 48 MB çalışma sonucu, 15 MB docs ikilisi, arşiv WIP kopyaları; numba için numpy üst sınırı yok; pyproject extras yok |

### 2.6 Test ve V&V süreci (TST)

| ID | Ö | Bulgu |
|---|---|---|
| TST-01 | K | CI fizik testlerinin neredeyse hiçbirini koşmuyor (public 36 dosya çoğu GUI; library kernel'sız; 20 `@slow` Monte Carlo testi, kampanya ve gravite verisi gerektirenler hiç koşmuyor) |
| TST-02 | B | Kehanet dağılımı: ~%22 bağımsız, ~%17 sonlu fark, ~%25 aynı-model gidiş-dönüş, ~%12 kendi çıktısının anlık görüntüsü, ~%24 duman. Pytest'ten hiçbir dış araç (erfa, Orekit, pyshtools) çağrılmıyor |
| TST-03 ✔ | B | Bilinen-yanlış davranışı sabitleyen testler: `test_filters.py:50-80` (doğrusal interpolasyon), `test_frame_spice_validation.py:369-460`, `test_harden_boundaries.py:310`, four-event başlığı, `test_doppler_model_review` p06/p07 |
| TST-04 | B | Gevşek toleranslar: Ay dönüş hızı %100 bağıl; gürültüsüz kurtarma 10–25 m; NEES 6 deneme delta=3; "adım taramasının en iyisi" sonlu farklar |
| TST-05 | O | Sıra bağımlılığı: `kclear()` test gövdelerinde, conftest import anında DE421'e geçiyor, modül-kapsamlı fixture paylaşılan havuzu değiştiriyor; zaman ölçen hız testi |
| TST-06 | O | Doğrudan test dosyası olmayan modüller: `radiometrics.py` (62 KB, en büyük boşluk), `propagation_contract.py`, `measurement_contracts.py`, `constants.py`, `rf/*` |
| TST-07 | O | Negatif testler eksik (UTC/TT etiket hatası, çerçeve merkezi takası, kernel kapsam sınırı, model uyumsuzluğunda iyimser kovaryans, bozuk UTDF); en iyi negatif takım `test_harden_mutants` kampanyaya bağlı → CI'da atlanıyor |
| TST-08 | O | Gereksinim/ID/izlenebilirlik matrisi/ICD yok; hata bütçesi yalnız sayısal; `docs/external_cross_validation.md` yok; kernel manifest/SHA yok; ruff/mypy/coverage yok |
| TST-09 ✔ (kapandı) | B | `main` kendi R2/R3 koruma kapısını geçmiyor: PR #61 (a8f1bef) `dynamics.py`'yi sayısal çekirdek kimliğinden (`ea6f63`) sonra eklemeli re-freeze olmadan değiştirdi → 10 ERROR + 3 FAIL; kapı CI'da koşmadığı için merge'de görülmedi (bkz. TST-01) |
| TST-10 ✔ (kapandı) | O | D19 testleri 4ce66da'nın bilerek yaptığı Hermite kenar değişikliğinden sonra güncellenmedi (6 FAIL) |
| TST-11 ✔ (test kapandı; CRLF açık) | O | Köprü provenans testi sayısal çekirdek re-freeze'i sonrası eski onaylı baytları bekliyor (1 FAIL); koruma hash'leri `core.autocrlf`'e bağlı |

---

## 3. Hesaplar arası uyumsuzluk matrisi

| # | A | B | Fark | Sentetikte görünür? | Bulgu |
|---|---|---|---|---|---|
| X-01 | Efemeris DE440 | GM DE431, gravite çerçevesi DE421, başlangıç durumu DE421 | ~3 m/gün, ~1.65 m | Hayır | DYN-01/02, CFG-04 |
| X-02 | Ay J2 sabit kutup | Harmonik PA çerçevesi | ~1.5° | Hayır | DYN-03 |
| X-03 | Harmonik dönüş "en yakın örnek" | Exact-epoch DE440 modu | ≤8e-5 rad | Kısmen | DYN-05 |
| X-04 | Truth DOP853 (+Numba fastmath Pines) | BLS Adams + gevşek erken tol; UKF RK4 + doğrusal efemeris; 42-durum saf Python | mm–m | Kısmen | DYN-14/15, MEAS-05, EST-04 |
| X-05 | Batch Doppler (Hermite Dünya) | UKF/EKF (doğrusal Dünya + doğrusal dönüşüm) | ~1.7e-3 m/s | **Evet** | MEAS-04 |
| X-06 | SRIF mevcut maliyet (42-durum) | SRIF aday maliyeti (6-durum, önselsiz) | 1–7 m yörünge | **Evet** | EST-01/02 |
| X-07 | İki yönlü menzil (SSB+Shapiro+medya seçeneği) | Doppler/ΔDOR/tek yönlü (yalnız Ay merkezli) | m / 10 µm/s | Hayır | MEAS-01 |
| X-08 | `spice_ssb` aberasyon (SSB hızı) | Işık zamanı (Ay merkezli) | ~11–20″ | Hayır | MEAS-02 |
| X-09 | SSB yolu istasyon (ITRF2020+gelgit+topo TDB) | Varsayılan (ITRF93 predict) | 0.2–0.3 m | Hayır | MEAS-06 |
| X-10 | UKF P0 | Enjekte edilen soğuk başlangıç hatası | 1.8× / 3.3× | **Evet** | EST-19 |
| X-11 | Masaüstü (rtol 1e-8, 3600 s) | Kütüphane/tez (1e-10/1e-11, 600 s) | m | **Evet** | CFG-01 |
| X-12 | Kuvvet parmak izi | Gerçek yürütülen efemeris/dönüş | eşit hash farklı fizik | — | DYN-10 |
| X-13 | Gravite yükleyici etiketi | STM çerçeve kontrolü | çalışmıyor | — | DYN-12 |
| X-14 | BLS tekillik eşiği | SRIF tekillik eşiği | 1e15 vs ~1e28 | Evet | EST-07 |
| X-15 | Ay merkezli 2YM (transponder var) | SSB (reddediyor) | — | — | MEAS-08 |

**Ana sistemik sorun:** sentetik doğrulama "aynı-model gidiş-dönüş" ağırlıklı olduğu için X-01, X-02,
X-07, X-08, X-09 gibi fizik tutarsızlıkları testlerde birbirini götürüyor ve yalnız gerçek veride
sistematik hata olarak ortaya çıkıyor. Programın ilk ilkesi: **model-uyumsuzluk testleri (truth ≠
estimator) ve bağımsız dış kehanet zorunlu.**

---

## 4. Standartlara uygunluk

| Standart | Var | Eksik → hangi iş paketi |
|---|---|---|
| NASA-STD-7009A | Doğrulama kanıtı, ön-kayıt, sayısal hata bütçesi, kullanım sınırları kısmen | Intended use + kabul kriteri, 8 faktörlü güvenilirlik, UQ raporu, raporlama şablonu → WP0.3, WP8.x |
| NPR 7150.2 / NASA-STD-8739.8 | Sürüm, CHANGELOG, SHA-sabitli Actions, sürüm kapısı | Sınıflandırma, SMP, gereksinim + çift yönlü izlenebilirlik, test planı/prosedürü, statik analiz, kapsam → WP0.3, WP7.x |
| ECSS-E-ST-40C / Q-ST-80C | Kusur defteri, PR süreci | SRS, ICD, SVVP, SVR/STR, SPAP, nicel metrikler, NCR → WP0.3, WP7.x |
| Moyer (DESCANSO-2) / DSN 810-005 | Doğru modeller çalışma yolunda | Üretim yolunda tam zincir, rampalı uplink, bias'lar → F2 |
| IERS Conventions 2010 | Katı gelgit, dtdb (çalışma yolunda) | EOP, okyanus yüklemesi, kutup gelgiti, levha hareketi → WP2.4 |
| CCSDS 502.0-B (ODM), 503.0-B (TDM) | — | TDM okuyucu/yazıcı, OEM/OCM kovaryanslı yazıcı → F6 |
| NAIF iyi uygulama | DE440 sözleşmesi fail-closed | Kernel manifest + SHA, yüksek hassasiyetli Dünya PCK → WP2.4, WP7.4 |

---

## 5. Risk kaydı (NPR 8000.4 tarzı 5×5; O=olasılık, S=sonuç, Skor=O×S)

| ID | Risk | Bağlı bulgular | O | S | Skor | Azaltma |
|---|---|---|---|---|---|---|
| R-01 | Gerçek veride metre seviyesi sistematik hata (ışık zamanı/Shapiro/medya/istasyon) | MEAS-01/02/06/07 | 5 | 5 | **25** | F2 |
| R-02 | Sentetik testler tutarsızlıkları gizliyor → yanlış "doğrulandı" kanısı | X-01…X-09, TST-02/03 | 5 | 4 | **20** | F1, WP5.1, F7 |
| R-03 | Gerçek veri sonuçlarının bağımsızlığı bozuk (F3 aynı gün ayar+puan, hold-out yok, döngüsel önseller) | RD-01…RD-05 | 5 | 4 | **20** | WP0.2, F5 |
| R-04 | SRIF yanlış kabul testi → durma ve yanlı tahminci karşılaştırması | EST-01/02/03/07 | 5 | 4 | **20** | F4 |
| R-05 | Fizik testlerinin CI'da koşmaması → regresyonlar fark edilmeden main'e girer | TST-01/05 | 4 | 4 | **16** | F7 |
| R-06 | UKF/EKF ile batch arasında model farkı (Doppler interp, efemeris, P0) | MEAS-04/05, EST-19 | 4 | 4 | **16** | F1 |
| R-07 | Eksik kuvvetler (k2, albedo/IR, Dünya gölgesi) F2r kalıntısını açıklıyor olabilir | DYN-06/07/08, RD-10 | 4 | 4 | **16** | F3 |
| R-08 | Kovaryans gerçekçi değil (birim bağımlı kırpma, jitter, regülarizasyon, korelasyonlu gürültü beyaz) | EST-06/12/13/15, RD-06 | 4 | 4 | **16** | F4 |
| R-09 | Uzun yaylarda sayılan Doppler çözücü çöküşü | MEAS-03 | 4 | 3 | 12 | F1 |
| R-10 | Tekrarlanamayan koşular (tohum yok, kernel kaydı yok, masaüstü farklı ayarlar) | CFG-01…04,10 | 4 | 3 | 12 | WP1.6 |
| R-11 | Efemeris/GM/çerçeve nesil karışımı | DYN-01/02, CFG-04 | 5 | 3 | 15 | F1 |
| R-12 | Kuvvet parmak izi farklı fiziği eşit gösterir → parite iddiaları geçersiz | DYN-10/11 | 3 | 4 | 12 | WP1.5 |
| R-13 | Yakınsama gevşek toleransta ilan ediliyor; durma kriterleri birime bağlı | EST-04/05 | 4 | 3 | 12 | F4 |
| R-14 | DMC Q taşması geçiş boşluklarında | EST-09 | 3 | 4 | 12 | F4 (F3'ten önce!) |
| R-15 | İstasyon zinciri predict BPC, EOP yok; zamanla bozulan doğruluk | MEAS-06 | 3 | 4 | 12 | WP2.4 |
| R-16 | Gereksinim/izlenebilirlik yok → "NASA/ESA seviyesi" iddiası denetlenemez | TST-08 | 5 | 2 | 10 | WP0.3 |
| R-17 | Yerel main'in geride olması ve 20+ eski worktree → yanlış tabanda çalışma | — | 3 | 3 | 9 | WP0.1 |
| R-18 | Dış kehanet repoda kalıcı değil | TST-02/08 | 3 | 3 | 9 | WP5.1 |
| R-19 | Gizli (kullanılmayan) modüllerde yanlış fizik ileride kullanılır | MEAS-13 | 2 | 4 | 8 | WP2.6 |
| R-20 | CCSDS arayüzü yok → bağımsız araçlarla çapraz kontrol zor | — | 4 | 2 | 8 | F6 |
| R-21 | Depo şişmesi ve sıra bağımlı testler | CFG-12, TST-05 | 3 | 2 | 6 | F7 |
| R-22 | Bilinen-kırmızı tam regresyon: yeni regresyonlar mevcut 20 hatanın arasında kaybolur; koruma kapısı anlamını yitirir | TST-09/10/11, TST-01 | 5 | 3 | 15 | WP0.5 |

---

## 6. Program

İş paketleri, prosedür, testler ve kabul ölçütleri: [V&V programı](../assurance/vv-program.md).
