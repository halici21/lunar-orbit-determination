# F1 faz kapısı — 2026-11-02 (hesaplandığı gibi)

**Konu:** `origin/main` @ `be84e88` (F1 WP1.1, 1.2, 1.4–1.7, 1.9, 1.10 ve F4 WP4.x birleşmiş);
kapı dalı `f1/phase-gate` @ `17a07b5` (yalnız ön-kayıt eklenmiş, kod `be84e88` ile aynı).
WP1.3 ([#100](https://github.com/halici21/lunar-orbit-determination/pull/100)) ve WP1.8
([#101](https://github.com/halici21/lunar-orbit-determination/pull/101)) açık PR'dır ve ayrı değerlendirilir.

**Ön-kayıt:** [`studies/phase17/configs/f1_phase_gate_predeclaration.json`](../../studies/phase17/configs/f1_phase_gate_predeclaration.json),
hiçbir kapı hesaplanmadan önce commit'lendi (`17a07b5`). Kriterler sonradan değiştirilmedi;
[değişiklik günlüğüne](../assurance/amendment-log.md) satır eklenmedi.

## Özet

| Kapı | Sonuç | Kısa gerekçe |
|---|---|---|
| G1 dış kehanet (Orekit hizalı, `ref3`) | **PASS** | 5 durumun hepsinde \|Δr_son\| ≤ 0.028 mm; Orekit farkı değişmedi (24.4–25.6 mm) |
| G2 üretim varyantı (`prod`, VODE Adams 1e-11, 600 s ızgara) | **PASS (atıflı)** | 13–224 mm değişim; aynı ortamda tamamı `3705b15` (C4-I, yalnız D1b) ve `4ce66da` (Hermite Dünya) commit'lerine atfedildi; ayrıca ortam (SciPy 1.17↔1.18) aynı kaynakta 571 mm'ye kadar oynatıyor → yeni bulgu: üretim Adams 1e-11 profili 18–725 mm/gün |
| G3 eşli benchmark | **FAIL (hesaplandığı gibi), 3 alt kriterin 1'i** | başarı 1.0 ✓; SRIF sonlanma 0.25/0.44 → 1.00/1.00 ✓; BLS-LM medyanı −%1.84 (sınır %1) ✗ |
| G4 tam regresyon | **PASS** | çıkış kodu 0; JUnit 2589 test, 0 başarısız, 0 hata, 32 atlanan |
| G5 X kalemleri | **PARTIAL** | 13 kalemin 6'sı main'de kapalı; X-03 PR #100'de kapalı; X-01, X-04, X-12 kısmi; X-02, X-11 açık |

**Faz kapısı hükmü: PARTIAL.** F1 tamamlanmadı; açık kalemler aşağıda adlandırıldı.

## G1 — Orekit hizalı dinamik (`ref3`: doğrudan SPICE, DOP853 1e-14/1e-10)

Koşu: `agent-sandbox/orekit-gmat-oracle/harness` `run_lunar_od.run(case, "ref3")`, lunar_od bu
checkout'tan, çıktı ayrı dizinde (`outputs/f1_gate/oracle_rerun`); saklanan 2026-10-21 çıktılarına
(lunar_od `a5707e0`, Orekit) dokunulmadı.

| Durum | \|r_son(şimdi) − r_son(anlık görüntü)\| | şimdi vs Orekit | anlık görüntü vs Orekit |
|---|---|---|---|
| D1a | 0.000 mm | 25.628 mm | 25.628 mm |
| D1b | 0.028 mm | — | — |
| D2-50 | 0.000 mm | 25.614 mm | 25.614 mm |
| D2-100 | 0.000 mm | 25.503 mm | 25.503 mm |
| D3 | 0.000 mm | 24.386 mm | 24.386 mm |

**Not (D1b):** D1b "DE421" durumudur. Harness `load_spice_kernels(KERNEL_DIR)` çağırır ve
Phase C4-I'den (`3705b15`) beri bu çağrı varsayılan olarak `de440.bsp` yükler. Bu yüzden D1b
artık DE421 koşmuyor, D1a ile aynı efemerisi kullanıyor. 0.028 mm, DE421→DE440 farkının 1 günlük
etkisidir. D1b'nin DE421 anlamı harness'ta `ephemeris="DE421"` geçirilerek geri kazanılır;
saklanan çıktılar sandbox'a aittir ve değiştirilmedi.

## G2 — Üretim varyantı

Ön-kayıtlı kural: her durumda \|Δr_son\| ≤ 1 mm; daha büyük değişim adı konmuş, birleşmiş bir
değişikliğe atfedilmeli, yoksa FAIL.

**Ham sonuç (bu worktree'nin ortamı):** `prod`, saklanan anlık görüntüden 13–224 mm ayrışıyor
(D1a 224.05, D1b 13.46, D2-50 24.64, D2-100 39.06, D3 82.06 mm). Hiçbir durum 1 mm içinde değil.

**Atıf yöntemi:** `a5707e0..be84e88` aralığında `dynamics.py`, `ephemeris.py`, `accelerated.py`
veya `lunar_frames.py`'ye dokunan yedi commit'in `src/` ağaçları `git archive` ile ayrı dizinlere
çıkarıldı ve harness aynı `prod` ayarıyla her ağaçta koşuldu. Ölçüm iki ortamda yapıldı: bu
worktree'nin `.venv`'i (SciPy 1.17.1, NumPy 2.4.3, Numba 0.65.1) ve anlık görüntünün üretildiği
sandbox `.venv`'i (SciPy 1.18.1, NumPy 2.5.4, Numba 0.68.0). Sandbox'ta hiçbir dosya değiştirilmedi.

Anlık görüntü ortamında, \|r_son − r_son(saklanan prod)\| [mm]:

| ağaç | D1a | D1b | D2-50 | D2-100 | D3 |
|---|---|---|---|---|---|
| `a5707e0` (anlık görüntü kaynağı) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `4134806` (`4ce66da`'dan hemen önce) | 0.00 | **210.59** | 0.00 | 0.00 | 0.00 |
| `4ce66da` | **241.43** | 30.84 | **34.88** | **14.75** | **349.98** |
| `be84e88` (kapı) | 241.43 | 30.84 | 34.88 | 14.75 | 349.98 |

Bu tablo iki atfı ve bir ortam etkisini ortaya koyuyor:

1. **`3705b15` Phase C4-I (DE440 üretim efemerisi) → yalnız D1b.** Harness'ın "DE421" durumu artık
   DE440 yüklüyor (bkz. G1 notu); `4134806`'da D1b, D1a ile bit bit aynı.
2. **`4ce66da` "align Hermite histories…" → beş durumun hepsi.** Tek faktörlü sınama: kapı ağacında
   yalnız `sample_moon_centered_ephemeris(..., earth_interpolation="legacy_pchip")` seçilince D1a
   47.84 mm'ye döner (Hermite ile 173.53 mm, ref3'e göre). Hermite Dünya konumunu SPICE'a karşı
   54 µm içinde veriyor; eski PCHIP 48.6 m hata yapıyordu. Yani interpolasyon **daha doğru**.
3. **Ortam etkisi (birleşmiş bir değişiklik değil).** Aynı `a5707e0` kaynağı bu worktree'nin
   ortamında saklanan `prod`'dan 2.0–571 mm ayrışıyor (D3 571.12, D2-100 35.14 mm). `ref3` her iki
   ortamda 0.000 mm tekrar üretiliyor.

**Neden:** `prod` sonucu VODE Adams'ın rtol 1e-11 / atol 1e-12'deki kesme hatasıyla sınırlı;
µm düzeyindeki sağ taraf değişikliklerine ve SciPy sürümüne kaotik tepki veriyor. Kapı ağacında
D1a'da, Adams toleransı sıkılaştırıldıkça ref3'e farkı:

| Dünya interpolasyonu | rtol 1e-11 | 1e-12 | 1e-13 | rtol 1e-11, atol 1e-9 |
|---|---|---|---|---|
| Hermite (`4ce66da` sonrası) | 173.53 mm | 22.30 mm | **2.42 mm** | 127.91 mm |
| eski PCHIP | 47.84 mm | 15.54 mm | 3.50 mm | 47.43 mm |

Doğrudan SPICE ile Adams 1e-11 bile 18.1 mm veriyor (ablasyon `direct_adams`). Hermite sıkı
toleransta ref3'e daha yakın iniyor. Eski PCHIP'in 1e-11'deki küçük hatası, interpolasyon ve
entegratör hatalarının kısmen birbirini götürmesindendi.

**G2 hükmü: PASS (atıflı), ön-kayıttaki kurala göre.** Bu worktree'nin ortamında görülen değişim,
iki adlandırılmış birleşmiş değişiklik (`3705b15`, `4ce66da`) ile Python ortam farkının
toplamıdır. Anlık görüntü ortamında ise yalnız iki commit'e tam atfedildi (0.00 mm'ye kadar).

**Yeni bulgu → X-04 / DYN-14 / WP3.3:** üretim `propagate_state` varsayılanı (VODE Adams
1e-11/1e-12, 600 s efemeris) bu 1 günlük yaylarda ref3'e göre 18–725 mm konum hatası veriyor ve
sonucu ortam sürümüyle 0.5 m'ye kadar değişiyor. Orekit'e 25 mm uyum yalnız `ref3` profiline
aittir. Bu profil üretim tahmincilerinin kullandığı profil olmadığından, "Orekit ile mm uyumu"
iddiası üretim yolu için geçerli değildir. WP3.3 entegratör sertifikasında profil başına geçerlilik
aralığı ve tolerans seçimi buna göre yapılmalıdır.

## G3 — Eşli benchmark (`examples/configs/benchmark.json`, 32 tohum × 2 gürültü)

Koşu: `examples/benchmark_estimators.py --output outputs/benchmark_f1gate`, commit `17a07b5`,
`git_dirty = False`, 80.9 s. Karşılaştırma: `docs/figures/benchmark/summary.json` (`732fda8`).

| Grup | başarı | sonlanma | medyan konum hatası | NEES ort. |
|---|---|---|---|---|
| gürültü 1 / BLS-LM | 1.00 → 1.00 | 1.00 → 1.00 | 53.26 → 52.28 m (**−%1.84**) | 5.01 → 4.99 |
| gürültü 1 / SRIF | 1.00 → 1.00 | **0.25 → 1.00** | 52.28 → 52.28 m | 4.99 → 4.99 |
| gürültü 1 / SR-UKF | 1.00 → 1.00 | 1.00 → 1.00 | 52.25 → 52.28 m | 4.99 → 4.99 |
| gürültü 3 / BLS-LM | 1.00 → 1.00 | 1.00 → 1.00 | 99.00 → 96.92 m (−%2.10) | 5.31 → 5.18 |
| gürültü 3 / SRIF | 1.00 → 1.00 | **0.44 → 1.00** | 96.92 → 96.92 m | 5.17 → 5.18 |
| gürültü 3 / SR-UKF | 1.00 → 1.00 | 1.00 → 1.00 | 96.93 → 96.92 m | 5.18 → 5.18 |

- Kriter 1 (her grup başarı = 1.0): **PASS**.
- Kriter 2 (SRIF sonlanma ≥ 0.95, iki gürültü grubunda): **PASS** (1.00 / 1.00; WP4.1 kabulü sağlandı).
- Kriter 3 (BLS-LM medyanı küratörlü değerin %1'i içinde): **FAIL** (gürültü 1'de −%1.84).

**Tanı (kriteri değiştirmez):** küratörlü koşuda BLS-LM ile SRIF aynı tohumda en çok 1.95 m
(gürültü 1) ve 43.1 m (gürültü 3) ayrışıyordu; şimdi en çok 0.001 m / 0.003 m. BLS-LM'in
durma nedeni iki koşuda da 32/32 `J-Stab`. Yani BLS-LM artık SRIF ile aynı minimuma iniyor;
küratörlü BLS-LM sayısı erken durmuş bir çözümdü. Ön-kayıttaki "hiçbir F1 değişikliği bu durumda
BLS-LM'i hedeflemiyor" varsayımı yanlış çıktı: konu commit'i F4 değişikliklerini de içeriyor
(WP4.2 yakınsama çıkışı ve LM güncellemesi, EST-04/05/08). Hangi tek commit'in kaydırdığı bu
kapıda ayrıştırılmadı. Kriter hesaplandığı gibi FAIL kalır. Yeni BLS-LM değeri SRIF ile 1 mm
içinde örtüştüğü için bilimsel olarak iyileşmedir; küratörlü benchmark (`docs/figures/benchmark/`)
sahip onayıyla yeniden kürate edilmelidir.

## G4 — Tam regresyon

`QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -u -m pytest -p no:cacheprovider -q -rfE
--junitxml=... tests`, commit `17a07b5`, 1007 s: **çıkış kodu 0; 2589 test (JUnit sayımı, alt testler
dahil), 0 başarısız, 0 hata, 32 atlanan.** 2441 test toplanıyor.

Not: aynı commit'teki ilk koşu ilerleme çıktısında [100%]'e ulaştı, F ya da E göstermedi; ama
pytest özet satırı yazılmadı, kabuk zinciri de çıkış kodunu yuttu. Bu yüzden o koşu kanıt
sayılmadı ve koşu JUnit raporu ile çıkış kodu kaydedilerek tekrarlandı.

## G5 — Çapraz hesap kalemleri (F1 kapı listesi)

| X | Konu | Durum | Kanıt / kalan |
|---|---|---|---|
| X-01 | efemeris / GM / gravite çerçevesi nesli | **kısmi** | GM DE440 (#99, `tests/test_ephemeris_generation_wp1_2.py`), çekirdek DE440 varsayılan (C4-I). Kalan: senaryo `lunar_gravity_frame` varsayılanı hâlâ `MOON_PA_DE421` (`src/lunar_od/scenario_config.py:188`) |
| X-02 | Ay J2 sabit IAU kutbu, (J2, R_ref) çifti | **açık** (main'de) | #100'de `xfail(strict)` ile ölçüldü (kutup ~1.5°, J2'nin %2'si). Sahip 2026-11-02'de varsayılanın MOON_PA_DE440 kutbu + GRGM1200A (C20, 1738 km) olmasına karar verdi; uygulama #100'ün üstünde ayrı PR'da (`f1/wp1-3b-j2-pole-grail`), bu kapının konusu dışında |
| X-03 | harmonik yönelim "en yakın örnek" | **PR'da kapalı** | #100 (SLERP), DYN-05 kanıt testi geçiyor; main'de açık |
| X-04 | truth / tahminci yayılım profili farkı | **kısmi** | #101 profil kaydı + isteğe bağlı ret; DYN-14 (skaler atol) ve DYN-15 (Numba fastmath) ölçümle çürütüldü; MEAS-05 (UKF doğrusal efemeris) açık; retin varsayılan olması **sahip kararı**. G2'nin bulgusu bu kaleme eklenir |
| X-05 | batch / UKF sayılan Doppler | **kapalı** | WP1.1, `test_meas04_rotation_interpolation_against_spice_sxform` |
| X-06 | SRIF maliyet yolu | **kapalı** | WP4.1, `tests/test_srif_acceptance_wp4_1.py`; G3'te SRIF sonlanma 1.00 |
| X-08 | `spice_ssb` aberasyon çerçevesi | **kapalı** | WP1.10, SPICE CN+S'ye ~1 mas |
| X-10 | UKF P0 / soğuk başlangıç | **kapalı** | WP1.9, `test_est19_ukf_default_p0_matches_injected_cold_start_sigma` |
| X-11 | masaüstü gizli varsayılanları | **açık** | `desktop_app/models/scenario_model.py:14,42-43` hâlâ efemeris adımı 3600 s, rtol 1e-8, atol 1e-10 (REQ-SW-006 açık) |
| X-12 | kuvvet parmak izi | **kısmi** | WP1.5 efemeris kimliği hash'leniyor; Ay dönüş modu henüz hash'lenmiyor (REQ-DYN-004) |
| X-13 | gravite yükleyici etiketi / STM | **kapalı** | WP1.7 (#96), `tests/test_gravity_frame_label_wp1_7.py` |

Bu PR'da doğrulama matrisinde REQ-DYN-001 satırı WP1.2 kanıtıyla `partial` yapıldı (#99 matris
satırını güncellememişti).

## Sonraki adımlar

1. Sahip kararları: X-04 (profil reddi varsayılan mı), G3 benchmark'ın yeniden küratörlüğü.
   X-02 kararı verildi (varsayılan değişiyor); uygulaması ayrı PR'da.
2. Açık iş: X-01 gravite çerçevesi varsayılanı `MOON_PA_DE440` (DE421 tabanlarına etkisi
   ölçülerek), X-11 masaüstü varsayılanlarının kütüphaneyle hizalanması, X-12 dönüş modu hash'i.
3. #100 ve #101 birleşince X-03 kapanır ve X-04 ilerler; ikisi arasında bir satırlık
   çözümleyici çakışması beklenir.

## Ek — 2026-11-03 gece ilerlemesi ve ajan kararları

Yukarıdaki kapı sonuçları hesaplandığı gibi kalır; bu ek yalnız sonraki işleri kaydeder.
Sahip 2026-11-03 gecesi "sabah dokuza kadar herhangi bir kararı bana sorma, sen devam et" dedi.
Aşağıdaki **ajan kararları** bu yetkiyle verildi ve **sahip onayı bekliyor**. Her biri ayrı PR'dır;
reddedilen PR kapatılır.

| X | Durum | PR | Not |
|---|---|---|---|
| X-01 | **PR'da kapalı** | [#107](https://github.com/halici21/lunar-orbit-determination/pull/107) | senaryo `lunar_gravity_frame` varsayılanı `MOON_PA_DE440`; harmonik kapalıyken etkisiz olduğu için sayısal değişim yok; eski masaüstü arşivleri açılmaya devam ediyor |
| X-02 | **PR'da kapalı** | [#103](https://github.com/halici21/lunar-orbit-determination/pull/103) | sahip kararı (2026-11-02); MOON_PA_DE440 kutbu + GRGM1200A; eski model bit-bit seçilebilir; 2602 test / 0 hata |
| X-03 | **kapalı** | #100 (birleşti) | — |
| X-04 | **kısmi, karar verildi** | [#101](https://github.com/halici21/lunar-orbit-determination/pull/101) | aşağıdaki karar 1; MEAS-05 açık |
| X-11 | **PR'da kapalı** | [#104](https://github.com/halici21/lunar-orbit-determination/pull/104) | rtol/atol/efemeris adımı/λ0 kütüphaneden; bağımsız soğuk başlangıç tohumu; eski job.json'lar kayıtlı tohumlarla çalışıyor |
| X-12 | **PR'da kapalı** | [#106](https://github.com/halici21/lunar-orbit-determination/pull/106) | harmonik `rotation_mode` parmak izinde; donmuş harmoniksiz parmak izleri değişmedi |

**Ajan kararı 1 — X-04, profil paritesi reddi varsayılan olmasın (isteğe bağlı kalsın).**
Gerçek değer (truth) yörüngesinin tahminciden daha sıkı bir sayısal profille üretilmesi kasıtlı
ve standart bir uygulamadır. Varsayılan ret bu uygulamayı yasaklar ve mevcut bütün senaryo
koşularını kırar. Adil karşılaştırmanın asıl koşulu, karşılaştırılan tahmincilerin (BLS-LM / SRIF /
SR-UKF) aynı profili paylaşmasıdır. #101 bunu `shared_profile` ve `profile_differences` olarak her
manifeste yazıyor; ret isteyen çağıran açıkça açabiliyor. Kalan X-04 işi MEAS-05'tir (UKF hızlı
sigma yayıcısının doğrusal efemerisi); bu, F2/F4'te Hermite efemerisle kapatılacak.

**Ajan kararı 2 — G3 referansı yeniden kürate edildi** ([#105](https://github.com/halici21/lunar-orbit-determination/pull/105)).
Aynı plan temiz main'de (`4d8819a`) yeniden koşuldu. BLS-LM ile SRIF artık tohum başına ±1 mm
içinde ve SRIF 32/32 sonlanıyor. G3 FAIL'i bu raporda hesaplandığı gibi kalır; #105 yalnız bir
sonraki kapının referansını değiştirir.

**F1 hükmü (güncel):** F1 kalemlerinin hepsi ya main'de kapalı ya da açık bir PR'da kapalı. Tek
istisna X-04'ün MEAS-05 parçasıdır. #101, #103, #104, #105, #106 ve #107 birleşince F1 kapısı
G3 referansı yenilenmiş olarak yeniden koşulmalıdır (yeni ön-kayıtla). Programın bir sonraki fazı
F2'dir: WP2.1, üretim yolunda SSB ışık zamanı + Shapiro (MEAS-01, R-01).
