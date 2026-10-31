# V&V programı: fazlar ve iş paketleri

Kaynak: [bağımsız değerlendirme](../validation/independent-assessment-2026-10-31.md). Bulgu kimlikleri (DYN-, MEAS-, EST-, RD-, CFG-, TST-) ve uyumsuzluk kimlikleri (X-) o belgededir.

**Her WP için standart prosedür** (mevcut proje yöntemi + NASA/ESA öğeleri):
1. Worktree: `git -C python_port worktree add -b <dal> ../python_port_<ad> origin/main`, kendi `.venv`'i.
2. **Ön-kayıt** `studies/phase17/configs/<wp>_predeclaration.json`: hipotez, kapılar, eşikler, veri
   bölümü (fit/validation/hold-out), reddetme koşulu. Sonuç görüldükten sonra değişiklik → "amendment
   log"a yazılır ve sonuç **keşifsel** sayılır (RD-04).
3. **Önce başarısız olan test** (bulguyu kanıtlar), sonra düzeltme.
4. Test kapısı: hedefli testler + tam regresyon (~11 dk, kernel'lı).
5. Bilimsel kapı: etkilenen metrikler, baseline farkı, hata bütçesi satırı.
6. Gereksinim matrisi güncellemesi (REQ ↔ test ↔ kapı).
7. Sahip onayı → PR → merge sahipte. Korunan dosyalar (`accelerated.py` vb.) için açık "re-freeze" onayı.
8. "Hesaplandığı gibi FAIL" gizlenmez.

### F0 — Temel çizgi, yönetim ve acil önlemler (SRR eşdeğeri)

| WP | İş | Dosyalar | Kabul |
|---|---|---|---|
| WP0.1 | Yerel main'i `origin/main`'e ff-only getir (Codex UI işi orada → sahibe sor); birleşmiş worktree'lerin (C1–C5, D1, D2, dmc, ekf, partials, f2fix, f2stop, integrate, d18d19, phase17_integ, ui_calm) silinmesi için sahip onayı | — | `git worktree list` temiz; kernel'lı tam regresyon sayısı kaydı |
| WP0.2 | **ACİL — F3 düzeltmesi (RD-01) ve DMC Q (EST-09)**: σ* seçimini fit-günü verisine (leave-one-pass-out ya da ortalama NIS≈1) veya ayrı validasyon gününe taşı; gün 263 yalnız puanlama. DMC Q'yu dt/τ alt-adımlarıyla birleştir. Ön-kayıt amendment olarak işle, sweep'ten **önce** | `origin/feature/f3-lro-ekf`: `phase_f3_lro_dmc_smoother.py`, ön-kayıt JSON; `src/lunar_od/dmc.py` | Yeni test `test_dmc_q_long_gap_finite_psd` (dt=86400, τ=600; alt-adım referansına 1e-10 bağıl); F3 sonucu bağımsız gün ile |
| WP0.3 | Güvence belgeleri `docs/assurance/`: `intended-use.md` (§0 varsayımı), `requirements.md` (REQ-DYN/MEAS/EST/IO/SW/DATA kimlikleri, doğrulama yöntemi T/A/I/D), `verification-matrix.csv`, `risk-register.md` (§5), `assumptions-and-limits.md` (§3), `amendment-log.md`; değerlendirme raporu `docs/validation/independent-assessment-2026-10-31.md` (§1–§4); mkdocs gezinmesi | `docs/`, `docs/mkdocs.yml` | `mkdocs build --strict`; her REQ en az bir test ya da "açık" |
| WP0.4 | Hold-out politikası: LRO günleri 264–268'i **mühürle** (hiçbir betik okumasın; `lro_data/holdout/` ayrı, hash kaydı); ileride tek seferlik dondurulmuş değerlendirme | `studies/phase17/configs/lro_data_partition.json` | Ön-kayıtta veri bölümü zorunlu alan |

### F1 — Tutarlılık ve doğruluk düzeltmeleri (en yüksek getiri, kısa)

| WP | İş | Dosyalar | Önce-başarısız test (kehanet, tolerans) | Kabul |
|---|---|---|---|---|
| WP1.1 | UKF/EKF sayılan Doppler: Dünya için batch'in Hermite yardımcısı; dönüşüm için doğrudan SPICE `sxform` ya da quaternion SLERP + ω×r; hızlı sigma yayıcısında Hermite efemeris (MEAS-04/05) | `filters.py:1867-2118`, `dynamics.py:1363`, `accelerated.py` (re-freeze) | `test_ukf_local_transform_orthonormal_and_vs_exact_sxform` (SPICE; ortho < 1e-12, istasyon < 1 mm); `test_ukf_vs_batch_counted_doppler_prediction_parity` (< 1e-6 m/s); `test_filters.py:50-80` yeniden yazılır | X-05 kapalı |
| WP1.2 | Efemeris nesli birliği: GM → DE440 (`gm_de440.tpc`), gravite çerçevesi varsayılanı `MOON_PA_DE440`, başlangıç durumu DE440, scenario doğrulaması gezegen efemerisiyle çapraz kontrol; DE421 tarihsel mod açık etiketle korunur (DYN-01/02, CFG-04) | `constants.py`, `scenario_config.py:183,1160`, `spice_loader.py`, `fixtures/` | `test_ephemeris_generation_consistency` (karışık nesil → ValueError); `test_constants_match_de440_gm` | X-01 kapalı; DE421 bit-bit testi geçer |
| WP1.3 | Ay yönelimi: harmonik ve J2 için exact-epoch `moon_pa_de440_rotation_at_et` ya da SLERP; J2'yi C̄20 harmonik yoluna yönlendir; (J2, R_ref) GRAIL başlığından (DYN-03/04/05) | `lunar_frames.py:159-189`, `dynamics.py:57-82,971,1127,1565`, `constants.py` | `test_moon_rotation_continuous_at_grid_midpoint`; `test_moon_pa_rate_vs_iau_wdot` (%0.5; mevcut %100 testi değişir); `test_j2_r2_product_matches_grail_c20` (1e-6) | X-02, X-03 kapalı |
| WP1.4 | Sayılan Doppler çözücü: yakınsama/denklem artıklarını yerel ışık zamanlarıyla (`two_way_range.py:316-330` deseni) (MEAS-03) | `radiometrics.py:1068-1171` | `test_counted_doppler_converges_on_long_arcs` (ızgara 3e5 s ve 2.5e6 s başlangıçlı, 1000 rastgele t_mid) | Uzun yayda hata yok |
| WP1.5 | Kuvvet parmak izi: SPK SHA, ızgara adımı, interpolasyon türü, pertürbasyon tohumu/genliği, dönüş modu, J2 kutbu sayısal; devre dışı terimler "not_applicable"; `bind_spec_to_runtime` Earth J2/harmonikleri bağlasın; `propagation_contract` gerçek gölge durumunu raporlasın (DYN-10/11/16) | `force_contract.py:444,504,517,551`, `propagation_contract.py` | `test_fingerprint_differs_for_perturbed_ephemeris`; `test_fingerprint_differs_for_rotation_mode`; `test_earth_j2_contract_requires_runtime_binding` | X-12 kapalı |
| WP1.6 | Tekrarlanabilir senaryo: şemaya `measurement_seed`, `cold_start_seed` (noise açıkken zorunlu), elevasyon maskesi/boşluk/efemeris adımı/yay modu alanları; bilinmeyen anahtar → hata; şema dataclass'tan üretilsin; CLI tüm alanları geçirsin; çıktılara commit, dirty, config hash, tohumlar, kernel manifest, paket sürümleri (`quickstart_od.source_identity()` yeniden kullan); masaüstü gizli varsayılanları kütüphaneyle hizala, iki tohumu `SeedSequence.spawn(2)` ile ayır (CFG-01…06,09,10) | `scenario_config.py`, `examples/run_scenario_config.py`, `reporting.py`, `desktop_app/models/scenario_model.py`, `desktop_app/services/scenario_execution.py`, `scenario_artifacts.py` | `test_scenario_json_fully_defines_run` (aynı JSON → bit-bit aynı çıktı); `test_unknown_config_key_rejected`; `test_desktop_effective_config_matches_library_defaults` | X-11 kapalı |
| WP1.7 | Gravite yükleyici etiketi açık PA gerçekleştirmesi; scenario'daki eski "gradient yok" engeli kaldırılır; `enable_*` her zaman hata veren alanlar şemadan gizlenir (DYN-12, CFG-07) | `gravity_model_loader.py:443`, `scenario_config.py:~1173` | `test_loaded_gravity_model_accepted_by_stm_path` | X-13 kapalı |
| WP1.8 | Ortak yayılım profili: truth ve tüm tahminciler aynı `PropagationProfile` kullanmadıkça karşılaştırma reddedilir; konum/hız/STM için ayrı atol vektörü; Numba Pines `fastmath=False` ya da N≥300 parite testi (DYN-14/15) | `dynamics.py`, `propagation_contract.py`, `accelerated.py` (re-freeze) | `test_estimator_comparison_requires_shared_profile`; `test_pines_numba_parity_n300` | X-04 kapalı |
| WP1.9 | UKF P0'ı `THESIS_COLD_START_SIGMA_*` × `cold_start_scale`'dan kur ve kaydet (EST-19) | `scenarios.py:1427-1434` | `test_ukf_p0_matches_injected_cold_start` | X-10 kapalı; paired benchmark yeniden koşulur |
| WP1.10 | `spice_ssb` aberasyonu: ya SSB ışık zamanıyla tutarlı hale getir ya kaldır; belgeyi düzelt (MEAS-02) | `measurements.py:1354-1561`, `docs/theory/stellar_aberration_od_impact.md` | `test_apparent_direction_vs_spice_cn_s` (`spkpos "CN+S"`; < 0.1″) | X-08 kapalı |

**F1 faz kapısı:** X-01…X-06, X-08, X-10…X-13 kapalı; tam regresyon; paired benchmark ve 1 günlük
Orekit noktaları yeniden üretilir, farklar ön-kayıttaki aralıkta; hata bütçesine satırlar.

### F2 — Ölçüm modelinin üretimleştirilmesi (Moyer / DSN 810-005 / IERS 2010)

| WP | İş | Önce-başarısız test / kehanet | Kabul |
|---|---|---|---|
| WP2.1 | SSB ışık zamanı + Shapiro tüm gözlemlerde (sayılan Doppler, ΔDOR, tek yönlü, konum) üretim tahmincilerine; profil `scenario_config.ALLOWED_MEASUREMENT_MODEL_PROFILES`'a; Ay merkezli model "legacy_synthetic" etiketiyle; Shapiro'da cisim her uçta kendi epoch'unda; iki-döngü toleransı 1e-13 s + sayaç metadata (MEAS-01/11/12) | `test_one_way_lt_vs_spice_cn` (1 ps); `test_shapiro_vs_moyer_closed_form` (1 µm); `test_two_way_range_vs_mpmath_moyer` (1 mm); Orekit fixture'ları (2YM < 1 mm, Doppler < 1 µm/s) | X-07 kapalı |
| WP2.2 | Medya (Saastamoinen/IERS ZHD + Niell + iyonosfer) ve topocentric istasyon zamanı üretim yoluna; Az/El için kırılma (Bennett/Saastamoinen) ya da "kırılma düzeltilmiş" girdi zorunluluğu (MEAS-06/07) | `test_saastamoinen_vs_iers_tn36_example` (0.1 mm); `test_niell_mapping_vs_orekit` (1e-6); `test_refraction_vs_published_table` | — |
| WP2.3 | Rampalı uplink (ramp tablosu), iki yönlü menzil istasyon bias + zaman etiketi bias solve-for, SSB yolunda transponder gecikmesi, istasyon başına/geçiş başına bias seçeneği (MEAS-08, RD-07/08/09) | `test_ramped_doppler_vs_constant_limit`; `test_two_way_range_bias_solve_for_recovers` | X-15 kapalı |
| WP2.4 | İstasyon zinciri: ITRF2020 + IERS EOP (finals2000A) ya da yüksek hassasiyetli Dünya BPC; levha hareketi; kutup gelgiti; okyanus yüklemesi (HARDISP/BLQ); predict BPC kapsam dışı → fail-closed; NAIF DSN istasyon SPK'sı ile çapraz kontrol (MEAS-06) | `test_j2000_itrf_vs_erfa_c2t06a_with_eop` (1 mas); `test_solid_tide_vs_dehanttideinel_testcase` (1 µm); `test_dsn_station_state_vs_earthstns_spk` (1 mm) | X-09 kapalı |
| WP2.5 | Varsayılan range-rate profilini açıkça seç; Doppler modunda eşlik eden menzil/açılar da ışık zamanlı; konum gözlemlerinde varsayılan Jacobian `implicit_light_time` (MEAS-09/10) | Jacobian sonlu fark: Richardson ile yakınsama derecesi doğrulaması (en iyi adım seçimi değil) | — |
| WP2.6 | Gizli modüller: landmark optik (dC/dr, MOON_ME→PA, piksel merkezi) ve ΔDOR kuasar terimi (IERS consensus) düzelt ya da "deneysel" işaretle (MEAS-13) | `test_pinhole_jacobian_with_nadir_attitude_fd` | — |
| WP2.7 | Görünürlük: görünür (ışık zamanlı) geometri, LOLA DEM ile arazi örtmesi (opsiyonel), düzgün olmayan ızgara; `atan2` elevasyon ve az sarma (MEAS-14) | `test_visibility_nonuniform_grid`; `test_azel_edge_cases` | — |

**F2 faz kapısı:** gerçek LRO fit günlerinde artık RMS/bias'ın F2p'ye göre değişimi raporlanır (hold-out'a dokunmadan).

### F3 — Dinamik genişletme (önce etki büyüklüğü taraması, C1 tarzı)

| WP | İş | Kehanet / test | Kabul |
|---|---|---|---|
| WP3.1 | Materiality taraması: Ay k2 gelgiti (IERS tarzı ΔC2m(t), `tide_system` ayrıştırma), albedo/IR (Knocke), Dünya gölgesi, 1PN Schwarzschild, Venüs/Jüpiter, box-wing/panel SRP (LRO), Dünya J2 yönelimi, Güneş ışık zamanlı yön | her terim için ivme + 1 günlük yörünge etkisi, OD'de |δθ/σ| | Eşiği aşanlar WP3.2'ye |
| WP3.2 | Seçilen terimlerin uygulanması + analitik STM katkıları | `test_<term>_vs_published_amplitude`; complex-step STM (rel < 1e-9); Orekit fixture (< 1 mm/gün, terim başına) | Hata bütçesine satır |
| WP3.3 | Uzun yay entegratör sertifikası 1/3/7/28 gün, tüm profiller; referans segmentli sıkı DOP853 + Orekit | `test_two_body_10rev_closure` (1 mm; enerji 1e-11) | Profil başına geçerlilik aralığı belgelenir |
| WP3.4 | LRO dinamik hipotezleri (F2r'nin açık kalanları): GRGM900C N=600 kendi çerçevesinde, GL1800F N=600, k2, panel SRP, C_N/S_N serbest, zaman etiketi bias'ı, LOLA'dan bağımsız bias önselleri; F2s için sentetik konik ile tespit gücü (RD-07/10, RD-03) | Her biri ön-kayıtlı, yalnız fit/validasyon günleri | F2r kalıntısının açıklanan payı raporlanır |

### F4 — Tahminci sağlamlığı ve istatistiksel tutarlılık

| WP | İş | Önce-başarısız test | Kabul |
|---|---|---|---|
| WP4.1 | SRIF: mevcut ve aday maliyet aynı yol + aynı tolerans + MAP önseliyle (`_acceptance_residual`, `_weighted_map_cost` deseni); trust-region/LM sönümleme (EST-01/02/03) | `test_srif_zero_step_cost_identity` (step=0 → maliyetler eşit); `test_srif_converges_to_map_with_tight_prior` | Benchmark'ta SRIF yakınsama ≥ %95 |
| WP4.2 | Yakınsama yalnız istenen toleransta + GN azalması ≤ 0.1 ile ilan; adım normu posterior-σ biriminde; Nielsen λ güncellemesi, Moré D (EST-04/05/08) | `test_convergence_exit_at_requested_tolerance`; `test_heavy_damping_not_labelled_converged` | — |
| WP4.3 | Tüm posteriorlar `_square_root_covariance_from_design` ile; tek ve dengelenmiş koşul eşiği (~1e10 oranı) BLS/SRIF için ortak; gözlemlenebilirlik ölçekli Jacobian'dan (EST-06/07/18) | `test_station_angles_covariance_vs_qr_reference` (1e-8); `test_bias_classification_unit_invariant` | X-14 kapalı |
| WP4.4 | SR-UKF: downdate geri dönüşü (QR yeniden faktörleme), ortalama `X0 + Σ w_i (X_i − X0)`, varsayılan α gerekçeli, göreli jitter, regülarizasyon bir kez, dondurulmuş durumlar Schmidt-Kalman, süreç gürültüsü varsayılanı SNC (dt bağımlı) (EST-10…14) | `test_ukf_identity_process_preserves_P` (1e-9); `test_angle_update_sqrt_vs_joseph`; `test_regularized_state_keeps_prior_sigma`; `test_snc_q_independent_of_sampling` | Yakalanmamış istisna yok |
| WP4.5 | Korelasyonlu gürültü: istasyon başına AR(1) `ρ^(Δt/τ)`; blok-R Cholesky beyazlatma ya da etkili örnek boyutu ile ağırlık şişirme; consider kovaryans raporu (EST-15, RD-06) | `test_ar1_inter_pass_correlation_near_zero`; `test_whitened_residuals_white` | — |
| WP4.6 | Tanılar: ANEES/ANIS iki taraflı χ² aralıkları, ham/düzeltilmiş NIS ayrı, artık otokorelasyonu kanal ve zaman bazında, Ljung–Box, reddedilenler hariç ortalama NIS; batch ayıklama a posteriori σ ve iteratif (EST-16/17) | `test_anees_bounds_match_chi2` | — |
| WP4.7 | Monte Carlo: `SeedSequence([base, trial])` + ortak rastgele sayılar; 100+ deneme; model-uyumsuzluk kampanyası (truth 100×100, filtre 20×20 + consider) (EST-20) | `test_ukf_bls_mc100_nees_nis_chi2_bounds` (NEES ort. [5.37, 6.69]); `test_truth_100x100_filter_20x20_consider_nees` (%99) | — |
| WP4.8 | DMC'yi SR-UKF ve batch'e; RTN eksenli, ayrı along/cross σ, τ ~ yörünge periyodu, σ NIS ile seçilir (EST-21, RD-12) | `test_dmc_rtn_vs_inertial_equivalence` | — |

### F5 — Bağımsız doğrulama (IV&V) ve gerçek veri

| WP | İş | Kabul |
|---|---|---|
| WP5.1 | Orekit kehaneti repoda kalıcı: `tools/oracle/` üretim betikleri (Orekit isteğe bağlı), sabitlenmiş referans CSV'leri (provenans: Orekit sürümü, kernel SHA, sabitler), `tests/test_external_oracle.py` (Orekit gerektirmez), `docs/external_cross_validation.md`; sabit/çerçeve/zaman hizalaması belgelenir; GMAT ikincil (ölçüm fiziği) | Dinamik < 1 mm/gün (hizalı sabitlerle), 2YM < 1 mm, Doppler < 1 µm/s |
| WP5.2 | pyerfa, pyshtools, SPICE çağrılı L2 testleri (`test_grail_100x100_accel_vs_pyshtools` rel < 1e-12; `test_utc_tt_tdb_chain_vs_erfa_1972_2050` 1 ns) | CI'da kernel önbelleğiyle |
| WP5.3 | LRO dallarını kapat: F2r/F2s (`phase-f1-lro-residuals`) ve F3 (WP0.2 düzeltmesiyle) main'e ya da arşive | Açık dal yok |
| WP5.4 | **Tek seferlik hold-out değerlendirmesi**: F1–F4 sonrası dondurulmuş hattı (model, ayıklama, önsel, çözücü; commit SHA ön-kayıtta) günler 264–268'de koş; birincil metrik görülmemiş izleme verisine tahmin artıkları (beyazlık, χ²), ikincil LOLA radyal < 1 m ve 3B RMS | Kabul kriteri (§0) ya da gerekçeli FAIL |
| WP5.5 | İkinci veri seti: GRAIL veya LRO DSN TRK-2-34 (okuyucu), JPL yeniden yapılandırılmış SPK ile karşılaştırma; `test_grail_real_doppler_residual_rms` (L5; RMS < 0.1 mm/s, Ljung–Box p > 0.01) | ≥2 bağımsız gerçek yay |
| WP5.6 | UTDF sağlamlaştırma: sayım hızı/rollover kontrolü + gerekçe kodu, `pad_xmit == pad_rcv`, artık saniye fixture'ı, kayıttaki frekansların kullanımı, gerçek arşiv kaydıyla test (RD-11) | `test_utdf_decode_real_archived_record`; `test_utdf_leap_second_crossing` |

### F6 — Standart arayüzler (CCSDS)

| WP | İş | Test |
|---|---|---|
| WP6.1 | CCSDS TDM (503.0-B) KVN okuyucu/yazıcı; `measurement_ingestion` TDM'den | CCSDS örnek dosyalarıyla gidiş-dönüş; Orekit ayrıştırıcısıyla çapraz okuma |
| WP6.2 | OEM/OPM/OCM (502.0-B) yazıcı, kovaryanslı; masaüstünden dışa aktarma | Orekit ile okuma + konum farkı 0 |
| WP6.3 | TRK-2-34 okuyucu (WP5.5 ile ortak) | gerçek kayıt fixture |

### F7 — Yazılım güvencesi (NPR 7150.2 Sınıf D / ECSS-Q-ST-80C hafif)

| WP | İş | Kabul |
|---|---|---|
| WP7.1 | pytest işaretçileri: `analytic`(L0), `derivative`(L1), `external_oracle`(L2), `model_mismatch`(L3), `statistical`(L4), `real_data`(L5), `golden`(L6), `contract_negative`(L7), artı `spice`, `campaign`, `qt`, `numba`, `slow`; CI işaretçiye göre; L2+ kernel eksikse **FAIL** | `pyproject.toml` |
| WP7.2 | CI: her commit L0+L1+L7; her PR L2 (önbellekli kernel); nightly L3+L4+tam regresyon (self-hosted Windows ya da önbellekli Ubuntu); release L5 | `.github/workflows/scientific-nightly.yml` |
| WP7.3 | Test hijyeni: `kclear()` test gövdelerinden kaldır, SPICE havuzu kaydet/geri yükle fixture'ı; conftest'in import-anı DE421 geçişini fixture'a taşı; zamanlı hız testini benchmark işaretçisine; çalışma JSON'larına bağlı testleri `golden` işaretle; `test_harden_mutants`'ı kampanyadan bağımsız fixture'la CI'ya; bilinen-yanlış sabitleyen testleri (TST-03) yeniden yaz; gevşek toleransları sıkılaştır (gürültüsüz kurtarma < 1 mm; Ay dönüş hızı %0.5; Richardson FD) | Rastgele sırada (`pytest-randomly`) 3 koşu yeşil |
| WP7.4 | Kernel manifesti (ad + SHA-256 + kaynak URL) `spice_loader.py` yüklemede doğrular; tüm çıktılara yazılır | Yanlış kernel → hata |
| WP7.5 | ruff + mypy (önce `src/lunar_od` çekirdek), coverage (kritik modüller ≥ %80: radiometrics, estimators, filters, dynamics, relativistic_light_time), eksik test dosyaları (`test_radiometrics.py`, `test_propagation_contract.py`, `test_measurement_contracts.py`, `test_constants.py` IAU 2015/IERS 2010'a karşı) | CI'da eşik |
| WP7.6 | Bağımlılık: numba/numpy üst sınırları, `pyproject` extras (desktop, accelerated, dev, examples), hash'li kilit dosyası, Linux pin seti | `pip check` yeşil |
| WP7.7 | Tekilleştirme: `C_LIGHT` tek kaynak (`constants`), yetkili sayılan Doppler/efemeris uygulamaları belgelenir; büyük çalışma artefaktları release asset/LFS; `archive/` WIP kopyaları dala | Depo boyutu raporu |

### F8 — Güvenilirlik değerlendirmesi ve sürüm (NASA-STD-7009A)

| WP | İş |
|---|---|
| WP8.1 | 8 faktörlü güvenilirlik tablosu (verification, validation, input pedigree, results UQ, results robustness, use history, M&S management, people qualifications), her faktör için kanıt bağlantısı |
| WP8.2 | UQ raporu: sayısal hata bütçesini **ölçüm → yörünge doğruluğu** bütçesine genişlet (model hatası terimleri F2/F3'ten, consider analizi F4'ten) |
| WP8.3 | ECSS/NASA belge seti (hafif): SRS = `requirements.md`, ICD (veri formatları, çerçeve/zaman sözleşmeleri), SVVP (bu program), SVR (doğrulama matrisi çıktısı) |
| WP8.4 | Sürüm `0.2.0`: değerlendirme raporu, izlenebilirlik matrisi kapalı, risk kaydı güncel |

### Faz bağımlılıkları ve paralellik

```
F0 (WP0.1–0.4; WP0.2 ACİL)
 ├─► F1 ─┬─► F2 ─┐
 │       ├─► F3 ─┼─► F5 ─► F8
 │       └─► F4 ─┘
 ├─► F6 (bağımsız)
 └─► F7 (bağımsız; WP7.1–7.3 F1 ile paralel)
```
F1 içinde WP1.1, WP1.4, WP1.5, WP1.6, WP1.9 farklı dosyalara dokunur → paralel yürütülebilir; WP1.2/1.3/1.7/1.8 ortak dosyalar (`dynamics.py`, `scenario_config.py`) → sıralı.

---
