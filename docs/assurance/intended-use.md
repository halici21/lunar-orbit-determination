# Kullanım amacı ve kabul ölçütleri

## Kullanım amacı

Alçak Ay yörüngesindeki (LRO benzeri, 30–200 km) bir uzay aracı için yer istasyonu radyometrik
izlemesiyle (iki yönlü menzil, sayılan Doppler; isteğe bağlı açı ve ΔDOR) araştırma sınıfı yörünge
belirleme: tahminci karşılaştırması (BLS-LM, SRIF, SR-UKF, EKF/RTS), model etkisi çalışmaları ve
gerçek veriyle yeniden yapılandırma.

**Uçuş yazılımı değildir.** Operasyonel manevra planlaması, çarpışma kaçınma ya da görev güvenliği
kararları için kullanılamaz. NPR 7150.2 Sınıf D, ECSS kritiklik D muamelesi.

## Kabul ölçütleri

| Kimlik | Ölçüt | Kanıt |
|---|---|---|
| AC-1 | Sentetik, model-uyumlu koşulda ≥100 tohumlu Monte Carlo: ANEES ve ANIS iki taraflı %95 χ² aralığında | WP4.7 |
| AC-2 | Sentetik, model-uyumsuz koşulda (truth ≠ estimator): consider kovaryansı ile NEES %99 aralığında | WP4.7 |
| AC-3 | Bağımsız dış referansa (Orekit, hizalı sabit/çerçeve/zaman) karşı: dinamik < 1 mm/gün; iki yönlü menzil < 1 mm; sayılan Doppler < 1 µm/s | WP5.1 |
| AC-4 | Gerçek LRO verisi, **daha önce hiç okunmamış** günlerde (264–268) dondurulmuş hatla tahmin: artıklar beyaz (Ljung–Box p > 0.01) ve normalize RMS ≈ 1 | WP5.4 |
| AC-5 | Aynı günlerde NASA yayımlanmış yörüngesine karşı radyal < 1 m, 3B RMS ≤ 10 m (LOLA'nın kendi doğruluk tabanı; altındaki farklar sıralama için kullanılmaz) | WP5.4 |
| AC-6 | Her yayımlanan sayı commit, config hash, tohum, kernel manifesti ve paket sürümleriyle yeniden üretilebilir | WP1.6, WP7.4 |

## Kullanım dışı

- LOLA/SPK doğruluk tabanının (~10 m) altındaki yörünge farklarıyla model sıralaması.
- Yalnız aynı-model sentetik gidiş-dönüşe dayanan doğruluk iddiası (bkz.
  [varsayımlar](assumptions-and-limits.md)).
- `predict` Dünya yönelim kernel'inin kapsadığı tarih dışı istasyon zinciri.
- Masaüstü uygulamasının gizli varsayılanlarıyla (CFG-01) üretilmiş bilimsel sayılar.
