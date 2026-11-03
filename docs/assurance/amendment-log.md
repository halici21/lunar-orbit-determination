# Ön-kayıt değişiklik günlüğü

Bir ön-kayıtlı kapı, eşik, seçim kuralı ya da veri bölümü **sonuç görüldükten sonra** değiştiğinde
buraya bir satır eklenir. Böyle bir değişiklikten sonraki sonuç **keşifsel** sayılır ve
[kullanım amacındaki](intended-use.md) AC-4/AC-5 için ancak dokunulmamış hold-out verisinde
tekrarlanarak kanıt olur.

| Tarih | Çalışma | Değişiklik | Sonuçtan önce mi? | Durum |
|---|---|---|---|---|
| 2026-10 | F2i | G1 kuralı, G1 FAIL sonrası göreli (< 1e-3 χ²) yapıldı | Hayır | keşifsel |
| 2026-10 | F2j-G1 | A1/A2 değişiklikleri D0–D4 tanıları sonrası yazıldı | Hayır | keşifsel |
| 2026-10 | F2m | 5× medyan Doppler eşiği, bozuk geçişin (69 mm/s) ve diğerlerinin (4–8 mm/s) bilinmesiyle seçildi | Hayır | keşifsel; F2p'de etkisi ≤ 0.15σ |
| 2026-10 | F2r | Ölçüt 3–5 sayısal eşikleri E2 sonrası belirlendi | Hayır | keşifsel |
| 2026-10 | F2s → F2s-1 | Sonlu fark adımı yöntem kontrolü FAIL sonrası değişti | Hayır | keşifsel |
| 2026-10-31 | F3 | σ* gün 263'ün veriye karşı χ²'si ile seçildi ve H2 aynı günde puanlandı (RD-01). Düzeltme sweep'ten önce yetişmedi; sonuç F3_NO_DMC_GAIN (H2 sağlanmadı), yani yanlılık sahte bir PASS üretmedi. F3'ün 263 tahmin kazancı (χ² 69 900 → 3 643) seçim günüyle aynı günde ölçüldüğü için iyimser sayılır ve bağımsız kanıt değildir | Hayır | keşifsel; F3b/F3c bulguları da 263'e bağlı |
| 2026-10-31 | WP0.2 | Bundan sonraki her LRO ön-kaydı ayar (σ, eşik, model seçimi) için fit günlerini ya da ayrı validasyon gününü, puanlama için ayrı günü adlandırır; aynı gün ikisine kullanılamaz | Evet | kural |
| 2026-11-02 | WP4.7b (AC-1) | Ön-kayıtlı koşu (taban 20261102, 100 deneme, 3601 s konum yayı, 200 m / 0.2 m/s soğuk başlangıç): BLS ANEES 5.168, AC-1'in istatistik başına %95 aralığının [5.34, 6.70] dışında; UKF 5.509 ve ANIS 1.0007 içinde. Üç istatistik birlikte sınandığı için aralık istatistik başına %99'a (aile düzeyinde ~%97) genişletildi; BLS 5.168 %99 aralığında [5.15, 6.93]. Sonradan bağımsız tekrar (taban 7, 200 deneme): BLS 5.85, UKF 6.13, ANIS 1.0015, %95 aralığında | Hayır | keşifsel; AC-1 harfiyen (%95, istatistik başına) ilk koşuda BLS için sağlanmadı |
| 2026-11-03 | WP2.4 dilim 3 (G2–G5 tarihleri) | Sahip onayıyla yüksek hassasiyetli Dünya PCK'sı güncellendi (earth_000101_270128_261102.bpc, son ölçüm 2026-11-02; sürüm adıyla, eski rolling dosya LRO çalışmaları için dokunulmadı). Dilim 1'in G2–G5 kapıları eski sürümün tarihlerini sabit yazmıştı; kurallar aynı kalarak tarihler sabitlenen çekirdeğin kendi meta verisinden (pckcov kapsamı, yorumdaki son ölçüm) türetildi. Dilim 1 sonuçları eski sürüme karşı kayıtlı kalır. Keşifsel değil: değişiklik görülen bir kapı değerinden değil çekirdek güncellemesinden. |

Kaynak: [bağımsız değerlendirme RD-04](../validation/independent-assessment-2026-10-31.md).
Tarihler çalışma raporlarındaki commit tarihleridir; tam kimlikler `studies/phase17/reports/` içinde.
