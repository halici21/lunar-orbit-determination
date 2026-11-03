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
| 2026-11-03 | WP2.1 (G1, G4) | G1: SPICE CN'e karşı tek yönlü ışık zamanı toleransı 1e-12 s → 2e-11 s (ölçülen 4.0e-12 s; SPICE'ın kendi SSB durumları yuvarlanmış et'de: ~30 km/s × ulp(et) ≈ 3.6 mm kehanet tabanı). G4: Jacobian sonlu fark adımları 1 m / 1e-3 m/s → 10 m / 1e-2 m/s ve satır normuna göre tolerans (FD tahmini ~1/h ile kayıyordu: yeniden yayılım gürültüsü; nitelikli Ay merkezli Jacobian aynı tabanı gösteriyor). Ön-kayıt: `studies/phase17/configs/wp2_1_predeclaration.json` | Hayır | keşifsel (G1, G4)|
| 2026-11-03 | WP2.1 (G8) | Ay merkezli iki yönlü menzilin SSB verisine uyumsuzluk bandı 0.5–5 m → 7–12.5 m (ölçülen 11.5–12.2 m). Ön-kayıtlı bant yalnız C1 çerçeve terimini taşıyordu; C1'in kendi listelediği Güneş Shapiro gecikmesini (7.3–7.8 m) atlamıştı. Bağımsız tahmin ≈ 11.9 m. Shapiro payı kapalı formla ayrıca sınandı | Hayır | keşifsel (G8)|

Kaynak: [bağımsız değerlendirme RD-04](../validation/independent-assessment-2026-10-31.md).
Tarihler çalışma raporlarındaki commit tarihleridir; tam kimlikler `studies/phase17/reports/` içinde.
