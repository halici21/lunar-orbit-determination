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

Kaynak: [bağımsız değerlendirme RD-04](../validation/independent-assessment-2026-10-31.md).
Tarihler çalışma raporlarındaki commit tarihleridir; tam kimlikler `studies/phase17/reports/` içinde.
