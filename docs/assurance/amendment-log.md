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
| 2026-11-07 | WP3.1-R (A1) | G1(b)'nin "v, u boyunca" durumu "(1 − v/c) çarpanı" yerine "a_LT = (1 − 2v/c) a_SRP" oldu. Beyan edilen formül [(1 − v·u/c)u − v/c] bu durumda (1 − 2v/c)u verir: gelen akının Doppler çarpanı ve sapınım terimi u boyunca ayrı ayrı −v/c katar (Poynting–Robertson; Burns, Lamy & Soter 1979, denk. 2). Formül değişmedi; eski test cümlesi formülle çelişiyordu. Testler yazılırken, hiçbir test, kapı ya da tarama sayısı yokken bulundu | Evet | düzeltme; çalışma onaylayıcı kalır |
| 2026-11-07 | WP3.1-R (A2) | Taramadan önce, kuvvet terimi içermeyen bir entegratör tanısı O1×E1 hücresinin G4 tabanını ön kayıtlı profilde (tek DOP853 koşusu, shadow_events False) 1 günlük yayda 1.434 m ölçtü: G4 ön kayıtlı haliyle orada FAIL (> 1e-3 m). Üretimin tutarlı profilinde (PropagationProfile(): DOP853, shadow_events True) 6.0e-5 m. Neden: ay gölgesi ivmesinin her yarıgölge kontağındaki kırığı adımların içine düşüyor (D1: ~1.2 günlük yayda 0.1–1.9 m). O1×E1'in 1 saatlik smoke koşusu (tüm terimler) da görüldü: ön kayıtlı profilde G5'in K_SRP sütunu 0.17 sapıyor. Değişiklik: taramanın tüm yayılımları (taban, terimler, G3 referansı, G4, K duyarlılığı, G5 sonlu farkları) üretimin olay-bölütlemeli profiliyle ikinci bir tam koşu. Terimler, hücreler, kapılar, eşikler ve önemlilik kuralı aynı. Ön kayıtlı koşu da tam koşulur ve hesaplandığı gibi raporlanır | Hayır | keşifsel; REQ-DYN-007'yi D2 kararlarıyla kapatmak sahip kararı |

Kaynak: [bağımsız değerlendirme RD-04](../validation/independent-assessment-2026-10-31.md).
Tarihler çalışma raporlarındaki commit tarihleridir; tam kimlikler `studies/phase17/reports/` içinde.
