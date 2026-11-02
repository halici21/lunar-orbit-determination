# Varsayımlar ve kullanım sınırları

Bu sayfa, `main`'in bugünkü haliyle üretilen bir sonucu yorumlarken bilinmesi gerekenleri listeler.
Ayrıntı ve kanıt: [bağımsız değerlendirme §2–§3](../validation/independent-assessment-2026-10-31.md).
Bir madde ilgili iş paketiyle kapanınca buradan kaldırılır ve değerlendirmedeki satır güncellenir.

## Sentetik sonuçlar

1. **Aynı-model gidiş-dönüş tutarsızlıkları gizler.** Gerçek değer ve tahminci aynı basitleştirilmiş
   modeli paylaştığında X-01 (efemeris/GM/çerçeve nesli), X-02 (Ay J2 kutbu), X-07 (Ay merkezli ışık
   zamanı) ve X-09 (istasyon zinciri) birbirini götürür (X-08 aberasyon çerçevesi WP1.10'da kapatıldı). Sentetik doğruluk
   sayıları gerçek veri doğruluğu değildir.
2. **Tahminci karşılaştırmaları** şu farkları içerir: UKF hızlı yayıcıda
   RK4 + doğrusal efemeris (X-04, MEAS-05), BLS ve SRIF'in
   farklı tekillik eşikleri (X-14). (X-06, SRIF adım kabulündeki yol farkı, WP4.1'de kapatıldı.)
3. **Entegrasyon toleransları** yalnız 8 481 s'lik yaylar için sertifikalıdır; BLS ilk iterasyonlarda
   rtol 1e-8'e gevşer ve bu evrede yakınsama ilan edebilir.
4. **Kuvvet parmak izi** yürütülen efemeris kernel'ini, interpolasyonunu ve Ay dönüş modunu
   kapsamaz; "force parity PASS" bu eksenlerde eşitlik kanıtı değildir.
5. **Varsayılan Ay J2 terimi** (X-02, sahip kararı 2026-11-02) tam epoch'taki MOON_PA_DE440
   kutbu etrafındadır ve GRAIL GRGM1200A (C̄20, R_ref = 1738 km) çiftini kullanır
   (`J2_MOON_GRAIL_UNNORMALIZED`, `R_MOON_GRAIL_REF_M`). Kutup her sağ taraf çağrısında
   `pxform` ile alınır; hızlı UKF sigma yayıcısı aralık ortasındaki kutbu kullanır (1 günde
   2.2e-4 rad sürüklenme). Model yayın epoch'unu ister: `lunar_j2_epoch_et0`, harmonik epoch ya da
   efemeris nesnesinin `et0_s`'i; epoch yoksa ya da eski katsayı verilirse hata verir.
   Eski IAU sabit ortalama kutup modeli (`J2_MOON_UNNORMALIZED`, `R_MOON_M`; figür ekseninden
   1.47–1.62°, DYN-03; J2·R² 4.7e-4 yüksek, DYN-04) `lunar_j2_model="iau_mean_pole_legacy"` ya da
   `use_lunar_j2_model(...)` ile bit-bit aynı seçilebilir; 2026-11-02 öncesi dondurulmuş J2
   tabanları bu modelle yeniden üretilir (R0B `FROZEN_LUNAR_J2_FINGERPRINT` dahil). Izgaralı
   harmonik yolda Ay yönelimi WP1.3'ten beri SLERP ile süreklidir (DYN-05 kapandı).

## Uzun yaylar

5. ~~Tek-sekme sayılan Doppler çözücüsü yay başlangıç epoch'u ~1.3e5 s'yi aştığında giderek daha sık
   `converged=False` döndürür (2.5e6 s'de %94); 28 günlük yay-yay kampanyaları bu yoldan geçemez.~~
   **Kapandı (F1 WP1.4, MEAS-03):** yakınsama ve denklem artıkları yerel ışık zamanlarıyla
   hesaplanıyor; 3e5 s ve 2.5e6 s başlangıçlı ızgaralarda 1000/1000 nokta yakınsıyor.
6. ~~DMC süreç gürültüsü dt/τ ≳ 30 olduğunda hassasiyetini, ≳ 100'de pozitif yarı tanımlılığını kaybeder.~~
   Kapatıldı (WP0.2): βdt > 1 için kapalı form; dt/τ = 5000'e kadar 60 basamaklı kehanete ≤ 1.5e-12 (WP4.8a).

## Gerçek veri

7. Üretim gözlemleri Ay merkezli Newton ışık zamanı kullanır; Shapiro, medya, topocentric zaman ve
   güncel ITRF yalnız iki yönlü menzilin SSB çalışma yolundadır.
8. LRO günleri 262 ve 263 model seçimi, ayıklama ve tanı için kullanılmıştır; bunlar üzerindeki
   tahmin sonuçları bağımsız validasyon değildir. İstasyon bias önselleri LOLA'ya göre artıklardan
   türetilmiştir.
9. LOLA yörüngesinin beyan edilen doğruluğu (~10 m toplam, < 1 m radyal) altındaki farklarla model
   sıralaması yapılmaz.
10. Doppler artıkları güçlü otokorelasyonludur (lag-1 0.74–0.81); beyaz ağırlıklandırmayla verilen
    formal σ'lar 2.5–3× iyimserdir.

## Masaüstü ve CLI

11. Masaüstü uygulaması formda görünmeyen rtol 1e-8 / atol 1e-10 ve 3600 s efemeris adımı kullanır;
    bu ayarlarla üretilen sayılar kütüphane veya tez sayılarıyla karşılaştırılamaz.
12. CLI senaryo koşuları ölçüm gürültüsü tohumunu kaydetmez; `noise=true` koşuları tekrarlanamaz.

## Tahminci ayarları

13. UKF `process_noise` **adım başına ayrık varyanstır** (sahip kararı 2026-11-01, EST-14): Q,
    yayılım adımı dt'den bağımsız her adıma eklenir. Ölçüm aralığı ya da yayılım adımı değişince
    birim zamandaki gürültü değişir; Q taraması ve ayarlanmış Q değerleri yalnız ayarlandıkları
    örnekleme düzeninde geçerlidir. dt'ye bağlı SNC/DMC modeli seçimli olarak kullanılabilir.

14. BLS-LM varsayılan sönümleme takvimi **"fixed"** kalır (sahip kararı 2026-11-02, EST-08; λ/5, ×10, D = diag(AᵀA)):
    dondurulmuş sonuçlar ve kapılar bu takvimle üretildi. Kazanç oranlı Nielsen/Moré takvimi
    `damping_schedule="nielsen"` ile seçilir (WP4.2b); aynı minimuma daha az iterasyonla ulaşır.
    Nielsen seçimli kalır; dondurulmuş sonuçlar yeniden üretilmez.
