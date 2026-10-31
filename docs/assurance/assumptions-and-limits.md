# Varsayımlar ve kullanım sınırları

Bu sayfa, `main`'in bugünkü haliyle üretilen bir sonucu yorumlarken bilinmesi gerekenleri listeler.
Ayrıntı ve kanıt: [bağımsız değerlendirme §2–§3](../validation/independent-assessment-2026-10-31.md).
Bir madde ilgili iş paketiyle kapanınca buradan kaldırılır ve değerlendirmedeki satır güncellenir.

## Sentetik sonuçlar

1. **Aynı-model gidiş-dönüş tutarsızlıkları gizler.** Gerçek değer ve tahminci aynı basitleştirilmiş
   modeli paylaştığında X-01 (efemeris/GM/çerçeve nesli), X-02 (Ay J2 kutbu), X-07 (Ay merkezli ışık
   zamanı), X-08 (aberasyon çerçevesi) ve X-09 (istasyon zinciri) birbirini götürür. Sentetik doğruluk
   sayıları gerçek veri doğruluğu değildir.
2. **Tahminci karşılaştırmaları** şu farkları içerir: UKF/EKF sayılan Doppler'de doğrusal Dünya ve
   dönüşüm interpolasyonu (X-05; ızgara ortasında istasyon konumu 15–244 m kayar), UKF hızlı yayıcıda
   RK4 + doğrusal efemeris (X-04), UKF P0 ile enjekte başlangıç hatası uyumsuzluğu (X-10), BLS ve SRIF'in
   farklı tekillik eşikleri (X-14). SRIF adım kabulü iki farklı yayılım yolunu karşılaştırır (X-06;
   rtol 1e-10'da 0.21 m, 1e-8'de 2.95 m fark).
3. **Entegrasyon toleransları** yalnız 8 481 s'lik yaylar için sertifikalıdır; BLS ilk iterasyonlarda
   rtol 1e-8'e gevşer ve bu evrede yakınsama ilan edebilir.
4. **Kuvvet parmak izi** yürütülen efemeris kernel'ini, interpolasyonunu ve Ay dönüş modunu
   kapsamaz; "force parity PASS" bu eksenlerde eşitlik kanıtı değildir.

## Uzun yaylar

5. ~~Tek-sekme sayılan Doppler çözücüsü yay başlangıç epoch'u ~1.3e5 s'yi aştığında giderek daha sık
   `converged=False` döndürür (2.5e6 s'de %94); 28 günlük yay-yay kampanyaları bu yoldan geçemez.~~
   **Kapandı (F1 WP1.4, MEAS-03):** yakınsama ve denklem artıkları yerel ışık zamanlarıyla
   hesaplanıyor; 3e5 s ve 2.5e6 s başlangıçlı ızgaralarda 1000/1000 nokta yakınsıyor.
6. DMC süreç gürültüsü dt/τ ≳ 30 olduğunda hassasiyetini, ≳ 100'de pozitif yarı tanımlılığını kaybeder.

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
