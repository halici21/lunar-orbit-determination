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
3. **Entegrasyon toleransları** yalnız 8 481 s'lik yaylar için sertifikalıdır; BLS-LM (konum, range-rate,
   iki yönlü menzil) F4 WP4.2-R'den beri her iterasyonu çağıranın toleransında skorlar; SRIF yolları ilk
   iterasyonlarda hâlâ rtol 1e-8'e gevşer ve bu evrede yakınsama ilan edebilir (taranmadı, açık kalem).
4. **Kuvvet parmak izi** yürütülen efemeris kernel'ini, interpolasyonunu ve Ay dönüş modunu
   kapsamaz; "force parity PASS" bu eksenlerde eşitlik kanıtı değildir.
5. **Varsayılan Ay J2 terimi** IAU sabit ortalama kutbunu kullanır; bu kutup MOON_PA_DE440
   figür ekseninden 1.47–1.62° uzaktadır (DYN-03). 160 km'de fark J2 ivmesinin %2.0'ı
   (7.5e-6 m/s², SRP'nin ~500 katı). (J2, R) çifti GRAIL GRGM1200A'nın (C̄20, R_ref) çiftinden
   J2·R² olarak 4.7e-4 yüksektir (DYN-04). Dondurulmuş J2 tabanları bu varsayılanla üretildiği için
   değiştirilmedi; doğru kutuplu derece-2 alan harmonik yolda DIRECT modda (nmax=2,
   `harmonic_epoch_et0`) kullanılabilir. Izgaralı harmonik yolda Ay yönelimi WP1.3'ten beri
   SLERP ile süreklidir (DYN-05 kapandı).

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
15. **Range-rate varsayılanları** anlık geometrik range-rate ve anlık eşlik menzil/açılarıdır (WP2.5'te
    değişmedi; sahip kararı bekliyor). Barycentric sayılan Doppler yanında anlık eşlik menzili SSB tek yönlü
    menzilden 35.5–39.5 km uzaktır; kendi içinde tutarlı sentetik modeldir, ışık zamanı iddiası yoktur.
    Işık zamanlı eşlik için `companion_geometry='apparent_one_way_ssb'` (yalnız `barycentric_dsn`);
    Ay merkezli `apparent_one_way` barycentric Doppler yanında reddedilir.
16. **Işık zamanı Jacobian'ı** WP2.5'ten beri varsayılan olarak `implicit_light_time`'dır (konum profilleri
    ve range-rate eşlik satırları). SSB profillerinde (`one_way_light_time_ssb[_aberrated]`,
    `apparent_one_way_ssb`) Jacobian Ay merkezli çözümden gelir: satır başına ~1.5e-4, v/c mertebesi.
    Birinci derece eşleme (%4.6'ya kadar) adıyla seçilebilir; DMC batch implicit modeli reddettiği için
    orada adıyla istenmelidir. 1 saatlik barycentric Doppler yayındaki BLS-LM 2–3.6 cm tabanı erken
    iterasyon gevşetmesiymiş; F4 WP4.2-R'den beri 0.19 mm.
17. **Landmark optik** (`lunar_landmark_optical`, F2 WP2.6): enlem/boylam varsayılan olarak `MOON_ME_DE440_ME421`
    çerçevesinde okunur (LOLA katalogları); WP2.6 öncesi sessizce `MOON_PA_DE440` idi (163–875 m fark) ve Phase 17
    R1O-OPT / HARDEN sonuçları PA ile üretildi. Kamera tutumu varsayılan olarak dış girdidir (yıldız izleyici);
    tutum tahmin edilen konumdan kuruluyorsa `attitude_source='nadir_from_state'` gerekir (aksi halde Jacobian
    %5.8 eksik). Piksel kuralı `pixel_origin` ile açıktır (varsayılan 'corner', W/2). Mercek bozulması, aydınlatma
    ve küresel olmayan Ay şekli modellenmez.
18. **ΔDOR**: `lunar_od.delta_dor` gözlemi `legacy_synthetic`'tir (Ay merkezli D_S, statik düzlem dalga D_Q;
    SSB modelinden 0.6–1 µs). Üretim gözlemi `relativistic_light_time.delta_dor_ssb`: SSB uzay aracı gecikmesi ve
    aynı modelde kuasar gecikmesi, istasyon zamanında; ortak istasyon yaklaşımları ΔDOR'da sadeleşir. IERS 2010
    konsensüs gecikmesi (`quasar_delay_consensus`) bağımsız kontroldür. ERFA dtdb'nin topocentric terimi DSN
    kompleksleri arasında kesin V·b/c²'den 0.47–0.69 ns sapar: tek tek gecikmeler bu kadar hatalıdır, ΔDOR değil
    (açık kalem). Kuasar doğrultusundaki medya, kuasar yapısı ve Güneş kenarı yüksek mertebe terimi modellenmez;
    hiçbir üretim tahmincisi ΔDOR tüketmez.
19. **Görünürlük maskesi** (F2 WP2.7, MEAS-14) varsayılan olarak geometriktir: uzay aracı, istasyon ve Ay aynı
    epoch'ta. Işık zamanlı maske `analyze_visibility_gap_with_transforms(geometry='light_time_downlink' |
    'light_time_two_way')` ile seçilir (6×6 dönüşüm ve hızlı durum geçmişi gerekir); hangi modun doğru olduğu
    gözleme bağlıdır (tek yönlü ya da iki yönlü) ve fonksiyon bunu bilmez. Geometrik maskede Ay örtmesi sınırları
    ~τ ≈ 1.33–1.35 s, iki yönlü yükselişler ~2τ ≈ 2.6–2.9 s erkendir. Ay küresel (yarıçap yapılandırmadan),
    arazi yoktur: kenar rölyefinin km'si başına 2.7–3.5 s örtme sınırı kayması (10 km'de ~30 s, ışık zamanı
    etkisinin ~20 katı; LOLA DEM ertelendi, sahip kararı bekliyor). Kırılma maskede yoktur. Düzgün olmayan
    ızgarada boşluklar zamanla köprülenir; düzgün ızgarada örnek sayısıyla (davranış değişmedi). Üretilen
    gürültülü azimutlar [0, 2π) içine sarılmaz; artıklar `wrap_to_pi` ile sarılır.

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
