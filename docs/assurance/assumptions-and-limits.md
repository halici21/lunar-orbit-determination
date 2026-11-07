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
   RK4 + doğrusal efemeris (X-04, MEAS-05; doğrusal Dünya konumu ızgara ortasında ≤117 m sapar, ama yörünge farkı 60 s'de 4.6e-6 m, 1 günde 4.1e-2 m — ölçüm gürültüsünün çok altında, `tests/test_meas05_sigma_ephemeris_fidelity.py` ile sabitlendi), BLS ve SRIF'in
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

## Kuvvet modeli ve yayılım

Madde 15–17'nin kaynağı [WP3.1-R önemlilik taraması](../../studies/phase17/reports/wp3_1_materiality_survey.md)
(REQ-DYN-007). Ön kayıtlı koşu kapı vetosuyla WP3_1_INCONCLUSIVE kaldı; aşağıdaki sayılar
olay-bölütlemeli A2 koşusundandır. A2 sonuç görüldükten sonra eklendiği için keşifseldir
([değişiklik günlüğü](amendment-log.md) 2026-11-07). Karar (2026-11-08, sahip ajana devretti): A2'nin
sınıflandırması WP3.2 kapsamının temelidir; REQ-DYN-007, WP3.2'nin ön kayıtlı hold-out tekrarından sonra kapanır.
WP3.2 terimleri seçimli (opt-in) ekler; varsayılanlar ve dondurulmuş sonuçlar değişmez.
Sınırlar 100 km'lik dairesel yörüngeler, 1 günlük yay, üç DSN istasyonu, menzil ve menzil hızı
ve çözülen (x0, K_SRP) için geçerlidir. Daha uzun yay ya da farklı çözülen parametre kümesi yeni bir tarama ister.

15. **Üretimde olmayan ve OD'yi eşiğin üstünde saptıran kuvvetler** (WP3.2'ye kadar). Gerçek veri
    sonuçlarında bunlar model hatasıdır; aynı modeli paylaşan sentetik gidiş-dönüş testleri onları göstermez (madde 1).
    Yanıtlar 1 günlük yayda, 0.1σ eşiğine karşı en kötü hücre, tasarım ve gürültü düzeyindendir:

    | Terim | En büyük yanıt | Not |
    |---|---|---|
    | Ay gelgiti (Dünya ve Güneş, derece 2–3) | 230σ | |
    | Ay albedo/IR basıncı | 165σ | yokluğunda kestirilen K_SRP ~%21 düşük çıkar |
    | Dünya gölgesi | 200σ, yalnız tutulma yaylarında | yokluğunda K_SRP %19–25 düşük çıkar |
    | SRP'de ışık zamanı ve sapınım | 0.11σ, sınırda | Ön kayıtlı terimle 1.4σ çıktı, ama o terim gölgeyi 100 km'deki kontaklarda 3.07 kat uzağa koyuyor. Işın-geçiş geometrisiyle (keşifsel tanı) 0.109. Sapınım ve gölge ışık zamanı ters işaretli; ikisi birlikte uygulanmalı |
    | Dünya J2'si | 0.11σ, sınırda | seçimli, varsayılan kapalı |

16. **Modellenmeden sınırlandırılan kuvvetler.** Ay 1PN Schwarzschild terimi ≤ 0.029σ'dır: günde 2.5 cm,
    ivme 1.3e-10 m/s². Venüs ve Jüpiter ≤ 0.006σ'dır: günde ≤ 0.25 mm, entegrasyon tabanının 10 katının altında.
    Üretimin Dünya J2'sindeki birim çerçeve (DYN-19'un o kısmı) ITRF93 kutbuna göre ≤ 2.2e-4σ fark yaratır.
17. **Gölgeli alçak yörüngelerde olaysız entegrasyonun sayısal tabanı.** Ay gölgesi ivmesi her
    yarıgölge kontağında kırılır. Olay bölütlemesi yoksa entegratör bu kırıkların üzerinden adım atar;
    `propagate_state` varsayılanı da `shadow_events=False`'tur. Gölgeli 100 km'lik yörüngelerde tek DOP853
    koşusunun tabanı (rtol 1e-12'ye karşı 1e-13) günde 0.2–1.4 m'dir. Olay-bölütlemeli `PropagationProfile()`
    ile taban ≤ 7.5e-5 m'dir. Varsayılan Adams yolunun tabanı burada ölçülmedi. Santimetre ve altındaki
    karşılaştırmalar gölgeli yörüngelerde `PropagationProfile()` ile yapılmalıdır. Profil başına
    sertifika WP3.3'tedir (REQ-DYN-006).

## İstasyon konumları

18. `config.py` istasyon tabloları yuvarlatılmıştır; "Goldstone DSN" DSS-14'ten 15.7 km, en yakın antenden 6.0 km,
    "Madrid DSN" 0.22 km, "Canberra DSN" 0.29 km uzaktadır (WP2.4 dilim 4). Bu tablolar kendi içinde tutarlı sentetik
    veri içindir; gerçek veride `lunar_od.dsn_stations.dsn_station` (DSN 810-005 301 Rev. P) kullanılır.
19. `dsn_station` konumu verilen epoch'ta dondurur (`Station` statiktir): gün başına < 0.2 mm hata. Kaynağın beyan ettiği
    mutlak doğruluk 0.1 m'ye kadardır (1σ). Levha hareketi 2027'de Goldstone 0.47 m, Canberra 1.25 m, Madrid 0.74 m birikir.
20. Kutup gelgiti modellenmez: etkisi 1.2 mm, OD'de ≤ 0.004σ (Phase C3). Okyanus yüklemesi modellenmez ve ölçülmemiştir
    (BLQ katsayıları gerekir). DSN dışı istasyonlar (LRO'nun USN sahaları, RD-09) bu kaynakta yoktur.
