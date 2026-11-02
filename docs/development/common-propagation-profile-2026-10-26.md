# Ortak kuvvet ve sayısal profil aktarımı — 26 Ekim 2026

Başlangıç: `c1cf233`, `codex/numerical-core-reliability`. Bu paket main'e
entegre edilmedi. Amaç, R6'nın profil/kuvvet aktarımındaki somut eksiklerini
kapatmak; tam bilimsel kabul, veri/runtime manifesti ve C1–C5 ortak dispatch'i
bu kayıtta tamamlandı sayılmıyor.

## Değişiklikler

- `propagation_contract.py`: model seçenekleri ile integratör seçeneklerinin
  ayrı ve açık aktarımı. Profil doğrulaması, SRP ile ortak D2 seçimi ve etkin
  sayısal metadata tek yerde. Geçersiz integratör/olay birleşimleri reddediliyor.
- `ScenarioConfig`: `srp` ve `propagation_profile` JSON gidiş/dönüşü. Desteklenmeyen
  iç seçenekler, yanlış bool ve geçersiz sayılar sessizce yok sayılmıyor.
- Position/range-rate BLS ve SRIF: SRP/J2/profil nominal, aday ve posterior/
  observability yollarına taşındı. Açık profil altında erken iterasyon toleransı
  gevşetilmiyor; nominal, aday ve posterior aynı caller toleranslarını kullanıyor.
- Handoff: hot, formal ve sqrt-formal aynı kuvvet ve profili taşıyor. Seri ve
  process işçileri aynı seçenekleri alıyor. Profil seçildiğinde doğrulanmamış
  RK4 sigma hızlandırıcısı araya girmiyor. Process sayısı görev ve Windows
  sınırına göre sınırlanıyor.
- UKF: haricî fast sağlayıcının açık profili sessizce geçersiz kılması reddediliyor.
  Gölgeli SRP + Taylor3 talebinde kullanılan ODE, etkin metadata'da belirtiliyor.
- Kuvvet kimliği: SRP için K, gölge, örtücü yarıçap, basınç, AU ve Güneş yarıçapı
  `v2` kontratına dahil. Runtime bağlantısı eksik/farklı SRP'yi reddediyor.
  Statik bir SRP sabiti kontratta değiştirilip başka sabitle hesap yapılamıyor.
  SRP içermeyen `v1` kanonik payload ve fingerprint'ler korunuyor.
- CLI ve gerçek masaüstü analiz işçisi aynı seçenekleri truth ve kestiriciye
  aktarıyor. Masaüstü truth/ölçüm önbelleği artık farklı kuvvet, profil veya
  toleransı paylaşmıyor; aynı bilimsel ayarlı estimator varyantları paylaşabiliyor.
- Etkin truth/kestirici profilleri, force manifestinden ayrı
  `*_numerical_profile.json` dosyasına yazılıyor. Bu dosya tam veri/runtime
  manifesti olarak sunulmuyor; eksik kestirici metadata'sı null kalıyor.

## Tarihsel uyumluluk ve kabul sınırı

Doğrudan iki yönlü BLS/SRIF API'sinde `propagation_profile=None` tarihsel
ADAMS/adaptif davranışı korur. Yeni ortak senaryo yolu seçilen D2 profilini
açıkça gönderir. Profil seçilmemiş, SRP kapalı tarihsel truth/kestirici
varsayılanları da değiştirilmedi; ayrı sayısal metadata bunları olduğu gibi gösterir.

İlk uygulamada iki yönlü doğrudan API'nin None varsayılanı da DOP853'e dönmüştü.
Bu, bağımsız ADAMS yeniden kurulumlarıyla 5 kovaryans uyuşmazlığı doğurdu;
karma koşu 600 s'de tamamlanmadı. Bu kayıt başarısız/timeout olarak korunuyor.
Varsayılan geri alındı; testler, eski eşikler ve fixture'lar değiştirilmedi.
İki yolun seçimini ayrıca kontrol eden dört regresyon eklendi.

Yeni SRP kontratının kullanılabilir tüketicileri
`pending_common_profile_validation`, fast sigma tüketicisi `unsupported` olarak
kaydedilir. Yeni profilli koşu, eski `r1.fd-sweep.v1` etiketiyle otomatik
sertifikalandırılmaz; `pending_common_profile_derivative_qualification` yazılır.
Tüm yollarda aynı rtol/atol verilmesi, aynı sayısal hata veya görev doğruluğu
elde edildiğini tek başına kanıtlamaz.

## Kanıt

- `shared-profile-baseline`: 9 başarısız; gerçek eksik aktarımın ilk kaydı.
- `shared-profile-core`: 27 başarılı.
- `profile-force-regressions`: 191 başarılı, 20 yavaş test atlandı, 39 alt test başarılı.
- `srp-identity`: 65 başarılı, 12 alt test başarılı.
- `shared-profile-end-to-end`: 59 başarılı, 4 alt test başarılı. Gürültüsüz iki
  yayda cold/hot/formal/sqrt-formal; gerçek process işçilerinde seri eşitliği;
  gerçek CLI dispatch'inde truth ve estimator aktarımı, JSON ve manifest dosyası.
- `shared-profile-desktop`: 50 başarılı; gerçek headless işçi, kuvvet/tolerans
  önbellek ayrımı, mevcut desktop ve DE440 hata sözleşmeleri.
- `profile-estimator-covariance-regressions`: timeout; 5 F işareti var, tamamlanmış
  JUnit yok. Kaynak varsayılan hatası yukarıda açıklandı; PASS'a çevrilmedi.
- `profile-legacy-posterior-recheck`: D25/D28'in 10 testi başarılı.
- `profile-final-core-regressions`: 213 başarılı; K BLS/SRIF/SRUKF, square-root
  kovaryans, R1 posterior/observability, D25/D28/D29 ve SRP regresyonları dahil.
  Koşu yaklaşık 277 s duvar saati sürdü. FD referansının yavaş olduğunu belirten
  beklenen bir RuntimeWarning var.
- Son kaynak koruma/CLI/desktop kontrolü: 17 başarılı. İlk kaynak koruma
  kontrolü, CLI dosyası için eski hash'e düşen bir kayıt nedeniyle başarısızdı;
  gözden geçirilen iki entry-point hash'i ayrı receipt ile eklendi. Eski
  kaynak hash tabloları ve tarihsel sonuçlar korunuyor.
- Repository kontrolü geçti: 597 yol, 409 kaynak kimliği, 146 belge / 405 bağlantı.
- Gruplar örtüşür; bu sayılar tekil test toplamı değildir. Komut, import,
  kaynak hash'leri, süre ve gerçek sonuçlar ayrı JSON/log kayıtlarındadır.

## Sonraki matematiksel kapanışlar

1. Position/range-rate MAP kabul maliyetinin açık önsellerle uyumunu bağımsız
   doğrula. Adım/prior ve posterior terimleri mevcut; aday maliyetinin veri
   artığı üzerinden kurulması kod incelemesinde sözleşme riski olarak kaldı.
2. Gerçek LRO temas penceresinde yüksek dereceli 6/42/48 ve K/ampirik sütunları
   çoklu FD/tolerans/grid ile sınansın. Ucuz SPK geometrisi taraması,
   2009-09-17 00:40:40.683–00:40:51.187 UTC tutulma çıkışını seçti; ±45 s'de
   9–11 mevcut menzil kaydı var. İlk 30 s penceresinin tümü umbradaydı; sıfır
   K sütunları bu yüzden kapanış kanıtı değildi. Bu seçim hesabı, propagatör
   doğrulaması veya bağımsız LOLA doğruluk kabulü değildir.
3. Tam kuvvet/ölçüm/sayısal/veri/runtime manifesti; C1–C5/DE440 ortak akış
   desteği; yeni profillerin birleşik ölçüm ve 0,1 formal sigma bütçesi.
4. Tarihsel F2g G4e FAIL ve uzun çoklu başlangıç/holdout kapıları açık kalır.

Main entegrasyonu ve varsayılanların bilimsel kabulü ayrı aşamadır.
