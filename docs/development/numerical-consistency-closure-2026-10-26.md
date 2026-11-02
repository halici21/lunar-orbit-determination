# 26 Ekim 2026 yeniden tarama ve düzeltme durumu

Bu kayıt, ilk yeniden taramadaki bulguları güncel düzeltme ağacıyla karşılaştırır.
Ana hedef hâlâ açık: etkin fizik, yayılım, ölçüm, türev, çözüm ve kovaryansın
birlikte doğrulanması. Aşağıdaki kısa testler tam bilimsel kabul değildir.

## Tabanların ayrımı

- Main: `a285cf1748c558271de41aa20a7d15fbb4c82011`.
- Güncel F: `45e14be241c2ade4ecf8f152a9f5fc05d249c2ac`; F2h sonuçları ve F2i kodu var.
- Düzeltme ağacının bu tur başlangıcı: `4ce66da`.
- Güncel F adaptörlerini alan birleşim: `f207e4c`.
- Bu turdaki yeni kaynaklar, deneylerin JSON kayıtlarındaki SHA-256 değerleriyle tanımlanır.
- F ve main çalışma ağaçları değiştirilmedi. `sources/` salt okunur tutuldu.
- Tarihsel deney dosyaları, FAIL kapıları ve kabul eşikleri değiştirilmedi.

## Önem sıralı bulgular ve mevcut durum

| Öncelik | Bulgu ve sınıf | Yapılan / kalan |
|---|---|---|
| P0 | `np.isclose` gerçek zaman aralıklarını sıfır sayıyordu; yerel geçmişte önceki satır midpoint ile örtüşüyordu. **Yeniden üretilmiş kusur.** | Dinamik zaman eşitliği tam eşitlik oldu; indeksler ayrıldı. Serbest hareket ve analitik kovaryansla hot/formal/sqrt-formal yolları 0, 86400, 100000 ve 1e9 s başlangıçlarında sınandı. |
| P0 | Sabit SRP, UKF STM ve yerel ölçüm dinamiğinde eksikti; K-only sigma noktaları aynı ölçüm önbelleğine düşebiliyordu. **Yeniden üretilmiş kusur.** | SRP/J2 ve sayısal profil ilgili yollara taşındı; K önbellek anahtarına alındı. Aktif SRP ile doğrulanmamış haricî fast sağlayıcı reddediliyor. Yüksek dereceli UKF ve tüm profil birleşimlerinin kabulü açık. |
| P0 | F2i kabul edilen küçük adımdan sonra bağımsız son nokta kontrolü olmadan Converged diyebiliyordu. **Yeniden üretilmiş kusur.** | Güncel F2h/i adaptörleri bounded TRF çekirdeğine bağlandı; 15 parametre, ampirik ivme düzeni, önseller ve process işçileri korundu. Dışbükey K=0 karşı örneği geçti. Gerçek veride türev oracle'ı ve çoklu başlangıç kabulü ayrıca gerekli. |
| P1 | Grup/faz ortam gecikmesi değerde vardı; durum/saat Jacobian zincirinde yoktu. **Yeniden üretilmiş kusur.** | Native Niell/Black-Eisner/iyonosfer türevleri ve olay-zamanı zinciri menzil/Doppler/ölçülmüş Delta-DOR'a eklendi. Analitik ve gerçek SPICE örneklerinde çoklu FD sınandı. Haricî eski sağlayıcılarda yerel FD yöntemi açıkça tanımlı. |
| P1 | Gölge minimum araması büyük mutlak zamanlarda 1 s giriş–çıkış çiftini kaçırıyordu. **Yeniden üretilmiş sınır kusuru.** | Minimum araması yerel zaman değişkenine taşındı. 18,94 s ve 1 s örnekleri, fazlar ve 1e9 s'ye kadar zaman kaymaları sınandı. Genel olay tamlığı teoremi veya tüm yüksek derece yaylarının kabulü iddia edilmiyor. |
| P1 | Ortak JSON/CLI/desktop runner, D2 profili ve SRP'yi doğrudan kütüphane yolları gibi taşımıyor. **Kodla doğrulanan eksik.** | Açık. `ScenarioConfig`, `run_batch_arc_sequence`, truth/handoff/estimator/posterior tüketicileri birlikte bağlanmalı; desteklenmeyen birleşimler reddedilmeli. Harmonik runner reddi zaten açık, sessiz fallback değil. |
| P1 | Kuvvet kimliği SRP'yi kapsamıyor; kod hash'i veri/EOP/önsel/runtime manifesti değil. **Kodla doğrulanan eksik.** | F2 kod koruması tüm kütüphane modülleri ve kullanılan F1 okuyucularını kapsayacak şekilde genişletildi. ForceModelContract ve tam çalışma manifesti hâlâ açık. |
| P1 | Güncel F2g G4e ve F2f/g/h bilimsel kabul kapıları açık. **Kayıtlı deney bulgusu / kanıt eksiği.** | Önceki G4e Doppler 1,1e-3 / 1,5e-3 FAIL korunuyor. Kısa yeni testler bunu PASS yapmaz; aynı tarihsel yapılandırmada çoklu FD/tolerans/cadence ve referans yakınsaması gerekir. F2i tamamlanmış birleşik sonuç bu taramada görülmedi. |
| P2 | Kaynak koruma manifesti güncel düzeltmeler ve önceden gelen C4-I exports ile uyuşmuyordu. **Yeniden üretilmiş metadata kusuru.** | Önce başarısız repository kontrolü kaydedildi. Yeni kimlikler, önceki hash'ler korunarak açık gerekçeyle kaydedildi. Bu metadata güncellemesi bilimsel requalification sayılmadı. |
| P2 | Doppler gözlem paydası UTC etiket farkı; hesaplanan uçlar ve payda TDB farkı kullanıyor. **Ölçülmüş sözleşme farkı / saat gerçekleşimi için kanıt eksiği.** | Üç geçerli gerçek aralıkta yalnız paydanın etkisi 1,52e-7–2,25e-7 m/s. LRO 2e-5 m/s bütçesinin altında. Uçların zamanı ve istasyon sayım süresi ayrı tanımlanmalı; bu fark büyük hata diye sunulmamalı. |

## Deney kanıtları

- `baseline-time-shadow`: 7 başarısız, 8 başarılı; kusurlu başlangıç kaydı.
- `corrected-time-shadow`: 25 başarılı; önceki olay ve güncel F2 adaptör testleri dahil.
- `baseline-ukf-forces-import-fixed`: 3 başarısız; `corrected-ukf-forces`: 17 başarılı.
- `baseline-media-chain`: 11 başarısız; `baseline-ddor-chain`: 5 başarısız.
- `corrected-all-media`: 84 başarılı; native model grubu ve eski relativistik ölçüm regresyonları dahil.
- `filter-regressions`: 75 başarılı, 11 yavaş test atlandı, 23 subtest başarılı.
- `extended-core-regressions-v2`: 286 başarılı, 9 yavaş test atlandı, 12 subtest başarılı.
- Bu gruplar örtüşür; tekil test sayısı diye toplanmamalı.
- `real-media-chain-v2`: mevcut 2009 LRO/DE440 geometrilerinden üç satır;
  saat FD adımları 0,01 / 0,1 / 1 s; altı durum sütununda iki FD adımı.
  Menzil ortam saat türevi en büyük uyuşmazlık `2,1793e-9 m/s`,
  Doppler ortam saat türevi `1,2606e-11 m/s²`.
  Bunlar türev hatalarıdır; gözlem artıklarına ve sayısal ölçüm bütçesine doğrudan uygulanmaz.
- İlk probe'da eksik `n_par` test sarmalayıcısı ve ilk geniş test komutunda
  yanlış dosya adı düzeltildi; ilk başarısız komutlar kayıtlarda kaldı.
- İlk sentetik hız FD denemesinde 0,1 adımda büyük menzil çıkarımından
  yuvarlama tabanı görüldü. İki daha büyük adım birlikte sınandı; tek iyi adım seçilmedi.

Her `*.log` için eşleşen `*.json` komut, import, başlangıç commit'i, kaynak
hash'leri ve süreyi taşır. Detaylı gerçek ortam sonuçları `real-media-chain-result.json` içindedir.

## Yüksek derece kontrolü ve ek regresyon

- `high-degree-short`: 600 s gözlem penceresi, N=200; 600 s duvar saati
  sınırında timeout. İki tolerans döngüsünün log satırı var, tamamlanmış
  sayısal sonuç artefaktı yok. **Doğrulanamadı**, kabul için kullanılmadı.
- `high-degree-30s`: N=200 korunarak 30 s gözlem penceresi, 10 s epoch lead;
  6 menzil ve 6 Doppler, 10 s çıktı grid'i, üç tolerans seviyesi.
  Yaklaşık 115 s'de tamamlandı. 6/42/48 yollarında en büyük menzil farkı
  `2,9802e-7 m`, Doppler farkı `2,6627e-8 m/s`.
  Nominal–en sıkı profil farkı menzilde `1,1921e-7 m`, Doppler'de `2,6627e-8 m/s`.
  Son iki referans gözlemi kayan nokta çözünürlüğünde eşit çıktı; bu sıfır
  fiziksel/nümerik hata anlamına gelmez.
- Sıfır ampirik ivme her üç profilde üretim durum yayılımıyla bit düzeyinde eşitti.
  **Bu yayda temas yok; iki FD adımında K gözlem sütunları da sıfır.** Bu
  sonuç, gölge teması veya tanımlanabilir K türevi kapanışı değildir.
  Son tasarım/formal sigma etkisi bu deneyde hesaplanmadı; ortak bütçe tam kapanmadı.
- Son kod taraması, yeni ortam-türev yardımcı fonksiyonunun vacuum dalında
  genişletilmiş ampirik duyarlılık sütunlarını altı/yedi sütunluk işaretçiyle
  boyutlandırdığını ortaya çıkardı. **Bu düzeltme turunda yakalanan regresyon**;
  `vacuum-extra-columns-baseline` menzil ve Doppler için iki kez başarısız oldu.
  Sıfır ortam türevi gerçek duyarlılık matrisinin sütun sayısından kurulunca
  `final-media-extra-columns`: 96 test geçti. Ek ampirik sütunlar korunuyor.
- `current-source-provenance`: 13 test geçti; 79 test seçilmedi.
- Repository kontrolü, tarihsel kimlikler korunarak güncel kaynak kimlikleri
  kaydedildikten sonra geçti: 597 yol, 409 kaynak kimliği, 144 belge / 405 bağlantı.

R1, R3 ve ölçüm zincirindeki dar kusurların regresyon kanıtı var; R6/R7 ve
tüm profillerde birleşik bilimsel kabul açık. Bu ayrım sonraki yol haritasının temelidir.

## Kalan kapanış koşulları

1. Yüksek dereceli kısa yaylarda 6/42/48, ampirik ivme ve olay etkilerini
   aynı profil altında ölç. Gölge içermeyen bir yay, gölge kapanışı sayılmasın.
2. F2g G4e'yi aynı ölçüm/senaryo yapılandırmasında çoklu FD adımıyla yeniden
   ele al; Jacobian ve büyük mutlak ET yuvarlama tabanını ayrı ölç.
3. Ortak profil/kuvvet sözleşmesini truth, handoff, nominal, aday ve posteriora
   uçtan uca taşı. Taylor3 talebinin gölgeli SRP için fiilen ODE kullanması
   sonuç metadata'sında belirtilmeli.
4. Tam kuvvet–ölçüm–sayısal–veri/runtime manifesti, koşullu bias önselleri,
   rank ve bağımsız posterior doğrulaması birlikte ele alınmalı.
5. PROJECT, DSN_REALISTIC ve LRO sayısal ölçüm bütçeleri yeni etkin modelle
   yeniden doğrulanmalı; eski düşük derece geçer sonuçlar otomatik taşınmamalı.
6. Ardından uzun çoklu başlangıç, kalibrasyon/holdout ayrımı ve menzil+Doppler
   doğrulaması. Main'e entegrasyon ayrı incelenebilir aşama olacak.

Bu kayıt, bütün hesapların artık tutarlı olduğu veya görevin kapandığı iddiası içermez.
