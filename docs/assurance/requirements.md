# Gereksinimler

Doğrulama yöntemi: **T** test, **A** analiz, **I** inceleme, **D** gösterim. Durum ve bağlı testler
[doğrulama matrisinde](verification-matrix.csv). "Açık" gereksinimler bir iş paketine (WP) bağlıdır.

## Dinamik (REQ-DYN)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-DYN-001 | Bir koşudaki tüm efemeris, GM ve Ay yönelim girdileri aynı JPL efemeris neslinden (varsayılan DE440) gelir; karışık nesil reddedilir. | T | 1.2 |
| REQ-DYN-002 | Ay gravite çerçevesi kuvvet değerlendirme epoch'unda sürekli bir fonksiyondur (exact-epoch ya da interpolasyonlu). | T | 1.3 |
| REQ-DYN-003 | Ay J2 terimi harmonik alanla aynı gövde çerçevesinde ve aynı (C̄20, R_ref) çiftiyle değerlendirilir. | T/A | 1.3 |
| REQ-DYN-004 | Kuvvet modeli parmak izi yürütülen efemeris kernel'ini, ızgarayı, interpolasyon türünü, dönüş modunu ve pertürbasyonları kapsar. | T | 1.5 |
| REQ-DYN-005 | Gerçek değer ve tüm tahminciler, karşılaştırma yapılan bir koşuda aynı yayılım profilini kullanır. | T | 1.8 |
| REQ-DYN-006 | Her yayılım profili için uzunluk aralığına göre (1/3/7/28 gün) entegrasyon hatası belgelenir ve hata bütçesinde yer alır. | A | 3.3 |
| REQ-DYN-007 | Etki büyüklüğü taramasında OD eşiğini aşan kuvvetler (Ay k2 gelgiti, albedo/IR, Dünya gölgesi, 1PN, …) modellenir ya da varsayımlarda sınırlandırılır. | A/T | 3.1–3.2 |

## Ölçüm (REQ-MEAS)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-MEAS-001 | Üretim gözlemleri (menzil, sayılan Doppler, ΔDOR, tek yönlü) Güneş sistemi barisentrik ışık zamanı ve Shapiro gecikmesiyle hesaplanır (Moyer / DSN 810-005). | T | 2.1 |
| REQ-MEAS-002 | Işık zamanı ve aberasyon aynı çerçevede hesaplanır. | T | 1.10 |
| REQ-MEAS-003 | Işık zamanı çözücüleri yakınsamayı yerel ışık zamanı nicelikleriyle test eder ve yay başlangıç epoch'undan bağımsız yakınsar (≥ 3e6 s). | T | 1.4 |
| REQ-MEAS-004 | Tüm tahminciler (batch ve ardışık) aynı gözlem fonksiyonunu ve aynı Dünya/istasyon interpolasyonunu kullanır; dönüşüm matrisleri ortonormal kalır. | T | 1.1 |
| REQ-MEAS-005 | Troposfer, iyonosfer ve topocentric istasyon zamanı üretim gözlemlerinde uygulanır. | T | 2.2 |
| REQ-MEAS-006 | İstasyon konumu güncel ITRF, IERS EOP, katı gelgit, kutup gelgiti, okyanus yüklemesi ve levha hareketiyle hesaplanır; predict yönelim kernel'i kapsamı dışında hata verilir. | T | 2.4 |
| REQ-MEAS-007 | Rampalı uplink frekansı, istasyon/geçiş bias'ları ve zaman etiketi bias'ı tahmin edilebilir. | T | 2.3 |

## Tahmin (REQ-EST)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-EST-001 | Adım kabulü mevcut ve aday noktayı aynı yayılım yolu, toleransı ve amaç fonksiyonuyla (MAP) değerlendirir. | T | 4.1 |
| REQ-EST-002 | Yakınsama yalnız istenen entegratör toleransında ve ölçek bağımsız bir ölçütle (GN azalması) ilan edilir. | T | 4.2 |
| REQ-EST-003 | Posterior kovaryans ve tekillik kararları dengelenmiş (birimden bağımsız) tasarım matrisinden ve tüm tahminciler için ortak eşikle verilir. | T | 4.3 |
| REQ-EST-004 | SR-UKF sayısal olarak sağlamdır: downdate geri dönüşü, göreli jitter, tek seferlik regülarizasyon, dt'ye bağlı süreç gürültüsü. | T | 4.4 |
| REQ-EST-005 | DMC süreç gürültüsü tüm boşluk uzunluklarında sonlu, pozitif yarı tanımlı ve 1e-8 bağıl doğruluktadır. | T | 0.2 |
| REQ-EST-006 | UKF başlangıç kovaryansı enjekte edilen başlangıç hatasıyla tutarlıdır ve kaydedilir. | T | 1.9 |
| REQ-EST-007 | NIS/NEES iki taraflı χ² aralıklarıyla raporlanır; adaptif R altında ham ve düzeltilmiş NIS ayrı tutulur. | T | 4.6 |
| REQ-EST-008 | Korelasyonlu ölçüm gürültüsü beyazlatılır ya da etkili örnek boyutuyla ağırlıklandırılır. | T | 4.5 |

## Veri ve yöntem (REQ-DATA)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-DATA-001 | Her gerçek veri çalışması ön-kayıtta fit / validasyon / hold-out bölümünü tanımlar; ayar parametreleri puanlama gününde seçilmez. | I | 0.2, 0.4 |
| REQ-DATA-002 | Hold-out günleri dondurulmuş hat değerlendirilene kadar hiçbir betik tarafından okunmaz. | I | 0.4 |
| REQ-DATA-003 | Önseller puanlama referansından türetilmez. | I | 3.4 |
| REQ-DATA-004 | Sonuç görüldükten sonraki ön-kayıt değişiklikleri [değişiklik günlüğüne](amendment-log.md) yazılır ve sonuç keşifsel sayılır. | I | 0.3 |
| REQ-DATA-005 | UTDF okuyucu sayım tutarlılığını, verici/alıcı istasyon eşliğini ve artık saniyeleri denetler. | T | 5.6 |

## Arayüz (REQ-IO)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-IO-001 | CCSDS TDM (503.0-B) okunur ve yazılır. | T | 6.1 |
| REQ-IO-002 | Yörünge ve kovaryans CCSDS OEM/OCM (502.0-B) olarak yazılır. | T | 6.2 |
| REQ-IO-003 | Bir senaryo JSON'u koşuyu tam tanımlar; bilinmeyen anahtar reddedilir; aynı JSON bit-bit aynı çıktıyı verir. | T | 1.6 |

## Yazılım (REQ-SW)

| Kimlik | Gereksinim | Yöntem | WP |
|---|---|---|---|
| REQ-SW-001 | Her çıktı commit, dirty durumu, config hash, tohumlar, kernel manifesti (SHA-256) ve paket sürümlerini kaydeder. | T | 1.6, 7.4 |
| REQ-SW-002 | Fizik testleri katmanlara göre işaretlenir ve CI'da koşar: commit (L0/L1/L7), PR (L2), nightly (L3/L4), sürüm (L5). | I/D | 7.1–7.2 |
| REQ-SW-003 | Kernel gerektiren L2+ testler kernel yoksa başarısız olur; atlanmaz. | T | 7.1 |
| REQ-SW-004 | Testler sıradan bağımsızdır (rastgele sırada üç koşu yeşil). | D | 7.3 |
| REQ-SW-005 | Kritik modüllerde satır kapsamı ≥ %80; ruff ve mypy CI'da. | D | 7.5 |
| REQ-SW-006 | Masaüstü uygulaması kütüphane varsayılanlarından farklı bilimsel ayar kullanmaz ya da farkı formda gösterir. | T | 1.6 |
