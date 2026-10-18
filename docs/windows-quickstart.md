# Windows kurulum ve kullanım

## İlk kurulum

1. Python.org üzerinden **64 bit Python 3.13** kurun. Python Launcher seçeneğini açık bırakın.
2. GitHub Releases üzerinden kaynak ZIP paketini indirin ve bir klasöre çıkarın; veya repoyu klonlayın.
3. Klasördeki setup_windows.cmd dosyasına çift tıklayın.
4. Kurulum tamamlanınca launch_lunar_od.cmd dosyasına çift tıklayın.

Kurulum bu klasörün içinde .venv oluşturur ve sabitlenmiş bağımlılıkları yükler.
`src/lunar_od` paketini düzenlenebilir kurulumla yükler. Güncel kaynakları
aldıktan sonra mevcut ortamda da `setup_windows.cmd` komutunu yeniden çalıştırın.
İnternet bağlantısı gerekir. Bilgisayarınızdaki genel Python paketlerini değiştirmez.
Mevcut .venv kullanılamıyorsa onu otomatik silmez; hata mesajındaki yönlendirmeyi izleyin.
Kurulum dosyası her çalıştırıldığında bağımlılık profilini yeniden uygular.

Bu paket Python gerektiren bir kaynak dağıtımıdır. Python içeren bağımsız bir EXE
kurulum paketi değildir. Başlatıcı yeni klonda veya ZIP'ten çıkarılmış klasörde
çalışır; geliştiricinin bilgisayarına ait mutlak yollar içermez.

## Uygulamayı kullanma

- **Görev görünümü:** yörünge, Dünya istasyonları ve kayıtlı ölçüm Gantt’ını birlikte izleyin.
- **İstasyonlar / Temas planı / Yüzey izi:** katalog, ayrı görünürlük hesabı ve Ay yüzeyindeki izdüşüm.
- **Senaryo:** ayarları düzenleyin; yeni UUID ile ayrı bir koşu oluşturulur.
- **Çalıştır:** işlem günlüğünü izleyin. Hesaplama ayrı süreçte çalışır.
- **Sonuçlar:** kayıtlı sonuçları, birimleri, epoku ve kaynak bilgilerini inceleyin.
- **Arşiv:** önceki sonuç klasörünüzü seçin. Dosyalar yerinde okunur; taşınmaz.
- **Karşılaştırma:** arşivde Ctrl ile iki koşu seçin; raporu dışa aktarabilir ve yeniden açabilirsiniz.

İlk kurulumda sonuç klasörü bulunmayabilir; arşiv boş görünür.
Geliştiricinin sekiz deneme koşusu pakete dahil değildir.
Seçilen arşiv klasörü normal kapanıştan sonra hatırlanır.

**İstasyonlar** sayfasında ad veya ağ koduyla (DSN, ESA…) arayın. **Ctrl+F**
arama alanına geçer ve sorguyu seçer; alanın temizleme düğmesi tüm satırları
geri getirir. Alt metin görünen/toplam ve seçili istasyon sayısını gösterir.
**Export CSV**, seçili satırları değil, filtrelenmiş tablonun tamamını aktarır;
eşleşme yoksa düğme devre dışıdır. Bu sayfadaki harita tüm kataloğu gösterir.

Dar pencerede istasyon araçları iki satıra yerleşir: arama/CSV/ekleme ve
seçimden ağ kaydetme/düzenleme/silme. Sayfayı aşağı kaydırarak tabloya erişin;
koordinat ve hata sütunları için tablonun kendi yatay kaydırıcısını kullanın.
Pencere boyutu değiştiğinde arama ve seçili istasyon korunur.

Haritanın altındaki **Ağlar** satırı nokta renklerini açıklar; arama sırasında
da tüm haritanın anahtarıdır. **★ Özel istasyon** işaretin türünü belirtir;
yıldızın rengi yine ait olduğu ağı gösterir.

Haritadaki **noktaya veya istasyon adına** tıklayarak tablodaki kaydı seçin.
Adlar birbirini ve noktaları kapatmadan yerleşir; ince çizgiler adın ait olduğu
konumu gösterir. Noktaların koordinatları değişmez. Çok yoğun/uzun özel adlar
sığmazsa gizlenebilir; bütün istasyonlara tablodan erişebilirsiniz.

Tablo başlıklarına tıklayarak görünümü sıralayabilirsiniz. Düzenleme, silme ve
seçimden ağ kaydetme işlemleri ekranda seçili istasyonları kullanır. Aramayı
değiştirdiğinizde seçim temizlenir; işlemden önce satırı yeniden seçin.

Enlem/boylam ve belirsizlik sütunları kayıtlı sayısal duyarlıkla sıralanır.
Hücrenin üzerine gelerek tam değeri ve birimini görün. **—**, tanımlı olmayan
değerdir; artan sıralamada sonda, azalan sıralamada baştadır. Sıfırdan ayrıdır.

## Bağımsız temas planı

**Temas planı**, sabit örnek başlangıç yörüngesinden SPICE ile ayrı görünürlük
hesabı yapar (10 dk örnekleme). İstasyonları ve birimli ayarları seçip
**Görünürlüğü hesapla** düğmesine basın. Alt+S/E/B ilgili parametreye geçer.
Hesap sürerken girişler kilitlenir. Sonuç başlığı T0/UTC ve kullanılan ayarları
gösterir; istasyon adları başlığın araç ipucunda bulunur. Ayarları değiştirirseniz
grafik/dışa aktarımın önceki hesaba ait olduğu belirtilir. Hesap veya grafik
hatasında önceki eşleşen sonuç korunur. Grafik/CSV ham görünürlük maskesidir.
Kayıtlı koşunun ölçüm Gantt’ını **Görev görünümü** sayfasından açın.
Dar pencerede ayarlar ve Gantt alt alta yerleşir; aşağı kaydırarak hesaplama,
dışa aktarım ve grafiğin alt eksenine erişin. Bu sayfada sağa-sola kaydırmak gerekmez.
İstasyon listesinin kendi kaydırıcısı korunur. Geniş pencerede paneller yan yanadır.

## Yörünge, harita ve Gantt

Bu ekran **main** sürümündedir; v0.1.0-alpha.1 ZIP paketi bu değişikliği içermez.
Yeni kurulumda **Görev görünümü** açılır. Geçerli kaydedilmiş oturum varsa son
arşiv/sonuç/karşılaştırma/görev sayfası geri açılabilir.

Henüz koşu açılmamışsa yörünge ve Gantt alanları başlangıç adımlarını gösterir;
haritadaki noktalar istasyon kataloğudur. **Koşu aç** mevcut sonuçları,
**Yeni senaryo** yeni koşu ayarlarını açar. Bir sonuç yüklendikten sonra
"durum vektörü bulunmuyor" veya "ölçüm zamanları kaydedilmemiş" yazıyorsa,
o sonuçta ilgili veri eksiktir.

1. **Görev görünümü → Koşu aç** ile arşive gidin; sonuç klasörünü seçin.
2. Bir koşu seçip **Harita ve yörüngeyi aç** düğmesine basın.
   Koşu adının altında işlem durumu ve kaynak CSV’deki yay başarı alanları görünür.
   **İşlem tamamlandı**, tamamlanma kaydının bulunduğunu gösterir; yakınsama veya
   bilimsel doğruluk onayı değildir. **Koşu ayrıntıları**, aynı sonucun hatalarını,
   duruş nedenlerini, ayarlarını ve kaynaklarını açar.
3. İstasyonu listeden seçin veya haritadaki noktasına / Gantt satır adına
   tıklayın: üç seçim eşleşir. Listede ok tuşlarıyla da seçim yapabilirsiniz.
   **Tüm istasyonlar** vurguyu kaldırır. Alt metin, seçili epokta ölçüm kaydı
   olup olmadığını gösterir; eski koşularda veri eksikliği ayrıca belirtilir.
4. Gantt’ın zaman alanına tıklayın veya kaydırıcıyı hareket ettirin:
   en yakın kayıtlı epok seçilir. Satır adına tıklamak zamanı değiştirmez.
5. **Oynat / Duraklat**, hız seçimi ve **Başa dön** aynı zaman imlecini yönetir.
   Başka sayfaya geçmek oynatmayı durdurur. Yörüngeyi sürükleyerek döndürün;
   tekerlekle yakınlaştırın.
6. **Yörünge değerleri ne anlama geliyor?** açıklamasını açın: yükseklik,
   eğim, yarı büyük eksen, dışmerkezlik ve periyot tanımları gösterilir.

Dar pencerede yörünge ve harita alt alta yerleşir; aşağı kaydırarak haritaya,
Gantt’a ve oynatma kontrollerine erişin. Geniş pencerede yan yana gösterilir.

Dolu istasyon noktası, seçili epokun kaydedilmiş ölçüm aralığında olduğunu
belirtir. Gantt, hazırlanmış yaylarda üretilen ölçüm örneklerini gösterir;
tüm süre boyunca geometrik/radyo görünürlüğü veya kestirim başarısı değildir.
Yeni koşular istasyon adlarını ve ölçüm aralıklarını ayrıca kaydeder.
Eski kayıtlar için eksik veri mesajı görünür; yay sınırları istasyon teması
olarak kullanılmaz. **Görünürlük hesabı**, ayrı bir hesaplama aracını açar;
sonuçlarını tarihsel koşunun temas takvimi olarak etiketlemez.

Kayıtlı durum ve kuvvet manifesti yoksa yörünge/periyot uydurulmaz. Eğim J2000
kaydı için J2000 XY düzlemine göredir; Ay ekvatoruna göre eğim değildir.
Truth yörüngesi senaryonun referans hareketidir, gerçek görev telemetrisi değildir.

## Yüzey izi: iki ayrı model

**Yörünge tasarımı**, analitik iki cisim ön incelemesidir. Ay dönüşü yaklaşık,
sabit hızlı Z dönüşüdür; t=0 yerel başlangıçtır ve UTC epoku yoktur. İstasyon
görünürlüğü hesaplanmaz. Bu nedenle “Visibility not computed” gösterilir;
analitik CSV iletişim uygunluğu için kullanılamaz.

**SPICE örneğini hesapla**, sabit örnek başlangıçtan ayrı bir hesap yapar;
tasarımcıdaki elemanları devralmaz. Kaynak epok, IAU_MOON ve 4 dk örnekleme
grafiğin üstünde belirtilir. “Visible samples”, görünür bulunan geometrik
örneklerin oranıdır; kesintisiz temas süresi veya gerçek görev telemetrisi değildir.
Kayıtlı koşunun yörüngesini **Görev görünümü** üzerinden inceleyin.
Yüzey izi sayfasında dar pencerede mevcut yatay/dikey kaydırıcılar kullanılır.
Sayfa veya sekme değiştirince yüzey izi oynatması durur; dönüşte **Play** ile
aynı konumdan devam edin. Gizli yörünge tasarımı önizlemesi sürekli dönmez;
sekmesi açılınca dönmeye başlar. **Reset**, oynatmayı durdurup başa alır.

## Ortam kontrolü

Açılışta alt çubuktaki **Ortam** düğmesi kontrol durumunu gösterir.
Düğmeye tıklayın veya **Ayarlar → Hesaplama ortamı** bölümünü açın.

- **Kernel klasörü seç:** seçimi kaydeder; sonraki masaüstü hesaplamaları bu yolu kullanır.
- **Otomatik bul:** kayıtlı seçimi kaldırır; uygulamanın açılışında devraldığı ortam
  değişkeni ve yükleyicinin standart arama sırası yeniden kullanılır.
- **Yeniden kontrol et:** paket importlarını, referans sürümleri, gerekli on dosyanın
  okunabilirliğini ve SPICE yükleme/örnek epok sorgularını ayrı süreçte denetler.

Kayıtlı özel klasör yoksa başka klasöre sessizce geçilmez; eksik olarak raporlanır.
Dosya adı ve okunabilirlik kontrolü tek başına başarılı SPICE yüklemesi sayılmaz.
Referans paket sürümünden farklılık uyarıdır; import veya kernel hatası koşuyu engeller.
Kontrol sürerken arşiv ve sonuç ekranları kullanılabilir.

**Senaryo → Koşuyu başlat**, seçilen süre ve örnek adımlarıyla yeni bir kontrol yapar.
Başarısızsa koşu klasörü oluşturmadan rapora yönlendirir. Çıktı yolunda yalnızca
mevcut üst klasörün izinleri kontrol edilir; yazma başarısı gerçek koşu oluşturulurken
ayrıca doğrulanır. Devam eden hesaplamanın kernel seçimi değiştirilmez.

SPICE kontrolü başlangıç, orta ve son örnek epoklara bakar; tüm zaman aralığında
kesintisiz kernel kapsamını, bilimsel doğruluğu veya çözücü yakınsamasını kanıtlamaz.
Yüksek dereceli çekim ve dış kampanya verileri bu başlangıç kontrolünün kapsamına girmez.

## SPICE kernels

Arayüzü açmak ve kayıtlı sonuçları incelemek için kernel gerekmez.
SPICE kullanan bilimsel koşular için kernel dosyalarını ayrıca hazırlayın.

PowerShell'de:

    $env:LUNAR_OD_KERNEL_DIR = "C:\path\to\kernels"
    .\.venv\Scripts\python.exe desktop_app/main.py

Çift tıklayarak başlatmada **Ayarlar → Hesaplama ortamı → Kernel klasörü seç**
üzerinden seçim yapabilirsiniz. Bu masaüstü tercihi LUNAR_OD_KERNEL_DIR değerinden
önceliklidir; komut satırı araçları kendi ortam değişkenlerini kullanır.

| Dosya | Görev |
| --- | --- |
| naif0012.tls.txt | Artık saniyeler |
| moon_de440_220930.txt | DE440 Ay principal-axis çerçevesi |
| moon_pa_de440_200625.bpc | DE440 Ay principal-axis yönelimi |
| de421.bsp | Gezegen efemerisi |
| earth_assoc_itrf93.tf.txt | Dünya ITRF93 çerçevesi |
| moon_080317.tf.txt | Ay çerçevesi |
| earth_2025_250826_2125_predict.bpc | Dünya yönelimi |
| moon_pa_de421_1900-2050.bpc | Ay principal-axis yönelimi |
| gm_de431.tpc.txt | Çekim sabitleri |
| pck00010.tpc.txt | Gezegen sabitleri |

Liste, [güncel yükleyicideki](../src/lunar_od/spice_loader.py) REQUIRED_KERNELS sırasını izler.
DE440 çerçeve ve yönelim dosyaları birlikte gereklidir; iki Ay gerçekleşimini
koruyan yükleme sırasını uygulamanın yükleyicisi yönetir.

Değişken ayarlanmadığında yükleyici önce ~/Documents/mice/kernels, sonra
projenin kernels klasörünü arar. Kernel dosyaları GitHub paketinde bulunmaz.
Dış kampanya ve yüksek dereceli çekim verilerinin yolları için
[doğrulama rehberine](validation/verification.md) bakın.

## Sorun giderme

- **Python bulunamadı:** 64 bit Python 3.13 kurun; setup_windows.cmd dosyasını tekrar çalıştırın.
- **Kurulum başarısız:** konsoldaki pip hata mesajını inceleyin; ağ/proxy ayarını kontrol edin.
- **Arayüz açılmıyor:** setup_windows.cmd dosyasını tekrar çalıştırın. Komut satırında
  launch_lunar_od.cmd --check ile sürüm ve bağımlılık importlarını denetleyin.
- **Kayıt yok:** Arşiv → Arşiv klasörü seç ile kendi sonuç klasörünüzü açın.
- **Koşu bitti fakat başarı “Hayır”:** süreç tamamlanması, çözücünün operasyonel
  başarı veya yakınsama ölçütünü sağlaması anlamına gelmez. Kaydedilmiş duruş nedenini inceleyin.

Normal masaüstü kullanımında QT_QPA_PLATFORM=offscreen ayarlamayın; bu ayar
görünür pencere oluşturmayan otomatik testler içindir.
