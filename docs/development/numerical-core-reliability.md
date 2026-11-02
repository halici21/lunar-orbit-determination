# Sayısal çekirdek: ilk uygulama ve sınırlı doğrulama

Temiz F tabanı `64b273beaf28810ab7e3fa203cc94c40c5e4bfcf` kontrol edildi;
uygulama ayrı `codex/numerical-core-reliability` çalışma ağacında yapıldı.
Main'e birleştirme, veri indirme ve tam F2d/e/f/g kampanyası yapılmadı.
Tarihsel sonuçlar, FAIL kayıtları, fizik sabitleri ve gravite dereceleri korunur.

Bu teslim seçilmiş kısa yayları yeterli bulur; bütün yörüngeleri, yüksek derece
graviteyi, UKF'yi veya gerçek görev doğruluğunu yeterli ilan etmez.

## Üç incelenebilir paket

| Paket | Commit | Değişiklik |
|---|---|---|
| A | `4134806` | Kısa/sıyıran temas çiftleri ve son yörüngede temas kontrolü |
| B | `b441958` | Sınırlı TRF, son nokta tanıları, SVD kovaryans, tarihsel kayıt koruması |
| C | Bu teslimin üçüncü commit'i | Hermite tutarlılığı, ölçüm türev zinciri ve ortak F2 sayısal ağı |

Bağımlılık sırası A→B→C'dir. F tabanındaki D2, C2, C4-I ve F2g altyapısı
gereklidir; bu commit'lerin eski main'e tek başına uygulanması yeterli değildir.

### A — gölge olayları

[Olay bulucu](../../src/lunar_od/dynamics.py) dış/iç temas denklemlerini korur:
`gamma-(a_sun+a_occ)` ve `gamma-abs(a_sun-a_occ)`. Yoğun çıktı üzerindeki yerel
minimum/maksimum taraması aynı adımda kalan giriş–çıkış çiftlerini bulur.
Yuvarlanmış aydınlanma katsayısı kök referansı değildir.

Ön geçiş toleransı 100 kat sıkılaştırıldı; makine sınırı gözetilir. Kesme
noktaları son 6/42/48 ve F2g yayılımında kontrol edilir. Geometrik artık
`1e-9 rad` sınırını aşarsa açık hata verilir; bu kontrol tek başına menzil
doğruluğu sertifikası değildir. Arayüz, yay içi temas ve ters zaman sıralaması
korundu.

1, 5, 18,941 ve 60 saniyelik tutulmalar dört fazda ve iki yönde sınandı.
18,941 saniyelik örnekte ayrıca 60 saniyelik çevrimin bütün tam-saniye fazları
iki yönde tarandı: 120/120 çift bulundu; analitik dairesel K=0 referansıyla en
büyük zaman farkı `1,23e-9 s`, kabul sınırı `1 ms` oldu. Normal tutulma, tam
Güneş, teğet ve yay sınırı örnekleri de kaydedildi. Sonlu tarama keyfî kuvvet
ve efemeris geçmişleri için matematiksel tamlık ispatı değildir.

### B — çözüm ve kovaryans

[Ortak çekirdek](../../src/lunar_od/bounded_least_squares.py),
[F2 sürücüsünün](../../studies/phase17/scripts/phase_f2_lro_od.py) residual,
Jacobian, önsel ve süreç adaptörlerini kullanır. Residual `y-h`, türevi `-H`,
amaç `0.5*||R_white||²` olur. Önseller açık beyazlatılmış satırlardır.

SciPy yoğun `trf/exact`, parametre ölçekleri ve doğrudan `K>=0` sınırı
kullanılır. Küçük maliyet/adım değişimine göre duruş kapalıdır. Native duruş
kodu ayrıca saklanır. `MAX_ITER=30` yeni yolda fonksiyon değerlendirme
bütçesidir; geçmiş Jacobian doğrusallaştırmalarını sayar, eski iterasyonlarla
eşdeğer sayılmaz.

Son noktada ölçekli gradyan, uygulanabilir yön/KKT, aktif sınırlar, sınırlı GN
kalan düzeltmesi ve tahmini maliyet azalması yeniden hesaplanır.
`STATIONARY` yalnız bu tanılar 0,1 formal-sigma sayısal ölçütünü sağladığında
verilir. Durağanlık, tekillik/rank ve veriye uyum ayrıdır. Büyük indirgenemez
residual yanında küçük GN azalmasının çıkarma sırasında kaybolması da
regresyonla kapatıldı.

Kovaryans son beyazlatılmış tasarımın SVD'sinden kurulur. Rank eşiği
`eps*max(m,n)*s_max` kaydedilir. Rank eksikse kovaryans/sigmalar verilmez;
null yönleri ve koşulluluk raporlanır. Sınırdaki tam-rank kovaryans yerel,
kısıtsız formal kovaryanstır; kesilmiş dağılım veya gerçek doğruluk değildir.

Parametre sırası korunur: durum `0..5`, K `6`, menzil biasları `7..9`, Doppler
biası `10`, ampirik ivmeler `11..14`. K=0 karşı örneği, lineer/koşullu/rank
eksik referanslar, ölçek değişimi ve yanlış yakınsama etiketi sınandı.

`--solver legacy` eski çözüm döngüsünü korur; legacy etiketleri yeni G1
kapısını geçiremez. Tam tarihsel sayılar ayrıca özgün commit/kernel/model
kimliklerini gerektirir. Yeni sonuçlar `outputs/numerical-core/{trf,legacy}/`
altına yazılır. F2f/g parçaları ve F2g kontrolleri kod kimliği uyuşmadan
birleştirilemez; eski kontroller yeni koşuya sessizce taşınmaz.
Aynı günün F1 bias önselleri bağımsız kalibrasyon yerine koşullu önsel olarak
etiketlenir. Hata durumunda süreç havuzu kapatılır.

### C — interpolasyon ve türev zinciri

[Dünya efemerisi](../../src/lunar_od/ephemeris.py) tek konum–hız Hermite eğrisi
kullanır; hız eğrinin türevidir. Güneş konum PCHIP'i korunur. Açık
`earth_interpolation='legacy_pchip'` eski MATLAB/DE421 kaydı içindir; fixture
değerleri/eşikleri değiştirilmedi, test eski sözleşmeyi açıkça seçer.

[Durum/STM yardımcısı](../../src/lunar_od/radiometrics.py) izin verilen sınır
devamında uç Hermite polinomunu kullanır. İç formül ve tam uç örnekleri
korundu. K ve ampirik sütunları aynı durum yardımcısını kullanır. Mevcut
destek, iki-ULP temsil toleransı ve genel 10 saniyelik devam sınırı
genişletilmedi. F2'nin 120 saniyelik ön desteği, Doppler uçları ve son desteği
korundu; destek dışı sorgular reddedilir.

İkinci/üçüncü derece analitik polinomlar düğüm, iç aralık ve uçlarda sınandı.
DE440 SPICE 60/240/600 saniyelik tablolarda `v-dr/dt` yapı gereği sıfırdır.
60 saniyede SPICE'a göre en büyük konum/hız farkları yaklaşık `1,23e-7 m` ve
`3,05e-9 m/s` oldu; ince gridde yuvarlama tabanı görülür. Bu değerler doğrudan
gözlem hata bütçesi sayılmadı.

C2 son satırındaki yaklaşık `0,03 m/s_clock` fark giderildi. Kalan farkın
yalnız interpolasyondan gelmediği gösterildi: hareketli Shapiro gövdesinin
açık zaman türevi ve TT/TDB aralık ölçeğinin orta-nokta bağımlılığı eksikti.
[Mevcut modelin türevleri](../../src/lunar_od/relativistic_light_time.py)
bu zinciri artık taşır. Shapiro zaman katkısı
`-(dS/dr_tx+dS/dr_rx) dot v_body` olur. Aralık `tau*(1-D'(t3-tau/2))` için
durum çarpanı `1-D'+tau*D''/2`, saat ek terimi `-tau*D''` olur.
Yeni fizik modeli veya nominal gözlem formülü eklenmedi.

`D''`, analitik hız bağıntısının göreli zamanda 1 saniyelik farkından alınır;
C2'de bağımsız 10 saniyelik farkla katkı farkı yaklaşık `3e-13 m/s_clock`
oldu. Hareketli-gövde ve karesel saat bağıntılı regresyonlar önce başarısız,
düzeltmeden sonra geçer oldu.

Dört adımın tümü saklandı: `1e-4, 1e-3, 1e-2, 1e-1 s`. Son satırda 0,1
saniyenin artığı O2'de `2,91e-7`, O3'te `-6,61e-7 m/s_clock` oldu. En küçük
adımda yaklaşık `6e-4 m/s_clock` fark kalır; büyük menzil değerlerini çıkarıp
küçük adıma bölmek tabanı büyütür. Bu türev tanısı counted-Doppler değeri
değildir; Doppler bütçesiyle geçer sayılmadı, tek iyi adım seçilmedi.

## Birleşik bütçe ve gerçek veri sınırı

[Hata bütçesi](numerical-core/error_budget.csv),
[nicel kanıt](numerical-core/qualification.json) ve
[komut/süre/kimlik kaydı](numerical-core/experiments.json) teslimin parçasıdır.
Sistematik katkılar RSS ile birleştirilmedi. Birleşik gözlem farkı doğrudan
ölçüldü; parametre etkisi beyazlatılmış tasarımın tanımlanabilir SVD yönlerinde
değerlendirildi.

Kısa sentetik deney: 8.481 saniyelik gölgeli/gölgesiz yaylar, 24'er menzil ve
Doppler örneği, 6/42/48 yolları, nokta-Ay+SRP, gerçek DE440 istasyonları ve C2
SSB modeli. Ortam/bias/yüksek graviteyi yeterli saymaz. Referansın 5→2,5
saniye ve sıkı tolerans değişimi ayrıca kaydedildi.

| Profil | rtol | Menzil sınırı | Doppler sınırı |
|---|---:|---:|---:|
| PROJECT | 1e-11 | 0,05 m | 1e-6 m/s |
| DSN_REALISTIC | 1e-12 | 0,01 m | 3e-7 m/s |
| LRO S-band | 1e-10 | 0,01 m | 2e-5 m/s |

Üç profil kısa deneydeki gözlem ve toplam 0,1 formal-sigma bütçelerini sağladı.
Bütün eski integratör varsayılanları için geçerlik iddiası yoktur.

Gerçek LRO altkümesi: mevcut UTDF'den 600 saniye, 12 menzil/23 Doppler, N=2,
15 parametre. Üç değerlendirmeyle sınırlı çözüm doğru biçimde `MaxIter` döner;
durağan/tek/görev doğruluğunda ilan edilmez. Sıfır ampirik ivme üretim
yayılımını bit düzeyinde yeniden üretir.

20 saniyelik ağ Doppler'de `2,96e-5 m/s` ile **FAIL** oldu; kanıtta korunur.
Yeni TRF yolu 10 saniyelik ağı süreçlere ve tahmine açıkça taşır; legacy 20
saniyeyi korur. 10 saniyede farklar yaklaşık `5,42e-5 m`, `2,98e-6 m/s` ve
`0,00187 formal sigma` oldu. Referansın 10→5 saniye/tolerans değişimi ayrıca
raporlanır. Kesin son değerler CSV/JSON'dadır.

Ek LOLA SPK deneyi sabit 36 menzil/30 Doppler örneği, veri/kernel hashleri ve
5→10 saniye farkını kaydeder. Yörünge kestirimi veya bağımsız gerçeklik
doğrulaması değildir. Yaklaşık 485 m ayarlanmamış menzil RMS'sine 0,01 m
sayısal eşik uygulanmaz. F1 detrend, F2 fit ve LOLA konum farkları ayrıdır.

## Korunan FAIL'ler ve sonraki kapanışlar

Yeni sayısal çalışma, başarısız denemeler ve ortam hataları dahil 1.800 saniye
sınırı içinde tutuldu. Kesin süre ve son test sayıları deney kaydındadır.
Checkout korumasının ortak sanal ortamın paket yolunu da kaldırdığı ilk
toplama hatası yerel paket alias'ıyla giderildi. Bütün `lunar_od` importları
bu ayrı çalışma ağacına sabitlendi; gerçek bağımlılık yolları kayıtlıdır.

Depo koruma kontrolünde başlangıçtan kalan tek açık: değiştirilmemiş
`src/lunar_od/__init__.py` ile manifest hash'i uyuşmaz. Başlangıç/son hash
aynı `2e702373...`, manifest `bc3a5597...` değeridir. Bu FAIL geçer yapılmadı.
Düzeltilen dosyaların yeni kimlikleri önceki kimlikler korunarak eklenmiştir.

F2d/e tabanda tamamlanmış tarihsel deneylerdir; eski "F2d bilinmiyor" ifadesi
güncel değildir. Tamamlanma bilimsel yeterlilik değildir. Eski F2d
`STATIONARY` etiketleri yeni tanıları geriye dönük kazanmaz. F2f/g tam
yeterliliği bu görevde doğrulanmadı.

Sonraki sıra ve kapanış koşulları:

1. **Main bağımlılıkları ve kaynak kapısı:** __init__ manifest farkının C4-I
   geçmişini ayrı incele; main/F/D2/C2/C4-I/F2g eşlemesini çıkar. A→B→C'yi
   ortak uyumlu tabanda sınamadan birleştirme. Main entegrasyonu yapılmadı.
2. **Yüksek derece/paralellik:** N=200/300 gölgeli/sıyıran kısa altkümede
   6/42/48/ampirik yollar, tolerans/ağ ve serial/process eşitliğini sına.
   Başlangıç için 15–60 dakika kaynak payı bir planlama varsayımıdır;
   N=2 süresi yüksek derece için garanti değildir. Kapanış: aynı bütçeler,
   temas tamlığı ve bağımsız referans yakınsaması.
3. **Tam F2d/e/f/g:** çoklu başlangıç, K=0, son KKT/GN, rank/kovaryans,
   menzil+Doppler ve tutulmuş gün tahmini birlikte raporlansın. Eski F2d
   koşuları 1,24/26,32 saat; F2e 0,88/1,48 saat sürmüş. Bunlar yeni TRF/N300
   süresi değildir. İlk yeni doğrusallaştırmayı ölçerek süre çıkar; zor
   başlangıç için gün mertebesinde pay gerekebilir. Bu koşular ilk teslimin
   bitiş koşulu yapılmadı.
4. **Ortak bilimsel profil:** kuvvet/ölçüm/zaman/frame/GM/veri/önsel/sayısal
   kimlikleri birlikte taşınsın. SRP'nin ForceModelContract kimlik açığı ve
   desteklenmeyen kombinasyonlar ayrıca kapatılsın. Eski DE421/PCHIP açık
   yeniden üretim profili olsun; sessiz model düşürme olmasın.
5. **Gerçek veri bağımsızlığı:** kalibrasyon/doğrulama geçişlerini ayır;
   koşullu F1 önsellerini bağımsız sayma. LOLA/gravite korelasyonları, kovaryans
   kalibrasyonu ve gerçek konum doğruluğu ayrıca doğrulansın. Performans ve
   arayüz değişimleri bilimsel kapanışlardan sonra ele alınsın.

Hiçbir tarihsel FAIL yeni kod geçtiği için PASS'a dönüştürülmez.
