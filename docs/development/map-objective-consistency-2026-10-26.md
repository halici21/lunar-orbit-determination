# Açık önsel ve aday kabul maliyeti — 26 Ekim 2026

Taban `71e088f`. Position/range-rate BLS ve SRIF adımı, açık önsel terimini
içeriyordu; aday kabul maliyeti yalnız veri artığını kullanıyordu. Böylece veri
maliyetini biraz artırıp toplam MAP amacını azaltan doğru adım reddedilebiliyordu.

Bağımsız karşı örnek: `h(x)=x²`, `y=4`, önsel ortalama `1`, varyans `1/2`.
Amaç `(4-x²)² + 2(x-1)²`; pozitif optimum `x³-3x-1=0` denkleminin
`1,8793852415718` köküdür. Diğer beş durum yönü tam rank lineer gözlemlerle
korunur. Üretim QR/LM ve kabul döngüsü aynen çalışır; yalnız yayılım ve ölçüm
adaptörleri bu analitik modele bağlanır. İlk koşuda dört yol da başarısız:
SRIF yaklaşık `2`, BLS yaklaşık `1,99337748` noktasında kalır.

Düzeltme: mevcut beyazlatma faktörüyle önsel karesi nominal ve aday maliyetine
eklenir. Aynı amaç adım, kabul ve raporlanan `final_cost` için kullanılır.
Önsel verilmediyse sayısal adım düzenleyicisi maliyete eklenmez; veri-only
hesap sırası korunur. Kestirici aileleri arasındaki maliyet anlamı açıktır:
batch'te veri + açık önsel amacı; UKF'de biriken NIS.

Son karşı örnekler, covariance ve sqrt-information girdilerinin ikisinde de
optimumu, son amaç değerini ve aynı son noktada kurulan formal Gauss–Newton
bilgi/kovaryansını bağımsız ifadelerle kontrol eder. Bu formal kovaryans,
nonlineer problemin tam Bayesian posterior varyansı veya görev doğruluğu değildir.

Kanıt: `map-objective-baseline` dört FAIL; ilk uygulamadaki eksik BLS
`prior_sqrt_info` değişkeni de ayrı başarısız logda korunur.
`map-objective-and-posterior`: 93 PASS, 9 yavaş test atlandı. Profil aktarımı,
mevcut estimator/scenario/R1 kovaryans regresyonları dahil; beklenen FD-yavaşlık
uyarısı var. Tarihsel sonuç/fixture/eşik değiştirilmedi.

Bu düzeltme bütün nonlineer sonlandırmaları, robust ağırlıkların son noktadaki
yeniden kurulmasını veya tam gerçek veri kabulünü sertifikalandırmaz.
R5/R7'nin bilgilendirici gölge/K/ampirik deneyleri ve R8 bütçe/manifest kapanışı açık.
