# Risk kaydı

Yaşayan belge. İlk sürüm [bağımsız değerlendirme](../validation/independent-assessment-2026-10-31.md) §5'ten; NPR 8000.4 tarzı 5×5 (O olasılık, S sonuç, Skor O×S). Bir iş paketi bir riski azalttığında aynı PR'da O/S, durum ve kanıt güncellenir; kapanan risk silinmez.

Durum: `open` açık, `mitigating` azaltma sürüyor, `closed` kapandı (kanıt bağlantısıyla), `accepted` sahip kabul etti.

| ID | Risk | Bağlı bulgular | O | S | Skor | Azaltma | Durum |
|---|---|---|---|---|---|---|---|
| R-01 | Gerçek veride metre seviyesi sistematik hata (ışık zamanı/Shapiro/medya/istasyon) | MEAS-01/02/06/07 | 5 | 5 | **25** | F2 | open |
| R-02 | Sentetik testler tutarsızlıkları gizliyor → yanlış "doğrulandı" kanısı | X-01…X-09, TST-02/03 | 5 | 4 | **20** | F1, WP5.1, F7 | open |
| R-03 | Gerçek veri sonuçlarının bağımsızlığı bozuk (F3 aynı gün ayar+puan, hold-out yok, döngüsel önseller) | RD-01…RD-05 | 5 | 4 | **20** | WP0.2, F5 | open |
| R-04 | SRIF yanlış kabul testi → durma ve yanlı tahminci karşılaştırması | EST-01/02/03/07 | 5 | 4 | **20** | F4 | open |
| R-05 | Fizik testlerinin CI'da koşmaması → regresyonlar fark edilmeden main'e girer | TST-01/05 | 4 | 4 | **16** | F7 | open |
| R-06 | UKF/EKF ile batch arasında model farkı (Doppler interp, efemeris, P0) | MEAS-04/05, EST-19 | 4 | 4 | **16** | F1 | open |
| R-07 | Eksik kuvvetler (k2, albedo/IR, Dünya gölgesi) F2r kalıntısını açıklıyor olabilir | DYN-06/07/08, RD-10 | 4 | 4 | **16** | F3 | open |
| R-08 | Kovaryans gerçekçi değil (birim bağımlı kırpma, jitter, regülarizasyon, korelasyonlu gürültü beyaz) | EST-06/12/13/15, RD-06 | 4 | 4 | **16** | F4 | open |
| R-09 | Uzun yaylarda sayılan Doppler çözücü çöküşü | MEAS-03 | 4 | 3 | 12 | F1 | open |
| R-10 | Tekrarlanamayan koşular (tohum yok, kernel kaydı yok, masaüstü farklı ayarlar) | CFG-01…04,10 | 4 | 3 | 12 | WP1.6 | open |
| R-11 | Efemeris/GM/çerçeve nesil karışımı | DYN-01/02, CFG-04 | 5 | 3 | 15 | F1 | open |
| R-12 | Kuvvet parmak izi farklı fiziği eşit gösterir → parite iddiaları geçersiz | DYN-10/11 | 3 | 4 | 12 | WP1.5 | open |
| R-13 | Yakınsama gevşek toleransta ilan ediliyor; durma kriterleri birime bağlı | EST-04/05 | 4 | 3 | 12 | F4 | open |
| R-14 | DMC Q taşması geçiş boşluklarında | EST-09 | 3 | 4 | 12 | F4 (F3'ten önce!) | open |
| R-15 | İstasyon zinciri predict BPC, EOP yok; zamanla bozulan doğruluk | MEAS-06 | 3 | 4 | 12 | WP2.4 | open |
| R-16 | Gereksinim/izlenebilirlik yok → "NASA/ESA seviyesi" iddiası denetlenemez | TST-08 | 5 | 2 | 10 | WP0.3 | open |
| R-17 | Yerel main'in geride olması ve 20+ eski worktree → yanlış tabanda çalışma | — | 3 | 3 | 9 | WP0.1 | open |
| R-18 | Dış kehanet repoda kalıcı değil | TST-02/08 | 3 | 3 | 9 | WP5.1 | open |
| R-19 | Gizli (kullanılmayan) modüllerde yanlış fizik ileride kullanılır | MEAS-13 | 2 | 4 | 8 | WP2.6 | open |
| R-20 | CCSDS arayüzü yok → bağımsız araçlarla çapraz kontrol zor | — | 4 | 2 | 8 | F6 | open |
| R-21 | Depo şişmesi ve sıra bağımlı testler | CFG-12, TST-05 | 3 | 2 | 6 | F7 | open |
| R-22 | Bilinen-kırmızı tam regresyon: yeni regresyonlar mevcut 20 hatanın arasında kaybolur; koruma kapısı anlamını yitirir | TST-09/10/11, TST-01 | 5 | 3 | 15 | WP0.5 | open |

## Değişiklik geçmişi

| Tarih | Risk | Değişiklik | Kanıt |
|---|---|---|---|
| 2026-10-31 | hepsi | İlk kayıt | bağımsız değerlendirme |
| 2026-10-31 | R-22 | Tam regresyon sonrası eklendi (10 FAIL / 10 ERROR, `origin/main`'de mevcut) | değerlendirme §0.1 |
