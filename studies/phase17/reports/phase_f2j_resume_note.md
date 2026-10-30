# Phase F2j — devam notu (2026-10-26)

Workspace `HANDOFF_lunar_od_roadmap.md` §3'ün kopyası; hesap değişikliği öncesi kaydedildi.

## ŞU AN (2026-10-26 ~17:30): F2j (üç günlük yay) KOŞUYOR — kaldığım yer

> Hesap değişikliği öncesi kayıt. Aynı not depoda da var:
> `python_port_f1/studies/phase17/reports/phase_f2j_resume_note.md`.

**Ne koşuyor:**
- F2j: gerçek LRO verisiyle üç günlük yay (2009-260, 261, 262) ve bir günlük tahmin testi (2009-263).
- Ön-kayıt ve giriş noktası commit edildi ve push edildi: `9cc3873` (worktree `python_port_f1`, dal
  `phase-f1-lro-residuals`).
- Ön-kayıt dosyası: `studies/phase17/configs/phase_f2j_lro_three_day_arc_predeclaration.json`.
- Model F2h'ninki:
  - N = 300;
  - yalnızca yörünge boyunca ampirik terimler (C_T, S_T); C_N ve S_N 1e-11'e sabitlendi;
  - `F2H_PRIORS`;
  - başlangıç F2h S1 çözümü;
  - durağanlık eşiği **göreli**: Gauss–Newton azalması < 1e-3·χ².
- 8 işçi, 15:16'da başladı.
- Süreç: python PID 10080 (alt süreç 31656). Bu Claude oturumunun arka plan kabuğundan başlatıldı.

**Son durum (17:00 civarı):**

| iterasyon | menzil RMS | K | geçen süre |
|---|---|---|---|
| 1 | 3.905 m | 0.0550 | 1834 s |
| 2 | 3.265 m | 0.0554 | 3419 s |
| 3 | 2.702 m | 0.0563 | 4297 s |
| 4 | 2.548 m | 0.0564 | 5951 s |

- İterasyon başına 15–27 dakika sürüyor.
- Tahminen 3–6 saat daha var, sonra tahmin adımı için ~20 dakika.

**Süreç hâlâ yaşıyor mu? Kontrol et:**

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'f2j' }
Get-Content C:\Users\erayh\Documents\Python\lunar-orbit-determination\f2j1_progress_N300_p.log
```

- İlerleme günlüğü: `<workspace>/f2j1_progress_N300_p.log`. Workspace kökünde, çünkü günlük cwd'ye göre yazılıyor.
- Çıktı: `<workspace>/lro_data/f2j_run.txt`.
- Canlı izleme: `powershell -ExecutionPolicy Bypass -File lro_data\f2_status.ps1 -Tag j`.

**Süreç öldüyse (uygulama ya da oturum kapanınca ölebilir), yeniden başlat.**
- Checkpoint yok; koşu baştan başlar.
- Workspace kökünden çalıştır:

```powershell
cd C:\Users\erayh\Documents\Python\lunar-orbit-determination
Remove-Item f2j1_progress_N300_p.log -ErrorAction SilentlyContinue
& .\python_port_f1\.venv\Scripts\python.exe .\python_port_f1\studies\phase17\scripts\phase_f2_lro_od.py --f2j --workers 8 *> .\lro_data\f2j_run.txt
```

- 8 işçiden fazlasını kullanma. Başka uygulamalar ~30 GB commit tutuyor ve 14 işçide MemoryError olmuştu.
- Makinenin uykuya geçmesini engelle; F2d bir kez uykudan dolayı bir gün takılmıştı.

**Bitince yapılacaklar:**
1. Sonuç dosyası: `studies/phase17/results/phase_f2j_lro_three_day_arc.json`. Çıktının sonunda kapı JSON'u yazar.
2. Kapıları hesaplandığı gibi raporla:
   - G1: Converged ya da STATIONARY (göreli kural);
   - G3: menzil ≤ 2 m ve Doppler ≤ 5 mm/s.
3. Raporlanacaklar, kapı değil:
   - yay boyunca ve 263 tahmin gününde LOLA'ya göre RIC;
   - tahmin artıkları ve fit sonundan bu yana geçen saat;
   - K ve C_T/S_T değerleri, F2h ile karşılaştırmalı.
4. Raporu `studies/phase17/reports/phase_f2_lro_od.md` dosyasına **§16** olarak ekle. Biçim §15 gibi olsun:
   ön-kayıt, kanıt, kapı tablosu, okuma.
5. Commit ve push yap. Mesaj sonuna `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` ekle.
6. Bu handoff'u, `memory/lunar-od-nasa-roadmap.md` dosyasını ve canlı panoyu güncelle.

**Sahip kararı bekleyen F2 maddeleri:**
- F2i G1 için göreli eşiği gerekçeli kabul etmek. F2j bu eşiği baştan ön-kayıtla kullanıyor; F2i'nin sonucu değişmedi.
- F2g G4e: FD kâhin tabanı.
- Faz E onayı: üretime ampirik ivmeler ve numba harmonik STM.

**F2j'den sonra yol haritası:**
- daha uzun yaylar ve tahmin testleri;
- box-wing SRP;
- seri adımların paralelleştirilmesi (aday yayılımı, tahmin);
- Faz D: smoother, DMC, çok yay, Monte Carlo;
- Faz B: Orekit aracı;
- veriye bağlı C maddeleri: BLQ, EOP, WEA hava dosyaları.

