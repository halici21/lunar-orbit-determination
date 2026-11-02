"""Compact, source-based explanations; no inferred solver success."""
from collections import Counter
from statistics import median


def number(value):
    return "kayıtlı değil" if value is None else f"{value:.4g}"


def result_status(result, details):
    """Recorded process/success headline without computing diagnostic statistics."""
    arcs = result.updates if result.metadata.get("record_kind") == "arc_solution" else result.arcs
    known = [r.get("operational_success") for r in arcs if r.get("operational_success") is not None]
    success = f"{sum(v is True for v in known)}/{len(known)} yay kaydı başarılı" if known else "operasyonel başarı bilinmiyor"
    if arcs and len(known) < len(arcs):
        success += f" · {len(arcs)-len(known)} kayıtta başarı alanı yok"
    status = details.get("status", "İşlem durumu bilinmiyor")
    headline = f"{status} · {success}"
    if status == "Tamamlanma kaydı yok":
        headline += " · Koşu sürüyor veya yarım kalmış olabilir; Çalıştır ekranını kontrol edin, bitince sonucu yeniden açın."
    return headline


def result_summary(result, details):
    rows = result.updates
    arcs = rows if result.metadata.get("record_kind") == "arc_solution" else result.arcs
    headline = result_status(result, details)
    stops = Counter(r.get("stop_reason") or "Bilinmiyor" for r in arcs)
    stop_text = ", ".join(f"{key} ({value})" for key, value in sorted(stops.items())) or "kayıtlı değil"
    errors = "Tüm kaynak · N=" + str(len(rows)) + " · Konum hata medyanı " + number(median(r["position_error_m"] for r in rows)) + " m · Hız hata medyanı " + number(median(r["velocity_error_mps"] for r in rows)) + " m/s"
    nis = sum(r.get("normalized_nis") is not None for r in rows)
    diagnostics = f"Yakınsama: ayrı bir doğrulama alanı yok · Duruş: {stop_text}\n"
    diagnostics += f"Normalize NIS: {nis}/{len(rows)} kayıtta mevcut" if nis else "Normalize NIS: kaynakta yok"
    truth = bool(result.states) and result.metadata.get("state_role") == "truth"
    diagnostics += " · Referans yörünge: mevcut" if truth else " · Referans yörünge: yok veya rolü doğrulanmamış"
    diagnostics += " · Çalışma süresi: doğrulanmış kayıt yok"
    return headline, errors, diagnostics


def comparison_summary(comparison):
    changes = comparison.differences
    def value(v):
        text = str(v).replace("\n", " ")
        return text if len(text) <= 48 else text[:45] + "…"
    ordered = sorted(changes, key=lambda d: (not d["field"].startswith(("config.", "options.")), d["field"]))
    fields = [f'{d["field"]}: {value(d["A"])} → {value(d["B"])}' for d in ordered[:3]]
    settings = ("Değişen ayarlar: " + "; ".join(fields)) if fields else "Kayıtlı ayarlarda fark yok."
    if len(changes) > 3:
        settings += f" · +{len(changes)-3} fark: Ayarlar ve kaynaklar sekmesinde"
    if not all(s.get("config") for s in comparison.snapshots):
        settings += " · Ayar kaynağı eksik; ayar eşitliği doğrulanamadı."
    if not comparison.time_compatible or not comparison.pairs:
        effect = "Hata değişimi: karşılaştırılamıyor; doğrulanmış ortak kayıt yok."
    else:
        values = []
        for key, title, unit in (("position_error_m", "Konum", "m"), ("velocity_error_mps", "Hız", "m/s")):
            stats = comparison.metrics["paired"][key]
            a, b = stats["A"]["median"], stats["B"]["median"]
            delta = b-a
            percentage = f"{100*delta/a:+.2f}%" if a else "A=0; yüzde tanımsız"
            values.append(f"{title} {delta:+.4g} {unit} ({percentage})")
        effect = f"Eşleşen N={len(comparison.pairs)} · Hata medyanlarının farkı B−A: " + " · ".join(values)
        effect += " · Negatif değer daha düşük kayıtlı hata."
    scope_a = comparison.left.metadata.get("nis_scope")
    scope_b = comparison.right.metadata.get("nis_scope")
    complete_nis = all(all(r.get("normalized_nis") is not None for r in source.updates) for source in (comparison.left, comparison.right))
    nis = "NIS farkı: kapsam eksik/farklı veya tanılama kayıtları eksik." if not (scope_a and scope_a == scope_b and complete_nis) else "NIS: aynı kayıtlı kapsam; ayrıntılar grafiklerde."
    return settings + "\n" + effect + "\n" + nis + " Çalışma süresi farkı: doğrulanmış süre kaydı yok."
