"""Readable source metadata and settings, with plain-text tree values."""
import json
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QHeaderView


def display(value):
    if value is None:
        return "Bilinmiyor"
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class RunDetailsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.heading = QLabel("Henüz koşu seçilmedi")
        self.heading.setTextFormat(Qt.PlainText)
        self.heading.setWordWrap(True)
        self.heading.setObjectName("detailHeading")
        self.context = QLabel("Arşivden bir koşu açın veya sonuç klasörü seçin.")
        self.context.setTextFormat(Qt.PlainText)
        self.context.setWordWrap(True)
        layout.addWidget(self.heading)
        layout.addWidget(self.context)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Alan", "Kayıtlı değer"])
        self.tree.setColumnWidth(0, 145)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.setAlternatingRowColors(False)
        self.tree.setRootIsDecorated(True)
        self.tree.setWordWrap(True)
        self.tree.setAccessibleName("Koşunun kayıtlı ayarları ve kaynak bilgileri")
        layout.addWidget(self.tree, 1)

    def clear(self):
        self.tree.clear()
        self.heading.setText("Henüz koşu seçilmedi")
        self.context.setText("Arşivden bir koşu açın veya sonuç klasörü seçin.")

    def show_details(self, name, metadata, data, sources=(), records=None):
        self.tree.clear()
        self.heading.setText(name)
        estimator = metadata.get("estimator_type", "Bilinmiyor")
        self.context.setText(f"{estimator} · {records if records is not None else '—'} kayıt\n{data.get('status', 'Bilinmiyor')}")
        job = data.get("job", {})
        def group(title, values):
            top = QTreeWidgetItem(self.tree, [title])
            def add(parent, key, value):
                if isinstance(value, dict) and value:
                    node = QTreeWidgetItem(parent, [str(key)])
                    for child, item in value.items():
                        add(node, child, item)
                else:
                    text = display(value)
                    item = QTreeWidgetItem(parent, [str(key), text])
                    item.setToolTip(0, str(key))
                    item.setToolTip(1, text)
            for key, value in values.items():
                add(top, key, value)
            top.setExpanded(title in {"Koşu", "Referans ve birimler", "Kaynak koşu ve değişiklikler"})
        group("Koşu", {"Kimlik": job.get("run_id"), "Zaman / UTC": job.get("created_at"),
            "Seed": metadata.get("seed"), "Çözücü": estimator,
            "Üretici": metadata.get("producer"), "Kod sürümü": metadata.get("commit")})
        group("Referans ve birimler", {key: data.get("units", metadata.get(key)) if key == "units" else metadata.get(key) for key in
            ("epoch", "time_scale", "frame", "origin", "units", "record_kind", "state_role")})
        for field, title in (("config", "Etkin ayarlar"), ("options", "Zamanlama ve başlangıç"), ("force", "Kuvvet modeli")):
            group(title, data.get(field) or {"Bilgi": "Kaynakta yok"})
        origin = job.get("derived_from")
        if isinstance(origin, dict):
            changes = origin.get("form_changes", [])
            group("Kaynak koşu ve değişiklikler", {"Kaynak kimliği": origin.get("run_id"),
                "Kaynak dosyası": origin.get("result_path")})
            top = QTreeWidgetItem(self.tree, ["Değiştirilen alanlar", "Önce → sonra"])
            for change in changes if isinstance(changes, list) else ():
                if isinstance(change, dict):
                    QTreeWidgetItem(top, [str(change.get("field", "Bilinmiyor")),
                        display(change.get("before")) + " → " + display(change.get("after"))])
            top.setExpanded(True)
        group("Kaynak dosyaları", {str(i+1): {"Dosya": s.get("path"), "SHA-256": s.get("sha256"), "Bayt": s.get("bytes")}
            for i, s in enumerate(sources)})
        group("Notlar", {str(i+1): value for i, value in enumerate(data.get("notes", ()))})
