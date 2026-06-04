# IGQK — Technisches Gutachten

**Gutachter-Perspektive:** Informationsgeometrie, numerische Optimierung, LLM-Inferenz-Effizienz
**Gegenstand:** `Mirkulix/IGQK` (Feb 2026, Python-Paket `igqk/`).
**Datum:** 2026-06-04
**Methodik:** Quellcode-Review gegen die behauptete Theorie. Jede Bewertung mit Datei + Zeile belegt.

> **Hinweis:** Dieses Repo ist die frühere, **Python-only**-Variante. Der Nachfolger `Mirkulix/qland` enthält dieselben Python-Module **plus** eine Rust-„IGQK Math Engine" (`qlang/`, ~31k LoC). Die Rust-spezifischen Befunde (totes Quantenmodul, Finite-Differenzen-Training, Theorem-„Verifikation") stehen im `EVALUATION.md` von `qland`. Hier: der Python-Kern, der in **beiden** Repos byte-identisch ist.

---

## Zusammenfassung (TL;DR)

Was das README verspricht — eine information­geometrisch *begründete* Quantisierung, getragen von Quantengradientenfluss mit Konvergenz-/Kompressionsschranken — ist im Code **nicht realisiert**. Tatsächlich tut der Code:

1. **Magnitude-Threshold-Ternarisierung** `{-1,0,+1}` per fester Schwelle — algorithmisch identisch zu *Ternary Weight Networks* (Li & Liu, 2016).
2. Standard-Sparse (Top-k) und Low-Rank (Truncated SVD).

Die einzig potenziell neue Idee — *die Fisher-Metrik soll die Quantisierungsentscheidung steuern* — fehlt: Die Projektion ignoriert die Metrik komplett und arbeitet euklidisch. Dazu kommt eine **aspirationale README**, die ~12 Module auflistet, die nicht existieren, und „Validierungs"-Berichte mit MNIST auf **Zufallsniveau (10,40 %)**, gemeldet als „0 % Verlust, perfekt".

---

## PHASE 1 — Bestandsaufnahme

**Mathematik-Module (`igqk/`):**
- `igqk/core/manifold.py` — Statistische Mannigfaltigkeit, Fisher-Metrik, „Riemann-Distanz", Geodäten
- `igqk/core/quantum_state.py` — Dichtematrix ρ (Eigenzerlegung, Entropie)
- `igqk/core/measurement.py` — Messoperatoren / Kollaps auf diskrete Gewichte
- `igqk/core/evolution.py` — Quantengradientenfluss
- `igqk/compression/projection.py` — „Optimale Projektion" Π: M → N
- `igqk/integration/pytorch.py` — Trainer-Anbindung
- **Theorie-Quelle:** `Entwicklung_der_IGQK-Theorie_Mathematische_Details.pdf`, `README.md`

**(a) Metrik:** Fisher-Information (`manifold.py:7-14`, `:57`) — berechnet, aber als Geometrie ungenutzt.
**(b) Zielfunktion:** behauptet `dρ/dt = -i[H,ρ] - γ{G⁻¹∇L,ρ}` (`README.md:22`); real euklidische Verzerrung `‖W-Π(W)‖²` (`projection.py:133`).
**(c) Schema:** ternär per Schwelle `0.5·std` (`projection.py:253`, `measurement.py:90`); zusätzlich Low-Rank/Sparse.
**(d) Sätze:** 5.1 Konvergenz `≤ min L + O(ℏ)`, 5.2 Verzerrung `D ≥ (n-k)/(2β)·ln(1+β σ²_min)`, 5.3 `E_gen ≤ E_train + √(I(A:B)/n)` (`README.md:184-216`).

**Was die Methode TATSÄCHLICH tut:** Post-hoc Magnitude-Threshold-Quantisierung eines klassisch trainierten Modells. Keine information­geometrische Entscheidung, kein Quanteneffekt, kein quantisierungs-bewusstes Training.

---

## PHASE 2 — Korrektheitsprüfung

### 2.1 Informationsgeometrie nicht implementiert (zentraler Mangel)
- `manifold.py:117-152` — `riemannian_distance` gibt für **alle** Metriken (`hellinger`/`kl`/`euclidean`) denselben euklidischen `‖θ1-θ2‖` zurück; Hellinger wird explizit als euklidisch „approximiert" (`:140-144`).
- `manifold.py:172-192` Geodäte = Gerade (TODO Christoffel); `:194-212` Exp-Map = `θ+t·v`; `:214-233` Paralleltransport = Identität; `:154-170` Tangentialprojektion = Identität.
- ⇒ Riemann-Struktur kollabiert zu flachem ℝⁿ. Die Metrik fließt **nicht** in die Projektion (`grep fisher/riemannian_distance` über `compression/` + `measurement.py` ⇒ leer). `_project_ternary` (`projection.py:249-259`) ist reines euklidisches Schwellwertverfahren = TWN.

### 2.2 Numerische Umsetzung
- **Fisher falsch:** `manifold.py:106-109` akkumuliert `torch.outer(grad, grad)` mit dem **Batch-Loss-Gradient** ⇒ `(E[g])(E[g])ᵀ` (Rang 1), nicht `E[g·gᵀ]`. Zudem dichte `dim×dim`-Allokation (`:77`) — für reale Netze nicht handhabbar.
- **Kein Straight-Through-Estimator:** Ternarisierung post-hoc auf Parameter (`projection.py:63-84`), kein Gradientendurchgriff ⇒ kein QAT.
- **Latenter Bug:** `projection.py:105` nutzt `np.sqrt` ohne `numpy`-Import ⇒ Low-Rank-Pfad wirft `NameError`.
- **„Optimale Projektion" ist nicht optimal:** Docstring behauptet `Π(θ)=argmin_{θ'∈N} d_M(θ,θ')` (`projection.py:231`), implementiert ist eine feste Schwelle ohne jede Minimierung über `d_M`.

### 2.3 README beschreibt nicht-existente Module
`README.md:153-180` listet u. a. `compression/ternary.py`, `lowrank.py`, `sparse.py`, `geometry/fisher.py`, `laplacian.py`, `geodesic.py`, `theory/hlwt.py`, `tlgt.py`, `fchl.py`, `integration/optimizer.py`. **Keines** dieser Module existiert; `igqk/geometry/` und `igqk/theory/` enthalten nur leere `__init__.py`. Die „Propositionen 6.1–6.3" (HLWT/TLGT/FCHL als Spezialfälle, `README.md:316-325`) haben keinerlei Code-Grundlage.

### 2.4 Empirische Claims irreführend
`IGQK_Complete_Package/VALIDATION_REPORT.md:114-116` / `IGQK_TEST_REPORT.md:141`: beste MNIST-Genauigkeit **10,40 %** (= Zufall bei 10 Klassen), gemeldet als „nach Kompression 10,40 %, Verlust 0,00 % 🎯". „0 % Verlust" ist trivial beim Komprimieren eines nicht-lernenden Modells.

### 2.5 Versteckte Annahmen
Diagonale/flache Geometrie, reeller statt komplexer Hilbertraum, euklidische Ersetzung der Riemann-Distanz, implizite Gauß-Annahmen in Th. 5.2/5.3.

---

## PHASE 3 — Einordnung

- **BitNet b1.58:** trainiert ternär from-scratch mit STE in LLM-Skala. IGQK: kein STE, kein Skalierungsnachweis, Modell auf Zufallsniveau.
- **GPTQ/AWQ:** Fehlerkompensation 2. Ordnung. IGQK: reine Magnitude-Schwelle, keine Kompensation ⇒ strikt schwächer.
- **QuIP#:** weit jenseits dessen, was hier existiert.
- **Natural Gradient (Amari/K-FAC):** real nützlich; hier nur als (ungenutzte, falsch berechnete) Fisher angedeutet.
- **Rate-Distortion:** Th. 5.2 ist eine schwächere (untere-Schranken-)Umformulierung.

**Wirklich neu:** demonstrierbar nichts. Der einzige neue Beitrag wäre die *geometrische Begründung* der Quantisierung — genau der nicht implementierte Teil.

---

## PHASE 4 — Ökonomie (Tokens-pro-Watt)

Ternär adressiert prinzipiell **Bandbreite** (32→~1,58 bit) und **Compute** (Mult→Add) — aber nur mit Low-Bit-Kernel. Hier werden Ternärgewichte als `f32` gespeichert (`projection.py:255`), es gibt keinen gepackten Speicher und keinen add-Matmul ⇒ **null** reale Einsparung; „1/16" ist reine Bit-Theorie (`projection.py:96-99`). Als Implementierung adressiert die Methode **weder** Bandbreite **noch** Compute. Der konzeptionelle Wert von Low-Bit-Inferenz wird bereits von BitNet/GPTQ/AWQ + existierenden Kerneln eingefahren; IGQK fügt nichts hinzu. **Realistischer Hebel aus diesem Repo: keiner.**

---

## Fazit

### 3 stärkste Punkte
1. Saubere, lesbare Modulstruktur; korrekte Standard-Bausteine (SVD-Low-Rank `projection.py:261-284`, Top-k-Sparse `:286-301`).
2. Legitime Forschungsrichtung (Quantisierung als Projektion unter Informationsmetrik) — *falls* implementiert.
3. PyTorch-Integrationsgerüst vorhanden, leicht erweiterbar.

### 3 größte Risiken
1. **Theorie–Code-Lücke nahezu total:** Information­geometrie und „Quanten"-Mechanik stehen im README/PDF, nicht im Code; die Projektion ist euklidisches TWN.
2. **Irreführende Empirie + fiktive README-Module:** MNIST auf Zufallsniveau als „perfekt"; ~12 dokumentierte Module existieren nicht.
3. **Keine Skalierbarkeit / keine Kernel / kein STE:** dichte Fisher, f32-Ternär, post-hoc-Quantisierung ⇒ kein realer Nutzen.

### 3 wichtigste nächste Schritte
1. **Geometrie tatsächlich entscheiden lassen:** Fisher-gewichtete ternäre Zuweisung (`(W-Q)ᵀ G (W-Q)` minimieren) implementieren und **gegen GPTQ/AWQ** an realem Modell messen.
2. **Echtes QAT mit STE in Skala:** BitNet-b1.58-Parität auf kleinem LLM reproduzieren, reale Genauigkeit berichten.
3. **Low-Bit-Inferenz-Kernel** liefern und gemessene Tokens/Watt vs. fp16/INT8 zeigen — sonst bleibt „16×" Fiktion.

---
*Erstellt durch automatisiertes Code-Review. Alle Aussagen sind über Datei + Zeile nachprüfbar.*
