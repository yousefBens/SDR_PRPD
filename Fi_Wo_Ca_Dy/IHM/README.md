# 📡 IHM SDR Détection Décharges Partielles (GE Vernova)

Interface graphique PyQt6 / Matplotlib pour l'analyse spectrale et la caractérisation des **Décharges Partielles (DP / PRPD)** à l'aide d'un SDR Ettus USRP B200 / B210 (plage 100 MHz – 2 GHz).

---

## 📄 Sommaire
1. [Algorithme d'Auto-Gain (AGC)](#-1-algorithme-dauto-gain-agc)
2. [Pourquoi le gain 76 dB donne un meilleur SNR qu'un gain de 40 dB à 2 GHz](#-2-pourquoi-le-gain-de-76-db-donne-un-meilleur-snr)
3. [Calibration Numérique dBFS → dBm](#-3-calibration-numérique-dbfs--dbm)
4. [Structure du Code](#-4-structure-du-code)
5. [Guide de Lancement rapide](#-5-guide-de-lancement-rapide)

---

## ⚙️ 1. Algorithme d'Auto-Gain (AGC)

L'algorithme de contrôle automatique de gain (`core/auto_gain.py` & `core/usrp_backend.py`) ajuste de manière autonome le gain RF analogique du récepteur B200 pour chaque fréquence scannée.

```
                    ┌───────────────────────────────┐
                    │  Départ : Gain initial 40 dB  │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │  Acquisition des échantillons │
                    │     I/Q (durée utiles = 21ms)  │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │    Calcul des métriques :     │
                    │ - max_dbfs                    │
                    │ - clipping_fraction           │
                    │ - robust_dbfs (99.99%)        │
                    │ - median_dbfs (bruit fond)    │
                    └──────────────┬────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌──────────────────────────────┐           ┌──────────────────────────────────┐
│  SATURATION ?                │           │  SIGNAL TROP FAIBLE ?            │
│  - max_dbfs > -3 dBFS  OU    │           │  - (robust_dbfs - median_dbfs)   │
│  - clipping_fraction > 1e-4  │           │    < 12.0 dB                     │
└────────┬─────────────────────┘           └────────┬─────────────────────────┘
         │ OUI                                      │ OUI
         ▼                                          ▼
┌──────────────────────────────┐           ┌──────────────────────────────────┐
│  Diminuer le gain :          │           │  Augmenter le gain :             │
│  40 dB → 20 dB → 0 dB        │           │  40 dB → 60 dB → 76 dB           │
└────────┬─────────────────────┘           └────────┬─────────────────────────┘
         │                                          │
         └───────────────────┬──────────────────────┘
                             │
                             ▼
            ┌──────────────────────────────────┐
            │ NOUVELLE ACQUISITION USRP AVEC   │
            │      LE NOUVEAU GAIN RF          │
            └──────────────────────────────────┘
```

### 📋 Procédure étape par étape :
1. **Initialisation** : Chaque fréquence commence avec le gain médian de **40 dB**.
2. **Mesure** : L'USRP effectue une acquisition d'échantillons I/Q complexes. On calcule :
   - `max_dbfs` : L'amplitude crête absolue du signal en dBFS (0 dBFS = plein échelle ADC).
   - `clipping_fraction` : La proportion d'échantillons saturés ($|IQ| \ge 0.999$).
   - `robust_dbfs` : Le niveau de puissance crête représentatif (99.99ème percentile).
   - `median_dbfs` : Le niveau médian du bruit de fond.
3. **Évaluation de la saturation** : Si `max_dbfs > -3.0 dBFS` ou s'il y a du clipping (`> 1e-4`), l'ADC est en sur-régime. L'algorithme **réduit le gain** d'un cran ($40 \to 20 \to 0$ dB) et **relance immédiatement l'acquisition**.
4. **Évaluation du rapport Signal/Bruit (SNR)** : Si l'écart entre le pic de signal et le bruit est faible (`robust_dbfs - median_dbfs < 12.0 dB`), l'algorithme **augmente le gain** ($40 \to 60 \to 76$ dB) pour amplifier le signal au maximum sans saturer l'ADC et **relance l'acquisition**.
5. **Validation** : Dès qu'aucun changement de gain n'est requis, la valeur courante est validée et retenue.

---

## 📶 2. Pourquoi le gain de 76 dB donne un meilleur SNR

### 💡 Explication physique & matérielle :
- **À haute fréquence (~1.9 GHz - 2.0 GHz)**, l'atténuation dans le milieu de propagation et les câbles est très importante. Le signal de Décharge Partielle capté par l'antenne est extrêmement faible (ex: -75 à -85 dBm).
- **Le Facteur de Bruit (Noise Figure) du B200 / AD9361** s'améliore à haut gain RF :
  - **À Gain 40 dB** : L'amplification analogique est moyenne. Le bruit de quantification propre à l'ADC 12-bits (bruit numérique) domine la mesure. Le plancher de bruit équivalent se situe vers **-82 dBm**, ce qui donne un **SNR de ~10 dB**.
  - **À Gain 76 dB (Max)** : L'amplificateur faible bruit (LNA) analogique est au maximum. Le signal HF est amplifié *avant* d'atteindre l'ADC. Le plancher de bruit thermique équivalent descend vers **-92.5 dBm**, faisant émerger les impulsions de DP avec un **SNR de ~15 dB**.

### 🛠️ Comment régler ce comportement :
1. **Dans le code (`core/auto_gain.py`)** : Nous avons ajusté la consigne `SNR_MIN_DB` de `6.0 dB` à `12.0 dB`. Désormais, lors des scans automatiques, le système sélectionne automatiquement **76 dB** pour toutes les fréquences élevées ou faibles sans bloquer à 40 dB.
2. **Dans le Panneau PRPD (IHM)** : Vous pouvez également sélectionner directement le gain **76 dB** dans la liste déroulante *Gain PRPD (dB)* lors de l'analyse d'une fréquence spécifique.

---

## 📐 3. Calibration Numérique dBFS → dBm

### 🔹 Qu'est-ce que le dBFS ?
Le **dBFS (Decibels relative to Full Scale)** est une unité purement numérique interne à l'ADC du SDR :
$$P_{\text{dBFS}} = 20 \log_{10} (|IQ|_{\text{max}})$$
- $0 \text{ dBFS}$ représente la saturation maximale absolue de l'ADC.
- Le dBFS ne dépend pas de la fréquence ni de la puissance en Watts reçue à la prise antenne.

### 🔹 La Conversion vers le dBm (Puissance réelle en mW)
Pour connaître la vraie puissance reçue sur la prise antenne RX2 en **dBm**, nous utilisons la table d'étalonnage mesurée en laboratoire :
```
rx2_power_calibration_final.pickle
```

La relation fondamentale de calibration est :
$$P_{\text{RX2, dBm}}(f, G) = P_{\text{dBFS}} + C(f, G)$$

Où $C(f, G)$ est le coefficient d'étalonnage spécifique à la fréquence $f$ (en Hz) et au gain analogique $G$ (en dB).

### 🔹 Interpolation Linéaire :
Si la fréquence centrale $f_{\text{mesurée}}$ ne correspond pas exactement aux fréquences étalonnées dans le fichier pickle (ex: $1978.686 \text{ MHz}$), l'IHM effectue une **interpolation linéaire** (`core/calibration.py` -> `interpolate_reference_dbm`) :

$$C(f, G) = C(f_1, G) + \frac{f - f_1}{f_2 - f_1} \cdot \left[ C(f_2, G) - C(f_1, G) \right]$$

Où $f_1$ et $f_2$ sont les deux fréquences d'étalonnage encadrant $f$.

---

## 📁 4. Structure du Code

- `ihm_main.py` : Point d'entrée de l'application PyQt6.
- `core/` :
  - `auto_gain.py` : Logique de décision de gain (seuils saturation / SNR).
  - `calibration.py` : Chargement du pickle et conversion dBFS $\to$ dBm.
  - `usrp_backend.py` : Pilote UHD B200, threads d'acquisition Scan et PRPD.
  - `prpd_processor.py` : Traitement temporel et extraction de phase PRPD.
- `ui/` :
  - `spectrum_panel.py` : Composant graphique du spectre avec zoom/pan/coordonnées.
  - `prpd_panel.py` : Composant graphique du scatter plot PRPD (plein écran).
  - `config_panel.py` : Panneau latéral de contrôle (fréquences, gains, durées).

---

## 🚀 5. Guide de Lancement rapide

Activer l'environnement virtuel local et lancer l'application :
```bash
source .venv/bin/activate
python ihm_main.py
```
