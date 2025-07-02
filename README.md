# Atlante Personalizzato Multiatlante per Analisi dell'Alzheimer

Un sistema avanzato di analisi di neuroimmagini che utilizza un approccio multiatlante per identificare e caratterizzare le regioni cerebrali rilevanti per l'Alzheimer.


## Obiettivo del Progetto

Questo progetto implementa una soluzione multiatlante personalizzata per l'analisi di immagini cerebrali, con particolare focus sull'identificazione e caratterizzazione delle aree implicate nella malattia di Alzheimer. 

Il sistema combina tre diversi atlanti cerebrali complementari per ottenere una panoramica più completa e dettagliata delle strutture cerebrali, permettendo un'analisi approfondita delle regioni potenzialmente interessate da processi neurodegenerativi.

## Metodologia

La soluzione utilizza un approccio multiatlante innovativo che integra:

1. **AAL (Automated Anatomical Labeling)**: atlante di riferimento standard che fornisce una segmentazione completa del cervello in 116 regioni anatomiche
   - Focus su: ippocampo, amigdala, regioni temporali, precuneo, cingolo e altre strutture rilevanti per l'Alzheimer

2. **ASHS (Advanced Hippocampal Subfield Segmentation)**: atlante specializzato che fornisce una segmentazione dettagliata dei sottocampi dell'ippocampo
   - Focus su: CA1, CA2, CA3, subiculum, giro dentato e corteccia entorinale, strutture fondamentali nei primi stadi dell'Alzheimer

3. **Desikan-Killiany**: atlante della corteccia cerebrale con 68 regioni corticali
   - Focus su: corteccia entorinale, parahippocampale, precuneo, cingolato posteriore e aree temporali

Il sistema integra i risultati di questi atlanti, assegnando priorità alle regioni più rilevanti per l'Alzheimer in base alla letteratura scientifica. Per ogni regione cerebrale vengono calcolate statistiche dettagliate e generate visualizzazioni avanzate, facilitando l'identificazione di pattern anomali.

## Dipendenze

Il progetto richiede le seguenti librerie Python:

```
nilearn>=0.10.0
nibabel>=5.0.0
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.6.0
scikit-image>=0.19.0
plotly>=5.10.0
seaborn>=0.12.0
```

## Installazione

1. Clona il repository:
   ```bash
   git clone https://github.com/username/atlas-personalized.git
   cd atlas-personalized
   ```

2. Crea un ambiente virtuale (consigliato):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Per Linux/macOS
   venv\Scripts\activate     # Per Windows
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements-multiatlante.txt
   ```

4. (Opzionale) Prepara una directory per le tue immagini cerebrali:
   ```bash
   mkdir -p images
   # Copia le tue immagini NIfTI (.nii o .nii.gz) nella directory 'images'
   ```

## Utilizzo

### Utilizzo Base

```python
from src.postprocess.custom_multiatlante import main

# Analisi di un'immagine locale
results = main(image_path="path/to/your/brain_image.nii.gz")

# Oppure, utilizzando le immagini di esempio e il percorso di output predefinito
results = main()
```

### Utilizzo Avanzato

```python
from src.postprocess.custom_multiatlante import (
    setup_dataset, 
    apply_multiatlante_to_image,
    integrate_atlas_results,
    create_advanced_visualizations,
    save_results
)

# 1. Scarica/prepara gli atlanti
mri_path = "path/to/your/brain_image.nii.gz"
output_dir = "my_results"
mri_path, atlases_dict, labels_dict = setup_dataset(output_dir="multiatlante_data_cache")

# 2. Applica gli atlanti all'immagine
multiatlante_results = apply_multiatlante_to_image(mri_path, atlases_dict, labels_dict)

# 3. Integra i risultati con priorità alle regioni Alzheimer
integrated_results = integrate_atlas_results(multiatlante_results)

# 4. Crea visualizzazioni
vis_paths = create_advanced_visualizations(
    mri_path, multiatlante_results, integrated_results, output_dir
)

# 5. Salva risultati come CSV
results_csv = save_results(integrated_results, output_dir)

print(f"Analisi completata. Risultati in: {output_dir}")
print(f"Risultati CSV: {results_csv}")
print(f"Visualizzazioni: {vis_paths}")
```

### Esecuzione da Riga di Comando

```bash
# Esegui l'analisi utilizzando l'immagine locale in 'images' (se presente)
python -m src.postprocess.custom_multiatlante

# Specifica un'immagine e una directory di output
python -c "from src.postprocess.custom_multiatlante import main; main('path/to/image.nii.gz', 'output_folder')"
```

### Esecuzione dei Test

```bash
# Esegui i test unitari
python -m unittest src.postprocess.test_multiatlante
```

## Visualizzazioni Generate

Il sistema genera diverse visualizzazioni avanzate per facilitare l'interpretazione dei risultati:

1. **Confronto Multiatlante (multiatlante_comparison.png)**:
   - Visualizza i tre atlanti fianco a fianco nelle viste sagittale, coronale e assiale
   - Permette il confronto diretto delle segmentazioni fornite dai diversi atlanti

2. **Heatmap delle Regioni Alzheimer (ad_regions_heatmap.png)**:
   - Mostra le metriche normalizzate (media, mediana, deviazione standard, volume) delle regioni rilevanti per l'Alzheimer
   - Colori differenti indicano a quale atlante appartiene ciascuna regione (AAL, ASHS, Desikan)
   - Le regioni sono ordinate per rilevanza nell'Alzheimer (più rilevanti in alto)

3. **Visualizzazioni 3D Interattive (HTML)**:
   - `3d_aal_ad_regions.html`: Regioni rilevanti per l'Alzheimer dall'atlante AAL
   - `3d_ashs_ad_regions.html`: Sottocampi dell'ippocampo e regioni correlate dall'atlante ASHS
   - `3d_desikan_ad_regions.html`: Regioni corticali rilevanti dall'atlante Desikan-Killiany
   - `integrated_3d_ad_regions.html`: Visualizzazione integrata di tutte le regioni rilevanti dai tre atlanti

4. **CSV dei Risultati (multiatlante_results.csv)**:
   - Contiene statistiche dettagliate per tutte le regioni cerebrali
   - Include metriche come media, mediana, deviazione standard, volume
   - Indica il punteggio di rilevanza per l'Alzheimer di ciascuna regione

### Interpretazione dei Risultati

- **Punteggi di rilevanza Alzheimer**: Valori più alti indicano regioni più comunemente associate ai processi patologici dell'Alzheimer
- **Volume regionale**: Può indicare atrofia (riduzione di volume) nelle regioni colpite dalla malattia
- **Intensità del segnale**: Variazioni significative nei valori di intensità (media, mediana) possono indicare anomalie strutturali

## Limitazioni e Sviluppi Futuri

- Il sistema attualmente supporta principalmente immagini strutturali T1
- L'integrazione con dati clinici e altri biomarcatori è prevista per sviluppi futuri
- Si prevede l'implementazione di analisi longitudinali per monitorare i cambiamenti cerebrali nel tempo


## Contatti

Per domande o suggerimenti, contattare giovanniiorio@proton.me .
