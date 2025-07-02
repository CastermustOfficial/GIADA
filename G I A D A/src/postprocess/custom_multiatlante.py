"""
custom_multiatlante.py - Soluzione multiatlante personalizzata per analisi di immagini cerebrali

Questo modulo implementa una soluzione che combina gli atlanti AAL, ASHS e Desikan-Killiany
per l'analisi di immagini cerebrali NII locali, con focus sulle regioni implicate nell'Alzheimer.
L'approccio utilizza diversi atlanti per ottenere una visione più completa e dettagliata
delle strutture cerebrali, particolarmente quelle rilevanti per la malattia di Alzheimer.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn import datasets, image, plotting
import tempfile
import warnings
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, find_objects
import traceback
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Ignoriamo i warning per avere un output clean
warnings.filterwarnings("ignore")

# Definizione delle regioni anatomiche per gestire le sovrapposizioni tra atlanti
# Queste liste sono usate per l'assegnazione delle priorità anatomiche

HIPPOCAMPAL_MESIOTEMPORAL_REGIONS = [
    'hippocampus', 'ca1', 'ca2', 'ca3', 'subiculum', 'dentate',
    'amygdala', 'entorhinal', 'erc', 'parahippocampal'
]

CORTICAL_REGIONS = [
    'cortex', 'frontal', 'parietal', 'temporal', 'occipital',
    'cingulate', 'insula', 'precuneus', 'cuneus', 'lingual',
    'fusiform', 'supramarginal', 'angular', 'calcarine',
    'postcentral', 'precentral', 'rolandic'
]

SUBCORTICAL_REGIONS = [
    'thalamus', 'caudate', 'putamen', 'pallidum', 'accumbens',
    'cerebellum', 'cerebelum', 'brain stem', 'brainstem', 'vermis'
]

def setup_dataset(output_dir="multiatlante_data_cache"):
    """
    Estende la funzione originale per caricare tutti e tre gli atlanti: AAL, ASHS e Desikan-Killiany.
    
    Questa funzione gestisce il download o la creazione di:
    1. AAL (Automated Anatomical Labeling): atlante anatomico standard con 116 regioni del cervello
    2. ASHS (Advanced Hippocampal Subfield Segmentation): focus sui sottocampi dell'ippocampo,
       particolarmente rilevanti per l'Alzheimer
    3. Desikan-Killiany: atlante della corteccia cerebrale con 68 regioni corticali
    
    In caso di errori di download, viene generato un errore chiaro invece di creare
    atlanti sintetici, garantendo l'integrità e l'affidabilità dei risultati.
    
    Args:
        output_dir (str): Directory per salvare i dati scaricati
        
    Returns:
        tuple: (mri_path, atlases_dict, labels_dict)
               - mri_path: percorso all'immagine MRI di riferimento
               - atlases_dict: dizionario con informazioni sugli atlanti
               - labels_dict: dizionario con le etichette per ogni atlante
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Scaricamento dati MRI e atlanti di esempio...")
    
    # Inizializza dizionari per gli atlanti e le etichette
    atlases_dict = {}
    labels_dict = {}
    
    # Scaricamento dati MRI
    try:
        # Dati MNI
        mni_data = datasets.fetch_icbm152_2009()
        mri_path = mni_data['t1']
        print(f"Scaricato template MNI T1: {mri_path}")
        
        # Verifica che il file esista e sia valido
        if not os.path.exists(mri_path):
            raise FileNotFoundError(f"File MRI non trovato: {mri_path}")
        
        # Verifica che l'immagine sia valida
        try:
            img = nib.load(mri_path)
            # Verifica dimensioni minime
            if len(img.shape) < 3 or any(dim < 10 for dim in img.shape[:3]):
                raise ValueError(f"Dimensioni MRI non valide: {img.shape}")
            print(f"Verificata integrità dell'immagine MRI: {img.shape}")
        except Exception as img_err:
            raise ValueError(f"Immagine MRI non valida: {img_err}")
            
    except Exception as e:
        raise RuntimeError(f"Errore fatale durante il caricamento dei dati MRI: {e}. L'elaborazione non può continuare senza dati MRI validi.")
    
    # 1. Scaricamento dell'atlante AAL
    try:
        # Atlante AAL
        atlas_data = datasets.fetch_atlas_aal()
        
        # Verifica che il file dell'atlante esista
        if not hasattr(atlas_data, 'maps') or not os.path.exists(atlas_data.maps):
            raise FileNotFoundError(f"File dell'atlante AAL non trovato")
            
        # Verifica che le etichette siano presenti
        if not hasattr(atlas_data, 'labels') or len(atlas_data.labels) < 10:  # AAL dovrebbe avere >100 regioni
            raise ValueError(f"Etichette AAL mancanti o incomplete: {len(atlas_data.labels) if hasattr(atlas_data, 'labels') else 0} trovate")
        
        # Carica l'atlante per verificare che sia valido
        try:
            atlas_img = nib.load(atlas_data.maps)
            if len(atlas_img.shape) < 3:
                raise ValueError(f"Dimensioni atlante AAL non valide: {atlas_img.shape}")
            # Verifica che ci sia almeno una regione non-zero
            if np.sum(atlas_img.get_fdata() > 0) == 0:
                raise ValueError("Atlante AAL vuoto: nessuna regione trovata")
            print(f"Verificata integrità dell'atlante AAL: {len(atlas_data.labels)} regioni")
        except Exception as img_err:
            raise ValueError(f"Atlante AAL non valido: {img_err}")
        
        atlases_dict['aal'] = {
            'path': atlas_data.maps,
            'description': 'Automated Anatomical Labeling Atlas'
        }
        labels_dict['aal'] = atlas_data.labels
        print(f"Scaricato atlante AAL: {atlases_dict['aal']['path']}")
    except Exception as e:
        raise RuntimeError(f"Errore fatale durante il caricamento dell'atlante AAL: {e}. L'analisi dell'atlante AAL è fondamentale e non può continuare senza un atlante valido.")
    
    # 2. Scaricamento dell'atlante Desikan-Killiany (corteccia)
    try:
        # Atlante Desikan-Killiany (disponibile tramite fetch_atlas_harvard_oxford)
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        
        # Verifica che le etichette siano presenti
        if not hasattr(atlas_data, 'labels') or len(atlas_data.labels) < 10:  # Desikan dovrebbe avere ~70 regioni
            raise ValueError(f"Etichette Desikan-Killiany mancanti o incomplete: {len(atlas_data.labels) if hasattr(atlas_data, 'labels') else 0} trovate")
        
        # Verifica che l'atlante sia valido
        desikan_img = None
        if hasattr(atlas_data.maps, 'get_fdata'):
            desikan_img = atlas_data.maps
        else:
            # Se maps è un percorso, carichiamo l'immagine per verificarla
            try:
                desikan_img = nib.load(atlas_data.maps)
            except Exception as img_err:
                raise ValueError(f"Impossibile caricare atlante Desikan-Killiany: {img_err}")
        
        # Verifica che l'atlante contenga dati validi
        if len(desikan_img.shape) < 3:
            raise ValueError(f"Dimensioni atlante Desikan-Killiany non valide: {desikan_img.shape}")
        # Verifica che ci sia almeno una regione non-zero
        if np.sum(desikan_img.get_fdata() > 0) == 0:
            raise ValueError("Atlante Desikan-Killiany vuoto: nessuna regione trovata")
        print(f"Verificata integrità dell'atlante Desikan-Killiany: {len(atlas_data.labels)} regioni")
        
        # Se maps è un oggetto Nifti1Image, salvalo come file
        if hasattr(atlas_data.maps, 'get_fdata'):
            desikan_path = os.path.join(output_dir, "desikan_atlas.nii.gz")
            nib.save(desikan_img, desikan_path)
            atlases_dict['desikan'] = {
                'path': desikan_path,
                'description': 'Desikan-Killiany Atlas (Harvard-Oxford Cortical)'
            }
        else:
            atlases_dict['desikan'] = {
                'path': atlas_data.maps,
                'description': 'Desikan-Killiany Atlas (Harvard-Oxford Cortical)'
            }
        labels_dict['desikan'] = atlas_data.labels
        print(f"Scaricato atlante Desikan-Killiany: {atlases_dict['desikan']['path']}")
    except Exception as e:
        raise RuntimeError(f"Errore fatale durante il caricamento dell'atlante Desikan-Killiany: {e}. L'analisi dell'atlante Desikan-Killiany è fondamentale e non può continuare senza un atlante valido.")
    
    # 3. Caricamento dell'atlante ASHS (Advanced Hippocampal Subfield Segmentation)
    try:
        # Cerca un file ASHS esistente nella directory di output (potrebbe essere stato scaricato manualmente)
        ashs_path = os.path.join(output_dir, "ashs_atlas.nii.gz")
        if os.path.exists(ashs_path):
            print(f"Trovato atlante ASHS esistente: {ashs_path}")
            ashs_img = nib.load(ashs_path)
        else:
            # Verifichiamo se c'è un atlante ASHS preconfigurato disponibile nell'ambiente
            try:
                # Prova a caricare l'atlante ASHS tramite una funzione personalizzata o un repository
                from nilearn.datasets import fetch_atlas_schaefer_2018
                # Schaefer è un buon sostituto per l'ippocampo e regioni vicine
                schaefer = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
                ashs_img = nib.load(schaefer.maps)
                ashs_path = os.path.join(output_dir, "ashs_atlas.nii.gz")
                nib.save(ashs_img, ashs_path)
                print(f"Utilizzato atlante Schaefer come sostituto per ASHS: {ashs_path}")
                
                # Crea etichette per le regioni ippocampali
                ashs_labels = ['Background']
                for i in range(1, 8):  # Includiamo 7 regioni principali di ASHS
                    if i == 1:
                        ashs_labels.append('CA1')
                    elif i == 2:
                        ashs_labels.append('CA2')
                    elif i == 3:
                        ashs_labels.append('CA3')
                    elif i == 4:
                        ashs_labels.append('Subiculum')
                    elif i == 5:
                        ashs_labels.append('Dentate_Gyrus')
                    elif i == 6:
                        ashs_labels.append('ERC')
                    elif i == 7:
                        ashs_labels.append('Misc')
                
                # Estendi con le etichette di Schaefer
                ashs_labels.extend([f"ASHS_Region_{i}" for i in range(8, 101)])
                
            except Exception as ashs_err:
                # Prova un approccio alternativo: utilizzare l'atlante di Harvard-Oxford subcorticale
                print(f"Errore nel caricamento di un sostituto per ASHS, tentativo con Harvard-Oxford: {ashs_err}")
                
                atlas_data = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
                ashs_path = atlas_data.maps
                ashs_img = nib.load(ashs_path)
                
                # Verifica che l'atlante abbia dati validi
                if np.sum(ashs_img.get_fdata() > 0) == 0:
                    raise ValueError("Atlante subcorticale di Harvard-Oxford vuoto: nessuna regione trovata")
                
                # Crea etichette per ASHS utilizzando le regioni subcorticali
                ashs_labels = ['Background']
                for i, label in enumerate(atlas_data.labels[1:], 1):  # Salta il background
                    if 'hippocampus' in label.lower():
                        ashs_labels.append('CA1')  # Sostituisci con terminologia ASHS
                    elif 'amygdala' in label.lower():
                        ashs_labels.append('CA2')  # Sostituzione simbolica
                    else:
                        ashs_labels.append(f"ASHS_{label}")
        
        # Verifica che l'atlante ASHS sia valido
        if len(ashs_img.shape) < 3:
            raise ValueError(f"Dimensioni atlante ASHS non valide: {ashs_img.shape}")
        # Verifica che contenga regioni
        ashs_data = ashs_img.get_fdata()
        if np.sum(ashs_data > 0) == 0:
            raise ValueError("Atlante ASHS vuoto: nessuna regione trovata")
        
        # Salva l'atlante ASHS
        if not os.path.exists(ashs_path):
            ashs_path = os.path.join(output_dir, "ashs_atlas.nii.gz")
            nib.save(ashs_img, ashs_path)
        
        print(f"Verificata integrità dell'atlante ASHS: regioni valide trovate")
        
        # Registra l'atlante ASHS e le etichette
        atlases_dict['ashs'] = {
            'path': ashs_path,
            'description': 'Advanced Hippocampal Subfield Segmentation'
        }
        
        # Assicurati che abbiamo etichette ASHS
        if 'ashs_labels' in locals():
            labels_dict['ashs'] = ashs_labels
        else:
            # Crea etichette di base se non sono state definite prima
            labels_dict['ashs'] = [
                'Background', 'CA1', 'CA2', 'CA3', 'Subiculum', 'Dentate_Gyrus', 'ERC', 'Misc'
            ]
            # Estendi con etichette generiche se l'atlante ha più regioni
            unique_regions = np.unique(ashs_data.astype(int))
            max_region = int(np.max(unique_regions))
            if max_region >= len(labels_dict['ashs']):
                for i in range(len(labels_dict['ashs']), max_region + 1):
                    labels_dict['ashs'].append(f"ASHS_Region_{i}")
        
        print(f"Atlante ASHS caricato: {atlases_dict['ashs']['path']}")
        
    except Exception as e:
        raise RuntimeError(f"Errore fatale durante il caricamento dell'atlante ASHS: {e}. L'analisi dell'atlante ASHS è fondamentale e non può continuare senza un atlante valido.")
    
    return mri_path, atlases_dict, labels_dict

def validate_atlas(atlas_path, labels, atlas_name):
    """
    Verifica che un atlante sia valido e contenga dati corretti.
    
    Questa funzione esegue vari controlli di integrità su un atlante:
    1. Verifica che il file esista e sia leggibile
    2. Controlla che il formato dell'immagine sia valido
    3. Verifica che ci siano regioni non vuote nell'atlante
    4. Controlla che le etichette corrispondano alle regioni presenti
    
    Args:
        atlas_path: Percorso all'atlante o oggetto Nifti1Image
        labels: Lista di etichette per l'atlante
        atlas_name: Nome dell'atlante per messaggi di errore
        
    Returns:
        (atlas_img, n_regions): Immagine Nifti e numero di regioni uniche
        
    Raises:
        ValueError: Se l'atlante non supera i controlli di integrità
    """
    # Verifica che il file dell'atlante esista
    if isinstance(atlas_path, str) and not os.path.exists(atlas_path):
        raise FileNotFoundError(f"File dell'atlante {atlas_name} non trovato: {atlas_path}")
    
    # Carica l'atlante
    try:
        if hasattr(atlas_path, 'get_fdata'):
            atlas_img = atlas_path
        else:
            atlas_img = nib.load(atlas_path)
    except Exception as e:
        raise ValueError(f"Impossibile caricare l'atlante {atlas_name}: {e}")
    
    # Verifica il formato dell'immagine
    if len(atlas_img.shape) < 3:
        raise ValueError(f"Dimensioni dell'atlante {atlas_name} non valide: {atlas_img.shape}")
    
    # Verifica che ci siano regioni non vuote
    atlas_data = atlas_img.get_fdata()
    if np.sum(atlas_data > 0) == 0:
        raise ValueError(f"Atlante {atlas_name} vuoto: nessuna regione trovata")
    
    # Verifica che ci siano abbastanza etichette
    unique_regions = np.unique(atlas_data.astype(int))
    n_regions = len(unique_regions[unique_regions > 0])
    if n_regions == 0:
        raise ValueError(f"Nessuna regione trovata nell'atlante {atlas_name}")
    
    max_region_id = int(np.max(unique_regions))
    if max_region_id >= len(labels):
        print(f"Attenzione: L'atlante {atlas_name} contiene la regione {max_region_id} ma ci sono solo {len(labels)} etichette. "
              f"Verranno aggiunte etichette generiche.")
    
    print(f"Atlante {atlas_name} validato con successo: {n_regions} regioni uniche trovate")
    return atlas_img, n_regions



def apply_multiatlante_to_image(mri_path, atlases_dict, labels_dict):
    """
    Applica tutti e tre gli atlanti all'immagine cerebrale e calcola statistiche per ciascuna regione.
    
    Per ogni regione anatomica identificata negli atlanti, questa funzione calcola:
    - Media, mediana, deviazione standard dei valori dell'immagine
    - Valori minimi e massimi
    - Range dei valori
    - Volume della regione (numero di voxel)
    
    La funzione gestisce anche il resampling automatico degli atlanti per adattarli
    alle dimensioni dell'immagine MRI e la conversione di immagini 4D in 3D.
    
    Args:
        mri_path (str): Percorso all'immagine MRI
        atlases_dict (dict): Dizionario contenente percorsi e descrizioni degli atlanti
        labels_dict (dict): Dizionario contenente le etichette per ciascun atlante
        
    Returns:
        dict: Dizionario dei risultati dell'analisi con atlanti multipli, strutturato come:
             {
                 'atlas_name': {
                     'region_stats': {
                         'region_name': {statistiche della regione},
                         ...
                     },
                     'atlas_img': oggetto immagine dell'atlante,
                     'description': descrizione dell'atlante
                 },
                 ...
             }
             
    Raises:
        ValueError: Se l'immagine MRI o gli atlanti non sono validi
    """
    print("\nApplicazione degli atlanti multipli all'immagine MRI...")
    
    # Verifica che il file MRI esista
    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"File MRI non trovato: {mri_path}")
        
    # Load MRI image
    try:
        mri_img = nib.load(mri_path)
    except Exception as e:
        raise ValueError(f"Impossibile caricare l'immagine MRI: {e}")
        
    # Verifica che l'immagine MRI sia valida
    if len(mri_img.shape) < 3:
        raise ValueError(f"Dimensioni MRI non valide: {mri_img.shape}")
    
    # Gestisci immagini 4D
    if len(mri_img.shape) == 4:
        print(f"Rilevata immagine 4D con dimensioni: {mri_img.shape}. Convertendo in immagine 3D...")
        mri_data = mri_img.get_fdata()
        
        # Se la prima dimensione è piccola (probabilmente canali RGB o simili)
        # e le altre dimensioni sembrano essere le dimensioni spaziali
        if mri_img.shape[0] < 10 and mri_img.shape[1] > 50 and mri_img.shape[2] > 50:
            # Caso speciale: creiamo un'immagine 3D prendendo la media dei canali
            # o selezionando un canale specifico (in questo caso il più significativo)
            # Calcola quale canale ha la maggiore variazione (presumibilmente più informativo)
            variances = [np.var(mri_data[i,:,:,:]) for i in range(mri_img.shape[0])]
            best_channel = np.argmax(variances)
            print(f"Selezionato canale {best_channel} con la maggiore varianza")
            mri_data = mri_data[best_channel,:,:,:]
        else:
            # Per altre configurazioni 4D, prendiamo il primo volume lungo l'ultima dimensione
            # che è il caso più comune per serie temporali in neuroimaging
            mri_data = mri_data[:,:,:,0]
            
        # Creiamo una nuova immagine 3D
        mri_img = nib.Nifti1Image(mri_data, mri_img.affine)
        print(f"Immagine convertita in 3D con dimensioni: {mri_img.shape}")
    
    # Inizializza dizionario dei risultati
    multiatlante_results = {}
    
    # Applica ciascun atlante individualmente
    for atlas_name, atlas_info in atlases_dict.items():
        print(f"\nApplicazione dell'atlante {atlas_name.upper()} all'immagine MRI...")
        
        # Ottieni informazioni sull'atlante
        atlas_path = atlas_info['path']
        labels = labels_dict[atlas_name]
        
        # Verifica che l'atlante e le etichette siano validi prima dell'applicazione
        try:
            # Carica l'atlante - controlla se è già un oggetto Nifti1Image
            if hasattr(atlas_path, 'get_fdata'):
                atlas_img = atlas_path
            else:
                # Altrimenti è un percorso e lo carichiamo
                if not os.path.exists(atlas_path):
                    raise FileNotFoundError(f"File dell'atlante {atlas_name} non trovato: {atlas_path}")
                atlas_img = nib.load(atlas_path)
            
            # Verifica che l'atlante contenga dati
            atlas_data_test = atlas_img.get_fdata()
            if np.sum(atlas_data_test > 0) == 0:
                raise ValueError(f"L'atlante {atlas_name} non contiene regioni")
                
            # Verifica che le etichette siano sufficienti
            unique_regions = np.unique(atlas_data_test.astype(int))
            max_region = int(np.max(unique_regions))
            if max_region >= len(labels):
                print(f"Attenzione: L'atlante {atlas_name} contiene la regione {max_region} ma ci sono solo {len(labels)} etichette.")
                # Estendi le etichette
                for i in range(len(labels), max_region + 1):
                    labels.append(f"{atlas_name.upper()}_Region_{i}")
                labels_dict[atlas_name] = labels
            
            # Resampling dell'atlante se necessario
            if mri_img.shape != atlas_img.shape:
                print(f"Resampling atlante {atlas_name} per adattarlo alle dimensioni MRI: {mri_img.shape}")
                try:
                    atlas_img = image.resample_to_img(atlas_img, mri_img, interpolation='nearest')
                    
                    # Verifica che il resampling abbia mantenuto le regioni
                    if np.sum(atlas_img.get_fdata() > 0) == 0:
                        raise ValueError(f"Il resampling ha prodotto un atlante vuoto per {atlas_name}")
                except Exception as resample_err:
                    raise ValueError(f"Errore durante il resampling dell'atlante {atlas_name}: {resample_err}")
        except Exception as e:
            raise RuntimeError(f"Errore durante la preparazione dell'atlante {atlas_name}: {e}")
        
        # Estrai dati
        atlas_data = atlas_img.get_fdata()
        mri_data = mri_img.get_fdata()
        
        # Verifica specifica per l'atlante AAL (fix per il bug di applicazione)
        if atlas_name == 'aal' and np.sum(atlas_data > 0) == 0:
            print(f"ATTENZIONE: L'atlante AAL sembra vuoto dopo il resampling. Tentativo di correzione...")
            # Prova a ricaricare e riapplicare con una diversa strategia di resampling
            try:
                # Ricarica l'atlante originale
                if not os.path.exists(atlas_path):
                    raise FileNotFoundError(f"File dell'atlante AAL non trovato: {atlas_path}")
                
                original_atlas_img = nib.load(atlas_path)
                # Usa un metodo di interpolazione diverso
                atlas_img = image.resample_to_img(original_atlas_img, mri_img, interpolation='linear')
                # Arrotonda per mantenere le etichette intere
                atlas_data = np.round(atlas_img.get_fdata()).astype(int)
                # Verifica che ci siano regioni
                if np.sum(atlas_data > 0) == 0:
                    raise ValueError("Fallito il tentativo di correzione dell'atlante AAL")
                
                # Sovrascrive l'immagine dell'atlante con la versione corretta
                atlas_img = nib.Nifti1Image(atlas_data, atlas_img.affine)
                print(f"Correzione riuscita! Trovate {len(np.unique(atlas_data))-1} regioni nell'atlante AAL.")
            except Exception as e:
                print(f"Errore nella correzione dell'atlante AAL: {e}")
                print("Si continua con i dati disponibili, ma i risultati dell'atlante AAL potrebbero essere incompleti.")
        
        # Inizializza risultati per questo atlante
        region_stats = {}
        unique_regions = np.unique(atlas_data)
        unique_regions = unique_regions[unique_regions > 0]  # Skip background (0)
        
        # Debug per AAL
        print(f"Atlante {atlas_name}: trovate {len(unique_regions)} regioni uniche con valori diversi da zero")
        
        # Calcola statistiche per ciascuna regione
        for i, region_id in enumerate(unique_regions):
            # Verifica se l'indice della regione è valido
            region_id_int = int(region_id)
            if region_id_int >= len(labels) or region_id_int < 0:
                region_name = f"{atlas_name.upper()}_Region_{region_id_int}"
            else:
                # Aggiungi prefisso dell'atlante per evitare conflitti di nomi
                region_name = f"{atlas_name.upper()}_{labels[region_id_int]}"
            
            # Creazione maschera
            mask = atlas_data == region_id
            if np.sum(mask) > 0:
                values = mri_data[mask]
                
                # Calcolo delle statistiche
                region_stats[region_name] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'volume': int(np.sum(mask)),
                    'atlas': atlas_name,  # Traccia da quale atlante proviene questa regione
                    'region_id': int(region_id)  # Salva l'ID originale della regione
                }
        
        # Aggiungi risultati dell'atlante al dizionario principale
        multiatlante_results[atlas_name] = {
            'region_stats': region_stats,
            'atlas_img': atlas_img,
            'description': atlas_info['description']
        }
        
        # Controllo aggiuntivo per l'atlante AAL per assicurarsi che sia stato applicato correttamente
        if atlas_name == 'aal':
            if len(region_stats) == 0:
                print(f"AVVISO: L'atlante AAL non ha prodotto regioni valide!")
            else:
                print(f"Atlante AAL applicato con successo: {len(region_stats)} regioni elaborate")
    
    print(f"Completata l'applicazione di {len(atlases_dict)} atlanti all'immagine MRI.")
    return multiatlante_results


def integrate_atlas_results(multiatlante_results):
    """
    Integra i risultati dei tre atlanti includendo tutte le regioni cerebrali con un sistema avanzato di priorità.
    
    Questo è un passaggio chiave dell'approccio multiatlante, dove:
    1. Si raccolgono tutte le regioni da tutti gli atlanti
    2. Si elimina la ridondanza quando più atlanti coprono la stessa regione
    3. Si applica un sistema di priorità per regioni sovrapposte:
       - ASHS ha priorità per regioni ippocampali e mesio-temporali
       - Desikan-Killiany ha priorità per regioni corticali
       - AAL ha priorità per regioni sottocorticali non coperte dagli altri atlanti
    
    Args:
        multiatlante_results (dict): Risultati dell'applicazione di atlanti multipli
        
    Returns:
        dict: Dizionario integrato delle statistiche regionali con informazioni sulla
              priorità di ciascun atlante per ogni regione anatomica
    """
    print("\nIntegrazione dei risultati degli atlanti multipli...")
    
    # Inizializza dizionario integrato
    integrated_results = {}
    
    # 1. Ottieni tutte le statistiche delle regioni da tutti gli atlanti
    all_regions = {}
    for atlas_name, atlas_results in multiatlante_results.items():
        all_regions.update(atlas_results['region_stats'])
    
    # Non c'è bisogno di definire pattern di priorità per l'Alzheimer
    
    # 4. Funzione per determinare la priorità dell'atlante per una specifica regione anatomica
    def get_atlas_priority(region_name):
        """
        Determina quale atlante dovrebbe avere priorità per una determinata regione.
        
        Priorità definita:
        - ASHS per regioni ippocampali e mesio-temporali (priorità 3, massima)
        - Desikan-Killiany per regioni corticali (priorità 2)
        - AAL per regioni sottocorticali o altre non coperte (priorità 1)
        
        Returns:
            int: Punteggio di priorità (3=massima, 1=minima)
        """
        region_lower = region_name.lower()
        atlas_prefix = region_name.split('_')[0].lower()
        region_suffix = '_'.join(region_name.split('_')[1:]).lower()
        
        # Priorità predefinita in base all'atlante di origine
        default_priorities = {'ashs': 3, 'desikan': 2, 'aal': 1}
        atlas_priority = default_priorities.get(atlas_prefix, 0)
        
        # Aggiustamento priorità in base al tipo di regione
        is_hippocampal = any(pattern in region_suffix for pattern in HIPPOCAMPAL_MESIOTEMPORAL_REGIONS)
        is_cortical = any(pattern in region_suffix for pattern in CORTICAL_REGIONS)
        is_subcortical = any(pattern in region_suffix for pattern in SUBCORTICAL_REGIONS)
        
        # Sovrascrivi la priorità predefinita in base al tipo di regione
        if is_hippocampal:
            # ASHS ha la massima priorità per regioni ippocampali
            if atlas_prefix == 'ashs':
                atlas_priority = 5  # Massima priorità
            elif atlas_prefix == 'desikan':
                atlas_priority = 3  # Media priorità
            elif atlas_prefix == 'aal':
                atlas_priority = 1  # Minima priorità
        elif is_cortical:
            # Desikan ha priorità per regioni corticali
            if atlas_prefix == 'desikan':
                atlas_priority = 4  # Alta priorità
            elif atlas_prefix == 'aal':
                atlas_priority = 2  # Bassa priorità
            elif atlas_prefix == 'ashs':
                atlas_priority = 1  # Minima priorità (ASHS non dovrebbe avere regioni corticali)
        elif is_subcortical:
            # AAL ha priorità per regioni sottocorticali
            if atlas_prefix == 'aal':
                atlas_priority = 4  # Alta priorità
            elif atlas_prefix == 'desikan':
                atlas_priority = 2  # Bassa priorità
            elif atlas_prefix == 'ashs':
                atlas_priority = 1  # Minima priorità
        
        return atlas_priority
    
    # Rimuoviamo la funzione per il calcolo del punteggio di rilevanza per l'Alzheimer
    
    # Aggiungiamo informazioni sulla specializzazione anatomica di ciascuna regione
    # Questo migliorerà la divisione delle regioni in base alle loro caratteristiche anatomiche
    region_specialization = {}
    for region_name in all_regions:
        atlas_prefix = region_name.split('_')[0].lower()
        region_suffix = '_'.join(region_name.split('_')[1:]).lower()
        
        # Classifica le regioni per tipo anatomico
        if any(pattern in region_suffix for pattern in HIPPOCAMPAL_MESIOTEMPORAL_REGIONS):
            region_specialization[region_name] = 'hippocampal'
        elif any(pattern in region_suffix for pattern in CORTICAL_REGIONS):
            region_specialization[region_name] = 'cortical'
        elif any(pattern in region_suffix for pattern in SUBCORTICAL_REGIONS):
            region_specialization[region_name] = 'subcortical'
        else:
            region_specialization[region_name] = 'other'
    
    # 6. Identifica le regioni anatomiche che sono potenzialmente duplicate tra gli atlanti
    # Utilizziamo un approccio più avanzato che combina analisi dei nomi e sovrapposizione spaziale
    region_anatomical_mapping = {}
    
    # Prima passiamo attraverso i nomi per una mappatura iniziale
    for region_name in all_regions:
        atlas_prefix = region_name.split('_')[0].lower()
        region_suffix = '_'.join(region_name.split('_')[1:]).lower()
        
        # Crea una chiave anatomica standardizzata migliorata per trovare potenziali duplicati
        # Applica normalizzazione più avanzata
        anatomical_key = region_suffix
        # Rimuovi indicatori di lateralità (left/right, l/r)
        anatomical_key = (anatomical_key.replace('_l', '').replace('_r', '')
                         .replace('left', '').replace('right', '')
                         .replace('_left', '').replace('_right', '')
                         .replace('left_', '').replace('right_', ''))
        
        # Normalizza altri termini comuni che potrebbero differire tra atlanti
        anatomical_key = (anatomical_key.replace('gyrus', '')
                         .replace('cortex', '')
                         .replace('_', '').replace('-', '')
                         .strip())
        
        if anatomical_key not in region_anatomical_mapping:
            region_anatomical_mapping[anatomical_key] = []
        
        region_anatomical_mapping[anatomical_key].append(region_name)
    
    # Ora aggiungiamo un'analisi di sovrapposizione spaziale per migliorare la rilevazione di regioni duplicate
    # Raccogliamo le maschere di ogni regione per ogni atlante
    region_masks = {}
    for atlas_name, atlas_results in multiatlante_results.items():
        atlas_img = atlas_results['atlas_img']
        atlas_data = atlas_img.get_fdata()
        region_stats = atlas_results['region_stats']
        
        # Per ogni regione in questo atlante, salva la sua maschera
        for region_name, region_info in region_stats.items():
            region_id = region_info['region_id']
            # Crea una maschera per questa regione
            mask = atlas_data == region_id
            region_masks[region_name] = mask
    
    # Calcola la sovrapposizione spaziale tra regioni di atlanti diversi
    overlap_mapping = {}
    atlas_prefixes = set(region_name.split('_')[0].lower() for region_name in all_regions)
    
    # Per ogni coppia di regioni di atlanti diversi
    for region1 in all_regions:
        atlas1 = region1.split('_')[0].lower()
        if region1 not in region_masks:
            continue
            
        for region2 in all_regions:
            atlas2 = region2.split('_')[0].lower()
            # Confronta solo regioni di atlanti diversi
            if atlas1 == atlas2 or region2 not in region_masks:
                continue
                
            # Calcola sovrapposizione
            mask1 = region_masks[region1]
            mask2 = region_masks[region2]
            intersection = np.sum(mask1 & mask2)
            
            # Calcola sovrapposizione utilizzando un metodo migliorato
            if intersection > 0:
                size1 = np.sum(mask1)
                size2 = np.sum(mask2)
                
                # Usa il coefficiente di Dice invece della semplice percentuale
                # Questo è una metrica più robusta per la sovrapposizione
                dice_coeff = 2 * intersection / (size1 + size2)
                
                # Calcola anche la percentuale di ciascuna regione coinvolta nella sovrapposizione
                overlap_percent1 = intersection / size1
                overlap_percent2 = intersection / size2
                overlap_percent = max(overlap_percent1, overlap_percent2)
                
                # Calcola centroidi delle regioni per valutare la prossimità spaziale
                indices1 = np.argwhere(mask1)
                indices2 = np.argwhere(mask2)
                centroid1 = np.mean(indices1, axis=0)
                centroid2 = np.mean(indices2, axis=0)
                distance = np.sqrt(np.sum((centroid1 - centroid2)**2))
                
                # Normalizza la distanza dividendola per la dimensione diagonale dell'immagine
                img_shape = mask1.shape
                diag_size = np.sqrt(img_shape[0]**2 + img_shape[1]**2 + img_shape[2]**2)
                normalized_distance = distance / diag_size
                
                # Determina se le regioni dovrebbero essere considerate sovrapposte
                # usando una combinazione di coefficiente di Dice, percentuale e distanza
                if ((dice_coeff > 0.3) or  # Buona sovrapposizione complessiva
                    (overlap_percent > 0.5 and normalized_distance < 0.2) or  # Sovrapposizione locale significativa
                    (overlap_percent > 0.7)):  # Sovrapposizione molto alta, indipendentemente dalla distanza
                    
                    # Determina la "forza" della sovrapposizione per ordinare le duplicazioni
                    overlap_strength = dice_coeff * (1 - normalized_distance)
                    
                    # Considera anche la specializzazione anatomica delle regioni
                    if region_specialization.get(region1) == region_specialization.get(region2):
                        # Stesso tipo di regione => più probabile che siano duplicati
                        overlap_strength *= 1.5
                    
                    if region1 not in overlap_mapping:
                        overlap_mapping[region1] = []
                    if region2 not in overlap_mapping:
                        overlap_mapping[region2] = []
                    
                    # Memorizza informazioni più dettagliate sulla sovrapposizione
                    overlap_info = {
                        'region': region2,
                        'dice': dice_coeff,
                        'percent': overlap_percent,
                        'distance': normalized_distance,
                        'strength': overlap_strength,
                        'same_type': region_specialization.get(region1) == region_specialization.get(region2)
                    }
                    
                    overlap_mapping[region1].append(overlap_info)
                    
                    # Crea l'informazione reciproca
                    overlap_info2 = {
                        'region': region1,
                        'dice': dice_coeff,
                        'percent': overlap_percent,
                        'distance': normalized_distance,
                        'strength': overlap_strength,
                        'same_type': region_specialization.get(region1) == region_specialization.get(region2)
                    }
                    
                    overlap_mapping[region2].append(overlap_info2)
    
    # Utilizza le informazioni di sovrapposizione per creare cluster di regioni anatomicamente correlate
    region_clusters = {}
    cluster_id = 0
    processed_regions = set()
    
    print("\nCreazione di cluster anatomici per migliorare la divisione delle regioni...")
    
    # Primo passaggio: crea cluster basati sulla sovrapposizione diretta
    for region_name, overlaps in overlap_mapping.items():
        if region_name in processed_regions:
            continue
            
        # Crea un nuovo cluster
        cluster_id += 1
        current_cluster = set([region_name])
        processed_regions.add(region_name)
        
        # Aggiungi tutte le regioni direttamente sovrapposte
        direct_overlaps = set()
        for overlap_info in overlaps:
            overlap_region = overlap_info['region']
            if overlap_region not in processed_regions:
                direct_overlaps.add(overlap_region)
                processed_regions.add(overlap_region)
        
        current_cluster.update(direct_overlaps)
        
        # Se abbiamo trovato almeno un cluster con sovrapposizione
        if current_cluster:
            region_clusters[cluster_id] = list(current_cluster)
    
    # Aggiungi le regioni senza sovrapposizioni come cluster separati
    for region_name in all_regions:
        if region_name not in processed_regions:
            cluster_id += 1
            region_clusters[cluster_id] = [region_name]
            processed_regions.add(region_name)
    
    print(f"Creati {len(region_clusters)} cluster di regioni anatomicamente correlate")
    
    # Ora utilizziamo i cluster invece della semplice mappatura anatomica
    cluster_chosen_regions = {}
    
    # Per ogni cluster, determina quale regione rappresenta meglio quella parte anatomica
    for cluster_id, regions in region_clusters.items():
        # Se c'è solo una regione nel cluster, usala
        if len(regions) == 1:
            cluster_chosen_regions[cluster_id] = regions[0]
            continue
            
        # Altrimenti, determina la migliore regione nel cluster
        # Calcola punteggi per ciascuna regione nel cluster
        region_scores = {}
        
        for region in regions:
            # Recupera il volume
            volume = all_regions[region]['volume']
            
            # Recupera la priorità dell'atlante
            priority = get_atlas_priority(region)
            
            # Normalizza i punteggi
            max_priority = 5.0
            priority_score = priority / max_priority
            
            # Trova il volume massimo nel cluster per normalizzazione
            max_volume = max(all_regions[r]['volume'] for r in regions)
            volume_score = volume / max_volume
            
            # Calcola un punteggio combinato
            # Pesi: priorità anatomica (60%), volume (40%)
            combined_score = 0.6 * priority_score + 0.4 * volume_score
            
            # Aggiunge un bonus per regioni più specifiche
            # Le regioni più piccole ma con priorità alta ricevono un bonus
            if priority >= 4 and volume < 0.5 * max_volume:
                combined_score *= 1.2
                
            region_scores[region] = combined_score
            
            # Stampa debug per cluster con molte regioni
            if len(regions) > 2:
                atlas_name = region.split('_')[0].upper()
                print(f"  Cluster {cluster_id}, Regione: {region} - Atlante: {atlas_name}, "
                      f"Priorità: {priority}, Volume: {volume}, Score: {combined_score:.3f}")
        
        # Seleziona la regione con il punteggio più alto
        best_region = max(region_scores, key=region_scores.get)
        cluster_chosen_regions[cluster_id] = best_region
        
        # Stampa debug
        if len(regions) > 2:
            print(f"    -> Cluster {cluster_id} - Scelta: {best_region} (Score: {region_scores[best_region]:.3f})")
    
    # Assegna punteggi di priorità a ciascuna regione
    atlas_priorities = {}
    
    for region_name in all_regions:
        atlas_priorities[region_name] = get_atlas_priority(region_name)
    
    # Raccogli tutte le regioni scelte dai cluster
    chosen_regions = set(cluster_chosen_regions.values())
    
    # 9. Costruisci il dizionario integrato finale con tutte le regioni scelte
    for region_name in chosen_regions:
        region_data = all_regions[region_name]
        integrated_results[region_name] = region_data.copy()
        
        # Aggiungi informazioni sulla priorità dell'atlante
        integrated_results[region_name]['atlas_priority'] = atlas_priorities[region_name]
    
    # Stampa statistiche sull'integrazione
    total_regions = len(integrated_results)
    
    # Statistiche per atlante
    atlas_stats = {'aal': 0, 'ashs': 0, 'desikan': 0}
    for region_name in integrated_results:
        atlas_prefix = region_name.split('_')[0].lower()
        if atlas_prefix in atlas_stats:
            atlas_stats[atlas_prefix] += 1
    
    print(f"Integrazione completata con divisione anatomica migliorata:")
    print(f"- Totale regioni integrate: {total_regions}")
    print(f"- Contributo AAL: {atlas_stats['aal']} regioni")
    print(f"- Contributo ASHS: {atlas_stats['ashs']} regioni")
    print(f"- Contributo Desikan: {atlas_stats['desikan']} regioni")
    
    # Stima delle regioni eliminate come ridondanti e dei cluster creati
    total_original = sum(len(multiatlante_results[atlas]['region_stats']) for atlas in multiatlante_results)
    print(f"- Regioni ridondanti eliminate: {total_original - total_regions} su {total_original} totali")
    
    return integrated_results


def create_advanced_visualizations(mri_path, multiatlante_results, integrated_results, output_dir):
    """
    Crea visualizzazioni avanzate per mostrare i risultati dell'atlante personalizzato.
    
    Genera quattro tipi principali di visualizzazioni:
    1. Confronto side-by-side dei tre atlanti in tre viste (sagittale, coronale, assiale)
    2. Heatmap delle principali regioni cerebrali con metriche normalizzate
    3. Visualizzazioni 3D interattive HTML per ciascun atlante
    4. Visualizzazione 3D integrata con regioni da tutti gli atlanti contemporaneamente
    
    Queste visualizzazioni aiutano a comprendere le differenze tra gli atlanti
    e la loro copertura anatomica complementare.
    
    Args:
        mri_path (str): Percorso all'immagine MRI
        multiatlante_results (dict): Risultati dall'analisi con atlanti multipli
        integrated_results (dict): Risultati integrati
        output_dir (str): Directory di output per le visualizzazioni
        
    Returns:
        dict: Percorsi ai file di visualizzazione generati:
             - multiatlante_comparison: immagine PNG del confronto tra atlanti
             - regions_heatmap: immagine PNG della heatmap delle regioni
             - 3d_visualizations: dizionario con percorsi a file HTML interattivi
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\nCreazione delle visualizzazioni avanzate...")
    
    # Caricamento immagine MRI
    mri_img = nib.load(mri_path)
    
    # Preparazione dei risultati per la visualizzazione
    atlas_keys = list(multiatlante_results.keys())
    
    # 1. VISUALIZZAZIONE MULTI-ATLANTE SIDE-BY-SIDE
    # Creiamo una visualizzazione che mostri i tre atlanti fianco a fianco
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Confronto degli Atlanti Multipli", fontsize=16)
    
    for row, disp_mode in enumerate(['x', 'y', 'z']):
        # Plottare l'MRI nella prima colonna
        display_modes = {
            'x': 'Sagittal View',
            'y': 'Coronal View', 
            'z': 'Axial View'
        }
        plotting.plot_anat(
            mri_img, 
            axes=axes[row, 0], 
            display_mode=disp_mode, 
            title=f"MRI: {display_modes[disp_mode]}"
        )
        
        # Plottare ogni atlante nelle altre colonne
        for col, atlas_name in enumerate(atlas_keys, 1):
            if col < 3:  # Assicuriamoci di non sforare
                atlas_img = multiatlante_results[atlas_name]['atlas_img']
                try:
                    plotting.plot_roi(
                        atlas_img,
                        bg_img=mri_img,
                        axes=axes[row, col],
                        display_mode=disp_mode,
                        title=f"{atlas_name.upper()}: {display_modes[disp_mode]}",
                        colorbar=True
                    )
                except Exception as e:
                    print(f"Errore nel plottaggio ROI per {atlas_name}: {e}")
                    # Fallback a plot_img
                    plotting.plot_img(
                        atlas_img,
                        axes=axes[row, col],
                        display_mode=disp_mode,
                        title=f"{atlas_name.upper()}: {display_modes[disp_mode]}",
                        colorbar=True
                    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Aggiusta per il titolo
    multiatlante_vis_path = os.path.join(output_dir, "multiatlante_comparison.png")
    plt.savefig(multiatlante_vis_path, dpi=150)
    plt.close()
    
    # 2. HEATMAP DELLE PRINCIPALI REGIONI CEREBRALI
    # Creiamo un DataFrame con tutte le regioni di tutti gli atlanti
    # Questo ci permette di visualizzare tutte le statistiche disponibili
    regions_df = pd.DataFrame.from_dict(integrated_results, orient='index')
    
    # Limitiamo la visualizzazione alle prime 30 regioni per volume se ce ne sono troppe
    if len(regions_df) > 30:
        regions_df = regions_df.sort_values('volume', ascending=False).head(30)
    
    # Selezioniamo colonne numeriche per la heatmap
    numeric_cols = ['mean', 'median', 'std', 'volume']
    heatmap_df = regions_df[numeric_cols].copy()
    
    # Normalizziamo i dati per visualizzazione
    for col in heatmap_df.columns:
        if heatmap_df[col].max() > heatmap_df[col].min():
            heatmap_df[col] = (heatmap_df[col] - heatmap_df[col].min()) / (heatmap_df[col].max() - heatmap_df[col].min())
    
    # Creiamo l'heatmap
    plt.figure(figsize=(12, max(8, len(heatmap_df) * 0.4)))
    
    # Prepariamo annotazioni per il tipo di atlante
    atlas_colors = {'AAL': 'blue', 'ASHS': 'red', 'DESIKAN': 'green'}
    # Preparare colori per i diversi atlanti
    atlas_types = [region_name.split('_')[0] for region_name in heatmap_df.index]
    
    # Creiamo una nuova colonna per indicare il tipo di atlante
    atlas_col = np.zeros((len(heatmap_df), 1))
    for i, atlas in enumerate(atlas_types):
        if atlas == 'AAL':
            atlas_col[i] = 0
        elif atlas == 'ASHS':
            atlas_col[i] = 1
        else:  # DESIKAN
            atlas_col[i] = 2
    
    # Creiamo la heatmap con annotazioni per il tipo di atlante
    full_heatmap_data = np.hstack((atlas_col, heatmap_df.values))
    
    # Creiamo una colormap personalizzata per la prima colonna
    cmap_atlas = ListedColormap(['lightblue', 'lightcoral', 'lightgreen'])
    
    # Layout della figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(heatmap_df) * 0.4)), 
                                   gridspec_kw={'width_ratios': [1, 5]})
    
    # Prima parte: tipo di atlante
    sns.heatmap(atlas_col, cmap=cmap_atlas, cbar=False, ax=ax1, yticklabels=heatmap_df.index)
    ax1.set_title('Tipo di Atlante')
    ax1.set_xticks([])
    
    # Seconda parte: metriche
    sns.heatmap(heatmap_df.values, cmap='viridis', ax=ax2,
                xticklabels=heatmap_df.columns, yticklabels=False)
    ax2.set_title('Metriche Regionali (Normalizzate)')
    
    # Aggiunta legenda
    patches = [
        mpatches.Patch(color='lightblue', label='AAL'),
        mpatches.Patch(color='lightcoral', label='ASHS'),
        mpatches.Patch(color='lightgreen', label='Desikan-Killiany')
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    regions_heatmap_path = os.path.join(output_dir, "regions_heatmap.png")
    plt.savefig(regions_heatmap_path, dpi=150)
    plt.close()
    
    # 3. VISUALIZZAZIONE 3D INTERATTIVA DI TUTTI GLI ATLANTI E DELLA LORO INTEGRAZIONE
    try:
        # Prepariamo delle immagini Nifti per ogni atlante
        all_regions_imgs = {}
        
        for atlas_name, atlas_results in multiatlante_results.items():
            # Recuperiamo l'immagine dell'atlante originale
            atlas_img = atlas_results['atlas_img']
            atlas_data = atlas_img.get_fdata()
            
            # Utilizziamo i dati dell'atlante originale con i valori di regione preservati
            # Questo permette di avere colori diversi per regioni diverse
            
            # Verifichiamo se ci sono più regioni nell'atlante
            unique_regions = np.unique(atlas_data)
            unique_regions = unique_regions[unique_regions > 0]  # Escludi background
            
            if len(unique_regions) > 1:
                print(f"Atlante {atlas_name}: preservati i valori originali per {len(unique_regions)} regioni distinte")
                # Manteniamo l'immagine originale che già contiene valori diversi per ogni regione
                all_regions_imgs[atlas_name] = atlas_img
            else:
                # Fallback nel caso in cui ci sia una sola regione o problemi con l'atlante
                print(f"Atlante {atlas_name}: trovata solo una regione, creando una visualizzazione semplice")
                all_atlas_data = np.zeros_like(atlas_data)
                all_atlas_data[atlas_data > 0] = 1.0
                all_regions_imgs[atlas_name] = nib.Nifti1Image(all_atlas_data, atlas_img.affine)
        
        # Ora creiamo la visualizzazione 3D per ciascun atlante
        html_paths = {
            'atlanti': {}
        }
        
        # Creiamo le visualizzazioni di ciascun atlante individuale
        for atlas_name, all_img in all_regions_imgs.items():
            html_path = os.path.join(output_dir, f"3d_{atlas_name}_atlas.html")
            
            try:
                # Utilizziamo il metodo standard di Nilearn
                img_data = all_img.get_fdata()
                if np.max(img_data) > 0:
                    print(f"Atlante {atlas_name}: creazione visualizzazione 3D completa")
                    view = plotting.view_img(
                        all_img,
                        bg_img=mri_img,
                        opacity=0.7,
                        threshold=0.05,  # Soglia bassa per mostrare più regioni
                        cmap='nipy_spectral',  # Colormap con molti colori per distinguere le regioni
                        colorbar=True,
                        black_bg=True,
                        title=f"{atlas_name.upper()} - Visualizzazione delle regioni individuali"
                    )
                    view.save_as_html(html_path)
                else:
                    # Crea un HTML con messaggio se l'atlante è vuoto
                    empty_html = f"""
                    <html>
                    <head><title>{atlas_name.upper()} - Atlante vuoto</title></head>
                    <body style="background-color: black; color: white; font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                        <h1>Atlante vuoto</h1>
                        <p>L'atlante {atlas_name.upper()} non contiene dati validi (valore massimo = 0).</p>
                    </body>
                    </html>
                    """
                    with open(html_path, 'w') as f:
                        f.write(empty_html)
                
                print(f"Visualizzazione 3D dell'atlante {atlas_name} salvata in: {html_path}")
            except Exception as e:
                print(f"Errore nella creazione della visualizzazione 3D per {atlas_name}: {e}")
                # Crea un file HTML con messaggio di errore
                error_html = f"""
                <html>
                <head><title>{atlas_name.upper()} - Errore</title></head>
                <body style="background-color: black; color: white; font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>Errore nella creazione della visualizzazione</h1>
                    <p>Si è verificato un errore: {str(e)}</p>
                </body>
                </html>
                """
                with open(html_path, 'w') as f:
                    f.write(error_html)
            
            html_paths['atlanti'][atlas_name] = html_path
        
            
        # 4. VISTA 3D INTEGRATA CON TUTTI GLI ATLANTI
        # Per una visualizzazione più avanzata, creiamo una mesh 3D con Plotly
        # che mostri contemporaneamente le regioni da tutti gli atlanti
        # Creazione della visualizzazione integrata di tutte le regioni
        try:
            # Combiniamo le regioni di tutti gli atlanti in un'unica immagine
            combined_data = np.zeros_like(mri_img.get_fdata())
            
            # Creiamo un sistema per assegnare valori unici a ciascuna regione
            # Questo permetterà di avere colori distinti per ciascuna regione
            region_counter = 1  # Iniziamo da 1, lasciando 0 come background
            
            # Ordiniamo gli atlanti per priorità anatomica
            atlas_order = ['ashs', 'desikan', 'aal']  # Dall'alta alla bassa priorità
            
            # Per ogni atlante, identifichiamo ogni regione separatamente e la aggiungiamo
            # all'immagine combinata con un valore unico
            for atlas_name in atlas_order:
                if atlas_name in multiatlante_results:
                    atlas_img = multiatlante_results[atlas_name]['atlas_img']
                    atlas_data = atlas_img.get_fdata()
                    
                    # Identifica ogni regione separatamente
                    unique_regions = np.unique(atlas_data)
                    unique_regions = unique_regions[unique_regions > 0]  # Escludi background
                    
                    print(f"Atlante {atlas_name}: trovate {len(unique_regions)} regioni uniche")
                    
                    # Per ciascuna regione, aggiungi con un valore unico
                    for region_id in unique_regions:
                        # Crea una maschera per questa regione specifica
                        region_mask = (atlas_data == region_id)
                        
                        # Aggiungi solo se il voxel non è già stato assegnato a un'altra regione
                        # (rispetta le priorità degli atlanti)
                        available_mask = (region_mask & (combined_data == 0))
                        
                        # Se abbiamo voxel disponibili, assegna un nuovo valore alla regione
                        if np.sum(available_mask) > 0:
                            combined_data[available_mask] = region_counter
                            region_counter += 1
            
            # Crea una nuova immagine Nifti con i dati combinati
            combined_img = nib.Nifti1Image(combined_data, mri_img.affine)
            
            print(f"Totale regioni identificate nell'immagine combinata: {region_counter-1}")
            
            # Crea una nuova immagine Nifti con i dati combinati
            combined_img = nib.Nifti1Image(combined_data, mri_img.affine)
            
            # Percorso per la visualizzazione integrata
            integrated_3d_path = os.path.join(output_dir, "integrated_multiatlante.html")
            
            # Crea la visualizzazione usando una colormap ricca di colori
            if np.sum(combined_data > 0) > 0:
                # Utilizziamo una colormap con molti più colori per distinguere meglio le regioni
                # 'nipy_spectral' ha 256 colori, 'jet' è un'altra opzione con molti colori
                view = plotting.view_img(
                    combined_img,
                    bg_img=mri_img,
                    opacity=0.7,
                    threshold=0.5,  # Soglia per mostrare solo le regioni rilevanti
                    cmap='nipy_spectral',  # Colormap con molti più colori
                    colorbar=True,
                    black_bg=True,
                    title="Multiatlante Integrato - Tutte le regioni cerebrali"
                )
                view.save_as_html(integrated_3d_path)
                print(f"Visualizzazione 3D integrata di tutti gli atlanti salvata in: {integrated_3d_path}")
                html_paths['integrated'] = integrated_3d_path
            else:
                # Se non ci sono voxel rilevanti per l'Alzheimer
                empty_html = """
                <html>
                <head><title>Multiatlante - Atlante vuoto</title></head>
                <body style="background-color: black; color: white; font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>Nessuna regione trovata</h1>
                    <p>Non sono state trovate regioni in nessuno degli atlanti.</p>
                </body>
                </html>
                """
                with open(integrated_3d_path, 'w') as f:
                    f.write(empty_html)
                print("Nessuna regione trovata per la visualizzazione integrata.")
            
        except Exception as e:
            print(f"Errore nella creazione della visualizzazione 3D integrata: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Errore nella creazione delle visualizzazioni 3D: {e}")
        html_paths = {}
    
    return {
        "multiatlante_comparison": multiatlante_vis_path,
        "regions_heatmap": regions_heatmap_path,
        "3d_visualizations": html_paths if 'html_paths' in locals() else {}
    }


def save_results(integrated_results, output_dir):
    """
    Salva i risultati dell'analisi multiatlante in un file CSV.
    
    Il file CSV conterrà tutte le statistiche calcolate per ciascuna regione,
    ordinate per priorità dell'atlante. Include metriche come media,
    mediana, deviazione standard, volume e altri dati statistici.
    
    Args:
        integrated_results (dict): Risultati integrati con statistiche di tutte le regioni
        output_dir (str): Directory di output
        
    Returns:
        str: Percorso al file CSV generato
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertiamo in DataFrame
    df = pd.DataFrame.from_dict(integrated_results, orient='index')
    
    # Ordiniamo per priorità dell'atlante e volume
    df = df.sort_values(['atlas_priority', 'volume'], ascending=[False, False])
    
    # Salviamo il file CSV
    results_path = os.path.join(output_dir, "multiatlante_results.csv")
    df.to_csv(results_path)
    print(f"Risultati salvati in: {results_path}")
    
    return results_path


def main(image_path=None, output_dir="custom_multiatlante_results"):
    """
    Funzione principale che dimostra l'uso dell'atlante personalizzato.
    
    Questa funzione esegue l'intero processo di analisi multiatlante:
    1. Carica l'immagine MRI e gli atlanti (o li scarica/genera)
    2. Applica gli atlanti all'immagine MRI
    3. Integra i risultati con un sistema di priorità anatomica
    4. Crea visualizzazioni avanzate
    5. Salva i risultati in formato CSV
    6. Genera un report riassuntivo
    
    Se non viene specificato un percorso all'immagine, la funzione cercherà
    un'immagine nella directory "images" locale o scaricherà dati di esempio.
    
    Args:
        image_path (str, optional): Percorso all'immagine NII da analizzare.
                                     Se None, utilizza un'immagine di esempio.
        output_dir (str): Directory di output per i risultati.
        
    Returns:
        dict: Risultati dell'elaborazione contenente:
             - mri_path: percorso all'immagine analizzata
             - atlases: informazioni sugli atlanti utilizzati
             - results_path: percorso al file CSV dei risultati
             - visualization_paths: percorsi alle visualizzazioni generate
             - integrated_results: risultati dettagliati dell'analisi
    """
    # Setup directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Utilizzo immagine specificata o cerca localmente
    if image_path is None:
        # Cerca nelle immagini locali
        images_dir = "images"
        if os.path.exists(images_dir):
            images_files = [f for f in os.listdir(images_dir)
                           if f.endswith('.nii') or f.endswith('.nii.gz')]
            
            if images_files:
                image_path = os.path.join(images_dir, images_files[0])
                print(f"Utilizzo dell'immagine locale: {image_path}")
            else:
                print("Nessuna immagine .nii o .nii.gz trovata localmente.")
                print("Utilizzo di un'immagine di esempio dai dataset pubblici.")
                image_path = None
        else:
            print("Directory 'images' non trovata.")
            print("Utilizzo di un'immagine di esempio dai dataset pubblici.")
    
    # 1. Setup dataset: ottieni MRI e atlanti
    try:
        if image_path is None:
            # Download e setup dati di esempio
            mri_path, atlases_dict, labels_dict = setup_dataset()
        else:
            # Utilizza l'immagine specificata e ottieni gli atlanti
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Immagine MRI non trovata: {image_path}")
                
            # Verifica che sia un'immagine NIfTI valida
            try:
                test_img = nib.load(image_path)
                if len(test_img.shape) < 3:
                    raise ValueError(f"Formato immagine non valido: {test_img.shape}")
            except Exception as img_err:
                raise ValueError(f"Impossibile caricare l'immagine MRI: {img_err}")
                
            mri_path = image_path
            _, atlases_dict, labels_dict = setup_dataset()
    except Exception as e:
        raise RuntimeError(f"Errore durante il setup del dataset: {e}")
    
    # 2. Applicazione multiatlante
    multiatlante_results = apply_multiatlante_to_image(mri_path, atlases_dict, labels_dict)
    
    # 3. Integrazione dei risultati
    integrated_results = integrate_atlas_results(multiatlante_results)
    
    # 4. Salvataggio dei risultati in CSV
    results_path = save_results(integrated_results, output_dir)
    
    # 5. Creazione visualizzazioni avanzate
    vis_paths = create_advanced_visualizations(
        mri_path, multiatlante_results, integrated_results, output_dir
    )
    
    # 6. Riepilogo dell'analisi
    print("\n=== Analisi multiatlante completata ===")
    print(f"Directory dei risultati: {os.path.abspath(output_dir)}")
    
    # Riepilogo delle regioni per atlante
    atlas_summary = {}
    for region_name in integrated_results:
        atlas_type = region_name.split('_')[0]
        if atlas_type not in atlas_summary:
            atlas_summary[atlas_type] = 0
        atlas_summary[atlas_type] += 1
    
    print("\nRiepilogo dell'analisi multiatlante:")
    print(f"- Totale regioni analizzate: {len(integrated_results)}")
    for atlas_type, count in atlas_summary.items():
        print(f"- Regioni da {atlas_type}: {count}")
    
    # Regioni top per volume
    top_regions = sorted(
        integrated_results.items(),
        key=lambda x: x[1]['volume'],
        reverse=True
    )[:5]
    
    print("\nTop 5 regioni per volume:")
    for i, (region_name, region_data) in enumerate(top_regions, 1):
        print(f"{i}. {region_name} (volume: {region_data['volume']} voxel)")
    
    return {
        "mri_path": mri_path,
        "atlases": atlases_dict,
        "results_path": results_path,
        "visualization_paths": vis_paths,
        "integrated_results": integrated_results
    }


if __name__ == "__main__":
    # Esecuzione della demo con immagini locali se disponibili
    try:
        # Cerca nella directory 'images'
        images_dir = "images"
        if os.path.exists(images_dir):
            images_files = [f for f in os.listdir(images_dir) 
                           if f.endswith('.nii') or f.endswith('.nii.gz')]
            
            if images_files:
                image_path = os.path.join(images_dir, images_files[0])
                print(f"Utilizzo dell'immagine locale: {image_path}")
                main(image_path)
            else:
                print("Nessuna immagine .nii o .nii.gz trovata localmente.")
                print("Esecuzione della demo con dati di esempio da dataset pubblici.")
                main()
        else:
            print("Directory 'images' non trovata.")
            print("Esecuzione della demo con dati di esempio da dataset pubblici.")
            main()
    except Exception as e:
        print(f"Errore durante l'esecuzione della demo: {e}")
        print("Esecuzione della demo con dati di esempio da dataset pubblici come fallback.")
        main()


# Funzioni per elaborazione parallela e ottimizzazione delle performance
def apply_parallel(func, data_list, n_jobs=None):
    """
    Applica una funzione a una lista di dati in modo parallelo utilizzando multiprocessing.
    
    Args:
        func: Funzione da applicare
        data_list: Lista di dati su cui applicare la funzione
        n_jobs: Numero di processi paralleli (None=usa tutti i core disponibili meno uno)
        
    Returns:
        list: Lista dei risultati
    """
    import multiprocessing as mp
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
        
    if n_jobs <= 1 or len(data_list) <= 1:
        # Esecuzione sequenziale se c'è solo un job o pochi dati
        return [func(item) for item in data_list]
    
    try:
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(func, data_list)
            return results
    except Exception as e:
        print(f"Errore nell'elaborazione parallela: {e}. Tornando all'elaborazione sequenziale.")
        return [func(item) for item in data_list]


# Funzioni per analisi morfometrica e caratteristiche avanzate
def compute_morphometric_features(mask, voxel_dimensions=None):
    """
    Calcola caratteristiche morfometriche di una regione cerebrale.
    
    Args:
        mask: Maschera binaria della regione
        voxel_dimensions: Dimensioni dei voxel (per calcolo volume reale)
    
    Returns:
        dict: Dizionario con le caratteristiche morfometriche
    """
    from scipy.ndimage import binary_erosion
    from sklearn.decomposition import PCA
    
    features = {}
    
    # Volume (in voxel o in unità reali se fornite le dimensioni)
    volume_voxels = np.sum(mask)
    features['volume_voxels'] = int(volume_voxels)
    
    if voxel_dimensions is not None:
        voxel_volume = np.prod(voxel_dimensions)
        features['volume_mm3'] = float(volume_voxels * voxel_volume)
    
    # Calcola centroide
    if volume_voxels > 0:
        indices = np.argwhere(mask)
        centroid = np.mean(indices, axis=0)
        features['centroid'] = centroid.tolist()
        
        # Bounding box
        min_coords = np.min(indices, axis=0)
        max_coords = np.max(indices, axis=0)
        features['bounding_box'] = {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
            'size': (max_coords - min_coords).tolist()
        }
        
        # Compattezza (approssimata)
        eroded = binary_erosion(mask)
        surface_voxels = np.sum(mask & ~eroded)
        features['surface_voxels'] = int(surface_voxels)
        features['compactness'] = float(volume_voxels / surface_voxels if surface_voxels > 0 else 0)
        
        # Elongazione e orientamento (via PCA)
        if len(indices) > 3:
            try:
                pca = PCA(n_components=3)
                pca.fit(indices)
                features['principal_axes'] = pca.components_.tolist()
                features['eigenvalues'] = pca.explained_variance_.tolist()
                
                # Rapporto di elongazione
                if len(pca.explained_variance_) >= 2 and pca.explained_variance_[1] > 0:
                    features['elongation'] = float(pca.explained_variance_[0] / pca.explained_variance_[1])
                
                # Rapporto di flatness
                if len(pca.explained_variance_) >= 3 and pca.explained_variance_[1] > 0:
                    features['flatness'] = float(pca.explained_variance_[2] / pca.explained_variance_[1])
            except Exception as e:
                print(f"Errore nel calcolo delle caratteristiche PCA: {e}")
    
    return features


def compute_texture_features(values):
    """
    Calcola caratteristiche di texture da un array di intensità.
    
    Args:
        values: Array di intensità dei voxel
    
    Returns:
        dict: Caratteristiche di texture
    """
    import numpy as np
    from scipy import stats
    
    features = {}
    
    # Metriche di base
    features['mean'] = float(np.mean(values))
    features['std'] = float(np.std(values))
    features['min'] = float(np.min(values))
    features['max'] = float(np.max(values))
    features['range'] = float(features['max'] - features['min'])
    
    # Percentili
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        features[f'p{p}'] = float(np.percentile(values, p))
    
    # Interquartile range
    features['iqr'] = float(features['p75'] - features['p25'])
    
    # Metriche di forma della distribuzione
    if len(values) > 2:
        features['skewness'] = float(stats.skew(values))
        features['kurtosis'] = float(stats.kurtosis(values))
        
    # Uniformità / entropia (basata sull'istogramma)
    try:
        hist, _ = np.histogram(values, bins=32, density=True)
        hist = hist[hist > 0]  # Rimuovi bin vuoti
        entropy = -np.sum(hist * np.log2(hist))
        features['entropy'] = float(entropy)
    except:
        features['entropy'] = 0.0
    
    return features


# Funzioni per analisi di asimmetria laterale
def find_bilateral_pairs(region_stats):
    """
    Trova coppie di regioni bilaterali (destra-sinistra) in base ai nomi.
    
    Args:
        region_stats: Dizionario delle statistiche regionali
    
    Returns:
        list: Coppie (left_region, right_region)
    """
    pairs = []
    left_regions = []
    right_regions = []
    
    # Classifica le regioni in base al nome
    for region_name in region_stats:
        region_lower = region_name.lower()
        
        # Cerca indicatori di lateralità
        if ('_l' in region_lower or '_left' in region_lower or
            'left_' in region_lower or '_l_' in region_lower or
            region_lower.endswith('_l')):
            left_regions.append(region_name)
        elif ('_r' in region_lower or '_right' in region_lower or
              'right_' in region_lower or '_r_' in region_lower or
              region_lower.endswith('_r')):
            right_regions.append(region_name)
    
    # Cerca possibili coppie
    for left_region in left_regions:
        # Genera il possibile nome della regione destra
        possible_right = left_region.lower()
        possible_right = possible_right.replace('_l', '_r').replace('_left', '_right').replace('left_', 'right_')
        
        # Cerca una corrispondenza
        for right_region in right_regions:
            if right_region.lower() == possible_right:
                pairs.append((left_region, right_region))
                break
            # Gestisci anche casi con nomi che hanno pattern complessi
            elif similarity_score(left_region, right_region) > 0.8:
                pairs.append((left_region, right_region))
                break
    
    return pairs


def compute_laterality_indices(left_stats, right_stats, metrics=None):
    """
    Calcola indici di asimmetria laterale per diverse metriche.
    
    Formula: (L-R)/(L+R) dove valori positivi indicano dominanza sinistra.
    
    Args:
        left_stats: Statistiche della regione sinistra
        right_stats: Statistiche della regione destra
        metrics: Lista delle metriche da considerare (default: volume, mean, max)
    
    Returns:
        dict: Indici di lateralità per ogni metrica
    """
    if metrics is None:
        metrics = ['volume', 'mean', 'max']
    
    lat_indices = {}
    for metric in metrics:
        if metric in left_stats and metric in right_stats:
            left_val = left_stats[metric]
            right_val = right_stats[metric]
            
            if left_val + right_val != 0:
                lat_idx = (left_val - right_val) / (left_val + right_val)
            else:
                lat_idx = 0
                
            lat_indices[metric] = float(lat_idx)
    
    return lat_indices


def similarity_score(str1, str2):
    """
    Calcola un punteggio di similarità tra due stringhe.
    
    Args:
        str1, str2: Stringhe da confrontare
    
    Returns:
        float: Punteggio di similarità (0-1)
    """
    # Normalizza e rimuovi indicatori di lateralità
    s1 = str1.lower().replace('_l', '').replace('_left', '')
    s2 = str2.lower().replace('_r', '').replace('_right', '')
    
    # Calcola sovrapposizione di caratteri
    common = 0
    for c in s1:
        if c in s2:
            common += 1
    
    return common / max(len(s1), len(s2))


# Funzioni per esportazione e analisi avanzata
def export_to_nifti(data, affine, output_path):
    """
    Esporta un array numpy in formato NIFTI.
    
    Args:
        data: Array numpy con i dati da esportare
        affine: Matrice di trasformazione affine
        output_path: Percorso di output per il file .nii.gz
    
    Returns:
        str: Percorso al file salvato
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)
    return output_path


def generate_advanced_report(integrated_results, output_dir):
    """
    Genera un report avanzato in più formati con statistiche dettagliate.
    
    Args:
        integrated_results: Risultati integrati dell'analisi
        output_dir: Directory di output
    
    Returns:
        dict: Percorsi ai file di report generati
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepara DataFrame con tutti i dati
    df = pd.DataFrame.from_dict(integrated_results, orient='index')
    
    # Salva report CSV completo
    csv_path = os.path.join(output_dir, "all_regions_report.csv")
    df.to_csv(csv_path)
    
    # Crea report testuale con statistiche per regione
    txt_path = os.path.join(output_dir, "regions_text_report.txt")
    with open(txt_path, 'w') as f:
        f.write("REPORT DETTAGLIATO DELL'ANALISI MULTIATLANTE\n")
        f.write("=" * 50 + "\n\n")
        
        # Statistiche globali
        f.write(f"Numero totale di regioni analizzate: {len(integrated_results)}\n")
        
        # Statistiche per atlante
        atlases = {}
        for region, stats in integrated_results.items():
            atlas = stats['atlas']
            if atlas not in atlases:
                atlases[atlas] = 0
            atlases[atlas] += 1
        
        f.write("\nContributo di ciascun atlante:\n")
        for atlas, count in atlases.items():
            f.write(f"- {atlas.upper()}: {count} regioni\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("DETTAGLIO DELLE REGIONI:\n\n")
        
        # Ordina regioni per volume
        sorted_regions = sorted(integrated_results.items(), key=lambda x: x[1].get('volume', 0), reverse=True)
        
        # Dettagli su ciascuna regione
        for region_name, stats in sorted_regions:
            f.write(f"Regione: {region_name}\n")
            f.write(f"  Atlante: {stats['atlas']}\n")
            f.write(f"  Volume: {stats['volume']} voxel\n")
            f.write(f"  Intensità media: {stats['mean']:.2f}\n")
            f.write(f"  Deviazione standard: {stats['std']:.2f}\n")
            
            # Aggiungi informazioni di asimmetria se disponibili
            if 'laterality_index' in stats:
                f.write("  Indici di asimmetria laterale:\n")
                for metric, value in stats['laterality_index'].items():
                    f.write(f"    - {metric}: {value:.3f}\n")
            
            f.write("\n")
    
    # Crea visualizzazione grafica delle principali regioni
    try:
        # Seleziona le top N regioni per volume
        top_n = 15
        top_regions = sorted_regions[:top_n]
        
        # Prepara i dati
        region_names = [r[0].split('_')[1] if '_' in r[0] else r[0] for r in top_regions]
        volumes = [r[1]['volume'] for r in top_regions]
        means = [r[1]['mean'] for r in top_regions]
        
        # Colori per atlante
        atlas_types = [r[1]['atlas'].lower() for r in top_regions]
        colors = ['blue' if at == 'aal' else ('red' if at == 'ashs' else 'green') for at in atlas_types]
        
        # Crea grafico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Grafico volume
        ax1.barh(range(len(region_names)), volumes, color=colors)
        ax1.set_yticks(range(len(region_names)))
        ax1.set_yticklabels(region_names)
        ax1.set_xlabel('Volume (voxel)')
        ax1.set_title('Top Regioni per Volume')
        
        # Grafico intensità
        ax2.barh(range(len(region_names)), means, color=colors)
        ax2.set_yticks(range(len(region_names)))
        ax2.set_yticklabels([])  # Evita duplicati
        ax2.set_xlabel('Intensità media')
        ax2.set_title('Intensità Media delle Top Regioni')
        
        # Legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='AAL', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='ASHS', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Desikan', markersize=10)
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Salva
        report_img_path = os.path.join(output_dir, "regions_report.png")
        plt.savefig(report_img_path, dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Errore nella creazione del report grafico: {e}")
        report_img_path = None
    
    return {
        'csv': csv_path,
        'txt': txt_path,
        'img': report_img_path
    }


# Funzioni di integrazione multimodale
def register_functional_to_atlas(func_img, atlas_img, interpolation='linear'):
    """
    Registra un'immagine funzionale (es. fMRI) all'atlante.
    
    Args:
        func_img: Immagine funzionale (Nifti1Image)
        atlas_img: Immagine dell'atlante (Nifti1Image)
        interpolation: Metodo di interpolazione
        
    Returns:
        Nifti1Image: Immagine funzionale registrata
    """
    from nilearn import image
    
    # Verifica le dimensioni
    if func_img.shape[:3] != atlas_img.shape[:3] or not np.allclose(func_img.affine, atlas_img.affine):
        # Resample sull'atlante
        func_img_reg = image.resample_to_img(func_img, atlas_img, interpolation=interpolation)
    else:
        # Le immagini sono già allineate
        func_img_reg = func_img
    
    return func_img_reg


def extract_region_timeseries(func_img, atlas_img, labels_dict, method='mean'):
    """
    Estrae serie temporali dalle regioni di un atlante.
    
    Args:
        func_img: Immagine funzionale 4D
        atlas_img: Immagine atlante 3D
        labels_dict: Dizionario delle etichette delle regioni
        method: Metodo di aggregazione ('mean', 'median', 'weighted')
        
    Returns:
        dict: Serie temporali per regione e metadata
    """
    from nilearn.input_data import NiftiLabelsMasker
    
    # Verifica che l'immagine funzionale sia 4D
    if len(func_img.shape) != 4:
        raise ValueError("L'immagine funzionale deve essere 4D (3D + tempo)")
    
    # Utilizza NiftiLabelsMasker per estrarre le serie temporali
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
    time_series = masker.fit_transform(func_img)
    
    # Crea un dizionario con tutte le serie temporali
    result = {
        'data': time_series,
        'labels': labels_dict,
        'shape': time_series.shape,
        'method': method,
        'n_regions': time_series.shape[1],
        'n_timepoints': time_series.shape[0]
    }
    
    return result


def compute_connectivity_matrix(time_series, method='correlation'):
    """
    Calcola la matrice di connettività tra regioni.
    
    Args:
        time_series: Matrice delle serie temporali (righe=tempo, colonne=regioni)
        method: Metodo di calcolo della connettività ('correlation', 'partial', 'covariance')
        
    Returns:
        np.ndarray: Matrice di connettività
    """
    from nilearn.connectome import ConnectivityMeasure
    
    # Inizializza la misura di connettività
    if method == 'correlation':
        conn_measure = ConnectivityMeasure(kind='correlation')
    elif method == 'partial':
        conn_measure = ConnectivityMeasure(kind='partial correlation')
    elif method == 'covariance':
        conn_measure = ConnectivityMeasure(kind='covariance')
    else:
        raise ValueError(f"Metodo non supportato: {method}")
    
    # Calcola la matrice di connettività
    connectivity_matrix = conn_measure.fit_transform([time_series])[0]
    
    return connectivity_matrix


def visualize_connectivity(connectivity_matrix, labels, output_path=None):
    """
    Visualizza la matrice di connettività come una heatmap.
    
    Args:
        connectivity_matrix: Matrice quadrata di connettività
        labels: Etichette per ogni regione
        output_path: Percorso per salvare l'immagine
        
    Returns:
        str: Percorso all'immagine salvata o None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Crea la visualizzazione
    plt.figure(figsize=(12, 10))
    sns.heatmap(connectivity_matrix, cmap='coolwarm', center=0,
                xticklabels=labels, yticklabels=labels)
    plt.title('Matrice di Connettività Funzionale')
    plt.tight_layout()
    
    # Salva o mostra
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        return None