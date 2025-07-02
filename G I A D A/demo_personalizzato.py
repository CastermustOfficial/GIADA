#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_personalizzato.py - Script dimostrativo per la soluzione multiatlante personalizzata

Questo script dimostra il funzionamento dell'implementazione aggiornata della soluzione
multiatlante personalizzata per l'analisi di immagini cerebrali, con un focus sulla
visualizzazione completa di tutte le regioni cerebrali.

Lo script:
1. Carica un'immagine NII reale locale (da 'images/atlas_optimized_view.nii.gz')
2. Applica la soluzione multiatlante che integra AAL, Desikan-Killiany e ASHS
3. Genera visualizzazioni 2D e 3D interattive dei risultati
4. Verifica che tutti i componenti funzionino correttamente (non utilizza dati sintetici)
5. Stampa un report dettagliato delle principali regioni cerebrali identificate

Requisiti:
- Python 3.8+
- Dipendenze specificate in requirements-multiatlante.txt
- Immagine NII locale in 'images/atlas_optimized_view.nii.gz'
"""

import os
import sys
import time
import argparse
import traceback
import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
from termcolor import colored
import warnings

# Ignoriamo avvisi per un output pulito
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Aggiungiamo la directory principale al path per l'importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importazione del modulo multiatlante personalizzato
try:
    from src.postprocess.custom_multiatlante import (
        setup_dataset,
        apply_multiatlante_to_image,
        integrate_atlas_results,
        create_advanced_visualizations,
        save_results,
        HIPPOCAMPAL_MESIOTEMPORAL_REGIONS,
        CORTICAL_REGIONS,
        SUBCORTICAL_REGIONS
    )
except ImportError as e:
    print(f"Errore di importazione: {e}")
    print("Assicurati di eseguire lo script dalla directory principale del progetto.")
    print("Per esempio: python demo_personalizzato.py")
    sys.exit(1)


def check_environment():
    """
    Verifica l'ambiente di esecuzione, controllando la presenza
    delle dipendenze necessarie e della struttura del progetto.
    
    Returns:
        bool: True se l'ambiente è configurato correttamente, False altrimenti.
    """
    print(colored("\n=== Verifica dell'ambiente di esecuzione ===", "cyan"))
    
    # Verifica che l'immagine NII sia presente
    image_path = "images/atlas_optimized_view.nii.gz"
    if not os.path.exists(image_path):
        print(colored(f"ERRORE: Immagine NII non trovata in '{image_path}'", "red"))
        print("Assicurati che l'immagine NII sia presente nella directory specificata.")
        return False
    
    # Controllo della validità dell'immagine NII
    try:
        img = nib.load(image_path)
        shape = img.shape
        print(colored(f"✓ Immagine NII trovata e validata: {image_path}", "green"))
        print(f"  Dimensioni: {shape}")
    except Exception as e:
        print(colored(f"ERRORE: L'immagine NII in '{image_path}' non è valida", "red"))
        print(f"  Dettaglio errore: {e}")
        return False
    
    # Verifica della struttura del progetto
    required_dirs = ["src/postprocess", "images"]
    for d in required_dirs:
        if not os.path.isdir(d):
            print(colored(f"ERRORE: Directory '{d}' non trovata", "red"))
            print("La struttura del progetto non è corretta.")
            return False
    
    # Verifica che il file custom_multiatlante.py sia presente
    if not os.path.exists("src/postprocess/custom_multiatlante.py"):
        print(colored("ERRORE: File 'src/postprocess/custom_multiatlante.py' non trovato", "red"))
        return False
    
    print(colored("✓ Ambiente verificato correttamente", "green"))
    return True


def print_progress(message, step=None, total=None):
    """
    Stampa un messaggio di progresso formattato, con eventuale indicazione
    del passo corrente su totale.
    
    Args:
        message (str): Messaggio da stampare
        step (int, optional): Passo corrente
        total (int, optional): Totale passi
    """
    if step is not None and total is not None:
        progress = f"[{step}/{total}] "
    else:
        progress = ""
    
    print(colored(f"\n>>> {progress}{message}", "blue"))


def print_section(title):
    """
    Stampa un titolo di sezione formattato.
    
    Args:
        title (str): Titolo della sezione
    """
    print("\n" + "=" * 80)
    print(colored(f"  {title}", "yellow"))
    print("=" * 80)


def create_regions_report(integrated_results, output_dir):
    """
    Crea un report dettagliato sulle principali regioni cerebrali identificate.
    
    Questo report include informazioni su tutte le regioni cerebrali identificate,
    con focus particolare su quelle con maggior volume.
    
    Args:
        integrated_results (dict): Risultati integrati dell'analisi multiatlante
        output_dir (str): Directory di output per il report
        
    Returns:
        str: Percorso del file di report generato
    """
    # Creiamo un dataframe con tutte le regioni
    df_all = pd.DataFrame.from_dict(integrated_results, orient='index')
    
    # Estrazione dell'atlante di origine per ogni regione
    df_all['atlas'] = df_all.index.str.split('_').str[0]
    
    # Aggiungiamo info sulla regione (parte del nome dopo il prefisso dell'atlante)
    df_all['region_name'] = df_all.index.map(lambda x: '_'.join(x.split('_')[1:]))
    
    # Classificazione delle regioni per tipo anatomico
    def classify_region_type(region_name):
        region_lower = region_name.lower()
        if any(pattern in region_lower for pattern in HIPPOCAMPAL_MESIOTEMPORAL_REGIONS):
            return 'Ippocampale/Mesiotemporale'
        elif any(pattern in region_lower for pattern in CORTICAL_REGIONS):
            return 'Corticale'
        elif any(pattern in region_lower for pattern in SUBCORTICAL_REGIONS):
            return 'Sottocorticale'
        else:
            return 'Altra'
            
    df_all['region_type'] = df_all['region_name'].apply(classify_region_type)
    
    # Ordiniamo per volume decrescente e poi per atlante
    df_all = df_all.sort_values(['volume', 'atlas_priority'],
                               ascending=[False, False])
    
    # Selezione e riordinamento delle colonne rilevanti per il report
    report_cols = ['atlas', 'region_name', 'region_type', 'atlas_priority', 'mean', 'median', 'std', 'volume']
    report_cols_basic = [col for col in report_cols if col in df_all.columns]
    
    report_df_all = df_all[report_cols_basic].copy()
    
    # Creazione di una figura con due subplot: tabella e grafico a barre
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, max(10, 6 + 4)),
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Creazione di un titolo principale
    fig.suptitle('Analisi Multiatlante - Principali regioni cerebrali', fontsize=14)
    
    # Creiamo una tabella con le regioni più voluminose e le loro statistiche
    top_regions = report_df_all.head(20)  # Limitiamo a 20 regioni per leggibilità
    cell_text = []
    for i, (idx, row) in enumerate(top_regions.iterrows(), 1):
        # Aggiungiamo il campo priorità dell'atlante se disponibile
        priority_val = row.get('atlas_priority', 'N/A')
        priority_str = f"{priority_val:.1f}" if isinstance(priority_val, (int, float)) else priority_val
        
        cell_text.append([
            i,
            row['atlas'],
            row['region_name'],
            row['region_type'],
            priority_str,
            f"{row['mean']:.1f}",
            f"{row['median']:.1f}",
            f"{row['std']:.1f}",
            f"{row['volume']:.0f}"
        ])
    
    # Imposta colori per evidenziare le righe in base all'atlante
    cell_colors = []
    for i, row in enumerate(cell_text):
        atlas = row[1]
        if atlas == 'AAL':
            color = '#f8e6e6'  # Rosso chiaro
        elif atlas == 'ASHS':
            color = '#e6f8e6'  # Verde chiaro
        else:  # DESIKAN
            color = '#e6e6f8'  # Blu chiaro
        cell_colors.append([color] * len(row))
    
    table = ax1.table(
        cellText=cell_text,
        colLabels=["#", "Atlante", "Regione", "Tipo", "Priorità", "Media", "Mediana", "DevStd", "Volume"],
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )
    
    # Rimuoviamo gli assi per la tabella
    ax1.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax1.set_title('Top 20 regioni cerebrali per volume', pad=20)
    
    # Grafico a barre per il conteggio di tutte le regioni per atlante
    all_atlas_counts = df_all['atlas'].value_counts()
    colors_all = {'AAL': '#ffcccc', 'ASHS': '#ccffcc', 'DESIKAN': '#ccccff'}
    ax2.bar(all_atlas_counts.index, all_atlas_counts.values, color=[colors_all[x] for x in all_atlas_counts.index])
    ax2.set_title('Distribuzione delle regioni per atlante')
    ax2.set_ylabel('Numero di regioni')
    # Aggiungi etichette di valore sopra ogni barra
    for i, v in enumerate(all_atlas_counts.values):
        ax2.text(i, v + 0.5, str(v), ha='center')
    
    # Aggiustiamo il layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Lascia spazio per il titolo principale
    
    # Salviamo la figura
    report_path = os.path.join(output_dir, "regions_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Salviamo il CSV con tutte le regioni
    all_regions_csv_path = os.path.join(output_dir, "all_regions_report.csv")
    report_df_all.to_csv(all_regions_csv_path)
    
    print(f"Salvato report CSV: tutte le regioni ({len(report_df_all)} righe)")
    
    return report_path


def generate_text_report(integrated_results, output_dir):
    """
    Genera un report testuale dettagliato su tutte le regioni cerebrali.
    
    Questo report include un riepilogo generale di tutte le regioni con focus
    sulle diverse tipologie anatomiche.
    
    Args:
        integrated_results (dict): Risultati integrati dell'analisi multiatlante
        output_dir (str): Directory di output per il report
        
    Returns:
        str: Percorso del file di report generato
    """
    # Ordiniamo tutte le regioni per tipo di atlante e priorità
    all_sorted_regions = sorted(
        integrated_results.items(),
        key=lambda x: (x[1]['atlas'], -x[1].get('atlas_priority', 0))
    )
    
    # Classificazione delle regioni per tipo anatomico
    def classify_region_type(region_name):
        region_lower = region_name.lower()
        if any(pattern in region_lower for pattern in HIPPOCAMPAL_MESIOTEMPORAL_REGIONS):
            return 'Ippocampale/Mesiotemporale'
        elif any(pattern in region_lower for pattern in CORTICAL_REGIONS):
            return 'Corticale'
        elif any(pattern in region_lower for pattern in SUBCORTICAL_REGIONS):
            return 'Sottocorticale'
        else:
            return 'Altra'
    
    # Classificazione delle regioni
    region_types = {}
    for region_name, region_data in integrated_results.items():
        region_lower = region_name.lower()
        region_type = classify_region_type(region_name)
        region_types[region_name] = region_type
    
    # Prepariamo il report
    report_path = os.path.join(output_dir, "regions_text_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" REPORT COMPLETO DELL'ANALISI MULTIATLANTE\n")
        f.write("=" * 80 + "\n\n")
        
        # Statistiche generali
        f.write(f"Totale regioni analizzate: {len(integrated_results)}\n\n")
        
        # Suddivisione per atlante di tutte le regioni
        total_atlas_counts = {}
        for region_name in integrated_results:
            atlas_type = region_name.split('_')[0]
            total_atlas_counts[atlas_type] = total_atlas_counts.get(atlas_type, 0) + 1
        
        f.write("Suddivisione di tutte le regioni per atlante:\n")
        for atlas_type, count in total_atlas_counts.items():
            f.write(f"- {atlas_type}: {count} regioni\n")
        
        # Suddivisione per tipo anatomico
        type_counts = {}
        for region_type in region_types.values():
            type_counts[region_type] = type_counts.get(region_type, 0) + 1
        
        f.write("\nSuddivisione delle regioni per tipo anatomico:\n")
        for region_type, count in type_counts.items():
            f.write(f"- {region_type}: {count} regioni\n")
            
        # Sistema di priorità applicato
        f.write("\n\n" + "-" * 80 + "\n")
        f.write(" SISTEMA DI PRIORITÀ MULTIATLANTE APPLICATO\n")
        f.write("-" * 80 + "\n\n")
        f.write("Il sistema multiatlante personalizzato utilizza la seguente gerarchia di priorità:\n")
        f.write("1. ASHS: massima priorità per regioni ippocampali e mesio-temporali\n")
        f.write("2. Desikan-Killiany: priorità per regioni corticali\n")
        f.write("3. AAL: priorità per regioni sottocorticali non coperte dagli altri atlanti\n\n")
        f.write("Questo approccio garantisce che, quando una regione anatomica è coperta da più atlanti,\n")
        f.write("vengano utilizzati i dati dall'atlante più specializzato per quella regione specifica.\n")
        
        # Lista delle tipologie anatomiche
        f.write("\nRiferimento delle categorie anatomiche usate per la prioritizzazione:\n")
        
        f.write("\nRegioni Ippocampali/Mesiotemporali:\n")
        for region in sorted(HIPPOCAMPAL_MESIOTEMPORAL_REGIONS):
            f.write(f"- {region}\n")
            
        f.write("\nRegioni Corticali:\n")
        for region in sorted(CORTICAL_REGIONS):
            f.write(f"- {region}\n")
            
        f.write("\nRegioni Sottocorticali:\n")
        for region in sorted(SUBCORTICAL_REGIONS):
            f.write(f"- {region}\n")
        
        # Dettagli per le regioni più voluminose
        f.write("\n\n" + "=" * 80 + "\n")
        f.write(" DETTAGLIO DELLE REGIONI CEREBRALI PRINCIPALI\n")
        f.write("=" * 80 + "\n\n")
        
        # Ordiniamo per volume decrescente per evidenziare le regioni più grandi
        top_regions = sorted(integrated_results.items(), key=lambda x: x[1]['volume'], reverse=True)[:20]
        
        for i, (region_name, region_data) in enumerate(top_regions, 1):
            # Estrai il nome della regione dopo il prefisso dell'atlante
            atlas_type, *region_parts = region_name.split('_')
            region_only = '_'.join(region_parts)
            
            f.write(f"#{i} - {region_only} ({atlas_type})\n")
            f.write(f"   Tipo: {region_types.get(region_name, 'N/A')}\n")
            f.write(f"   Priorità atlante: {region_data.get('atlas_priority', 'N/A')}\n")
            f.write(f"   Media: {region_data['mean']:.2f}\n")
            f.write(f"   Mediana: {region_data['median']:.2f}\n")
            f.write(f"   Deviazione standard: {region_data['std']:.2f}\n")
            f.write(f"   Volume: {region_data['volume']} voxel\n")
            f.write("   ---\n")
        
        # Elenco di tutte le regioni per referenza
        f.write("\n\n" + "=" * 80 + "\n")
        f.write(" ELENCO COMPLETO DI TUTTE LE REGIONI CEREBRALI\n")
        f.write("=" * 80 + "\n\n")
        
        # Raggruppa per atlante
        by_atlas = {}
        for region_name, region_data in all_sorted_regions:
            atlas_type = region_name.split('_')[0]
            if atlas_type not in by_atlas:
                by_atlas[atlas_type] = []
            by_atlas[atlas_type].append((region_name, region_data))
        
        # Scrivi regioni per ogni atlante
        for atlas_type, regions in by_atlas.items():
            f.write(f"\n--- {atlas_type} ({len(regions)} regioni) ---\n\n")
            for region_name, region_data in regions:
                # Estrai il nome della regione senza il prefisso dell'atlante
                region_only = '_'.join(region_name.split('_')[1:])
                # Aggiunge il tipo anatomico
                region_type = region_types.get(region_name, '')
                f.write(f"- {region_only} [{region_type}]\n")
    
    return report_path


def run_demo():
    """
    Esegue la demo del multiatlante personalizzato.
    
    Questa funzione esegue tutti i passaggi della demo:
    1. Verifica dell'ambiente
    2. Caricamento dell'immagine NII
    3. Setup degli atlanti
    4. Applicazione degli atlanti all'immagine
    5. Integrazione dei risultati
    6. Creazione delle visualizzazioni
    7. Generazione del report sulle principali regioni cerebrali
    
    Returns:
        dict: Risultati dell'elaborazione o None in caso di errore
    """
    start_time = time.time()
    
    # Directory di output
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Verifica dell'ambiente
    if not check_environment():
        print(colored("\nImpossibile procedere a causa di errori nell'ambiente.", "red"))
        return None
    
    try:
        # STEP 1: Caricamento dell'immagine locale
        print_section("1. CARICAMENTO IMMAGINE")
        print_progress("Caricamento dell'immagine MRI locale", 1, 7)
        
        image_path = "images/atlas_optimized_view.nii.gz"
        
        # Verifica dell'esistenza dell'immagine
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Immagine non trovata in {image_path}")
        
        # Verifica della validità dell'immagine
        try:
            img = nib.load(image_path)
            print(f"Immagine caricata con successo. Dimensioni: {img.shape}")
            if len(img.shape) < 3:
                raise ValueError(f"L'immagine ha dimensioni non valide: {img.shape}")
        except Exception as e:
            raise ValueError(f"Errore nel caricamento dell'immagine: {e}")
        
        # STEP 2: Setup degli atlanti
        print_section("2. SETUP ATLANTI")
        print_progress("Download e preparazione degli atlanti (AAL, ASHS, Desikan)", 2, 7)
        
        # Datadir per gli atlanti scaricati
        data_dir = "multiatlante_data_cache"
        
        try:
            # Utilizziamo l'immagine specificata ma scariniamo gli atlanti
            _, atlases_dict, labels_dict = setup_dataset(output_dir=data_dir)
            
            print(f"Atlanti caricati: {', '.join(atlases_dict.keys())}")
            
            # Mostra numero di regioni per atlante
            for atlas_name, labels in labels_dict.items():
                print(f"  Atlante {atlas_name}: {len(labels)-1} regioni")  # -1 per escludere il background
        except Exception as e:
            raise RuntimeError(f"Errore durante il setup degli atlanti: {e}")
        
        # STEP 3: Applicazione multiatlante
        print_section("3. APPLICAZIONE DEGLI ATLANTI")
        print_progress("Applicazione degli atlanti all'immagine MRI", 3, 7)
        
        try:
            multiatlante_results = apply_multiatlante_to_image(image_path, atlases_dict, labels_dict)
            
            # Statistiche sulle regioni identificate
            total_regions = sum(len(result['region_stats']) for result in multiatlante_results.values())
            print(f"Applicazione completata. Totale regioni identificate: {total_regions}")
            
            # Dettaglio per ogni atlante
            for atlas_name, result in multiatlante_results.items():
                region_count = len(result['region_stats'])
                print(f"  Atlante {atlas_name}: {region_count} regioni")
        except Exception as e:
            raise RuntimeError(f"Errore durante l'applicazione degli atlanti: {e}")
        
        # STEP 4: Integrazione dei risultati
        print_section("4. INTEGRAZIONE DEI RISULTATI")
        print_progress("Integrazione dei risultati con focus sulle regioni Alzheimer", 4, 7)
        
        try:
            integrated_results = integrate_atlas_results(multiatlante_results)
            
            # Statistiche sulle regioni
            print(f"Integrazione completata. Totale regioni: {len(integrated_results)}")
            
            # Statistiche per atlante
            atlas_counts = {}
            for region_name in integrated_results:
                atlas_type = region_name.split('_')[0]
                atlas_counts[atlas_type] = atlas_counts.get(atlas_type, 0) + 1
            
            for atlas_type, count in atlas_counts.items():
                print(f"  {atlas_type}: {count} regioni")
                
            # Mostra le top 5 regioni per volume
            top_regions = sorted(
                integrated_results.items(),
                key=lambda x: x[1]['volume'],
                reverse=True
            )[:5]
            
            print("\nTop 5 regioni per volume:")
            for i, (region_name, region_data) in enumerate(top_regions, 1):
                print(f"  {i}. {region_name} (volume: {region_data['volume']} voxel)")
        except Exception as e:
            raise RuntimeError(f"Errore durante l'integrazione dei risultati: {e}")
        
        # STEP 5: Creazione visualizzazioni
        print_section("5. CREAZIONE VISUALIZZAZIONI")
        print_progress("Creazione di visualizzazioni avanzate 2D e 3D", 5, 7)
        
        try:
            vis_paths = create_advanced_visualizations(
                image_path, multiatlante_results, integrated_results, output_dir
            )
            
            print("Visualizzazioni create:")
            for key, path in vis_paths.items():
                if key != "3d_visualizations":
                    print(f"  - {key}: {path}")
            
            if "3d_visualizations" in vis_paths:
                print("  - Visualizzazioni 3D:")
                for atlas_name, path in vis_paths["3d_visualizations"].items():
                    print(f"    * {atlas_name}: {path}")
        except Exception as e:
            raise RuntimeError(f"Errore durante la creazione delle visualizzazioni: {e}")
        
        # STEP 6: Salvataggio risultati
        print_section("6. SALVATAGGIO RISULTATI")
        print_progress("Salvataggio dei risultati in formato CSV", 6, 7)
        
        try:
            results_path = save_results(integrated_results, output_dir)
            print(f"Risultati salvati in: {results_path}")
        except Exception as e:
            raise RuntimeError(f"Errore durante il salvataggio dei risultati: {e}")
        
        # STEP 7: Creazione report regioni cerebrali
        print_section("7. REPORT REGIONI CEREBRALI")
        print_progress("Creazione del report sulle principali regioni cerebrali", 7, 7)
        
        try:
            # Crea un report visuale
            report_path = create_regions_report(integrated_results, output_dir)
            print(f"Report visuale creato: {report_path}")
            
            # Crea anche un report testuale dettagliato
            text_report_path = generate_text_report(integrated_results, output_dir)
            print(f"Report testuale creato: {text_report_path}")
        except Exception as e:
            raise RuntimeError(f"Errore durante la creazione del report: {e}")
        
        # Stampa il tempo di esecuzione
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTempo di esecuzione totale: {execution_time:.2f} secondi")
        
        # Riepilogo finale
        print_section("RIEPILOGO DELL'ANALISI MULTIATLANTE")
        print(f"Immagine analizzata: {image_path}")
        print(f"Atlanti utilizzati: {', '.join(atlases_dict.keys())}")
        print(f"Totale regioni identificate: {total_regions}")
        print(f"Tutti i risultati salvati in: {os.path.abspath(output_dir)}")
        
        # Proposta di apertura delle visualizzazioni
        print("\nPer visualizzare i risultati, puoi aprire uno dei seguenti file:")
        print(f"- Report visuale: {os.path.abspath(report_path)}")
        print(f"- Mappa regioni cerebrali: {os.path.abspath(vis_paths['regions_heatmap'])}")
        if vis_paths.get("3d_visualizations", {}).get("integrated"):
            print(f"- Visualizzazione 3D integrata: {os.path.abspath(vis_paths['3d_visualizations']['integrated'])}")
        
        # Aggiunta visualizzazione di tutte le regioni cerebrali
        all_regions_paths = vis_paths.get("3d_visualizations", {}).get("all_regions", {})
        if all_regions_paths:
            for atlas_name, path in all_regions_paths.items():
                if atlas_name == "integrated":
                    print(f"- Visualizzazione 3D integrata di tutte le regioni: {os.path.abspath(path)}")
                    break
        
        return {
            "mri_path": image_path,
            "atlases": atlases_dict,
            "results_path": results_path,
            "visualization_paths": vis_paths,
            "report_path": report_path,
            "integrated_results": integrated_results
        }
        
    except Exception as e:
        print(colored(f"\nERRORE DURANTE L'ESECUZIONE: {e}", "red"))
        print("\nTraceback completo:")
        traceback.print_exc()
        return None


def parse_arguments():
    """
    Parsing degli argomenti da linea di comando
    
    Returns:
        argparse.Namespace: Argomenti parsati
    """
    parser = argparse.ArgumentParser(
        description="Demo della soluzione multiatlante personalizzata per l'analisi di immagini cerebrali"
    )
    parser.add_argument(
        "--open-vis",
        action="store_true",
        help="Apre automaticamente le visualizzazioni 3D al termine dell'esecuzione"
    )
    parser.add_argument(
        "--vis-type",
        choices=["all", "region"],
        default="all",
        help="Tipo di visualizzazione da aprire: 'all' per tutte le regioni, 'region' per le regioni principali (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="demo_results",
        help="Directory di output per i risultati (default: demo_results)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse degli argomenti
    args = parse_arguments()
    
    # Directory di output personalizzabile
    output_dir = args.output_dir
    
    print_section("DEMO MULTIATLANTE PERSONALIZZATO")
    print("Questo script dimostra il funzionamento dell'implementazione aggiornata")
    print("della soluzione multiatlante personalizzata, che integra tre atlanti")
    print("per una visualizzazione completa di tutte le regioni cerebrali.\n")
    
    # Esecuzione della demo
    results = run_demo()
    
    # Apri le visualizzazioni se richiesto e se l'esecuzione ha avuto successo
    if args.open_vis and results:
        if "3d_visualizations" in results["visualization_paths"]:
            vis_paths = results["visualization_paths"]["3d_visualizations"]
            
            # Apri visualizzazione in base al parametro vis-type
            if args.vis_type == "all":
                # Cerca di aprire la visualizzazione integrata di tutte le regioni
                if "all_regions" in vis_paths and "integrated" in vis_paths["all_regions"]:
                    vis_path = vis_paths["all_regions"]["integrated"]
                    print(f"\nApertura visualizzazione 3D integrata di TUTTE le regioni cerebrali: {vis_path}")
                    webbrowser.open(f"file://{os.path.abspath(vis_path)}")
                # Se non disponibile, prova con una visualizzazione per atlante singolo
                elif "all_regions" in vis_paths and vis_paths["all_regions"]:
                    # Prendi il primo atlante disponibile
                    atlas_name, vis_path = list(vis_paths["all_regions"].items())[0]
                    print(f"\nApertura visualizzazione 3D di tutte le regioni dell'atlante {atlas_name.upper()}: {vis_path}")
                    webbrowser.open(f"file://{os.path.abspath(vis_path)}")
                # Se nemmeno queste sono disponibili, fallback sulla visualizzazione integrata
                elif "integrated" in vis_paths:
                    vis_path = vis_paths["integrated"]
                    print(f"\nVisualizzazione di atlanti singoli non disponibile.")
                    print(f"Fallback: apertura visualizzazione 3D integrata: {vis_path}")
                    webbrowser.open(f"file://{os.path.abspath(vis_path)}")
            else:  # args.vis_type == "region"
                # Cerca di aprire la visualizzazione integrata delle regioni principali
                if "integrated" in vis_paths:
                    vis_path = vis_paths["integrated"]
                    print(f"\nApertura visualizzazione 3D integrata delle principali regioni cerebrali: {vis_path}")
                    webbrowser.open(f"file://{os.path.abspath(vis_path)}")
                # Se non disponibile, prova con una visualizzazione per atlante singolo
                elif "atlanti" in vis_paths and vis_paths["atlanti"]:
                    # Prendi il primo atlante disponibile
                    atlas_name, vis_path = list(vis_paths["atlanti"].items())[0]
                    print(f"\nApertura visualizzazione 3D dell'atlante {atlas_name.upper()}: {vis_path}")
                    webbrowser.open(f"file://{os.path.abspath(vis_path)}")
                else:
                    print(f"\nNessuna visualizzazione delle regioni cerebrali disponibile.")