import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.cluster import KMeans
import os
import concurrent.futures
from tqdm import tqdm
import warnings

# Ignorar advertencias matemáticas triviales para limpiar la consola
warnings.filterwarnings('ignore')

# ==========================================
# PARÁMETROS GLOBALES
# ==========================================
SR_TARGET = 22050
N_FFT_SIZE = 1024
HOP_LENGTH = 512
VOLUMEN_RUIDO = 0.5
K_RANK = 20
ITERS_TRAIN = 300

DIR_VOZ = Path(r'musan/speech')
DIR_RUIDO = Path(r'musan/noise')
DIR_SALIDA = Path(r'dataset_procesado')

# ==========================================
# MATEMÁTICA PURA (WORKER)
# ==========================================
def bcgd_optimizer_step(X_real, W_mat, H_mat, vel_W, vel_H, learn_rate, momentum_beta, lambda_reg):
    M_mask = np.ones_like(X_real)
    
    # Nesterov para W
    W_lookahead = W_mat - learn_rate * momentum_beta * vel_W
    grad_W = (M_mask * (W_lookahead @ H_mat - X_real)) @ H_mat.T + lambda_reg * W_lookahead
    vel_W = momentum_beta * vel_W + grad_W
    W_mat = W_mat - learn_rate * vel_W
    W_mat = np.maximum(W_mat, 0)
        
    # Nesterov para H
    H_lookahead = H_mat - learn_rate * momentum_beta * vel_H
    grad_H = W_mat.T @ (M_mask * (W_mat @ H_lookahead - X_real)) + lambda_reg * H_lookahead
    vel_H = momentum_beta * vel_H + grad_H
    H_mat = H_mat - learn_rate * vel_H
    H_mat = np.maximum(H_mat, 0)

    return W_mat, H_mat, vel_W, vel_H

def process_audio_pair(args):
    """
    Función pura que será procesada en paralelo por cada núcleo del CPU.
    """
    idx, ruta_voz, ruta_ruido = args
    nombre_base = f"audio_{idx:03d}"
    out_dir = DIR_SALIDA / nombre_base
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Carga y Mezcla Dinámica
        onda_voz, _ = librosa.load(str(ruta_voz), sr=SR_TARGET, mono=True)
        onda_rui, _ = librosa.load(str(ruta_ruido), sr=SR_TARGET, mono=True)
        
        if len(onda_rui) < len(onda_voz):
            onda_rui = np.tile(onda_rui, int(np.ceil(len(onda_voz) / len(onda_rui))))
        onda_rui = onda_rui[:len(onda_voz)]
        onda_voz = onda_voz[:len(onda_rui)]
        
        mezcla_audio = onda_voz + VOLUMEN_RUIDO * onda_rui
        
        # 2. STFT
        espectro = librosa.stft(mezcla_audio, n_fft=N_FFT_SIZE, hop_length=HOP_LENGTH)
        X_target = np.abs(espectro)
        fase_target = np.angle(espectro)
        
        # 3. Escalador Dinámico de Gradiente
        frames_totales = X_target.shape[1]
        alpha_dinamico = 1e-3 * (215.0 / frames_totales)
        bins_freq = X_target.shape[0]
        
        # 4. Entrenamiento BCGD
        np.random.seed(idx) # Semilla independiente para cada hilo
        W = np.random.uniform(0, 1/np.sqrt(K_RANK), (bins_freq, K_RANK))
        H = np.random.uniform(0, 1/np.sqrt(K_RANK), (K_RANK, frames_totales))
        vW, vH = np.zeros_like(W), np.zeros_like(H)
        
        for _ in range(ITERS_TRAIN):
            W, H, vW, vH = bcgd_optimizer_step(X_target, W, H, vW, vH, alpha_dinamico, 0.9, 0.0)
            
        # 5. K-Means BSS
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(H)
        idx_S1 = np.where(labels == 0)[0]
        idx_S2 = np.where(labels == 1)[0]
        
        X_S1 = W[:, idx_S1] @ H[idx_S1, :]
        X_S2 = W[:, idx_S2] @ H[idx_S2, :]
        
        audio_S1 = librosa.istft(X_S1 * np.exp(1j * fase_target), hop_length=HOP_LENGTH)
        audio_S2 = librosa.istft(X_S2 * np.exp(1j * fase_target), hop_length=HOP_LENGTH)
        
        # 6. Selección de Voz vía RMSE
        limite = min(len(onda_voz), len(audio_S1), len(audio_S2))
        referencia, senal_1, senal_2 = onda_voz[:limite], audio_S1[:limite], audio_S2[:limite]
        
        if np.mean((referencia - senal_1)**2) < np.mean((referencia - senal_2)**2):
            audio_voz, audio_ruido = senal_1, senal_2
        else:
            audio_voz, audio_ruido = senal_2, senal_1
            
        # Reconstrucción de la mezcla completa (W x H)
        mezcla_rec = librosa.istft((W @ H) * np.exp(1j * fase_target), hop_length=HOP_LENGTH)[:limite]
            
        # 7. Guardado final
        sf.write(out_dir / f"{nombre_base}_mezcla_reconstruida.wav", mezcla_rec, SR_TARGET)
        sf.write(out_dir / f"{nombre_base}_voz_aislada.wav", audio_voz, SR_TARGET)
        sf.write(out_dir / f"{nombre_base}_ruido_residuo.wav", audio_ruido, SR_TARGET)
        
        return (True, nombre_base, "OK")
    
    except Exception as e:
        return (False, nombre_base, str(e))

# ==========================================
# CONTROLADOR PRINCIPAL
# ==========================================
if __name__ == '__main__':
    print("==================================================")
    print(" INICIANDO BATCH PROCESSOR PARALELO (NMF BSS)")
    print("==================================================\\n")
    
    DIR_SALIDA.mkdir(exist_ok=True)
    
    archivos_voz = sorted(DIR_VOZ.rglob('*.wav'))
    archivos_ruido = sorted(DIR_RUIDO.rglob('*.wav'))
    
    if not archivos_voz or not archivos_ruido:
        print("Error: No se encontraron archivos wav en musan/speech o musan/noise")
        exit()
        
    # Emparejamiento circular (si hay menos ruidos que voces)
    pares = []
    num_tareas = len(archivos_voz)
    for i in range(num_tareas):
        ruido_path = archivos_ruido[i % len(archivos_ruido)]
        pares.append((i+1, archivos_voz[i], ruido_path))
        
    # LIMITAMOS PARA PRUEBA A 2 AUDIOS
    import sys
    if '--full' not in sys.argv:
        print(">>> MODO PRUEBA: Procesando solo los primeros 2 audios. <<<")
        print(">>> Para procesar los 426, ejecuta: python batch_nmf_processor.py --full <<<\\n")
        pares = pares[:100]
        
    # nucleos = os.cpu_count()

    nucleos = max(1, os.cpu_count() - 2)

    print(f"Archivos a procesar : {len(pares)}")
    print(f"Núcleos detectados  : {nucleos}")
    print(f"Motor Paralelo      : ProcessPoolExecutor\\n")
    
    exitos = 0
    fallos = 0
    
    # Procesamiento Paralelo con Barra de Progreso
    with concurrent.futures.ProcessPoolExecutor(max_workers=nucleos) as executor:
        resultados = list(tqdm(executor.map(process_audio_pair, pares), total=len(pares), desc="Procesando Audios"))
        
    print("\\n==================================================")
    for exito, nombre, msg in resultados:
        if exito:
            exitos += 1
        else:
            fallos += 1
            print(f"[!] Error en {nombre}: {msg}")
            
    print(f"Procesamiento Completado: {exitos} Exitosos, {fallos} Fallidos.")
    print(f"Revisa la carpeta: {DIR_SALIDA.absolute()}")
