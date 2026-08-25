"""
Ultrasonic Feature Extraction and Dataset Utilities.
Extracts classical DSP features, time-domain statistics, Hilbert envelope metrics,
FFT spectral bands, and autocorrelation periodicity from ultrasonic pulse-echo waveforms.
Includes fast multi-threaded batch processing and disk caching.
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor

import seshizi_ml.ultrasonic as ultrasonic


def load_waveform(row, base_dir=None):
    """Bir metadata satırından veya dosya yolundan zaman (t_s) ve voltaj (v_volt) dizilerini okur."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if isinstance(row, (dict, pd.Series)):
        csv_rel = row["csv_path"]
    else:
        csv_rel = str(row)
    
    full_path = os.path.join(base_dir, "dataset", csv_rel)
    df_raw = pd.read_csv(full_path)
    t = df_raw["t_s"].to_numpy(dtype=np.float64)
    v = df_raw["v_volt"].to_numpy(dtype=np.float64)
    return t, v
def extract_features_from_waveform(t, v, thickness_mm=25.0, dsp_res=None, row=None):
    """Doğrudan zaman (t) ve voltaj (v) dizilerinden tüm DSP ve ML özelliklerini hesaplar (<1 ms)."""
    n_pts = len(t)
    dt = float(np.mean(np.diff(t))) if n_pts > 1 else 1e-9
    th_nominal_m = float(thickness_mm) / 1000.0
    ref_velocity = 5740.0 # 316 Paslanmaz Çelik referans ses hızı (m/s)

    # skip_first_packet=True ZORUNLU: kapalıyken (varsayılan False) uyarma
    # darbesi (bang) zarftan çıkarılmıyor, round-trip otokorelasyonu
    # bang-to-echo mesafesine kilitleniyor ve gerçek yankılar yanlış
    # ızgaraya göre elenip atılıyor -- bu depoda daha önce bulunup
    # ultrasonic.py içinde düzeltilen "Root cause #1" hatasının ta kendisi.
    # reference_velocity de eşleşen malzemeye (316 paslanmaz) verilmezse
    # iç diyagnostik yollar alüminyum varsayılanına düşüyor.
    #
    # max_echoes de sabit (varsayılan 4) BIRAKILAMAZ: bu veri seti TEK sabit
    # osiloskop penceresiyle (5us/bölme, 50us toplam) TÜM kalınlıklarda
    # toplandı -- pencereye sığan gidiş-dönüş sayısı kalınlığa göre 5.7
    # (25mm) ile 28.7 (5mm) arasında değişiyor. ultrasonic.py'deki
    # _capture_problem() kontrolü "trips > max_echoes + 3" olduğunda
    # kaydı doğrudan reddediyor (found=False) -- sabit max_echoes=4 ile
    # ince basamaklarda HER ZAMAN reddediliyordu. Pencereye gerçekte kaç
    # gidiş-dönüş sığdığını hesaplayıp buna göre veriyoruz.
    expected_round_trip = 2.0 * th_nominal_m / ref_velocity
    span_s = float(t[-1] - t[0]) if n_pts > 1 else 0.0
    trips_available = span_s / expected_round_trip if expected_round_trip > 0 else 4
    dynamic_max_echoes = int(min(40, max(4, trips_available + 1)))

    if dsp_res is None:
        try:
            dsp_res = ultrasonic.analyze(
                t, v, thickness_m=th_nominal_m, skip_first_packet=True,
                reference_velocity=ref_velocity, max_echoes=dynamic_max_echoes)
        except Exception:
            dsp_res = {}

    dsp_found = 1 if dsp_res.get('found', False) else 0
    dsp_coherence = float(dsp_res.get('coherence', 0.0) or 0.0)
    dsp_round_trip_s = float(dsp_res.get('envelope_round_trip_s', 0.0) or 0.0)
    dsp_velocity = float(dsp_res.get('velocity', 0.0) or 0.0)
    dsp_n_packets = len(dsp_res.get('packets', []))
    dsp_warnings_count = len(dsp_res.get('warnings', []))
    dsp_noise_level = float(dsp_res.get('noise_level', 0.0) or 0.0)

    dsp_thickness_est_mm = (ref_velocity * dsp_round_trip_s / 2.0) * 1000.0 if dsp_round_trip_s > 0 else 0.0

    v_abs = np.abs(v)
    max_abs_v = float(np.max(v_abs)) if n_pts > 0 else 0.0
    min_v = float(np.min(v)) if n_pts > 0 else 0.0
    max_v = float(np.max(v)) if n_pts > 0 else 0.0
    ptp_v = float(max_v - min_v)
    mean_v = float(np.mean(v)) if n_pts > 0 else 0.0
    std_v = float(np.std(v)) if n_pts > 0 else 0.0
    rms_v = float(np.sqrt(np.mean(v**2))) if n_pts > 0 else 0.0
    skew_v = float(skew(v)) if n_pts > 0 else 0.0
    kurt_v = float(kurtosis(v)) if n_pts > 0 else 0.0
    energy_v = float(np.sum(v**2) * dt)
    zcr = float(np.sum(np.diff(v > 0) != 0) / max(1, n_pts))

    fast_len = scipy.fft.next_fast_len(n_pts)
    analytic_signal = signal.hilbert(v, N=fast_len)[:n_pts]
    envelope = np.abs(analytic_signal)
    env_max = float(np.max(envelope)) if n_pts > 0 else 0.0
    env_mean = float(np.mean(envelope)) if n_pts > 0 else 0.0
    env_std = float(np.std(envelope)) if n_pts > 0 else 0.0
    env_auc = float(np.sum(envelope) * dt)

    t_rel = t - t[0] if n_pts > 0 else np.zeros(1)
    total_env = np.sum(envelope) + 1e-12
    env_centroid_t = float(np.sum(t_rel * envelope) / total_env)

    min_peak_dist = max(1, int(0.5e-6 / max(1e-12, dt)))
    peaks, _ = signal.find_peaks(envelope, distance=min_peak_dist, prominence=0.08 * max(1e-6, env_max))
    peak_times = t[peaks] if len(peaks) > 0 else np.array([0.0])
    peak_amps = envelope[peaks] if len(peaks) > 0 else np.array([0.0])

    p1_t = float(peak_times[0]) if len(peak_times) > 0 else 0.0
    p2_t = float(peak_times[1]) if len(peak_times) > 1 else p1_t
    p3_t = float(peak_times[2]) if len(peak_times) > 2 else p2_t

    delta_t_21 = max(0.0, p2_t - p1_t)
    delta_t_32 = max(0.0, p3_t - p2_t)
    p1_amp = float(peak_amps[0]) if len(peak_amps) > 0 else 0.0
    p2_amp = float(peak_amps[1]) if len(peak_amps) > 1 else 0.0
    p3_amp = float(peak_amps[2]) if len(peak_amps) > 2 else p2_amp
    amp_ratio_21 = p2_amp / (p1_amp + 1e-6)
    amp_ratio_32 = p3_amp / (p2_amp + 1e-6)

    env_centered = envelope - env_mean
    autocorr = signal.correlate(env_centered, env_centered, mode='full', method='fft')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_norm = autocorr / (autocorr[0] + 1e-12)

    min_lag_idx = int(0.8e-6 / max(1e-12, dt))
    if len(autocorr_norm) > min_lag_idx + 10:
        ac_peaks, _ = signal.find_peaks(autocorr_norm[min_lag_idx:], prominence=0.05)
        if len(ac_peaks) > 0:
            best_ac_idx = min_lag_idx + ac_peaks[0]
            autocorr_tau_s = float(best_ac_idx * dt)
            autocorr_peak_val = float(autocorr_norm[best_ac_idx])
        else:
            autocorr_tau_s = 0.0
            autocorr_peak_val = 0.0
    else:
        autocorr_tau_s = 0.0
        autocorr_peak_val = 0.0

    autocorr_thickness_est_mm = (ref_velocity * autocorr_tau_s / 2.0) * 1000.0

    fft_vals = np.abs(np.fft.rfft(v))
    fft_freqs = np.fft.rfftfreq(n_pts, d=dt)
    fft_power = fft_vals ** 2
    total_power = np.sum(fft_power) + 1e-12

    valid_mask = fft_freqs >= 0.2e6
    if np.sum(valid_mask) > 0:
        vf = fft_freqs[valid_mask]
        vp = fft_power[valid_mask]
        dom_idx = np.argmax(vp)
        dominant_freq_mhz = float(vf[dom_idx] / 1e6)
        spectral_centroid_mhz = float(np.sum(vf * vp) / (np.sum(vp) + 1e-12) / 1e6)
        spectral_spread_mhz = float(np.sqrt(np.sum(((vf/1e6 - spectral_centroid_mhz)**2) * vp) / (np.sum(vp) + 1e-12)))
    else:
        dominant_freq_mhz, spectral_centroid_mhz, spectral_spread_mhz = 0.0, 0.0, 0.0

    band_0_2mhz = float(np.sum(fft_power[(fft_freqs >= 0.0) & (fft_freqs < 2.0e6)]) / total_power)
    band_2_6mhz = float(np.sum(fft_power[(fft_freqs >= 2.0e6) & (fft_freqs < 6.0e6)]) / total_power)
    band_6_12mhz = float(np.sum(fft_power[(fft_freqs >= 6.0e6) & (fft_freqs < 12.0e6)]) / total_power)
    band_12_25mhz = float(np.sum(fft_power[(fft_freqs >= 12.0e6) & (fft_freqs < 25.0e6)]) / total_power)

    if row is not None and isinstance(row, (dict, pd.Series)):
        rec_id = row.get('id', 'live')
        variant = row.get('settings_variant', 'default') or 'default'
        gain = float(row.get('gain', 33))
        hp_raw = str(row.get('hp_filter_mhz', '1.0'))
        hp_val = 0.0 if 'OUT' in hp_raw.upper() else float(hp_raw)
        lp_val = float(row.get('lp_filter_mhz', 5))
        damping = float(row.get('damping', 1))
        prf = float(row.get('prf', 9))
        pulse_amp = float(row.get('pulse_amplitude', 9))
        averaging = float(row.get('averaging', 32))
    else:
        rec_id = 'live'
        variant = 'default'
        gain = 33.0
        hp_val = 1.0
        lp_val = 5.0
        damping = 1.0
        prf = 9.0
        pulse_amp = 9.0
        averaging = 32.0

    return {
        'id': rec_id,
        'thickness_mm': float(thickness_mm),
        'settings_variant': variant,
        # DSP
        'dsp_found': dsp_found,
        'dsp_coherence': dsp_coherence,
        'dsp_round_trip_s': dsp_round_trip_s,
        'dsp_velocity': dsp_velocity,
        'dsp_n_packets': dsp_n_packets,
        'dsp_warnings_count': dsp_warnings_count,
        'dsp_noise_level': dsp_noise_level,
        'dsp_thickness_est_mm': dsp_thickness_est_mm,
        # Zaman / Genlik
        'max_abs_v': max_abs_v,
        'ptp_v': ptp_v,
        'mean_v': mean_v,
        'std_v': std_v,
        'rms_v': rms_v,
        'skew_v': skew_v,
        'kurt_v': kurt_v,
        'energy_v': energy_v,
        'zcr': zcr,
        # Zarf
        'env_max': env_max,
        'env_mean': env_mean,
        'env_std': env_std,
        'env_auc': env_auc,
        'env_centroid_t': env_centroid_t,
        'p1_t': p1_t,
        'p2_t': p2_t,
        'p3_t': p3_t,
        'delta_t_21': delta_t_21,
        'delta_t_32': delta_t_32,
        'amp_ratio_21': amp_ratio_21,
        'amp_ratio_32': amp_ratio_32,
        # Otokorelasyon
        'autocorr_tau_s': autocorr_tau_s,
        'autocorr_peak_val': autocorr_peak_val,
        'autocorr_thickness_est_mm': autocorr_thickness_est_mm,
        # Spektral
        'dominant_freq_mhz': dominant_freq_mhz,
        'spectral_centroid_mhz': spectral_centroid_mhz,
        'spectral_spread_mhz': spectral_spread_mhz,
        'band_0_2mhz': band_0_2mhz,
        'band_2_6mhz': band_2_6mhz,
        'band_6_12mhz': band_6_12mhz,
        'band_12_25mhz': band_12_25mhz,
        # Cihaz ayarları
        'gain': gain,
        'hp_val': hp_val,
        'lp_val': lp_val,
        'damping': damping,
        'prf': prf,
        'pulse_amp': pulse_amp,
        'averaging': averaging
    }


def extract_all_features(row, base_dir=None):
    """Tek bir ölçüm kaydından hem DSP hem de istatistiksel/spektral ML özelliklerini çıkarır."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
    t, v = load_waveform(row, base_dir=base_dir)
    return extract_features_from_waveform(t, v, thickness_mm=float(row['thickness_mm']), row=row)


def extract_features_dataframe(df, base_dir=None, cache_path=None, force_recompute=False):
    """
    DataFrame içindeki tüm ölçümlerden çok iş parçacıklı (multi-threaded) olarak
    özellikleri çıkarır. cache_path verilirse CSV olarak önbellekler.
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        t0 = time.time()
        df_cached = pd.read_csv(cache_path)
        print(f"Features loaded from cache ({cache_path}) [{time.time()-t0:.2f} s, {len(df_cached)} records]")
        return df_cached

    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    num_workers = os.cpu_count() or 4
    print(f"Starting multi-threaded feature extraction ({num_workers} CPU workers)...")
    t0 = time.time()

    rows = [df.iloc[i].to_dict() for i in range(len(df))]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        features_list = list(executor.map(lambda r: extract_all_features(r, base_dir=base_dir), rows))

    df_features = pd.DataFrame(features_list)
    print(f"Feature extraction completed! Elapsed: {time.time()-t0:.2f} s ({len(df_features)} records)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_features.to_csv(cache_path, index=False)
        print(f"Features cached to disk: {cache_path}")

    return df_features


def stratified_multi_split(df, group_cols=['thickness_mm', 'settings_variant'], test_size=0.20, random_state=42):
    """
    Hem kalınlık (thickness_mm) hem de donanım ayar grubu (settings_variant) kombinasyonunu
    dengeli bir şekilde eğitim ve test kümelerine paylaştıran tabakalı bölücü.
    """
    rng = np.random.RandomState(random_state)
    train_indices = []
    test_indices = []
    
    group_df = df.copy()
    for col in group_cols:
        group_df[col] = group_df[col].fillna('default')
        
    for _, group in group_df.groupby(group_cols):
        idxs = list(group.index)
        rng.shuffle(idxs)
        n_test = int(np.round(len(idxs) * test_size))
        if len(idxs) >= 2 and n_test == 0 and test_size > 0:
            n_test = 1
        elif len(idxs) >= 2 and n_test == len(idxs):
            n_test = len(idxs) - 1
            
        test_indices.extend(idxs[:n_test])
        train_indices.extend(idxs[n_test:])
        
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)
