# cbr_engine.py (VERSI PERBAIKAN LENGKAP)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

class CBREngine:
    """
    Mesin Cerdas untuk Prediksi Berbasis Kasus yang Adaptif dan Probabilistik.
    Mengimplementasikan Retain sejati melalui Jarak Adaptif berbasis Error.
    """
    def __init__(self, data_filepath: str, log_filepath: str, end_date: str = None):
        self.data_filepath = data_filepath
        self.log_filepath = log_filepath
        self.end_date = end_date
        self.features = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
        self.target_col = 'Terakhir'
        self.date_col = 'Tanggal'
        self.scaler = MinMaxScaler()
        self.error_scaler = MinMaxScaler()
        
        self._load_and_process_knowledge()

    def _load_and_process_knowledge(self):
        # Tahap 1 Retain: Membaca Pengetahuan Faktual dan Pengalaman
        
        # 1a. Muat Data Faktual (Ground Truth)
        try:
            df = pd.read_csv(self.data_filepath, encoding='utf-8-sig')
        except FileNotFoundError:
            raise FileNotFoundError(f"File data tidak ditemukan di: {self.data_filepath}")
            
        df[self.date_col] = pd.to_datetime(df[self.date_col], format='%d/%m/%Y')
        df.sort_values(self.date_col, ascending=True, inplace=True)
        
        for col in self.features:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        
        if self.end_date:
            df = df[df[self.date_col] <= pd.to_datetime(self.end_date)]
        
        self.raw_df = df.reset_index(drop=True)
        if len(self.raw_df) < 1:
            raise ValueError("Tidak ada data yang tersedia untuk tanggal yang dipilih atau sebelumnya.")
        
        self.scaled_features = self.scaler.fit_transform(self.raw_df[self.features])
        self.scaled_df = pd.DataFrame(self.scaled_features, columns=self.features)
        self.scaled_df[self.date_col] = self.raw_df[self.date_col]

        # 1b. Muat & Gabungkan Pengetahuan Pengalaman (Log Error) - MODIFIKASI
        self.knowledge_df = self.scaled_df.copy()
        if os.path.exists(self.log_filepath) and os.path.getsize(self.log_filepath) > 0:
            log_df = pd.read_csv(self.log_filepath)
            log_df.rename(columns={'Tanggal_Prediksi': self.date_col}, inplace=True)
            log_df[self.date_col] = pd.to_datetime(log_df[self.date_col])
            
            if not log_df.empty and 'Error_Absolut' in log_df.columns:
                # Saring log untuk hanya menggunakan baris yang memiliki nilai error (bukan NaN)
                valid_error_logs = log_df.dropna(subset=['Error_Absolut']).copy()
                
                if not valid_error_logs.empty:
                    # Lakukan penskalaan error hanya pada data yang valid
                    valid_error_logs['E_norm'] = self.error_scaler.fit_transform(valid_error_logs[['Error_Absolut']])
                    self.knowledge_df = pd.merge(self.knowledge_df, valid_error_logs[[self.date_col, 'E_norm']], on=self.date_col, how='left')
                    self.knowledge_df['E_norm'].fillna(0, inplace=True)
                else:
                    self.knowledge_df['E_norm'] = 0 # Tidak ada log valid untuk dipelajari
            else:
                self.knowledge_df['E_norm'] = 0
        else:
            self.knowledge_df['E_norm'] = 0 # Log tidak ada, maka error dianggap 0

    def _create_case_base(self, window_size: int):
        if len(self.knowledge_df) <= window_size:
            raise ValueError("Data historis tidak cukup panjang untuk ukuran window.")
        
        case_base, targets, case_errors = [], [], []
        num_rows = len(self.knowledge_df)
        for i in range(num_rows - window_size):
            window_indices = range(i, i + window_size)
            target_index = i + window_size
            
            case_base.append(self.knowledge_df[self.features].iloc[window_indices].values)
            targets.append(self.raw_df[self.target_col].iloc[target_index])
            case_errors.append(self.knowledge_df['E_norm'].iloc[target_index])

        return np.array(case_base), np.array(targets), np.array(case_errors)

    def _adaptive_distance(self, window1, window2, past_error, lambda_val):
        d_chebyshev = np.mean(np.max(np.abs(window1 - window2), axis=1))
        return d_chebyshev * (1 + lambda_val * past_error)

    def predict(self, k: int, window_size: int, lambda_val: float):
        case_base, targets, case_errors = self._create_case_base(window_size)
        query_case = self.knowledge_df[self.features].iloc[-window_size:].values
        
        distances = [self._adaptive_distance(query_case, case, error, lambda_val) for case, error in zip(case_base, case_errors)]
        
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:k]
        
        k_distances = np.array(distances)[k_indices]
        k_targets = targets[k_indices]
        
        epsilon = 1e-6
        weights = 1 / (k_distances + epsilon)
        sum_of_weights = np.sum(weights)
        
        y_pred_awal = np.sum(weights * k_targets) / sum_of_weights
        
        weighted_variance = np.sum(weights * (k_targets - y_pred_awal)**2) / sum_of_weights
        weighted_std_dev = np.sqrt(weighted_variance)
        
        z_score = 1.96
        prediction_interval = (y_pred_awal - z_score * weighted_std_dev, y_pred_awal + z_score * weighted_std_dev)
        
        entropy = 0.5 * np.log(2 * np.pi * np.e * weighted_variance) if weighted_variance > 0 else -np.inf
        
        neighbor_info = pd.DataFrame({'Jarak Adaptif': k_distances, 'Harga Target Historis': k_targets})

        return {'prediction': y_pred_awal, 'uncertainty_std_dev': weighted_std_dev,
                'prediction_interval_95': prediction_interval, 'entropy': entropy,
                'k_neighbors_info': neighbor_info}

    def find_optimal_k(self, window_size: int, k_options: list, validation_days: int, lambda_val: float):
        if len(self.raw_df) < validation_days + window_size + 1:
            raise ValueError(f"Data tidak cukup untuk validasi.")
        
        errors = {}
        for k in tqdm(k_options, desc="Mencari K Optimal"):
            k_errors = []
            for i in range(1, validation_days + 1):
                validation_date = self.raw_df['Tanggal'].iloc[-i]
                train_end_date = validation_date - pd.Timedelta(days=1)
                
                try:
                    temp_engine = CBREngine(self.data_filepath, self.log_filepath, end_date=train_end_date.strftime('%Y-%m-%d'))
                    result = temp_engine.predict(k=k, window_size=window_size, lambda_val=lambda_val)
                    actual = self.raw_df[self.raw_df['Tanggal'] == validation_date][self.target_col].iloc[0]
                    k_errors.append(abs(result['prediction'] - actual))
                except (ValueError, IndexError): continue

            if k_errors: errors[k] = np.mean(k_errors)
        
        if not errors: return None
        return min(errors, key=errors.get)