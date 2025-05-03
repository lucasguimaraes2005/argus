
import os
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
import pytesseract



class ArgusSystem:
    """Sistema principal para detecção e verificação de placas de veículos."""
    
    def __init__(self, db_file='db_veiculos_roubados.csv'):
        """
        Inicializa o sistema Argus.
        
        Args:
            db_file (str): Caminho para arquivo CSV com database de veículos roubados.
        """
        print("Inicializando sistema Argus...")
        
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

        self.db_file = db_file
        self.stolen_vehicles = self._load_database()
        
        self.plate_cascade = None
        self._load_detector()
        
        self.alerts_dir = 'alertas'
        if not os.path.exists(self.alerts_dir):
            os.makedirs(self.alerts_dir)
    
    def _load_database(self):
        """Carrega a base de dados de veículos roubados."""
        try:
            if os.path.exists(self.db_file):
                df = pd.read_csv(self.db_file)
                print(f"Base de dados carregada com {len(df)} registros")
                return df
            else:
                df = pd.DataFrame(columns=['placa', 'modelo', 'cor', 'data_roubo'])
                df.to_csv(self.db_file, index=False)
                print("Nova base de dados criada")
                return df
        except Exception as e:
            print(f"Erro ao carregar base de dados: {str(e)}")
            return pd.DataFrame(columns=['placa', 'modelo', 'cor', 'data_roubo'])

    def _load_detector(self):
        """Carrega o detector de placas."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            
            if os.path.exists(cascade_path):
                self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                print("Detector de placas carregado com sucesso")
            else:
                self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Detector padrão carregado. Recomenda-se usar um detector específico para placas")
        except Exception as e:
            print(f"Erro ao carregar detector: {str(e)}")
            print("O sistema usará detecção básica de contornos")
    
    
    def add_stolen_vehicle(self, plate, model="Desconhecido", color="Desconhecido"):
        """
        Adiciona um veículo roubado à base de dados.
        
        Args:
            plate (str): Número da placa do veículo
            model (str): Modelo do veículo
            color (str): Cor do veículo
        """
        plate = self._normalize_plate(plate)
        
        if self.is_stolen(plate):
            print(f"A placa {plate} já está registrada na base de dados")
            return False
        
        new_data = pd.DataFrame({
            'placa': [plate],
            'modelo': [model],
            'cor': [color],
            'data_roubo': [datetime.now().strftime("%Y-%m-%d")]
        })
        
        self.stolen_vehicles = pd.concat([self.stolen_vehicles, new_data], ignore_index=True)
        self.stolen_vehicles.to_csv(self.db_file, index=False)
        print(f"Veículo com placa {plate} adicionado à base de dados")
        return True
    
    def remove_stolen_vehicle(self, plate):
        """
        Remove um veículo da base de dados de roubados.
        
        Args:
            plate (str): Número da placa do veículo
        """
        plate = self._normalize_plate(plate)
        
        if not self.is_stolen(plate):
            print(f"A placa {plate} não está na base de dados")
            return False
        
        self.stolen_vehicles = self.stolen_vehicles[self.stolen_vehicles['placa'] != plate]
        self.stolen_vehicles.to_csv(self.db_file, index=False)
        print(f"Veículo com placa {plate} removido da base de dados")
        return True
    
    def is_stolen(self, plate):
        """
        Verifica se um veículo está na base de dados de roubados.
        
        Args:
            plate (str): Número da placa do veículo
            
        Returns:
            bool: True se o veículo está registrado como roubado
        """
        plate = self._normalize_plate(plate)
        return plate in self.stolen_vehicles['placa'].values
    
    def _normalize_plate(self, plate):
        """
        Normaliza a placa para o formato padrão.
        
        Args:
            plate (str): Placa a ser normalizada
            
        Returns:
            str: Placa normalizada
        """
        plate = ''.join(c for c in plate if c.isalnum()).upper()
        return plate
        


def criar_db_exemplo():
    """Cria uma base de dados de exemplo com algumas placas."""
    df = pd.DataFrame({
        'placa': ['ABC1234', 'DEF5678', 'GHI9012', 'JKL3456'],
        'modelo': ['Volkswagen Gol', 'Fiat Uno', 'Chevrolet Onix', 'Toyota Corolla'],
        'cor': ['Prata', 'Vermelho', 'Branco', 'Preto'],
        'data_roubo': ['2025-04-15', '2025-04-20', '2025-04-25', '2025-04-30']
    })
    
    df.to_csv('db_veiculos_roubados.csv', index=False)
    print("Base de dados de exemplo criada com sucesso!")


if __name__ == "__main__":
    if not os.path.exists('db_veiculos_roubados.csv'):
        criar_db_exemplo()
    
    argus = ArgusSystem()