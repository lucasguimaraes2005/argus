
import os
import pandas as pd
from datetime import datetime


class ArgusSystem:
    """Sistema principal para detecção e verificação de placas de veículos."""
    
    def __init__(self, db_file='db_veiculos_roubados.csv'):
        """
        Inicializa o sistema Argus.
        
        Args:
            db_file (str): Caminho para arquivo CSV com database de veículos roubados.
        """
        print("Inicializando sistema Argus...")
        
        self.db_file = db_file
        self.stolen_vehicles = self._load_database()
        
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