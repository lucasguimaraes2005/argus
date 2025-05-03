
import os
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
import pytesseract



class ArgusSystem:
    
    def __init__(self, db_file='db_veiculos_roubados.csv'):

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

        plate = self._normalize_plate(plate)
        
        if not self.is_stolen(plate):
            print(f"A placa {plate} não está na base de dados")
            return False
        
        self.stolen_vehicles = self.stolen_vehicles[self.stolen_vehicles['placa'] != plate]
        self.stolen_vehicles.to_csv(self.db_file, index=False)
        print(f"Veículo com placa {plate} removido da base de dados")
        return True
    
    def is_stolen(self, plate):

        plate = self._normalize_plate(plate)
        return plate in self.stolen_vehicles['placa'].values
    
    def _normalize_plate(self, plate):

        plate = ''.join(c for c in plate if c.isalnum()).upper()
        return plate

    def _preprocess_plate_image(self, plate_img):

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    
    def _recognize_plate(self, plate_img):

        processed_img = self._preprocess_plate_image(plate_img)
        
        config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        text = pytesseract.image_to_string(processed_img, config=config)
        
        plate_text = ''.join(c for c in text if c.isalnum()).upper()
        
        return plate_text

    def _detect_plates_haarcascade(self, frame):

        if self.plate_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return plates
    
    def _detect_plates_contour(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blur, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        possible_plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if 1.5 <= aspect_ratio <= 5 and w > 60 and h > 20:
                possible_plates.append((x, y, w, h))
        
        return possible_plates

    def process_video(self, video_path, output_path=None, display=True):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {video_path}")
            return
        
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detected_plates = set()
        
        print(f"Processando vídeo: {video_path}")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0: 
                if display:
                    cv2.imshow('Argus - Detector de Placas', frame)
                if output_path:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            result_frame = frame.copy()
            
            plates = self._detect_plates_haarcascade(frame)
            
            if len(plates) == 0:
                plates = self._detect_plates_contour(frame)
            
            for (x, y, w, h) in plates:
                plate_img = frame[y:y+h, x:x+w]
                
                if plate_img.size == 0:
                    continue
                
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                plate_text = self._recognize_plate(plate_img)
                
                if len(plate_text) >= 6:  
                    is_stolen = self.is_stolen(plate_text)
                    
                    status_text = f"ALERTA: {plate_text}" if is_stolen else plate_text
                    color = (0, 0, 255) if is_stolen else (255, 0, 0)
                    cv2.putText(result_frame, status_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if is_stolen and plate_text not in detected_plates:
                        detected_plates.add(plate_text)
                        self._generate_alert(plate_text, plate_img)
            
            if display:
                cv2.imshow('Argus - Detector de Placas', result_frame)
            
            if output_path:
                out.write(result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processamento concluído. {len(detected_plates)} placas de veículos roubados detectadas.")
        return detected_plates
    
    def _generate_alert(self, plate_text, plate_img):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        img_path = os.path.join(self.alerts_dir, f"alerta_{plate_text}_{timestamp}.jpg")
        cv2.imwrite(img_path, plate_img)
        
        vehicle_info = self.stolen_vehicles[self.stolen_vehicles['placa'] == self._normalize_plate(plate_text)]
        
        with open(os.path.join(self.alerts_dir, "registros_alertas.txt"), "a") as f:
            f.write(f"\n--- ALERTA: Veículo Roubado Detectado ---\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Placa: {plate_text}\n")
            
            if not vehicle_info.empty:
                f.write(f"Modelo: {vehicle_info['modelo'].values[0]}\n")
                f.write(f"Cor: {vehicle_info['cor'].values[0]}\n")
                f.write(f"Data do Roubo: {vehicle_info['data_roubo'].values[0]}\n")
            
            f.write(f"Imagem salva em: {img_path}\n")
            f.write("-----------------------------------------\n")
        
        print(f"⚠️ ALERTA: Veículo roubado detectado - Placa: {plate_text}")


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


def menu_principal():
    
    if not os.path.exists('db_veiculos_roubados.csv'):
        criar_db_exemplo()
    
    argus = ArgusSystem()
    
    while True:
        print("\n" + "="*50)
        print("🔍 SISTEMA ARGUS - DETECTOR DE PLACAS 🔍")
        print("="*50)
        print("1. Processar vídeo")
        print("2. Adicionar veículo roubado à base de dados")
        print("3. Remover veículo da base de dados")
        print("4. Verificar placa específica")
        print("5. Listar veículos roubados")
        print("6. Sair")
        print("="*50)
        
        opcao = input("Escolha uma opção: ")
        
        if opcao == '1':
            video_path = input("Caminho do vídeo a ser processado: ")
            if not os.path.exists(video_path):
                print(f"Erro: O arquivo {video_path} não existe!")
                continue
                
            output = input("Salvar vídeo processado? (s/n): ").lower()
            output_path = None
            if output == 's':
                output_path = input("Caminho para salvar o vídeo processado: ")
            
            argus.process_video(video_path, output_path)
            
        elif opcao == '2':
            placa = input("Digite a placa do veículo roubado: ")
            modelo = input("Digite o modelo do veículo (ou deixe em branco): ")
            cor = input("Digite a cor do veículo (ou deixe em branco): ")
            
            if not modelo:
                modelo = "Desconhecido"
            if not cor:
                cor = "Desconhecido"
                
            argus.add_stolen_vehicle(placa, modelo, cor)
            
        elif opcao == '3':
            placa = input("Digite a placa do veículo a ser removido: ")
            argus.remove_stolen_vehicle(placa)
            
        elif opcao == '4':
            placa = input("Digite a placa a ser verificada: ")
            resultado = argus.is_stolen(placa)
            
            if resultado:
                print(f"⚠️ ATENÇÃO: A placa {placa} está registrada como veículo roubado!")
            else:
                print(f"✅ A placa {placa} NÃO está registrada como veículo roubado.")
                
        elif opcao == '5':
            if len(argus.stolen_vehicles) == 0:
                print("A base de dados está vazia!")
            else:
                print("\nVEÍCULOS ROUBADOS REGISTRADOS:")
                print("-"*60)
                print(f"{'PLACA':<10} | {'MODELO':<20} | {'COR':<10} | {'DATA DO ROUBO':<15}")
                print("-"*60)
                
                for _, row in argus.stolen_vehicles.iterrows():
                    print(f"{row['placa']:<10} | {row['modelo']:<20} | {row['cor']:<10} | {row['data_roubo']:<15}")
                
                print("-"*60)
                print(f"Total: {len(argus.stolen_vehicles)} veículos")
                
        elif opcao == '6':
            print("Encerrando o Sistema Argus. Até logo!")
            break
            
        else:
            print("Opção inválida! Por favor, escolha uma opção válida.")


if __name__ == "__main__":
    menu_principal()
