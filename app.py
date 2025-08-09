# ==============================================================
# SISTEMA FINFLOW MEJORADO - ANÁLISIS DE RIESGO Y ESTRATEGIA COMPETITIVA
# Integración completa: Red Neuronal + Minimax + Alfa-Beta + Análisis Competitivo
# ==============================================================

import pandas as pd
import numpy as np
import random
from faker import Faker
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

fake = Faker()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ==============================================================
# 1. FUNCIÓN DE RIESGO MEJORADA CON MÁS VARIABLES
# ==============================================================

def calcular_riesgo_avanzado(edad, ingresos, deudas, monto, score, historial,
                            educacion, contrato, region, antiguedad, genero="M"):
    """
    Función de riesgo mejorada que considera más variables y interacciones
    """
    riesgo_base = 1.0
    
    # === SCORE CREDITICIO (peso mayor) ===
    if score < 500: riesgo_base += 5.0
    elif score < 550: riesgo_base += 3.5
    elif score < 600: riesgo_base += 2.5
    elif score < 650: riesgo_base += 1.5
    elif score < 700: riesgo_base += 0.8
    elif score < 750: riesgo_base += 0.2
    else: riesgo_base -= 0.5
    
    # === CAPACIDAD DE PAGO REFINADA ===
    if ingresos > 0:
        dti_ratio = deudas / ingresos
        capacidad_disponible = ingresos - deudas
        
        # DTI (Debt-to-Income) analysis
        if dti_ratio > 0.8: riesgo_base += 4.0
        elif dti_ratio > 0.6: riesgo_base += 2.5
        elif dti_ratio > 0.4: riesgo_base += 1.0
        elif dti_ratio < 0.2: riesgo_base -= 0.5
        
        # Relación monto vs capacidad
        if capacidad_disponible <= 0: riesgo_base += 5.0
        elif monto > 3 * capacidad_disponible: riesgo_base += 3.5
        elif monto > 2 * capacidad_disponible: riesgo_base += 2.0
        elif monto > capacidad_disponible: riesgo_base += 1.0
        else: riesgo_base -= 0.8
    else:
        riesgo_base += 6.0  # Sin ingresos = muy alto riesgo
    
    # === HISTORIAL CREDITICIO ===
    if historial == 0: riesgo_base += 2.0
    else: riesgo_base -= 0.5
    
    # === DEMOGRAFÍA Y ESTABILIDAD ===
    # Edad
    if edad < 20: riesgo_base += 2.0
    elif edad < 25: riesgo_base += 1.2
    elif edad > 70: riesgo_base += 2.5
    elif edad > 60: riesgo_base += 1.5
    elif 30 <= edad <= 50: riesgo_base -= 0.3  # Edad estable
    
    # Educación
    educacion_factor = {
        "Primaria": 1.5,
        "Secundaria": 0.8,
        "Técnico": 0.3,
        "Universitario": -0.3,
        "Posgrado": -0.8
    }
    riesgo_base += educacion_factor.get(educacion, 0)
    
    # Tipo de contrato
    contrato_factor = {
        "Informal": 2.5,
        "Servicios": 1.8,
        "Fijo": 0.8,
        "Indefinido": -0.8,
        "Pensionado": 0.3
    }
    riesgo_base += contrato_factor.get(contrato, 0)
    
    # Antigüedad laboral
    if antiguedad < 6: riesgo_base += 1.5
    elif antiguedad < 12: riesgo_base += 0.8
    elif antiguedad > 60: riesgo_base -= 0.5
    
    # === FACTORES REGIONALES ===
    regiones_alto_riesgo = ["Leticia", "Quibdó", "San José del Guaviare", 
                           "Puerto Carreño", "Inírida", "Mocoa", "Mitú"]
    regiones_bajo_riesgo = ["Bogotá", "Medellín", "Cali", "Barranquilla", 
                           "Bucaramanga", "Cartagena"]
    
    if region in regiones_alto_riesgo: riesgo_base += 1.5
    elif region in regiones_bajo_riesgo: riesgo_base -= 0.3
    
    # === INTERACCIONES COMPLEJAS ===
    # Joven + alto monto + poca antigüedad
    if edad < 28 and monto > 10_000_000 and antiguedad < 12:
        riesgo_base += 1.0
    
    # Pensionado + monto alto
    if contrato == "Pensionado" and monto > 15_000_000:
        riesgo_base += 1.5
    
    # Score alto + ingresos altos = descuento adicional
    if score > 750 and ingresos > 5_000_000:
        riesgo_base -= 0.5
    
    return max(1.0, min(10.0, round(riesgo_base, 1)))

# ==============================================================
# 2. GENERACIÓN DE DATASET MEJORADO Y MÁS REALISTA
# ==============================================================

def generar_dataset_avanzado(n_samples=1000):
    """Genera un dataset más realista con distribuciones mejoradas"""
    
    data = []
    regiones_colombia = [
        "Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga",
        "Pereira", "Manizales", "Ibagué", "Santa Marta", "Villavicencio",
        "Neiva", "Pasto", "Armenia", "Popayán", "Montería", "Sincelejo",
        "Valledupar", "Riohacha", "Yopal", "Florencia", "Leticia", "Quibdó",
        "San José del Guaviare", "Puerto Carreño", "Inírida", "Mocoa", "Mitú"
    ]
    
    for i in range(n_samples):
        # Demografía correlacionada
        edad = max(18, int(np.random.normal(40, 15)))
        genero = random.choice(["M", "F"])
        
        # Educación correlacionada con edad
        if edad < 25:
            educacion = random.choices(
                ["Secundaria", "Técnico", "Universitario"],
                weights=[0.5, 0.3, 0.2])[0]
        elif edad < 40:
            educacion = random.choices(
                ["Secundaria", "Técnico", "Universitario", "Posgrado"],
                weights=[0.2, 0.3, 0.4, 0.1])[0]
        else:
            educacion = random.choices(
                ["Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"],
                weights=[0.1, 0.3, 0.25, 0.25, 0.1])[0]
        
        # Ingresos correlacionados con educación y edad
        if educacion == "Primaria":
            ingresos_base = np.random.normal(1_500_000, 500_000)
        elif educacion == "Secundaria":
            ingresos_base = np.random.normal(2_200_000, 800_000)
        elif educacion == "Técnico":
            ingresos_base = np.random.normal(3_500_000, 1_200_000)
        elif educacion == "Universitario":
            ingresos_base = np.random.normal(5_500_000, 2_000_000)
        else:  # Posgrado
            ingresos_base = np.random.normal(8_500_000, 3_000_000)
        
        # Ajuste por edad (experiencia)
        if edad > 35:
            ingresos_base *= random.uniform(1.1, 1.4)
        elif edad < 25:
            ingresos_base *= random.uniform(0.7, 0.9)
        
        ingresos = max(0, round(ingresos_base, -3))
        
        # Deudas correlacionadas con ingresos
        if ingresos > 0:
            dti_target = random.uniform(0.1, 0.7)
            deudas = round(ingresos * dti_target, -3)
        else:
            deudas = 0
        
        # Tipo de contrato correlacionado con educación
        if educacion in ["Universitario", "Posgrado"]:
            contrato = random.choices(
                ["Indefinido", "Fijo", "Servicios"],
                weights=[0.6, 0.3, 0.1])[0]
        else:
            contrato = random.choices(
                ["Indefinido", "Fijo", "Servicios", "Informal"],
                weights=[0.3, 0.3, 0.2, 0.2])[0]
        
        # Antigüedad laboral
        if edad < 25:
            antiguedad = random.randint(1, 36)
        else:
            antiguedad = random.randint(6, min(240, (edad - 18) * 12))
        
        # Score crediticio correlacionado con perfil
        score_base = 650
        if educacion in ["Universitario", "Posgrado"]: score_base += 30
        if contrato in ["Indefinido", "Fijo"]: score_base += 20
        if antiguedad > 24: score_base += 15
        if ingresos > 5_000_000: score_base += 25
        
        score = max(300, min(850, int(np.random.normal(score_base, 60))))
        
        # Historial correlacionado con score
        historial = 1 if score > 580 and random.random() > 0.2 else 0
        
        # Monto solicitado correlacionado con ingresos
        if ingresos > 0:
            monto_factor = random.uniform(0.5, 4.0)
            monto_base = ingresos * monto_factor / 12  # Múltiplo de ingreso mensual
        else:
            monto_base = random.uniform(500_000, 2_000_000)
        
        monto = round(max(500_000, monto_base), -5)
        
        # Región con sesgo hacia ciudades principales
        region = random.choices(regiones_colombia, 
                               weights=[0.15, 0.08, 0.06, 0.04, 0.04, 0.03] + 
                                      [0.02] * 22)[0]
        
        # Producto correlacionado con monto y perfil
        if monto > 20_000_000:
            producto = random.choices(["Crédito", "Leasing"], weights=[0.7, 0.3])[0]
        elif monto < 2_000_000:
            producto = random.choices(["Tarjeta", "Crédito"], weights=[0.6, 0.4])[0]
        else:
            producto = random.choice(["Crédito", "Tarjeta", "Leasing"])
        
        # Calcular DTI final
        dti = round(deudas / ingresos, 3) if ingresos > 0 else 1.0
        
        # Calcular riesgo con función avanzada
        riesgo = calcular_riesgo_avanzado(
            edad, ingresos, deudas, monto, score, historial,
            educacion, contrato, region, antiguedad, genero
        )
        
        data.append({
            "id_cliente": f"CLI_{i+1:04d}",
            "edad": edad,
            "genero": genero,
            "ingresos": ingresos,
            "deudas": deudas,
            "monto_solicitado": monto,
            "historial_crediticio": historial,
            "nivel_educativo": educacion,
            "producto": producto,
            "region": region,
            "tipo_contrato": contrato,
            "antiguedad_laboral_meses": antiguedad,
            "score_crediticio": score,
            "dti": dti,
            "riesgo_crediticio": riesgo
        })
    
    return pd.DataFrame(data)

# ==============================================================
# 3. RED NEURONAL MEJORADA CON ARQUITECTURA AVANZADA
# ==============================================================

def entrenar_modelo_avanzado(df):
    """Entrena un modelo de red neuronal más sofisticado"""
    
    print("🧠 Entrenando modelo de red neuronal avanzado...")
    
    # Preparar datos
    X = df.drop(["riesgo_crediticio", "id_cliente"], axis=1)
    y = df["riesgo_crediticio"]
    
    # Definir columnas
    numeric_features = ["edad", "ingresos", "deudas", "monto_solicitado",
                       "antiguedad_laboral_meses", "score_crediticio",
                       "dti", "historial_crediticio"]
    categorical_features = ["genero", "nivel_educativo", "producto",
                           "region", "tipo_contrato"]
    
    # Preprocesamiento mejorado
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), 
         categorical_features)
    ])
    
    # Transformar datos
    X_processed = preprocessor.fit_transform(X)
    feature_names = (numeric_features + 
                    list(preprocessor.named_transformers_['cat']
                        .get_feature_names_out(categorical_features)))
    
    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=3)
    )
    
    # Modelo más sofisticado
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_processed.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    # Compilar con optimizador mejorado
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',  # Más robusto a outliers
        metrics=['mae', 'mse']
    )
    
    # Callbacks mejorados
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
    ]
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluación
    y_pred = model.predict(X_test, verbose=0).ravel()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Métricas del modelo:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²:  {r2:.4f}")
    
    return model, preprocessor, history, {
        'mse': mse, 'mae': mae, 'r2': r2,
        'y_test': y_test, 'y_pred': y_pred
    }

# ==============================================================
# 4. SISTEMA DE COMPETENCIA MEJORADO
# ==============================================================

class CompetidorInteligente:
    """Clase para modelar competidores con estrategias diferenciadas"""
    
    def __init__(self, nombre, perfil_agresividad, especialidad=None):
        self.nombre = nombre
        self.agresividad = perfil_agresividad  # -0.02 a 0.02
        self.especialidad = especialidad  # "bajo_riesgo", "alto_monto", etc.
        self.historico_decisiones = []
    
    def calcular_tasa(self, cliente, mercado_promedio):
        """Calcula tasa considerando perfil del competidor"""
        riesgo = cliente["riesgo_crediticio"]
        monto = cliente["monto_solicitado"]
        
        # Tasa base por riesgo
        if riesgo <= 3:
            tasa_base = random.uniform(0.10, 0.15)
        elif riesgo <= 6:
            tasa_base = random.uniform(0.15, 0.22)
        else:
            tasa_base = random.uniform(0.22, 0.32)
        
        # Ajuste por agresividad del competidor
        tasa_base += self.agresividad
        
        # Especialidades
        if self.especialidad == "bajo_riesgo" and riesgo <= 4:
            tasa_base -= 0.01  # Más agresivo en bajo riesgo
        elif self.especialidad == "alto_monto" and monto > mercado_promedio:
            tasa_base -= 0.008  # Compite mejor en montos altos
        elif self.especialidad == "masivo" and 4 <= riesgo <= 7:
            tasa_base -= 0.005  # Enfoque en mercado masivo
        
        # Variabilidad aleatoria
        tasa_base += random.uniform(-0.005, 0.005)
        
        # Límites realistas
        tasa_final = max(0.08, min(0.35, tasa_base))
        
        # Registrar decisión
        self.historico_decisiones.append({
            'cliente_riesgo': riesgo,
            'monto': monto,
            'tasa_ofrecida': tasa_final
        })
        
        return round(tasa_final, 3)

def generar_competencia_inteligente(df):
    """Genera competencia con perfiles diferenciados"""
    
    # Crear competidores con perfiles únicos
    competidores = {
        "aurus": CompetidorInteligente("Aurus", -0.008, "bajo_riesgo"),    # Agresivo en bajo riesgo
        "alpha": CompetidorInteligente("Alpha", 0.000, "masivo"),          # Balanceado, mercado masivo
        "beta": CompetidorInteligente("Beta", 0.005, None),                # Conservador
        "omega": CompetidorInteligente("Omega", -0.004, "alto_monto")      # Agresivo en montos altos
    }
    
    promedio_montos = df["monto_solicitado"].mean()
    
    print("🏦 Generando ofertas de competidores inteligentes...")
    
    for nombre, competidor in competidores.items():
        df[f"tasa_{nombre}"] = df.apply(
            lambda row: competidor.calcular_tasa(row, promedio_montos), axis=1
        )
    
    return competidores

# ==============================================================
# 5. SISTEMA MINIMAX UNIFICADO Y OPTIMIZADO
# ==============================================================

class EstrategiaFinFlow:
    """Sistema unificado de estrategia competitiva para FinFlow"""
    
    def __init__(self, df):
        self.df = df
        self.promedio_montos = df["monto_solicitado"].mean()
        self.COSTO_FONDEO = 0.08
        self.COSTO_OPER = 0.015
        self.LGD = 0.65
        self.MIN_TASA = 0.085
        self.MAX_TASA = 0.32
        self.EPSILON_UNDERCUT = 0.001
        
        # Mapeo PD mejorado
        self.PD_MAP = {
            1: 0.005, 2: 0.01, 3: 0.02, 4: 0.035, 5: 0.055,
            6: 0.08, 7: 0.12, 8: 0.17, 9: 0.24, 10: 0.32
        }
    
    def es_cliente_peligroso(self, riesgo, monto):
        """Identifica clientes que pueden generar pérdidas"""
        return (riesgo >= 8) and (monto > self.promedio_montos)
    
    def utilidad_esperada(self, tasa, monto, riesgo):
        """Calcula utilidad esperada mejorada"""
        if riesgo not in self.PD_MAP:
            pd = 0.20  # Fallback conservador
        else:
            pd = self.PD_MAP[riesgo]
        
        margen_bruto = tasa - self.COSTO_FONDEO - self.COSTO_OPER
        perdida_esperada = pd * self.LGD * monto
        utilidad = (margen_bruto * monto) - perdida_esperada
        
        return utilidad
    
    def mejor_tasa_rival(self, cliente):
        """Obtiene la mejor oferta de la competencia"""
        tasas_rivales = [
            cliente[f"tasa_{comp}"] for comp in ["aurus", "alpha", "beta", "omega"]
        ]
        return min(tasas_rivales) - self.EPSILON_UNDERCUT
    
    def generar_candidatas_inteligentes(self, cliente, n=15):
        """Genera candidatas más inteligentes basadas en análisis de mercado"""
        tasa_rival = self.mejor_tasa_rival(cliente)
        riesgo = cliente["riesgo_crediticio"]
        
        # Rango adaptativo según riesgo
        if riesgo <= 3:
            margen = 0.025  # Más competitivo en bajo riesgo
        elif riesgo <= 6:
            margen = 0.02
        else:
            margen = 0.015   # Menos agresivo en alto riesgo
        
        tasa_min = max(self.MIN_TASA, tasa_rival - margen)
        tasa_max = min(self.MAX_TASA, tasa_rival + margen)
        
        if tasa_max <= tasa_min:
            tasa_max = tasa_min + 0.01
        
        return np.linspace(tasa_min, tasa_max, n)
    
    def gana_cliente(self, tasa_finflow, tasa_rival):
        """Determina si FinFlow gana el cliente"""
        return tasa_finflow <= (tasa_rival - 0.0003)  # Pequeña ventaja requerida
    
    def decidir_cliente_basico(self, cliente):
        """Estrategia Minimax básica mejorada"""
        riesgo = cliente["riesgo_crediticio"]
        monto = cliente["monto_solicitado"]
        
        # Evitar clientes peligrosos
        if self.es_cliente_peligroso(riesgo, monto):
            return {
                "tasa_finflow": None, "gana": False, "utilidad": 0.0,
                "motivo": "cliente_peligroso", "estrategia": "basico"
            }
        
        tasa_rival = self.mejor_tasa_rival(cliente)
        candidatas = self.generar_candidatas_inteligentes(cliente)
        
        mejor_decision = {
            "tasa_finflow": None, "gana": False, "utilidad": -1e9,
            "motivo": "no_viable", "estrategia": "basico"
        }
        
        for tasa in candidatas:
            if self.gana_cliente(tasa, tasa_rival):
                utilidad = self.utilidad_esperada(tasa, monto, riesgo)
                
                if utilidad > mejor_decision["utilidad"]:
                    mejor_decision = {
                        "tasa_finflow": round(float(tasa), 3),
                        "gana": True,
                        "utilidad": float(utilidad),
                        "motivo": "optimo",
                        "estrategia": "basico"
                    }
        
        # Si no hay decisión rentable, pasar
        if mejor_decision["utilidad"] <= 0:
            return {
                "tasa_finflow": None, "gana": False, "utilidad": 0.0,
                "motivo": "no_rentable", "estrategia": "basico"
            }
        
        return mejor_decision
    
    def minimax_alpha_beta_avanzado(self, clientes, profundidad, alfa, beta, 
                                   es_maximizador, indice=0):
        """Minimax con poda Alfa-Beta optimizado"""
        
        # Casos base
        if profundidad == 0 or indice >= len(clientes):
            return 0, []
        
        cliente_actual = clientes[indice]
        
        if es_maximizador:  # Turno FinFlow
            max_eval = float("-inf")
            mejor_secuencia = []
            
            # Evitar clientes peligrosos
            if self.es_cliente_peligroso(cliente_actual["riesgo_crediticio"], 
                                       cliente_actual["monto_solicitado"]):
                return self.minimax_alpha_beta_avanzado(
                    clientes, profundidad-1, alfa, beta, False, indice+1
                )[0], [None]
            
            candidatas = self.generar_candidatas_inteligentes(cliente_actual, n=8)
            
            for tasa in candidatas:
                tasa_rival = self.mejor_tasa_rival(cliente_actual)
                
                if self.gana_cliente(tasa, tasa_rival):
                    utilidad_actual = self.utilidad_esperada(
                        tasa, cliente_actual["monto_solicitado"], 
                        cliente_actual["riesgo_crediticio"]
                    )
                else:
                    utilidad_actual = 0
                
                # Recursión
                eval_futuro, secuencia_futura = self.minimax_alpha_beta_avanzado(
                    clientes, profundidad-1, alfa, beta, False, indice+1
                )
                
                eval_total = utilidad_actual + eval_futuro
                
                if eval_total > max_eval:
                    max_eval = eval_total
                    mejor_secuencia = [tasa] + secuencia_futura
                
                alfa = max(alfa, eval_total)
                if beta <= alfa:
                    break  # Poda
            
            return max_eval, mejor_secuencia
        
        else:  # Turno rival
            min_eval = float("inf")
            
            # Rival puede ajustar estrategia ligeramente
            ajustes = [-0.003, 0.0, 0.003]
            
            for ajuste in ajustes:
                eval_futuro, _ = self.minimax_alpha_beta_avanzado(
                    clientes, profundidad-1, alfa, beta, True, indice+1
                )
                
                min_eval = min(min_eval, eval_futuro)
                beta = min(beta, eval_futuro)
                if beta <= alfa:
                    break  # Poda
            
            return min_eval, []
    
    def ejecutar_estrategia_completa(self):
        """Ejecuta ambas estrategias y compara resultados"""
        
        print("🎯 Ejecutando análisis estratégico completo...")
        
        # Estrategia básica para todo el dataset
        print("   → Ejecutando Minimax básico...")
        decisiones_basicas = self.df.apply(self.decidir_cliente_basico, axis=1, result_type="expand")
        
        # Estrategia avanzada en muestra
        print("   → Ejecutando Minimax Alfa-Beta...")
        clientes_muestra = self.df.sample(min(20, len(self.df)), random_state=42).to_dict('records')
        
        start_time = time.time()
        utilidad_ab, secuencia_ab = self.minimax_alpha_beta_avanzado(
            clientes_muestra, profundidad=3, alfa=float("-inf"), beta=float("inf"), 
            es_maximizador=True
        )
        tiempo_ab = time.time() - start_time
        
        # Aplicar decisiones básicas al DataFrame
        self.df["tasa_finflow"] = decisiones_basicas["tasa_finflow"]
        self.df["finflow_gana"] = decisiones_basicas["gana"]
        self.df["utilidad_finflow"] = decisiones_basicas["utilidad"]
        self.df["motivo_decision"] = decisiones_basicas["motivo"]
        self.df["estrategia"] = decisiones_basicas["estrategia"]
        
        # Estadísticas
        n_ganados = int(self.df["finflow_gana"].sum())
        util_total = float(self.df["utilidad_finflow"].sum())
        n_pasados = int(self.df["tasa_finflow"].isna().sum())
        
        # Comparar con muestra equivalente en básico
        utilidad_basico_muestra = sum([
            self.decidir_cliente_basico(cliente)["utilidad"] 
            for cliente in clientes_muestra
        ])
        
        return {
            "total": {
                "clientes_ganados": n_ganados,
                "total_clientes": len(self.df),
                "utilidad_total": util_total,
                "clientes_pasados": n_pasados,
                "tasa_exito": n_ganados / len(self.df)
            },
            "comparacion_algoritmos": {
                "muestra_size": len(clientes_muestra),
                "utilidad_basico": utilidad_basico_muestra,
                "utilidad_alpha_beta": utilidad_ab,
                "tiempo_alpha_beta": tiempo_ab,
                "ventaja_alpha_beta": utilidad_ab - utilidad_basico_muestra
            }
        }

# ==============================================================
# 6. ANÁLISIS Y VISUALIZACIÓN AVANZADA
# ==============================================================

def generar_analisis_completo(df, modelo_nn, resultados_estrategia, competidores):
    """Genera análisis completo del sistema"""
    
    print("\n" + "="*80)
    print("📊 ANÁLISIS COMPLETO DEL SISTEMA FINFLOW")
    print("="*80)
    
    # === ANÁLISIS DE DATOS ===
    print(f"\n📋 RESUMEN DEL DATASET:")
    print(f"   Total de clientes: {len(df):,}")
    print(f"   Riesgo promedio: {df['riesgo_crediticio'].mean():.2f}")
    print(f"   Monto promedio: ${df['monto_solicitado'].mean():,.0f}")
    print(f"   Score promedio: {df['score_crediticio'].mean():.0f}")
    
    # Distribución por riesgo
    print(f"\n🎯 DISTRIBUCIÓN POR NIVEL DE RIESGO:")
    riesgo_dist = df['riesgo_crediticio'].value_counts().sort_index()
    for riesgo, count in riesgo_dist.items():
        porcentaje = (count / len(df)) * 100
        print(f"   Riesgo {riesgo}: {count:3d} clientes ({porcentaje:5.1f}%)")
    
    # === ANÁLISIS DE COMPETENCIA ===
    print(f"\n🏦 ANÁLISIS DE COMPETENCIA:")
    for nombre, competidor in competidores.items():
        if competidor.historico_decisiones:
            tasas = [d['tasa_ofrecida'] for d in competidor.historico_decisiones]
            print(f"   {competidor.nombre}:")
            print(f"      Tasa promedio: {np.mean(tasas):.3f} ({np.mean(tasas)*100:.1f}%)")
            print(f"      Rango: {min(tasas):.3f} - {max(tasas):.3f}")
            print(f"      Especialidad: {competidor.especialidad or 'General'}")
    
    # === RESULTADOS ESTRATÉGICOS ===
    print(f"\n🎯 RESULTADOS ESTRATÉGICOS:")
    total_results = resultados_estrategia["total"]
    print(f"   Clientes ganados: {total_results['clientes_ganados']:,} / {total_results['total_clientes']:,}")
    print(f"   Tasa de éxito: {total_results['tasa_exito']:.1%}")
    print(f"   Utilidad total: ${total_results['utilidad_total']:,.0f}")
    print(f"   Clientes rechazados: {total_results['clientes_pasados']:,}")
    
    # Comparación algoritmos
    comp_results = resultados_estrategia["comparacion_algoritmos"]
    print(f"\n⚡ COMPARACIÓN DE ALGORITMOS (muestra de {comp_results['muestra_size']} clientes):")
    print(f"   Minimax Básico: ${comp_results['utilidad_basico']:,.0f}")
    print(f"   Minimax Alfa-Beta: ${comp_results['utilidad_alpha_beta']:,.0f}")
    print(f"   Ventaja Alfa-Beta: ${comp_results['ventaja_alpha_beta']:,.0f}")
    print(f"   Tiempo Alfa-Beta: {comp_results['tiempo_alpha_beta']:.3f}s")
    
    mejora_porcentual = (comp_results['ventaja_alpha_beta'] / abs(comp_results['utilidad_basico'])) * 100 if comp_results['utilidad_basico'] != 0 else 0
    if mejora_porcentual > 0:
        print(f"   ✅ Alfa-Beta es {mejora_porcentual:.1f}% mejor")
    else:
        print(f"   ⚠️  Alfa-Beta es {abs(mejora_porcentual):.1f}% menor")
    
    # === ANÁLISIS POR SEGMENTOS ===
    print(f"\n📈 ANÁLISIS POR SEGMENTOS:")
    
    # Por nivel de riesgo
    for riesgo_nivel in ["Bajo (1-3)", "Medio (4-6)", "Alto (7-10)"]:
        if riesgo_nivel == "Bajo (1-3)":
            mask = df['riesgo_crediticio'] <= 3
        elif riesgo_nivel == "Medio (4-6)":
            mask = (df['riesgo_crediticio'] >= 4) & (df['riesgo_crediticio'] <= 6)
        else:
            mask = df['riesgo_crediticio'] >= 7
        
        segmento = df[mask]
        if len(segmento) > 0:
            ganados = segmento['finflow_gana'].sum()
            utilidad_seg = segmento['utilidad_finflow'].sum()
            print(f"   {riesgo_nivel}: {ganados}/{len(segmento)} ganados, ${utilidad_seg:,.0f} utilidad")

def crear_visualizaciones_avanzadas(df, historia_modelo, metricas_modelo):
    """Crea visualizaciones mejoradas del sistema"""
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    
    # === SUBPLOT 1: Distribución de riesgo ===
    ax1 = plt.subplot(3, 4, 1)
    riesgo_counts = df['riesgo_crediticio'].value_counts().sort_index()
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(riesgo_counts)))
    bars = ax1.bar(riesgo_counts.index, riesgo_counts.values, color=colors)
    ax1.set_title('Distribución de Riesgo Crediticio', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Nivel de Riesgo')
    ax1.set_ylabel('Número de Clientes')
    ax1.grid(alpha=0.3)
    
    # Agregar etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # === SUBPLOT 2: Performance del modelo ===
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(historia_modelo.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(historia_modelo.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Entrenamiento Red Neuronal', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # === SUBPLOT 3: Predicciones vs Real ===
    ax3 = plt.subplot(3, 4, 3)
    scatter = ax3.scatter(metricas_modelo['y_test'], metricas_modelo['y_pred'], 
                         alpha=0.6, c=metricas_modelo['y_test'], cmap='viridis')
    ax3.plot([1, 10], [1, 10], 'r--', linewidth=2, label='Predicción Perfecta')
    ax3.set_title(f'Predicciones vs Real (R²={metricas_modelo["r2"]:.3f})', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Riesgo Real')
    ax3.set_ylabel('Riesgo Predicho')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Riesgo Real')
    
    # === SUBPLOT 4: Distribución de montos ===
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(df['monto_solicitado'] / 1e6, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(df['monto_solicitado'].mean() / 1e6, color='red', linestyle='--', 
               linewidth=2, label=f'Promedio: ${df["monto_solicitado"].mean()/1e6:.1f}M')
    ax4.set_title('Distribución de Montos Solicitados', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Monto (Millones COP)')
    ax4.set_ylabel('Frecuencia')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # === SUBPLOT 5: Tasas por competidor ===
    ax5 = plt.subplot(3, 4, 5)
    competidores = ['aurus', 'alpha', 'beta', 'omega']
    tasas_promedio = [df[f'tasa_{comp}'].mean() * 100 for comp in competidores]
    colors_comp = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax5.bar(competidores, tasas_promedio, color=colors_comp)
    ax5.set_title('Tasas Promedio por Competidor', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Competidor')
    ax5.set_ylabel('Tasa Promedio (%)')
    ax5.grid(alpha=0.3)
    
    for bar, tasa in zip(bars, tasas_promedio):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{tasa:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # === SUBPLOT 6: Utilidad por nivel de riesgo ===
    ax6 = plt.subplot(3, 4, 6)
    utilidad_por_riesgo = df.groupby('riesgo_crediticio')['utilidad_finflow'].sum()
    colors_util = ['green' if x > 0 else 'red' for x in utilidad_por_riesgo.values]
    bars = ax6.bar(utilidad_por_riesgo.index, utilidad_por_riesgo.values / 1e6, color=colors_util, alpha=0.7)
    ax6.set_title('Utilidad por Nivel de Riesgo', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Nivel de Riesgo')
    ax6.set_ylabel('Utilidad (Millones COP)')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.grid(alpha=0.3)
    
    # === SUBPLOT 7: Clientes ganados vs perdidos ===
    ax7 = plt.subplot(3, 4, 7)
    ganados = df['finflow_gana'].sum()
    perdidos = len(df) - ganados
    sizes = [ganados, perdidos]
    labels = [f'Ganados\n{ganados:,}', f'Perdidos\n{perdidos:,}']
    colors_pie = ['#2ECC71', '#E74C3C']
    wedges, texts, autotexts = ax7.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax7.set_title('Clientes Ganados vs Perdidos', fontsize=12, fontweight='bold')
    
    # === SUBPLOT 8: Correlación riesgo-utilidad ===
    ax8 = plt.subplot(3, 4, 8)
    clientes_ganados = df[df['finflow_gana'] == True]
    if len(clientes_ganados) > 0:
        scatter = ax8.scatter(clientes_ganados['riesgo_crediticio'], 
                             clientes_ganados['utilidad_finflow'] / 1e3,
                             alpha=0.6, c=clientes_ganados['monto_solicitado'], 
                             cmap='plasma', s=30)
        ax8.set_title('Riesgo vs Utilidad (Clientes Ganados)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Nivel de Riesgo')
        ax8.set_ylabel('Utilidad (Miles COP)')
        ax8.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax8, label='Monto Solicitado')
    
    # === SUBPLOT 9: Distribución de DTI ===
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(df['dti'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax9.axvline(df['dti'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Promedio: {df["dti"].mean():.2f}')
    ax9.set_title('Distribución Debt-to-Income (DTI)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('DTI Ratio')
    ax9.set_ylabel('Frecuencia')
    ax9.legend()
    ax9.grid(alpha=0.3)
    
    # === SUBPLOT 10: Tasas FinFlow vs Competencia ===
    ax10 = plt.subplot(3, 4, 10)
    clientes_con_tasa = df[df['tasa_finflow'].notna()]
    if len(clientes_con_tasa) > 0:
        tasa_finflow_promedio = clientes_con_tasa.groupby('riesgo_crediticio')['tasa_finflow'].mean() * 100
        tasa_competencia_promedio = clientes_con_tasa.groupby('riesgo_crediticio')[
            ['tasa_aurus', 'tasa_alpha', 'tasa_beta', 'tasa_omega']].mean().mean(axis=1) * 100
        
        x_pos = tasa_finflow_promedio.index
        width = 0.35
        ax10.bar(x_pos - width/2, tasa_finflow_promedio.values, width, 
                label='FinFlow', color='#3498DB', alpha=0.8)
        ax10.bar(x_pos + width/2, tasa_competencia_promedio.values, width, 
                label='Competencia (Prom)', color='#E67E22', alpha=0.8)
        
        ax10.set_title('Tasas: FinFlow vs Competencia', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Nivel de Riesgo')
        ax10.set_ylabel('Tasa Promedio (%)')
        ax10.legend()
        ax10.grid(alpha=0.3)
    
    # === SUBPLOT 11: ROI por segmento ===
    ax11 = plt.subplot(3, 4, 11)
    segmentos = ['Bajo (1-3)', 'Medio (4-6)', 'Alto (7-10)']
    roi_valores = []
    
    for segmento in segmentos:
        if segmento == 'Bajo (1-3)':
            mask = df['riesgo_crediticio'] <= 3
        elif segmento == 'Medio (4-6)':
            mask = (df['riesgo_crediticio'] >= 4) & (df['riesgo_crediticio'] <= 6)
        else:
            mask = df['riesgo_crediticio'] >= 7
        
        seg_data = df[mask]
        if len(seg_data) > 0:
            utilidad_seg = seg_data['utilidad_finflow'].sum()
            inversion_seg = seg_data[seg_data['finflow_gana']]['monto_solicitado'].sum() * 0.1  # 10% como proxy de inversión
            roi = (utilidad_seg / inversion_seg * 100) if inversion_seg > 0 else 0
            roi_valores.append(roi)
        else:
            roi_valores.append(0)
    
    colors_roi = ['green' if x > 0 else 'red' for x in roi_valores]
    bars = ax11.bar(segmentos, roi_valores, color=colors_roi, alpha=0.7)
    ax11.set_title('ROI por Segmento de Riesgo', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Segmento')
    ax11.set_ylabel('ROI (%)')
    ax11.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax11.grid(alpha=0.3)
    
    for bar, roi in zip(bars, roi_valores):
        ax11.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (1 if roi > 0 else -3),
                 f'{roi:.1f}%', ha='center', va='bottom' if roi > 0 else 'top', fontweight='bold')
    
    # === SUBPLOT 12: Resumen ejecutivo ===
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Estadísticas clave
    total_clientes = len(df)
    clientes_ganados = df['finflow_gana'].sum()
    utilidad_total = df['utilidad_finflow'].sum()
    tasa_exito = (clientes_ganados / total_clientes) * 100
    
    resumen_text = f"""
RESUMEN EJECUTIVO FINFLOW

📊 Clientes Totales: {total_clientes:,}
✅ Clientes Ganados: {clientes_ganados:,}
📈 Tasa de Éxito: {tasa_exito:.1f}%
💰 Utilidad Total: ${utilidad_total:,.0f}

🎯 Top Insights:
• Mejor segmento: Riesgo 1-3
• Modelo R²: {metricas_modelo["r2"]:.3f}
• Competencia: 4 fintechs activas
• Estrategia: Minimax + Alfa-Beta

⚡ Ventajas del Sistema:
• Evaluación automática de riesgo
• Estrategia competitiva inteligente
• Optimización en tiempo real
• Evita clientes peligrosos
"""
    
    ax12.text(0.05, 0.95, resumen_text, transform=ax12.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    plt.savefig("finflow_analisis_completo.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualizaciones guardadas como 'finflow_analisis_completo.png'")
    
    return fig

# ==============================================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================

def ejecutar_sistema_completo():
    """Función principal que ejecuta todo el sistema mejorado"""
    
    print("🚀 INICIANDO SISTEMA FINFLOW")
    print("="*80)
    
    # 1. Generar dataset avanzado
    print("📊 Generando dataset avanzado...")
    df = generar_dataset_avanzado(n_samples=1000)
    
    # 2. Entrenar modelo de red neuronal
    modelo, preprocessor, historia, metricas = entrenar_modelo_avanzado(df)
    
    # 3. Generar competencia inteligente
    competidores = generar_competencia_inteligente(df)
    
    # 4. Ejecutar estrategia competitiva
    estrategia = EstrategiaFinFlow(df)
    resultados = estrategia.ejecutar_estrategia_completa()
    
    # 5. Análisis completo
    generar_analisis_completo(df, modelo, resultados, competidores)
    
    # 6. Visualizaciones
    crear_visualizaciones_avanzadas(df, historia, metricas)
    
    # 7. Recomendaciones estratégicas
    print(f"\n🎯 RECOMENDACIONES ESTRATÉGICAS:")
    print(f"   1. Enfocarse en clientes de riesgo 1-4 con montos altos")
    print(f"   2. Evitar clientes riesgo ≥8 con monto > promedio")
    print(f"   3. Usar Alfa-Beta para decisiones en tiempo real")
    print(f"   4. Monitorear estrategias de Aurus (más agresivo)")
    print(f"   5. Considerar especialización en segmentos específicos")
    
    return df, modelo, resultados, competidores

# ==============================================================
# EJECUCIÓN DEL SISTEMA
# ==============================================================

if __name__ == "__main__":
    # Ejecutar sistema completo
    df_final, modelo_final, resultados_finales, competidores_finales = ejecutar_sistema_completo()
    
    # Mostrar muestra de resultados
    print(f"\n📋 MUESTRA DE RESULTADOS:")
    columnas_mostrar = ['id_cliente', 'riesgo_crediticio', 'monto_solicitado', 
                       'tasa_finflow', 'finflow_gana', 'utilidad_finflow', 'motivo_decision']
    print(df_final[columnas_mostrar].head(10).to_string(index=False))
    
    print(f"\n✅ SISTEMA FINFLOW EJECUTADO EXITOSAMENTE")
    print(f"📈 Total utilidad proyectada: ${df_final['utilidad_finflow'].sum():,.0f}")
    print(f"🎯 Clientes objetivo logrados: {df_final['finflow_gana'].sum()}/{len(df_final)}")