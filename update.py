import pickle
from datetime import datetime
import hashlib

class MinimaxConMemoria(EstrategiaFinFlow):
    """Extensión de Minimax que aprende de experiencias pasadas"""
    
    def __init__(self, df):
        super().__init__(df)
        self.memoria_episodica = {}  # Diccionario de situaciones similares
        self.contador_decisiones = 0
        self.aciertos_memoria = 0
        
    def _generar_clave_situacion(self, cliente):
        """Genera clave única para situaciones similares"""
        # Discretizar variables para agrupación
        riesgo_bucket = min(10, max(1, cliente["riesgo_crediticio"]))
        monto_bucket = int(cliente["monto_solicitado"] / 5_000_000)  # Buckets de 5M
        
        # Crear clave compuesta
        clave = f"r{riesgo_bucket}_m{monto_bucket}"
        return clave
    
    def _buscar_experiencia_similar(self, cliente):
        """Busca experiencias similares en memoria episódica"""
        clave = self._generar_clave_situacion(cliente)
        
        if clave in self.memoria_episodica:
            experiencias = self.memoria_episodica[clave]
            if len(experiencias) >= 3:  # Mínimo 3 experiencias para confiar
                # Calcular promedio de utilidades exitosas
                utilidades_exitosas = [exp["utilidad"] for exp in experiencias if exp["exito"]]
                
                if utilidades_exitosas:
                    return {
                        "tiene_experiencia": True,
                        "utilidad_promedio": np.mean(utilidades_exitosas),
                        "tasa_recomendada": np.mean([exp["tasa"] for exp in experiencias if exp["exito"]]),
                        "probabilidad_exito": len(utilidades_exitosas) / len(experiencias),
                        "num_experiencias": len(experiencias)
                    }
        
        return {"tiene_experiencia": False}
    
    def _guardar_experiencia(self, cliente, tasa_utilizada, utilidad_obtenida, fue_exitoso):
        """Guarda experiencia en memoria episódica"""
        clave = self._generar_clave_situacion(cliente)
        
        if clave not in self.memoria_episodica:
            self.memoria_episodica[clave] = []
        
        experiencia = {
            "timestamp": datetime.now(),
            "tasa": tasa_utilizada,
            "utilidad": utilidad_obtenida,
            "exito": fue_exitoso,
            "riesgo_exacto": cliente["riesgo_crediticio"],
            "monto_exacto": cliente["monto_solicitado"]
        }
        
        self.memoria_episodica[clave].append(experiencia)
        
        # Mantener solo las últimas 20 experiencias por situación
        if len(self.memoria_episodica[clave]) > 20:
            self.memoria_episodica[clave] = self.memoria_episodica[clave][-20:]
    
    def decidir_con_memoria_episodica(self, cliente):
        """Decisión mejorada usando memoria episódica"""
        self.contador_decisiones += 1
        
        # 1. Buscar experiencia similar
        experiencia = self._buscar_experiencia_similar(cliente)
        
        if experiencia["tiene_experiencia"]:
            # Si tenemos experiencia y alta probabilidad de éxito, usar tasa recomendada
            if experiencia["probabilidad_exito"] > 0.7:
                tasa_recomendada = experiencia["tasa_recomendada"]
                tasa_rival = self.mejor_tasa_rival(cliente)
                
                # Verificar si la tasa recomendada sigue siendo competitiva
                if tasa_recomendada <= tasa_rival + 0.002:
                    utilidad_esperada = self.utilidad_esperada(
                        tasa_recomendada, 
                        cliente["monto_solicitado"], 
                        cliente["riesgo_crediticio"]
                    )
                    
                    if utilidad_esperada > 0:
                        self.aciertos_memoria += 1
                        print(f"💡 Usando memoria episódica: {experiencia['num_experiencias']} exp., "
                              f"éxito {experiencia['probabilidad_exito']:.1%}, tasa {tasa_recomendada:.3f}")
                        
                        return {
                            "tasa_finflow": round(float(tasa_recomendada), 3),
                            "gana": True,
                            "utilidad": float(utilidad_esperada),
                            "motivo": "memoria_episodica",
                            "estrategia": "episodica"
                        }
        
        # 2. Si no hay experiencia útil, usar estrategia normal
        decision_normal = self.decidir_cliente_basico(cliente)
        
        # 3. Guardar experiencia para futuro aprendizaje
        if decision_normal["tasa_finflow"] is not None:
            self._guardar_experiencia(
                cliente,
                decision_normal["tasa_finflow"],
                decision_normal["utilidad"],
                decision_normal["gana"]
            )
        
        return decision_normal
    
    def mostrar_estadisticas_memoria(self):
        """Muestra estadísticas del aprendizaje episódico"""
        total_situaciones = len(self.memoria_episodica)
        total_experiencias = sum(len(experiencias) for experiencias in self.memoria_episodica.values())
        
        if self.contador_decisiones > 0:
            tasa_uso_memoria = (self.aciertos_memoria / self.contador_decisiones) * 100
        else:
            tasa_uso_memoria = 0
        
        print(f"\n📚 ESTADÍSTICAS MEMORIA EPISÓDICA:")
        print(f"   Situaciones únicas aprendidas: {total_situaciones}")
        print(f"   Total experiencias almacenadas: {total_experiencias}")
        print(f"   Decisiones usando memoria: {self.aciertos_memoria}/{self.contador_decisiones}")
        print(f"   Tasa de uso de memoria: {tasa_uso_memoria:.1f}%")
        
        # Mostrar top 3 situaciones más exitosas
        if total_situaciones > 0:
            situaciones_exitosas = []
            for clave, experiencias in self.memoria_episodica.items():
                if len(experiencias) >= 3:
                    exitos = sum(1 for exp in experiencias if exp["exito"])
                    tasa_exito = exitos / len(experiencias)
                    utilidad_promedio = np.mean([exp["utilidad"] for exp in experiencias if exp["exito"]])
                    
                    situaciones_exitosas.append({
                        "situacion": clave,
                        "tasa_exito": tasa_exito,
                        "utilidad_promedio": utilidad_promedio,
                        "num_experiencias": len(experiencias)
                    })
            
            situaciones_exitosas.sort(key=lambda x: x["tasa_exito"], reverse=True)
            
            print(f"   Top 3 situaciones más exitosas:")
            for i, sit in enumerate(situaciones_exitosas[:3]):
                print(f"     {i+1}. {sit['situacion']}: {sit['tasa_exito']:.1%} éxito, "
                      f"${sit['utilidad_promedio']:,.0f} utilidad promedio")

# Ejemplo de uso integrado
def ejecutar_estrategia_con_memoria(df):
    """Ejecuta estrategia con memoria episódica"""
    estrategia_memoria = MinimaxConMemoria(df)
    
    print("🧠 Ejecutando estrategia con memoria episódica...")
    
    # Aplicar decisiones con memoria
    decisiones = df.apply(estrategia_memoria.decidir_con_memoria_episodica, axis=1, result_type="expand")
    
    # Actualizar DataFrame
    df["tasa_finflow"] = decisiones["tasa_finflow"]
    df["finflow_gana"] = decisiones["gana"]
    df["utilidad_finflow"] = decisiones["utilidad"]
    df["motivo_decision"] = decisiones["motivo"]
    df["estrategia"] = decisiones["estrategia"]
    
    # Mostrar estadísticas
    estrategia_memoria.mostrar_estadisticas_memoria()
    
    return estrategia_memoria