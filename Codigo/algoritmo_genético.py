import sys
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import folium
import osmnx as ox
import geopandas as gpd  # AGREGAR ESTA LÍNEA SI NO ESTÁ
import shutil  # AGREGAR ESTA LÍNEA SI NO ESTÁ
from generacion_inicial import GeneticAlgorithmRoutes
from folium.plugins import TimestampedGeoJson
from folium import Element
from datetime import datetime, timedelta
from pyproj import Transformer
from shapely.geometry import LineString, Point  # AGREGAR ESTA LÍNEA SI NO ESTÁ

# Fix Unicode encoding for console output
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python 2 or non-standard environment

CONFIG = {
    'MAX_CAPACITY': 9.0,        
    'MAX_SHIFT_TIME': 8.0,       
    'COLLECTION_TIME': 0.0,
    
    # VELOCIDADES CONFIGURABLES (km/h) - NUEVA SECCIÓN
    'SPEED_EMPTY_TO_ZONE': 40.0,      # Vacío: Garage → Primera recolección
    'SPEED_COLLECTING': 10.0,          # Recolectando en zona delimitada
    'SPEED_FULL_TO_LANDFILL': 30.0,   # Lleno: Última recolección → Relleno
    'SPEED_RETURN_EMPTY': 40.0,       # Vacío: Relleno → Garage
    
    # COMBUSTIBLE - NUEVA SECCIÓN
    'KM_PER_GALLON': 5.0,   # Galones por kilómetro
    'FUEL_PRICE_PER_GALLON': 2.70,    # Dólares por galón
    
    'POPULATION_SIZE': 60,
    'GENERATIONS': 400,
    'MUTATION_PROBABILITY': 0.25,
    'TOURNAMENT_SIZE': 5,
    
    'GARAGE_NODE': None,       
    'LANDFILL_NODE': None,
    
    'MAX_BASURA_SUBGRAFO': 57.59
}

class VRPSystem:
    def __init__(self, gpkg_file="cuenca_limpieza.gpkg"):
        print("INICIALIZANDO SISTEMA VRP DESDE GEOPACKAGE...")
        
        # ====================================================================
        # CARGAR GRAFO COMPLETO DESDE GEOPACKAGE
        # ====================================================================
        try:
            import geopandas as gpd
            
            print(f"\nCargando capas desde {gpkg_file}...")
            
            # Cargar nodos y aristas
            gdf_nodes = gpd.read_file(gpkg_file, layer='all_nodes')
            gdf_edges = gpd.read_file(gpkg_file, layer='all_edges')
            
            print(f"  [OK] Nodos cargados: {len(gdf_nodes)}")
            print(f"  [OK] Aristas cargadas: {len(gdf_edges)}")
            
            # Preparar índices para reconstrucción del grafo
            gdf_nodes = gdf_nodes.set_index('osmid')
            
            # Asegurar que las aristas tienen la columna 'key'
            if 'key' not in gdf_edges.columns:
                gdf_edges['key'] = 0
            
            # Configurar multi-index para aristas
            if 'u' in gdf_edges.columns and 'v' in gdf_edges.columns:
                gdf_edges = gdf_edges.set_index(['u', 'v', 'key'])
            
            # Reconstruir el grafo NetworkX
            self.city_graph = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
            
            print(f"  [OK] Grafo reconstruido: {len(self.city_graph.nodes)} nodos, "
                f"{len(self.city_graph.edges)} aristas")
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el GPKG: {e}")
            raise
        
        # ====================================================================
        # EXTRAER INFORMACIÓN DE NODOS ESPECIALES
        # ====================================================================
        
        # Obtener nodos clientes (con basura)
        self.customers = gdf_nodes[gdf_nodes['node_type'] == 'customer'].index.tolist()
        print(f"  [OK] Clientes (nodos con basura): {len(self.customers)}")
        
        # Obtener garage y landfill
        garage_nodes = gdf_nodes[gdf_nodes['node_type'] == 'garage'].index.tolist()
        landfill_nodes = gdf_nodes[gdf_nodes['node_type'] == 'landfill'].index.tolist()
        
        if not garage_nodes:
            raise ValueError("No se encontró el nodo 'garage' en el GPKG")
        if not landfill_nodes:
            raise ValueError("No se encontró el nodo 'landfill' en el GPKG")
        
        self.garage = garage_nodes[0]
        self.landfill = landfill_nodes[0]
        
        print(f"  [OK] Garage Node ID: {self.garage}")
        print(f"  [OK] Landfill Node ID: {self.landfill}")
        
        # Verificación básica
        if self.garage == self.landfill:
            print("[ALERTA] Garage y Landfill son el mismo nodo!")
        
        # ====================================================================
        # CREAR SUBGRAFO PARA VALIDACIÓN DE RUTAS
        # ====================================================================
        
        # Crear subgrafo con los nodos clientes + garage + landfill
        subgraph_nodes = set(self.customers + [self.garage, self.landfill])
        
        # Filtrar nodos que existen en el grafo principal
        valid_subgraph_nodes = [n for n in subgraph_nodes if n in self.city_graph.nodes]
        
        # Crear el subgrafo
        self.subgraph = self.city_graph.subgraph(valid_subgraph_nodes).copy()
        
        print(f"  [OK] Subgrafo creado: {len(self.subgraph.nodes)} nodos, "
            f"{len(self.subgraph.edges)} aristas")
        
        # ====================================================================
        # INICIALIZAR HELPER PARA VALIDACIÓN (OPCIONAL)
        # ====================================================================
        
        # Crear instancia de GeneticAlgorithmRoutes para usar sus métodos de validación
        self.ga_helper = GeneticAlgorithmRoutes(graph_file="cuenca_graph.graphml")
        self.ga_helper.graph = self.city_graph
        self.ga_helper.subgraph = self.subgraph

        # ====================================================================
        # PRECALCULAR MATRIZ DE DISTANCIAS (CRÍTICO PARA RENDIMIENTO)
        # ====================================================================
        self.precompute_distance_matrix()
        
        print("\n[OK] Sistema VRP inicializado correctamente desde GPKG\n")
    
    def precompute_distance_matrix(self):
        """
        Precalcula las distancias entre todos los nodos relevantes (garage, landfill, customers).
        Esto acelera enormemente las operaciones posteriores.
        """
        import time
        start_time = time.time()
        
        print("\n[PRECALCULANDO MATRIZ DE DISTANCIAS...]")
        
        # Nodos relevantes: garage, landfill, y todos los customers
        relevant_nodes = [self.garage, self.landfill] + self.customers
        total_nodes = len(relevant_nodes)
        
        print(f"  Calculando distancias para {total_nodes} nodos...")
        
        # Inicializar matriz de distancias como diccionario
        self.distance_matrix = {}
        
        # Calcular distancias usando shortest_path_length de NetworkX
        # Esto es mucho más eficiente que llamar get_dist_km repetidamente
        for i, node_from in enumerate(relevant_nodes):
            if (i + 1) % 50 == 0:  # Mostrar progreso cada 50 nodos
                print(f"  Progreso: {i+1}/{total_nodes} nodos procesados...")
            
            # Calcular distancias desde este nodo a todos los demás
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    self.city_graph, 
                    node_from, 
                    weight='length'
                )
                
                # Guardar solo las distancias a nodos relevantes
                for node_to in relevant_nodes:
                    if node_to in lengths:
                        # Convertir de metros a km
                        dist_km = lengths[node_to] / 1000.0
                        self.distance_matrix[(node_from, node_to)] = dist_km
                    else:
                        # Si no hay camino, usar distancia infinita
                        self.distance_matrix[(node_from, node_to)] = 999999.0
                        
            except Exception as e:
                print(f"  [ERROR] Calculando distancias desde nodo {node_from}: {e}")
                # En caso de error, llenar con valores muy grandes
                for node_to in relevant_nodes:
                    self.distance_matrix[(node_from, node_to)] = 999999.0
        
        elapsed = time.time() - start_time
        print(f"  [OK] Matriz de distancias precalculada en {elapsed:.2f} segundos")
        print(f"  Total de distancias almacenadas: {len(self.distance_matrix)}\n")
    
    def get_dist_km(self, node_from, node_to):
        """
        Calcula la distancia en kilómetros entre dos nodos.
        Usa la matriz precalculada si está disponible, si no calcula en tiempo real.
        """
        if node_from == node_to:
            return 0.0
        
        # Si tenemos matriz precalculada, usarla
        if hasattr(self, 'distance_matrix'):
            key = (node_from, node_to)
            if key in self.distance_matrix:
                return self.distance_matrix[key]
        
        # Fallback: calcular en tiempo real (solo si no está en la matriz)
        try:
            path_length = nx.shortest_path_length(
                self.city_graph, 
                source=node_from, 
                target=node_to, 
                weight='length'
            )
            return path_length / 1000.0
            
        except nx.NetworkXNoPath:
            return 999999.0
        except nx.NodeNotFound as e:
            print(f"[ERROR] Nodo no encontrado: {e}")
            return 999999.0
    
    def get_path(self, node_from, node_to):
        """
        Obtiene el camino completo (lista de nodos) entre dos puntos.
        """
        if node_from == node_to:
            return [node_from]
        
        try:
            path = nx.shortest_path(
                self.city_graph, 
                source=node_from, 
                target=node_to, 
                weight='length'
            )
            return path
            
        except nx.NetworkXNoPath:
            print(f"[WARNING] No hay camino entre {node_from} y {node_to}")
            return [node_from, node_to]
        except nx.NodeNotFound as e:
            print(f"[ERROR] Nodo no encontrado: {e}")
            return [node_from, node_to]


"""def evaluate_chromosome(chromosome, system):
    routes = []
    current_route = []
    
    current_load = 0.0
    current_time = 0.0
    total_distance = 0.0
    
    current_node = system.garage
    
    TRUCK_PENALTY = 0.0 
    
    for customer in chromosome:
        node_demand = system.city_graph.nodes[customer].get('demand', 1.0)
        
        dist_to_cust = system.get_dist_km(current_node, customer)
        
        # Determinar la velocidad correcta según el estado actual
        if current_node == system.garage:
            speed_to_customer = CONFIG['SPEED_EMPTY_TO_ZONE']
        elif current_node == system.landfill:
            speed_to_customer = CONFIG['SPEED_EMPTY_TO_ZONE']
        else:
            speed_to_customer = CONFIG['SPEED_COLLECTING']
        
        time_to_cust = dist_to_cust / speed_to_customer
        service_time = CONFIG['COLLECTION_TIME']
        
        dist_cust_to_dump = system.get_dist_km(customer, system.landfill)
        time_cust_to_dump = dist_cust_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
        
        dist_dump_to_garage = system.get_dist_km(system.landfill, system.garage)
        time_dump_to_garage = dist_dump_to_garage / CONFIG['SPEED_RETURN_EMPTY']
        
        # ════════════════════════════════════════════════════════════════════
        # VERIFICACIÓN ESTRICTA DE CAPACIDAD
        # ════════════════════════════════════════════════════════════════════
        
        # Caso 1: ¿Cabe en la ruta actual? (capacidad Y tiempo)
        time_if_add = (current_time + time_to_cust + service_time + 
                       time_cust_to_dump + time_dump_to_garage)
        
        if (current_load + node_demand <= CONFIG['MAX_CAPACITY']) and (time_if_add <= CONFIG['MAX_SHIFT_TIME']):
            # SÍ CABE - Agregar a la ruta actual
            current_route.append(customer)
            current_load += node_demand
            current_time += (time_to_cust + service_time)
            total_distance += dist_to_cust
            current_node = customer
            
        else:
            # NO CABE - Necesitamos decidir qué hacer
            
            # Opción A: ¿Podemos hacer descarga intermedia?
            # (ir al landfill, descargar, y seguir con este cliente)
            
            dist_to_dump = system.get_dist_km(current_node, system.landfill)
            time_to_dump = dist_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
            time_at_dump = current_time + time_to_dump
            
            dist_dump_to_cust = system.get_dist_km(system.landfill, customer)
            time_dump_to_cust = dist_dump_to_cust / CONFIG['SPEED_EMPTY_TO_ZONE']
            
            # Tiempo total SI hacemos descarga intermedia
            projected_total_time = (time_at_dump + time_dump_to_cust + 
                                    service_time + time_cust_to_dump + 
                                    time_dump_to_garage)
            
            # ════════════════════════════════════════════════════════════════
            # FIX CRÍTICO: Verificar capacidad DESPUÉS de descarga intermedia
            # ════════════════════════════════════════════════════════════════
            
            # Solo podemos hacer descarga intermedia si:
            # 1. El motivo es CAPACIDAD (no tiempo)
            # 2. El tiempo proyectado cabe
            # 3. El cliente SOLO cabe en un camión vacío (node_demand <= MAX_CAPACITY)
            
            can_do_intermediate_dump = (
                (current_load + node_demand > CONFIG['MAX_CAPACITY']) and  # Es por capacidad
                (projected_total_time <= CONFIG['MAX_SHIFT_TIME']) and      # El tiempo cabe
                (node_demand <= CONFIG['MAX_CAPACITY'])                      # El cliente cabe solo
            )
            
            if can_do_intermediate_dump:
                # DESCARGA INTERMEDIA - continuar la misma ruta
                current_route.append(system.landfill)
                
                total_distance += dist_to_dump
                total_distance += dist_dump_to_cust
                
                # RESETEAR CARGA (acabamos de descargar)
                current_load = 0.0  # ← FIX: Empezar desde 0, no desde node_demand
                current_time = time_at_dump + time_dump_to_cust + service_time
                
                # Ahora SÍ agregar el cliente
                current_route.append(customer)
                current_load += node_demand  # ← FIX: SUMAR la demanda correctamente
                current_node = customer
                
            else:
                # NUEVA RUTA - cerrar la actual y empezar una nueva
                
                # Cerrar ruta actual
                total_distance += system.get_dist_km(current_node, system.landfill)
                routes.append(current_route)
                
                total_distance += TRUCK_PENALTY
                
                # Iniciar nueva ruta
                current_route = [customer]
                current_load = node_demand  # ← Esto está bien
                
                dist_start = system.get_dist_km(system.garage, customer)
                total_distance += dist_start
                current_time = (dist_start / CONFIG['SPEED_EMPTY_TO_ZONE'] + CONFIG['COLLECTION_TIME'])
                current_node = customer

    # Cerrar última ruta
    if current_route:
        total_distance += system.get_dist_km(current_node, system.landfill)
        dist_return = system.get_dist_km(system.landfill, system.garage)
        total_distance += dist_return
        routes.append(current_route)

    # DIAGNÓSTICO: ¿Cuántos clientes visitamos?
    visited = set()
    for route in routes:
        for node in route:
            if node != system.landfill:
                visited.add(node)
    
    if len(visited) != len(chromosome):
        print(f"[WARNING] Solo visitamos {len(visited)}/{len(chromosome)} clientes")
        missing = set(chromosome) - visited
        print(f"Faltantes: {list(missing)[:5]}...")  # Mostrar primeros 5
        
    return total_distance, routes"""


"""
def evaluate_chromosome(chromosome, system, n_trucks=2):
    
    Evalúa un cromosoma dividiéndolo en exactamente n_trucks rutas.
    El punto de división es el 'gen' que separa los clientes entre camiones.
    Penaliza fuertemente si algún camión viola capacidad o tiempo.
    
    size = len(chromosome)

    # Dividir el cromosoma en n_trucks segmentos iguales (el GA aprenderá
    # a mover el punto de corte implícitamente a través de la ordenación)
    # Para 2 camiones: la primera mitad va al camión 1, la segunda al camión 2.
    # NOTA: el GA puede descubrir mejores divisiones explorando permutaciones.
    split = size // n_trucks
    segments = []
    for i in range(n_trucks):
        start = i * split
        end = (i + 1) * split if i < n_trucks - 1 else size
        segments.append(chromosome[start:end])

    routes = []
    total_distance = 0.0
    INFEASIBILITY_PENALTY = 99999.0  # Penalización por violar restricciones

    for seg in segments:
        if not seg:
            continue

        route = []
        current_load = 0.0
        current_time = 0.0
        current_node = system.garage
        feasible = True

        for customer in seg:
            node_demand = system.city_graph.nodes[customer].get('demand', 1.0)

            # Velocidad según estado
            if current_node in (system.garage, system.landfill):
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            else:
                speed = CONFIG['SPEED_COLLECTING']

            dist_to_cust = system.get_dist_km(current_node, customer)
            time_to_cust = dist_to_cust / speed
            service_time = CONFIG['COLLECTION_TIME']

            dist_cust_to_dump = system.get_dist_km(customer, system.landfill)
            time_cust_to_dump = dist_cust_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
            dist_dump_to_garage = system.get_dist_km(system.landfill, system.garage)
            time_dump_to_garage = dist_dump_to_garage / CONFIG['SPEED_RETURN_EMPTY']

            time_if_add = (current_time + time_to_cust + service_time +
                           time_cust_to_dump + time_dump_to_garage)

            if (current_load + node_demand <= CONFIG['MAX_CAPACITY'] and
                    time_if_add <= CONFIG['MAX_SHIFT_TIME']):
                # Cabe directo
                route.append(customer)
                current_load += node_demand
                current_time += time_to_cust + service_time
                total_distance += dist_to_cust
                current_node = customer
            else:
                # Intentar descarga intermedia (dentro del mismo camión)
                dist_to_dump = system.get_dist_km(current_node, system.landfill)
                time_to_dump = dist_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
                time_at_dump = current_time + time_to_dump
                dist_dump_to_cust = system.get_dist_km(system.landfill, customer)
                time_dump_to_cust = dist_dump_to_cust / CONFIG['SPEED_EMPTY_TO_ZONE']

                projected_time = (time_at_dump + time_dump_to_cust + service_time +
                                  time_cust_to_dump + time_dump_to_garage)

                can_intermediate = (
                        current_load + node_demand > CONFIG['MAX_CAPACITY'] and
                        projected_time <= CONFIG['MAX_SHIFT_TIME'] and
                        node_demand <= CONFIG['MAX_CAPACITY']
                )

                if can_intermediate:
                    route.append(system.landfill)
                    total_distance += dist_to_dump + dist_dump_to_cust
                    current_load = node_demand
                    current_time = time_at_dump + time_dump_to_cust + service_time
                    route.append(customer)
                    current_node = customer
                else:
                    # Imposible servir este cliente en este camión → INVIABLE
                    feasible = False
                    total_distance += INFEASIBILITY_PENALTY

        # Cerrar la ruta: último cliente → vertedero → garage
        if route:
            last = route[-1] if route[-1] != system.landfill else system.landfill
            total_distance += system.get_dist_km(last, system.landfill)
            total_distance += system.get_dist_km(system.landfill, system.garage)
            routes.append(route)

    return total_distance, routes"""

def evaluate_chromosome(chromosome, system, n_trucks=2):
    """
    Evalúa un cromosoma con gen de corte variable.
 
    El cromosoma tiene formato: [c1, c2, ..., cN, split_point]
    donde split_point define cuántos clientes van al camión 1.
 
    Penaliza fuertemente si algún camión viola capacidad o tiempo.
    """
    # Separar clientes del gen de corte
    clients = chromosome[:-1]
    split = chromosome[-1]
 
    # Asegurar que el split es válido
    n = len(clients)
    split = max(1, min(split, n - 1))
 
    segments = [clients[:split], clients[split:]]
 
    routes = []
    total_distance = 0.0
    INFEASIBILITY_PENALTY = 99999.0
 
    for seg in segments:
        if not seg:
            continue
 
        route = []
        current_load = 0.0
        current_time = 0.0
        current_node = system.garage
        feasible = True
 
        for customer in seg:
            node_demand = system.city_graph.nodes[customer].get('demand', 1.0)
 
            if current_node in (system.garage, system.landfill):
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            else:
                speed = CONFIG['SPEED_COLLECTING']
 
            dist_to_cust = system.get_dist_km(current_node, customer)
            time_to_cust = dist_to_cust / speed
            service_time = CONFIG['COLLECTION_TIME']
 
            dist_cust_to_dump = system.get_dist_km(customer, system.landfill)
            time_cust_to_dump = dist_cust_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
            dist_dump_to_garage = system.get_dist_km(system.landfill, system.garage)
            time_dump_to_garage = dist_dump_to_garage / CONFIG['SPEED_RETURN_EMPTY']
 
            time_if_add = (current_time + time_to_cust + service_time +
                           time_cust_to_dump + time_dump_to_garage)
 
            if (current_load + node_demand <= CONFIG['MAX_CAPACITY'] and
                    time_if_add <= CONFIG['MAX_SHIFT_TIME']):
                route.append(customer)
                current_load += node_demand
                current_time += time_to_cust + service_time
                total_distance += dist_to_cust
                current_node = customer
            else:
                # Intentar descarga intermedia
                dist_to_dump = system.get_dist_km(current_node, system.landfill)
                time_to_dump = dist_to_dump / CONFIG['SPEED_FULL_TO_LANDFILL']
                time_at_dump = current_time + time_to_dump
                dist_dump_to_cust = system.get_dist_km(system.landfill, customer)
                time_dump_to_cust = dist_dump_to_cust / CONFIG['SPEED_EMPTY_TO_ZONE']
 
                projected_time = (time_at_dump + time_dump_to_cust + service_time +
                                  time_cust_to_dump + time_dump_to_garage)
 
                can_intermediate = (
                    current_load + node_demand > CONFIG['MAX_CAPACITY'] and
                    projected_time <= CONFIG['MAX_SHIFT_TIME'] and
                    node_demand <= CONFIG['MAX_CAPACITY']
                )
 
                if can_intermediate:
                    route.append(system.landfill)
                    total_distance += dist_to_dump + dist_dump_to_cust
                    current_load = node_demand
                    current_time = time_at_dump + time_dump_to_cust + service_time
                    route.append(customer)
                    current_node = customer
                else:
                    feasible = False
                    total_distance += INFEASIBILITY_PENALTY
 
        if route:
            last = route[-1]
            total_distance += system.get_dist_km(last, system.landfill)
            total_distance += system.get_dist_km(system.landfill, system.garage)
            routes.append(route)
 
    return total_distance, routes

def create_individual_2trucks(node_list):
    """
    Cromosoma = permutación de clientes.
    La primera mitad va al camión 1, la segunda al camión 2.
    El GA explorará implícitamente qué clientes van en cada camión
    al reorganizar el orden.
    """
    ind = node_list.copy()
    random.shuffle(ind)
    return ind

def create_nn_individual_2trucks(system, randomness=0.17):
    """Nearest Neighbor para 2 camiones — igual que antes, el corte es implícito."""
    return create_nearest_neighbor_individual(system, randomness)

"""
def create_individual(node_list):
    ind = node_list.copy()
    random.shuffle(ind)
    return ind"""

def create_individual(node_list):
    """
    Cromosoma = permutación de clientes + gen de corte al final.
    El gen de corte indica dónde se divide la lista entre los 2 camiones.
    """
    n = len(node_list)
    ind = node_list.copy()
    random.shuffle(ind)
    split = random.randint(1, n - 1)  # Al menos 1 cliente por camión
    ind.append(split)                 # El último elemento es el punto de corte
    return ind


    """
def create_nearest_neighbor_individual(system, randomness=0.17):
    
    Crea un individuo usando heurística de Vecino Más Cercano.
    
    Args:
        system: VRPSystem instance
        randomness: 0.0 = estricto NN, 1.0 = completamente aleatorio
                   0.3 = elige entre los 3 vecinos más cercanos aleatoriamente
    
    Returns:
        Lista ordenada de clientes (cromosoma)
    
    unvisited = set(system.customers)
    route = []
    
    # Empezar desde el garage, ir al cliente más cercano
    current = system.garage
    
    while unvisited:
        # Calcular distancias a todos los no visitados
        distances = [(system.get_dist_km(current, node), node) for node in unvisited]
        distances.sort()  # Ordenar por distancia
        
        # Aplicar aleatoriedad: elegir entre los k vecinos más cercanos
        k = max(1, int(len(distances) * randomness))
        k = min(k, len(distances))  # No exceder la cantidad disponible
        
        # Elegir aleatoriamente entre los k más cercanos
        chosen_dist, chosen_node = random.choice(distances[:k])
        
        route.append(chosen_node)
        unvisited.remove(chosen_node)
        current = chosen_node
    
    return route
    """
    
def create_nearest_neighbor_individual(system, randomness=0.17):
    """
    Heurística Nearest Neighbor para 2 camiones con gen de corte variable.
 
    Construye el cromosoma ordenando clientes por proximidad desde el garage,
    luego elige un punto de corte aleatorio sesgado al centro (para que ambos
    camiones tengan carga razonable desde el inicio).
    """
    unvisited = set(system.customers)
    route = []
    current = system.garage
 
    while unvisited:
        distances = [(system.get_dist_km(current, node), node) for node in unvisited]
        distances.sort()
        k = max(1, int(len(distances) * randomness))
        k = min(k, len(distances))
        _, chosen_node = random.choice(distances[:k])
        route.append(chosen_node)
        unvisited.remove(chosen_node)
        current = chosen_node
 
    n = len(route)
    # Punto de corte sesgado al centro ±25% para equilibrar carga inicial
    centro = n // 2
    margen = max(1, n // 4)
    split = random.randint(max(1, centro - margen), min(n - 1, centro + margen))
    route.append(split)
    return route


def calculate_real_physics(vrp, routes):
    
    real_distance = 0.0
    real_load = 0.0
    
    for route in routes:
        if not route: continue
        
        real_distance += vrp.get_dist_km(vrp.garage, route[0])
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            real_distance += vrp.get_dist_km(u, v)
            real_load += vrp.city_graph.nodes[u]['demand']
            
        real_load += vrp.city_graph.nodes[route[-1]]['demand']
        
        real_distance += vrp.get_dist_km(route[-1], vrp.landfill)
        
        # --- MODIFICACION: Retorno al Garage ---
        real_distance += vrp.get_dist_km(vrp.landfill, vrp.garage)
        
    return real_distance, real_load

"""
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    
    current = (b + 1) % size
    p2_idx = (b + 1) % size
    
    while -1 in child:
        if parent2[p2_idx] not in child:
            child[current] = parent2[p2_idx]
            current = (current + 1) % size
        p2_idx = (p2_idx + 1) % size
    return child """

def ordered_crossover(parent1, parent2):
    """
    OX Crossover adaptado: opera solo sobre la parte de clientes.
    El gen de corte (último elemento) se hereda promediando ambos padres
    y aplicando una pequeña perturbación aleatoria.
    """
    # Separar clientes y gen de corte
    p1_clients = parent1[:-1]
    p2_clients = parent2[:-1]
    split1 = parent1[-1]
    split2 = parent2[-1]
 
    size = len(p1_clients)
    a, b = sorted(random.sample(range(size), 2))
    child_clients = [-1] * size
    child_clients[a:b+1] = p1_clients[a:b+1]
 
    current = (b + 1) % size
    p2_idx = (b + 1) % size
 
    while -1 in child_clients:
        if p2_clients[p2_idx] not in child_clients:
            child_clients[current] = p2_clients[p2_idx]
            current = (current + 1) % size
        p2_idx = (p2_idx + 1) % size
 
    # Gen de corte: promedio de padres + perturbación aleatoria pequeña
    avg_split = (split1 + split2) // 2
    perturbacion = random.randint(-max(1, size // 8), max(1, size // 8))
    child_split = max(1, min(size - 1, avg_split + perturbacion))
 
    child_clients.append(child_split)
    return child_clients

"""
def mutate(ind):
    Swap mutation: intercambia dos clientes aleatorios
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind
"""

def mutate(ind):
    """
    Swap mutation: intercambia dos clientes aleatorios.
    El gen de corte (último elemento) no se toca.
    """
    clients = ind[:-1]
    split = ind[-1]
    a, b = random.sample(range(len(clients)), 2)
    clients[a], clients[b] = clients[b], clients[a]
    clients.append(split)
    return clients

"""
def inversion_mutation(ind):
    
    2-Opt / Inversion mutation: invierte un segmento de la ruta.
    Esto ayuda a deshacer cruces (nudos) en la ruta.
    
    Ejemplo:
    Antes: [A, B, C, D, E, F]
    Seleccionar segmento [B, C, D] (índices 1-3)
    Después: [A, D, C, B, E, F]
    
    if len(ind) < 3:
        return ind
    
    # Seleccionar dos puntos aleatorios
    i, j = sorted(random.sample(range(len(ind)), 2))
    
    # Invertir el segmento entre i y j (inclusive)
    ind[i:j+1] = reversed(ind[i:j+1])
    
    return ind"""

def inversion_mutation(ind):
    """
    2-Opt / Inversion mutation: invierte un segmento de clientes.
    Con probabilidad 0.3 también muta el gen de corte (±1 a ±3 posiciones).
    """
    clients = ind[:-1]
    split = ind[-1]
    n = len(clients)
 
    if n >= 3:
        i, j = sorted(random.sample(range(n), 2))
        clients[i:j+1] = reversed(clients[i:j+1])
 
    # Mutar el gen de corte ocasionalmente para explorar divisiones distintas
    if random.random() < 0.3:
        delta = random.randint(-3, 3)
        split = max(1, min(n - 1, split + delta))
 
    clients.append(split)
    return clients

def split_mutation(ind):
    """
    Mutación exclusiva del gen de corte: asigna un punto de división
    completamente nuevo. Útil para escapar de divisiones localmente óptimas.
    """
    clients = ind[:-1]
    n = len(clients)
    new_split = random.randint(1, n - 1)
    clients.append(new_split)
    return clients
 
 
def tournament_selection(scored, tournament_size):
    """
    Selección por torneo real usando CONFIG['TOURNAMENT_SIZE'].
 
    Elige 'tournament_size' individuos al azar y retorna el de menor fitness.
    Mayor tournament_size → mayor presión selectiva.
    """
    contenders = random.sample(scored, min(tournament_size, len(scored)))
    winner = min(contenders, key=lambda x: x[0])
    return winner[1]  # Retorna el cromosoma (ind)

def detailed_audit(vrp, best_routes):
    print(f"\n{'='*70}")
    print("DETALLE AUDITORÍA DE RUTAS (Desglose de Kilómetros)")
    print(f"{'='*70}")
    
    total_fleet_service = 0
    total_fleet_commute = 0
    
    # Store rows for a summary table
    audit_table_rows = []

    for i, route in enumerate(best_routes):
        print(f"\n[CAMIÓN {i+1}]")
        dumps = 0
        
        # Breakdown distances
        d_start = vrp.get_dist_km(vrp.garage, route[0])
        commute_dist = d_start
        service_dist = 0
        
        print(f"   -> Salida Garaje a 1er cliente: {d_start:.2f} km (Traslado)")

        current_node = route[0]
        
        # Loop from 0 to len-2
        for idx in range(len(route)-1):
            u = route[idx]
            v = route[idx+1]
            dist = vrp.get_dist_km(u, v)
            
            if u == vrp.landfill or v == vrp.landfill:
                commute_dist += dist
            else:
                service_dist += dist
                
            if u == vrp.landfill:
                dumps += 1
                print(f"   >> DESCARGA #{dumps} (Tramo traslado: {dist:.2f} km)")
        
        last_node = route[-1]
        if last_node != vrp.landfill:
             d_end = vrp.get_dist_km(last_node, vrp.landfill)
             commute_dist += d_end
             # print(f"   -> Cliente final a Landfill: {d_end:.2f} km (Traslado)")
        
        d_return = vrp.get_dist_km(vrp.landfill, vrp.garage)
        commute_dist += d_return
        # print(f"   -> Landfill a Garaje: {d_return:.2f} km (Traslado)")
        
        total_dist = commute_dist + service_dist
        print(f"   Resultados Camión {i+1}:")
        print(f"     - Recolección: {service_dist:.2f} km")
        print(f"     - Traslados:   {commute_dist:.2f} km")
        print(f"     - TOTAL:       {total_dist:.2f} km")
        
        audit_table_rows.append([i+1, service_dist, commute_dist, total_dist])
        
        total_fleet_service += service_dist
        total_fleet_commute += commute_dist

    # Print Summary Table
    print(f"\n{'='*70}")
    print("RESUMEN TABULAR DE KILOMETRAJE (Recolección vs Traslados)")
    print(f"{'-'*70}")
    print(f"{'Camión':<8} | {'Recolección (km)':<18} | {'Traslado (km)':<15} | {'Total (km)':<12} | {'% Improd.':<10}")
    print(f"{'-'*70}")
    
    for row in audit_table_rows:
        truck_id, s_dist, c_dist, t_dist = row
        perc = (c_dist/t_dist)*100 if t_dist > 0 else 0
        print(f"{truck_id:<8} | {s_dist:<18.2f} | {c_dist:<15.2f} | {t_dist:<12.2f} | {perc:<9.1f}%")
        
    print(f"{'-'*70}")
    
    total_km = total_fleet_service + total_fleet_commute
    total_perc = (total_fleet_commute/total_km)*100 if total_km > 0 else 0
    
    print(f"{'TOTAL':<8} | {total_fleet_service:<18.2f} | {total_fleet_commute:<15.2f} | {total_km:<12.2f} | {total_perc:<9.1f}%")
    print(f"{'='*70}")
    
    if total_fleet_commute > total_fleet_service:
        print("\n[ANÁLISIS] El kilometraje de traslado supera al de servicio.")
        print("           Esto se debe a la ida y vuelta al Relleno Sanitario (Pichacay) y Garage.")

    print(f"\n{'='*60}")
    print("AUDITORÍA DE FUNCIONAMIENTO Y COBERTURA")
    print("="*60)
    
    serviced_set = set()
    transited_counter = {n: 0 for n in vrp.customers}
    
    for route in best_routes:
        for stop in route:
            if stop == vrp.landfill:
                continue
                
            if stop in serviced_set:
                 print(f"[ERROR] Cliente {stop} visitado MULTIPLES veces!")
            serviced_set.add(stop)
            
        full_path_nodes = [vrp.garage]
        curr = vrp.garage
        for stop in route:
            path = vrp.get_path(curr, stop)
            full_path_nodes.extend(path[1:])
            curr = stop
        full_path_nodes.extend(vrp.get_path(curr, vrp.landfill)[1:])
        
        for n in full_path_nodes:
            if n in transited_counter:
                transited_counter[n] += 1

    print(f"Total Clientes Esperados: {len(vrp.customers)}")
    print(f"Total Clientes Servidos:  {len(serviced_set)}")
    
    coverage_check = len(serviced_set) == len(vrp.customers)
    print(f"¿Cobertura Completa?      {'[SÍ]' if coverage_check else '[NO]'}")
    
    if not coverage_check:
        missing = set(vrp.customers) - serviced_set
        print(f"Faltan: {missing}")
    
    print("\n[DETALLE DE VISITAS]")
    print(f"{'Nodo ID':<15} | {'Estado':<10} | {'Veces Servido':<15} | {'Veces Transitado':<15}")
    print("-" * 65)
    
    for cust in vrp.customers:
        is_serviced = cust in serviced_set
        transit_count = transited_counter[cust]
        status = "OK" if is_serviced else "MISSING"
        serviced_count = 1 if is_serviced else 0
        
        print(f"{cust:<15} | {status:<10} | {serviced_count:<15} | {transit_count:<15}")
        
    print("-" * 65)
    print("NOTA: 'Veces Transitado' > 1 es NORMAL y ÓPTIMO. "
          "Significa que el camión pasó por esa esquina para llegar a otra calle, "
          "pero solo recolectó basura 1 vez.")
    print("="*60)

def detailed_trip_audit(vrp, routes):
    """
    Auditoría detallada de cada viaje al vertedero (dump trip).
    Muestra cuánta basura recogió el camión antes de cada descarga.
    """
    print(f"\n{'='*80}")
    print("AUDITORÍA DE VIAJES AL VERTEDERO (Carga por Descarga)")
    print(f"{'='*80}\n")
    
    fleet_total_load = 0.0
    fleet_total_time = 0.0
    
    for truck_idx, route in enumerate(routes, 1):
        print(f"{'─'*80}")
        print(f"CAMIÓN {truck_idx}")
        print(f"{'─'*80}")
        
        trip_number = 0
        current_trip_load = 0.0
        truck_total_load = 0.0
        truck_total_time = 0.0
        
        current_node = vrp.garage
        current_time = 0.0
        
        for i, node in enumerate(route):
            # Calcular distancia y tiempo al siguiente nodo
            dist = vrp.get_dist_km(current_node, node)
            
            # Determinar velocidad según contexto
            if current_node == vrp.garage:
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            elif current_node == vrp.landfill:
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            elif node == vrp.landfill:
                speed = CONFIG['SPEED_FULL_TO_LANDFILL']
            else:
                speed = CONFIG['SPEED_COLLECTING']
            
            travel_time = dist / speed
            current_time += travel_time
            
            if node == vrp.landfill:
                # Es una descarga intermedia
                trip_number += 1
                hours = int(current_time)
                minutes = int((current_time - hours) * 60)
                
                print(f"  Viaje #{trip_number} al Vertedero:")
                print(f"    Carga descargada: {current_trip_load:.2f} ton")
                print(f"    Tiempo acumulado: {hours}h {minutes:02d}m")
                print()
                
                truck_total_load += current_trip_load
                current_trip_load = 0.0  # Reset para el siguiente viaje
            else:
                # Es un cliente - recolectar basura
                demand = vrp.city_graph.nodes[node].get('demand', 0)
                current_trip_load += demand
                current_time += CONFIG['COLLECTION_TIME']
            
            current_node = node
        
        # Descarga final (si quedó carga)
        if current_trip_load > 0 or route[-1] != vrp.landfill:
            # Ir al landfill si no estamos ahí
            if route[-1] != vrp.landfill:
                dist = vrp.get_dist_km(route[-1], vrp.landfill)
                current_time += dist / CONFIG['SPEED_FULL_TO_LANDFILL']
            
            trip_number += 1
            hours = int(current_time)
            minutes = int((current_time - hours) * 60)
            
            print(f"  Viaje #{trip_number} al Vertedero (FINAL):")
            print(f"    Carga descargada: {current_trip_load:.2f} ton")
            print(f"    Tiempo acumulado: {hours}h {minutes:02d}m")
            print()
            
            truck_total_load += current_trip_load
        
        # Retorno al garage
        dist_return = vrp.get_dist_km(vrp.landfill, vrp.garage)
        current_time += dist_return / CONFIG['SPEED_RETURN_EMPTY']
        truck_total_time = current_time
        
        # Resumen del camión
        hours = int(truck_total_time)
        minutes = int((truck_total_time - hours) * 60)
        
        print(f"  RESUMEN CAMIÓN {truck_idx}:")
        print(f"    Total viajes al vertedero: {trip_number}")
        print(f"    Total basura recolectada:  {truck_total_load:.2f} ton")
        print(f"    Tiempo total de turno:     {hours}h {minutes:02d}m")
        print()
        
        fleet_total_load += truck_total_load
        fleet_total_time += truck_total_time
    
    # Resumen de flota
    print(f"{'='*80}")
    print(f"RESUMEN TOTAL DE LA FLOTA")
    print(f"{'='*80}")
    print(f"  Total camiones:           {len(routes)}")
    print(f"  Total basura recolectada: {fleet_total_load:.2f} ton")
    
    avg_time = fleet_total_time / len(routes) if routes else 0
    hours = int(avg_time)
    minutes = int((avg_time - hours) * 60)
    print(f"  Tiempo promedio/camión:   {hours}h {minutes:02d}m")
    print(f"{'='*80}\n")

def detailed_audit_initial(vrp, routes, title="SOLUCIÓN INICIAL"):
    """Auditoría detallada para solución inicial (antes de optimizar)"""
    print(f"\n{'='*70}")
    print(f"DETALLE AUDITORÍA DE RUTAS - {title}")
    print(f"{'='*70}")
    
    total_fleet_service = 0
    total_fleet_commute = 0
    
    audit_table_rows = []

    for i, route in enumerate(routes):
        dumps = 0
        
        # Breakdown distances
        d_start = vrp.get_dist_km(vrp.garage, route[0])
        commute_dist = d_start
        service_dist = 0
        
        current_node = route[0]
        
        # Loop from 0 to len-2
        for idx in range(len(route)-1):
            u = route[idx]
            v = route[idx+1]
            dist = vrp.get_dist_km(u, v)
            
            if u == vrp.landfill or v == vrp.landfill:
                commute_dist += dist
            else:
                service_dist += dist
                
            if u == vrp.landfill:
                dumps += 1
        
        last_node = route[-1]
        if last_node != vrp.landfill:
             d_end = vrp.get_dist_km(last_node, vrp.landfill)
             commute_dist += d_end
        
        d_return = vrp.get_dist_km(vrp.landfill, vrp.garage)
        commute_dist += d_return
        
        total_dist = commute_dist + service_dist
        
        audit_table_rows.append([i+1, service_dist, commute_dist, total_dist])
        
        total_fleet_service += service_dist
        total_fleet_commute += commute_dist

    # Print Summary Table
    print(f"\n{'='*70}")
    print("RESUMEN TABULAR DE KILOMETRAJE (Recolección vs Traslados)")
    print(f"{'-'*70}")
    print(f"{'Camión':<8} | {'Recolección (km)':<18} | {'Traslado (km)':<15} | {'Total (km)':<12} | {'% Improd.':<10}")
    print(f"{'-'*70}")
    
    for row in audit_table_rows:
        truck_id, s_dist, c_dist, t_dist = row
        perc = (c_dist/t_dist)*100 if t_dist > 0 else 0
        print(f"{truck_id:<8} | {s_dist:<18.2f} | {c_dist:<15.2f} | {t_dist:<12.2f} | {perc:<9.1f}%")
        
    print(f"{'-'*70}")
    
    total_km = total_fleet_service + total_fleet_commute
    total_perc = (total_fleet_commute/total_km)*100 if total_km > 0 else 0
    
    print(f"{'TOTAL':<8} | {total_fleet_service:<18.2f} | {total_fleet_commute:<15.2f} | {total_km:<12.2f} | {total_perc:<9.1f}%")
    print(f"{'='*70}\n")

def densify_path(path_coords, step_meters=10):
    dense_path = []
    for i in range(len(path_coords) - 1):
        p1 = np.array(path_coords[i])
        p2 = np.array(path_coords[i+1])
        dist = np.linalg.norm(p2 - p1) * 111139 
        num_steps = max(1, int(dist / step_meters))
        for s in range(num_steps):
            fraction = s / num_steps
            interp_point = p1 + (p2 - p1) * fraction
            dense_path.append(tuple(interp_point))
    dense_path.append(path_coords[-1]) 
    return dense_path

def calculate_route_metrics(vrp, route):
    """Wrapper que mantiene compatibilidad con código existente"""
    metrics = calculate_detailed_route_metrics(vrp, route)
    return metrics['total_distance'], metrics['total_load'], metrics['total_time']

def calculate_detailed_route_metrics(vrp, route):
    """
    Calcula métricas detalladas de una ruta con velocidades variables
    y consumo de combustible. MANEJA CORRECTAMENTE DESCARGAS INTERMEDIAS.
    """
    if not route:
        return {
            'dist_empty_to_zone': 0.0,
            'dist_collecting': 0.0,
            'dist_full_to_landfill': 0.0,
            'dist_return_empty': 0.0,
            'total_distance': 0.0,
            'total_load': 0.0,
            'total_time': 0.0,
            'customers_served': 0,
            'fuel_gallons': 0.0,
            'fuel_cost': 0.0
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # NUEVA LÓGICA: Procesar la ruta secuencialmente, respetando descargas
    # ════════════════════════════════════════════════════════════════════════
    
    total_distance = 0.0
    total_time = 0.0
    customers_served = 0
    current_load = 0.0
    max_load = 0.0  # La carga máxima en cualquier momento
    
    # Distancias por sección
    dist_empty_to_zone = 0.0
    dist_collecting = 0.0
    dist_full_to_landfill = 0.0
    dist_return_empty = 0.0
    
    # Estado inicial
    current_node = vrp.garage
    first_customer = True
    
    for i, node in enumerate(route):
        dist = vrp.get_dist_km(current_node, node)
        total_distance += dist
        
        # Determinar velocidad según contexto
        if current_node == vrp.garage:
            # Saliendo del garage vacío
            speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            dist_empty_to_zone += dist
            
        elif current_node == vrp.landfill:
            # Saliendo del landfill (vacío, ya descargamos)
            speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            dist_empty_to_zone += dist
            current_load = 0.0  # ← RESETEAR carga después de descargar
            
        elif node == vrp.landfill:
            # Yendo AL landfill (lleno)
            speed = CONFIG['SPEED_FULL_TO_LANDFILL']
            dist_full_to_landfill += dist
            
        else:
            # Entre clientes (recolectando)
            speed = CONFIG['SPEED_COLLECTING']
            dist_collecting += dist
        
        total_time += dist / speed
        
        # Procesar el nodo actual
        if node == vrp.landfill:
            # Es una descarga intermedia, no hacer nada más
            pass
        else:
            # Es un cliente - recolectar basura
            demand = vrp.city_graph.nodes[node].get('demand', 0)
            current_load += demand
            max_load = max(max_load, current_load)  # Rastrear máxima carga
            customers_served += 1
            total_time += CONFIG['COLLECTION_TIME']
        
        current_node = node
    
    # Último tramo: desde donde terminamos → Landfill → Garage
    if route[-1] != vrp.landfill:
        # Ir al landfill
        dist = vrp.get_dist_km(route[-1], vrp.landfill)
        dist_full_to_landfill += dist
        total_distance += dist
        total_time += dist / CONFIG['SPEED_FULL_TO_LANDFILL']
    
    # Retorno al garage
    dist = vrp.get_dist_km(vrp.landfill, vrp.garage)
    dist_return_empty += dist
    total_distance += dist
    total_time += dist / CONFIG['SPEED_RETURN_EMPTY']
    
    # COMBUSTIBLE
    fuel_gallons = total_distance / CONFIG['KM_PER_GALLON']
    fuel_cost = fuel_gallons * CONFIG['FUEL_PRICE_PER_GALLON']
    
    return {
        'dist_empty_to_zone': dist_empty_to_zone,
        'dist_collecting': dist_collecting,
        'dist_full_to_landfill': dist_full_to_landfill,
        'dist_return_empty': dist_return_empty,
        'total_distance': total_distance,
        'total_load': max_load,  # ← CAMBIO CRÍTICO: devolver la MÁXIMA carga, no la suma total
        'total_time': total_time,
        'customers_served': customers_served,
        'fuel_gallons': fuel_gallons,
        'fuel_cost': fuel_cost
    }

def print_detailed_solution_analysis(vrp, routes, title="ANÁLISIS DETALLADO"):
    """Imprime análisis completo con velocidades variables y combustible"""
    
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")
    
    # Acumuladores totales
    total_dist_empty_zone = 0.0
    total_dist_collecting = 0.0
    total_dist_full = 0.0
    total_dist_return = 0.0
    total_km = 0.0
    total_fuel = 0.0
    total_fuel_cost = 0.0
    total_customers = 0
    total_time = 0.0
    total_load_collected = 0.0
    
    # Análisis por camión
    for i, route in enumerate(routes, 1):
        metrics = calculate_detailed_route_metrics(vrp, route)
        
        print(f"{'─'*80}")
        print(f"CAMIÓN {i}")
        print(f"{'─'*80}")
        print(f"  Clientes servidos: {metrics['customers_served']}")
        
        # Calcular carga total recolectada (suma de todos los clientes)
        route_total_load = sum(vrp.city_graph.nodes[node].get('demand', 0) 
                               for node in route if node != vrp.landfill)
        print(f"  Carga recolectada: {route_total_load:.2f} ton")
        print(f"\n  DESGLOSE DE KILOMETRAJE:")
        print(f"    1. Garage → Zona (vacío):        {metrics['dist_empty_to_zone']:>8.2f} km  @ {CONFIG['SPEED_EMPTY_TO_ZONE']:.0f} km/h")
        print(f"    2. Recolección en zona:          {metrics['dist_collecting']:>8.2f} km  @ {CONFIG['SPEED_COLLECTING']:.0f} km/h")
        print(f"    3. Zona → Relleno (lleno):       {metrics['dist_full_to_landfill']:>8.2f} km  @ {CONFIG['SPEED_FULL_TO_LANDFILL']:.0f} km/h")
        print(f"    4. Relleno → Garage (vacío):     {metrics['dist_return_empty']:>8.2f} km  @ {CONFIG['SPEED_RETURN_EMPTY']:.0f} km/h")
        print(f"    {'─'*60}")
        print(f"    TOTAL DISTANCIA:                 {metrics['total_distance']:>8.2f} km")
        
        hours = int(metrics['total_time'])
        minutes = int((metrics['total_time'] - hours) * 60)
        print(f"\n  Tiempo total de ruta: {hours}h {minutes:02d}m")
        
        print(f"\n  COMBUSTIBLE:")
        print(f"    Consumo: {metrics['fuel_gallons']:.2f} galones ({CONFIG['KM_PER_GALLON']:.1f} gal/km)")
        print(f"    Costo:   ${metrics['fuel_cost']:.2f} (@ ${CONFIG['FUEL_PRICE_PER_GALLON']:.2f}/gal)")
        print()
        
        # Acumular totales
        total_dist_empty_zone += metrics['dist_empty_to_zone']
        total_dist_collecting += metrics['dist_collecting']
        total_dist_full += metrics['dist_full_to_landfill']
        total_dist_return += metrics['dist_return_empty']
        total_km += metrics['total_distance']
        total_fuel += metrics['fuel_gallons']
        total_fuel_cost += metrics['fuel_cost']
        total_customers += metrics['customers_served']
        total_time += metrics['total_time']
        total_load_collected += route_total_load
    
    # RESUMEN TOTAL
    print(f"{'='*80}")
    print(f"RESUMEN TOTAL DE LA FLOTA ({len(routes)} camiones)")
    print(f"{'='*80}")
    print(f"\n  KILOMETRAJE POR SECCIÓN:")
    print(f"    Garage → Zona (vacío):        {total_dist_empty_zone:>8.2f} km")
    print(f"    Recolección en zona:          {total_dist_collecting:>8.2f} km")
    print(f"    Zona → Relleno (lleno):       {total_dist_full:>8.2f} km")
    print(f"    Relleno → Garage (vacío):     {total_dist_return:>8.2f} km")
    print(f"    {'─'*60}")
    print(f"    TOTAL FLOTA:                  {total_km:>8.2f} km")
    
    print(f"\n  OPERACIÓN:")
    print(f"    Total clientes servidos:      {total_customers}")
    print(f"    Total basura recolectada:     {total_load_collected:.2f} ton")
    total_hours = int(total_time)
    total_mins = int((total_time - total_hours) * 60)
    print(f"    Tiempo acumulado:             {total_hours}h {total_mins:02d}m")
    
    print(f"\n  COMBUSTIBLE TOTAL:")
    print(f"    Consumo total:                {total_fuel:.2f} galones")
    print(f"    Costo total:                  ${total_fuel_cost:.2f}")
    print(f"    Costo promedio/camión:        ${total_fuel_cost/len(routes):.2f}")
    
    print(f"\n  EFICIENCIA:")
    print(f"    km de recolección / km total: {(total_dist_collecting/total_km*100):.1f}%")
    print(f"    km improductivos:             {((total_km - total_dist_collecting)/total_km*100):.1f}%")
    
    print(f"{'='*80}\n")
    
    # AUDITORÍA DE VIAJES AL VERTEDERO (integrada)
    print(f"{'='*80}")
    print("AUDITORÍA DE VIAJES AL VERTEDERO (Carga por Descarga)")
    print(f"{'='*80}\n")
    
    fleet_total_load_audit = 0.0
    
    for truck_idx, route in enumerate(routes, 1):
        print(f"{'─'*80}")
        print(f"CAMIÓN {truck_idx}")
        print(f"{'─'*80}")
        
        trip_number = 0
        current_trip_load = 0.0
        truck_total_load = 0.0
        
        current_node = vrp.garage
        current_time = 0.0
        
        for i, node in enumerate(route):
            dist = vrp.get_dist_km(current_node, node)
            
            if current_node == vrp.garage:
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            elif current_node == vrp.landfill:
                speed = CONFIG['SPEED_EMPTY_TO_ZONE']
            elif node == vrp.landfill:
                speed = CONFIG['SPEED_FULL_TO_LANDFILL']
            else:
                speed = CONFIG['SPEED_COLLECTING']
            
            travel_time = dist / speed
            current_time += travel_time
            
            if node == vrp.landfill:
                trip_number += 1
                hours = int(current_time)
                minutes = int((current_time - hours) * 60)
                
                print(f"  Viaje #{trip_number} al Vertedero:")
                print(f"    Carga descargada: {current_trip_load:.2f} ton")
                print(f"    Tiempo acumulado: {hours}h {minutes:02d}m")
                print()
                
                truck_total_load += current_trip_load
                current_trip_load = 0.0
            else:
                demand = vrp.city_graph.nodes[node].get('demand', 0)
                current_trip_load += demand
                current_time += CONFIG['COLLECTION_TIME']
            
            current_node = node
        
        # Descarga final
        if current_trip_load > 0 or route[-1] != vrp.landfill:
            if route[-1] != vrp.landfill:
                dist = vrp.get_dist_km(route[-1], vrp.landfill)
                current_time += dist / CONFIG['SPEED_FULL_TO_LANDFILL']
            
            trip_number += 1
            hours = int(current_time)
            minutes = int((current_time - hours) * 60)
            
            print(f"  Viaje #{trip_number} al Vertedero (FINAL):")
            print(f"    Carga descargada: {current_trip_load:.2f} ton")
            print(f"    Tiempo acumulado: {hours}h {minutes:02d}m")
            print()
            
            truck_total_load += current_trip_load
        
        # Retorno al garage
        dist_return = vrp.get_dist_km(vrp.landfill, vrp.garage)
        current_time += dist_return / CONFIG['SPEED_RETURN_EMPTY']
        
        hours = int(current_time)
        minutes = int((current_time - hours) * 60)
        
        print(f"  RESUMEN CAMIÓN {truck_idx}:")
        print(f"    Total viajes al vertedero: {trip_number}")
        print(f"    Total basura recolectada:  {truck_total_load:.2f} ton")
        print(f"    Tiempo total de turno:     {hours}h {minutes:02d}m")
        print()
        
        fleet_total_load_audit += truck_total_load
    
    print(f"{'='*80}")
    print(f"RESUMEN TOTAL DE LA FLOTA")
    print(f"{'='*80}")
    print(f"  Total camiones:           {len(routes)}")
    print(f"  Total basura recolectada: {fleet_total_load_audit:.2f} ton")
    
    avg_time = total_time / len(routes) if routes else 0
    hours = int(avg_time)
    minutes = int((avg_time - hours) * 60)
    print(f"  Tiempo promedio/camión:   {hours}h {minutes:02d}m")
    print(f"{'='*80}\n")

from folium import Element 
import math

def export_initial_solution_to_html(vrp, routes, filename="solucion_inicial.html"):
    print(f"\nGenerando mapa de SOLUCIÓN INICIAL: {filename}...")

    try:
        G_latlon = ox.project_graph(vrp.city_graph, to_crs='epsg:4326')
    except:
        G_latlon = vrp.city_graph

    def format_time(decimal_hours):
        h = int(decimal_hours)
        m = int(round((decimal_hours - h) * 60))
        if m == 60: h += 1; m = 0
        return f"{h}h {m:02d}m"

    if vrp.garage in G_latlon.nodes:
        gy, gx = G_latlon.nodes[vrp.garage]['y'], G_latlon.nodes[vrp.garage]['x']
    else:
        gy, gx = -2.9000, -79.0000

    m = folium.Map(location=[gy, gx], zoom_start=15, tiles='CartoDB positron')

    colors = [
        '#0033FF', '#F44336', '#6A0DAD', '#FF9800', '#9C27B0',
        '#00BCD4', '#E91E63', '#3F51B5', '#FFC107', '#795548'
    ]

    folium.Marker(
        [gy, gx], popup="<b>GARAGE (Inicio)</b>",
        icon=folium.Icon(color='green', icon='warehouse', prefix='fa')
    ).add_to(m)

    if vrp.landfill in G_latlon.nodes:
        ly, lx = G_latlon.nodes[vrp.landfill]['y'], G_latlon.nodes[vrp.landfill]['x']
        folium.Marker(
            [ly, lx], popup="<b>VERTEDERO (Fin/Descarga)</b>",
            icon=folium.Icon(color='black', icon='trash', prefix='fa')
        ).add_to(m)

    STEP_METERS = 25
    all_routes_js = []

    table_rows       = ""
    total_fleet_dist = 0.0
    total_fleet_load = 0.0
    total_fleet_time = 0.0

    for i, route in enumerate(routes):
        if not route:
            all_routes_js.append([])
            continue

        color        = colors[i % len(colors)]
        truck_number = i + 1

        r_dist, r_load, r_time = calculate_route_metrics(vrp, route)
        route_total_load = sum(
            vrp.city_graph.nodes[node].get('demand', 0)
            for node in route if node != vrp.landfill
        )

        total_fleet_dist += r_dist
        total_fleet_load += route_total_load
        total_fleet_time += r_time

        table_rows += f"""
        <tr style="border-bottom:1px solid #ddd;">
            <td style="padding:8px;">
                <span style="color:{color};font-weight:bold;">🚛 Camión {truck_number}</span>
            </td>
            <td style="padding:8px;">{route_total_load:.2f} t</td>
            <td style="padding:8px;">{r_dist:.2f} km</td>
            <td style="padding:8px;">{format_time(r_time)}</td>
        </tr>"""

        # Marcadores estáticos de clientes
        for cust in route:
            if cust == vrp.landfill or cust not in G_latlon.nodes:
                continue
            cy, cx = G_latlon.nodes[cust]['y'], G_latlon.nodes[cust]['x']
            folium.CircleMarker(
                [cy, cx], radius=3, color=color, weight=1,
                fill=True, fill_color='white', fill_opacity=0.7,
                popup=f"Cliente {cust} (Camión {truck_number})"
            ).add_to(m)

        # Ruta completa siguiendo calles
        full_path_nodes = []
        full_path_nodes.extend(vrp.get_path(vrp.garage, route[0]))
        for j in range(len(route) - 1):
            seg = vrp.get_path(route[j], route[j + 1])
            full_path_nodes.extend(seg[1:])
        if route[-1] != vrp.landfill:
            full_path_nodes.extend(vrp.get_path(route[-1], vrp.landfill)[1:])

        raw_coords = [
            (G_latlon.nodes[n]['y'], G_latlon.nodes[n]['x'])
            for n in full_path_nodes if n in G_latlon.nodes
        ]

        smooth = densify_path(raw_coords, step_meters=STEP_METERS)
        all_routes_js.append([[c[0], c[1]] for c in smooth])

    # ── Serializar a JSON ─────────────────────────────────────────────────────
    import json
    routes_json = json.dumps(all_routes_js)
    colors_json = json.dumps([colors[i % len(colors)] for i in range(len(routes))])
    names_json  = json.dumps([f"Camión {i+1}" for i in range(len(routes))])
    map_var     = m.get_name()

    # ── JavaScript animación ──────────────────────────────────────────────────
    js_code = f"""
    <script>
    (function() {{
        var checkMap = setInterval(function() {{
            if (typeof {map_var} === 'undefined') return;
            clearInterval(checkMap);

            var routesData  = {routes_json};
            var routeColors = {colors_json};
            var routeNames  = {names_json};

            var SPEED   = 1;
            var FPS     = 12;
            var playing = true;

            var markers     = [];
            var polylines   = [];
            var stepIndex   = [];
            var trailCoords = [];

            function truckIcon(color) {{
                var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">'
                    + '<rect x="2"  y="12" width="22" height="14" rx="2" fill="' + color + '" stroke="white" stroke-width="1.5"/>'
                    + '<rect x="24" y="16" width="10" height="10" rx="2" fill="' + color + '" stroke="white" stroke-width="1.5"/>'
                    + '<circle cx="8"  cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<circle cx="20" cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<circle cx="30" cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<rect x="4" y="14" width="8" height="6" rx="1" fill="rgba(255,255,255,0.35)"/>'
                    + '</svg>';
                return L.divIcon({{
                    html: '<div style="transform:translate(-18px,-18px)">' + svg + '</div>',
                    iconSize:   [36, 36],
                    iconAnchor: [18, 18],
                    className:  ''
                }});
            }}

            // Inicializar marcadores y polylines
            for (var i = 0; i < routesData.length; i++) {{
                if (!routesData[i] || routesData[i].length === 0) {{
                    markers.push(null);
                    polylines.push(null);
                    stepIndex.push(0);
                    trailCoords.push([]);
                    continue;
                }}
                var start = routesData[i][0];

                var mk = L.marker(start, {{ icon: truckIcon(routeColors[i]) }})
                          .bindTooltip(routeNames[i], {{
                              permanent:  true,
                              direction:  'right',
                              offset:     [15, -10],
                              className:  'truck-label'
                          }})
                          .addTo({map_var});
                markers.push(mk);

                var pl = L.polyline([], {{
                    color:   routeColors[i],
                    weight:  2.5,
                    opacity: 0.8
                }}).addTo({map_var});
                polylines.push(pl);

                stepIndex.push(0);
                trailCoords.push([start]);
            }}

            function resetAll() {{
                for (var i = 0; i < routesData.length; i++) {{
                    if (!routesData[i] || routesData[i].length === 0) continue;
                    stepIndex[i]   = 0;
                    trailCoords[i] = [routesData[i][0]];
                    polylines[i].setLatLngs([]);
                    markers[i].setLatLng(routesData[i][0]);
                }}
                document.getElementById('vrp-progress').value = 0;
                document.getElementById('vrp-pct').innerText  = '0%';
            }}

            function animate() {{
                if (!playing) return;

                var allDone = true;
                for (var i = 0; i < routesData.length; i++) {{
                    if (!routesData[i] || routesData[i].length === 0) continue;
                    var total = routesData[i].length;
                    var idx   = stepIndex[i];
                    if (idx < total) allDone = false;
                    if (idx >= total) continue;

                    var pos = routesData[i][idx];
                    markers[i].setLatLng(pos);
                    trailCoords[i].push(pos);
                    polylines[i].setLatLngs(trailCoords[i]);
                    stepIndex[i] = Math.min(idx + SPEED, total);
                }}

                // Actualizar barra de progreso (promedio de todos los camiones)
                var total_pct = 0;
                var cnt = 0;
                for (var i = 0; i < routesData.length; i++) {{
                    if (routesData[i] && routesData[i].length > 0) {{
                        total_pct += (stepIndex[i] / routesData[i].length) * 100;
                        cnt++;
                    }}
                }}
                var pct = cnt > 0 ? total_pct / cnt : 0;
                document.getElementById('vrp-progress').value = pct;
                document.getElementById('vrp-pct').innerText  = Math.round(pct) + '%';

                if (allDone) {{
                    resetAll();
                }}
            }}

            setInterval(animate, 1000 / FPS);

            // ── Controles ────────────────────────────────────────────────────
            document.getElementById('vrp-playpause').onclick = function() {{
                playing = !playing;
                this.innerText = playing ? '⏸️  Pausa' : '▶️  Play';
                this.style.background = playing ? '#4CAF50' : '#FF9800';
            }};

            document.getElementById('vrp-restart').onclick = function() {{
                resetAll();
                playing = true;
                document.getElementById('vrp-playpause').innerText  = '⏸️  Pausa';
                document.getElementById('vrp-playpause').style.background = '#4CAF50';
            }};

            document.getElementById('vrp-speed').oninput = function() {{
                SPEED = parseInt(this.value);
                document.getElementById('vrp-speed-label').innerText = 'x' + SPEED;
            }};
        }}, 200);
    }})();
    </script>

    <style>
    .truck-label {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: bold;
        font-size: 12px;
        color: #222;
        text-shadow: 1px 1px 2px white, -1px -1px 2px white;
    }}
    </style>
    """

    # ── Panel de controles HTML ───────────────────────────────────────────────
    controls_html = """
    <div id="vrp-controls" style="
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: rgba(25,25,25,0.93);
        color: white;
        padding: 10px 20px;
        border-radius: 35px;
        display: flex;
        align-items: center;
        gap: 14px;
        font-family: 'Roboto', sans-serif;
        font-size: 13px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.45);
        white-space: nowrap;
    ">
        <button id="vrp-playpause" style="
            background:#4CAF50; color:white; border:none;
            border-radius:20px; padding:7px 16px;
            cursor:pointer; font-size:13px; font-weight:bold;
            transition: background 0.2s;">
            ⏸️  Pausa
        </button>

        <button id="vrp-restart" style="
            background:#2196F3; color:white; border:none;
            border-radius:20px; padding:7px 16px;
            cursor:pointer; font-size:13px; font-weight:bold;">
            ↺  Reiniciar
        </button>

        <span style="opacity:0.7;">Progreso:</span>
        <input id="vrp-progress" type="range" min="0" max="100" value="0"
               style="width:140px; accent-color:#4CAF50; cursor:default;"
               readonly>
        <span id="vrp-pct" style="min-width:34px; text-align:right; font-weight:bold;">0%</span>

        <span style="opacity:0.7; margin-left:6px;">Velocidad:</span>
        <input id="vrp-speed" type="range" min="1" max="10" value="1"
               style="width:80px; accent-color:#FF9800;">
        <span id="vrp-speed-label" style="min-width:28px; font-weight:bold;">x1</span>
    </div>
    """

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    html_table = f"""
    <div style="
        position:fixed; top:50px; left:50px; width:350px;
        background:white; z-index:9999; border-radius:8px;
        box-shadow:0 0 15px rgba(0,0,0,0.2);
        font-family:'Roboto',sans-serif; font-size:13px; overflow:hidden;">

        <div style="background:#B71C1C;color:white;padding:10px;text-align:center;">
            <h4 style="margin:0;">SOLUCIÓN INICIAL</h4>
            <small>Estado Aleatorio (Sin Optimizar)</small>
        </div>

        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f2f2f2;text-align:left;">
                    <th style="padding:8px;">Vehículo</th>
                    <th style="padding:8px;">Carga</th>
                    <th style="padding:8px;">Dist.</th>
                    <th style="padding:8px;">Tiempo</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
                <tr style="background:#ffebee;font-weight:bold;border-top:2px solid #ccc;">
                    <td style="padding:8px;">TOTAL</td>
                    <td style="padding:8px;">{total_fleet_load:.2f} t</td>
                    <td style="padding:8px;">{total_fleet_dist:.2f} km</td>
                    <td style="padding:8px;">{format_time(total_fleet_time)}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    m.get_root().html.add_child(folium.Element(html_table))
    m.get_root().html.add_child(folium.Element(controls_html))
    m.get_root().html.add_child(folium.Element(js_code))
    folium.LayerControl().add_to(m)
    m.save(filename)
    print(f"[OK] Mapa Inicial guardado: {filename}")


def export_routes_to_html(vrp, routes, filename="rutas_optimizadas.html"):
    print(f"\nGenerando mapa interactivo UNIFICADO (Multi-Trip): {filename}...")

    try:
        G_latlon = ox.project_graph(vrp.city_graph, to_crs='epsg:4326')
    except:
        G_latlon = vrp.city_graph

    def format_time(decimal_hours):
        h = int(decimal_hours)
        m = int(round((decimal_hours - h) * 60))
        if m == 60: h += 1; m = 0
        return f"{h}h {m:02d}m"

    if vrp.garage in G_latlon.nodes:
        gy, gx = G_latlon.nodes[vrp.garage]['y'], G_latlon.nodes[vrp.garage]['x']
    else:
        gy, gx = -2.876, -78.989

    m = folium.Map(location=[gy, gx], zoom_start=15, tiles='CartoDB positron')

    colors = [
        '#0033FF', '#F44336', '#6A0DAD', '#FF9800', '#9C27B0',
        '#00BCD4', '#E91E63', '#3F51B5', '#FFC107', '#795548'
    ]

    folium.Marker(
        [gy, gx], popup="<b>GARAGE (Inicio)</b>",
        icon=folium.Icon(color='green', icon='warehouse', prefix='fa')
    ).add_to(m)

    if vrp.landfill in G_latlon.nodes:
        ly, lx = G_latlon.nodes[vrp.landfill]['y'], G_latlon.nodes[vrp.landfill]['x']
        folium.Marker(
            [ly, lx], popup="<b>VERTEDERO (Fin/Descarga)</b>",
            icon=folium.Icon(color='black', icon='trash', prefix='fa')
        ).add_to(m)

    STEP_METERS = 20

    table_rows       = ""
    total_fleet_dist = 0.0
    total_fleet_load = 0.0
    total_fleet_time = 0.0
    all_routes_js    = []

    for i, route in enumerate(routes):
        if not route:
            all_routes_js.append([])
            continue

        color        = colors[i % len(colors)]
        truck_number = i + 1

        # ── FeatureGroup para activar/desactivar desde LayerControl ──────────
        route_fg = folium.FeatureGroup(name=f"Camión {truck_number}", show=True)

        r_dist, r_load, r_time = calculate_route_metrics(vrp, route)
        route_total_load = sum(
            vrp.city_graph.nodes[node].get('demand', 0)
            for node in route if node != vrp.landfill
        )
        total_fleet_dist += r_dist
        total_fleet_load += route_total_load
        total_fleet_time += r_time

        table_rows += f"""
        <tr style="border-bottom:1px solid #ddd;">
            <td style="padding:8px;">
                <span style="color:{color};font-weight:bold;">🚛 Camión {truck_number}</span>
            </td>
            <td style="padding:8px;">{route_total_load:.2f} ton</td>
            <td style="padding:8px;">{r_dist:.2f} km</td>
            <td style="padding:8px;">{format_time(r_time)}</td>
        </tr>"""

        # Marcadores de clientes y descargas intermedias
        for cust in route:
            if cust == vrp.landfill:
                if vrp.landfill in G_latlon.nodes:
                    ly2, lx2 = G_latlon.nodes[vrp.landfill]['y'], G_latlon.nodes[vrp.landfill]['x']
                    folium.Marker(
                        [ly2, lx2],
                        popup=f"<b>Camión {truck_number}:</b> Descarga Intermedia",
                        icon=folium.Icon(color='gray', icon='refresh', prefix='fa')
                    ).add_to(route_fg)
                continue
            """if cust in G_latlon.nodes:
                cy, cx = G_latlon.nodes[cust]['y'], G_latlon.nodes[cust]['x']
                folium.CircleMarker(
                    [cy, cx], radius=6, color=color, weight=3,
                    fill=True, fill_color='white', fill_opacity=1,
                    popup=f"Cliente {cust}<br>Camión {truck_number}"
                ).add_to(route_fg)"""
            if cust in G_latlon.nodes:
                cy, cx = G_latlon.nodes[cust]['y'], G_latlon.nodes[cust]['x']
                folium.CircleMarker(
                    [cy, cx], radius=2, color=color, weight=1,
                    fill=True, fill_color=color, fill_opacity=0.4,
                    popup=f"Cliente {cust}<br>Camión {truck_number}"
                ).add_to(route_fg)

        # Ruta completa siguiendo calles
        full_path_nodes = []
        full_path_nodes.extend(vrp.get_path(vrp.garage, route[0]))
        for j in range(len(route) - 1):
            seg = vrp.get_path(route[j], route[j + 1])
            full_path_nodes.extend(seg[1:])
        if route[-1] != vrp.landfill:
            full_path_nodes.extend(vrp.get_path(route[-1], vrp.landfill)[1:])

        raw_coords = [
            (G_latlon.nodes[n]['y'], G_latlon.nodes[n]['x'])
            for n in full_path_nodes if n in G_latlon.nodes
        ]

        # Línea estática de la ruta
        """
        folium.PolyLine(
            raw_coords, color=color, weight=3, opacity=0.5,
            tooltip=f"Camión {truck_number}"
        ).add_to(route_fg)"""

        # Retorno landfill → garage
        return_path   = vrp.get_path(vrp.landfill, vrp.garage)
        return_coords = [
            (G_latlon.nodes[n]['y'], G_latlon.nodes[n]['x'])
            for n in return_path if n in G_latlon.nodes
        ]
        raw_coords.extend(return_coords)

        route_fg.add_to(m)

        # Coordenadas densificadas para la animación JS
        smooth = densify_path(raw_coords, step_meters=STEP_METERS)
        all_routes_js.append([[c[0], c[1]] for c in smooth])

    # ── Serializar a JSON ─────────────────────────────────────────────────────
    import json
    routes_json = json.dumps(all_routes_js)
    colors_json = json.dumps([colors[i % len(colors)] for i in range(len(routes))])
    names_json  = json.dumps([f"Camión {i+1}" for i in range(len(routes))])
    map_var     = m.get_name()

    # Generar barras de control: una por camión
    control_bars = ""
    for i in range(len(routes)):
        color = colors[i % len(colors)]
        control_bars += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 10px;
                    background:rgba(255,255,255,0.07);border-radius:6px;">
            <span style="font-size:18px;">🚛</span>
            <span style="color:{color};font-weight:bold;min-width:75px;">
                Camión {i+1}
            </span>
            <button onclick="seekTruck({i},-100)"
                style="background:none;border:none;color:white;
                       font-size:16px;cursor:pointer;padding:2px 5px;">⏮️</button>
            <button id="btn-{i}" onclick="toggleTruck({i})"
                style="background:none;border:none;color:white;
                       font-size:16px;cursor:pointer;padding:2px 5px;">⏸️</button>
            <button onclick="seekTruck({i},100)"
                style="background:none;border:none;color:white;
                       font-size:16px;cursor:pointer;padding:2px 5px;">⏭️</button>
            <input id="bar-{i}" type="range" min="0" max="1000" value="0"
                   oninput="scrubTruck({i},this.value)"
                   style="flex:1;accent-color:{color};cursor:pointer;">
            <span style="opacity:0.7;font-size:11px;">vel:</span>
            <input id="spd-{i}" type="range" min="1" max="5" value="1"
                   oninput="setSpeed({i},this.value)"
                   style="width:55px;accent-color:#FF9800;">
            <span id="spd-label-{i}"
                  style="min-width:28px;font-size:12px;color:#FF9800;">x1</span>
        </div>"""

    # ── JavaScript animación ──────────────────────────────────────────────────
    js_code = f"""
    <script>
    (function() {{
        var checkMap = setInterval(function() {{
            if (typeof {map_var} === 'undefined') return;
            clearInterval(checkMap);

            var routesData  = {routes_json};
            var routeColors = {colors_json};
            var routeNames  = {names_json};
            var N           = routesData.length;
            var FPS         = 12;

            var markers     = new Array(N).fill(null);
            var polylines   = new Array(N).fill(null);
            var stepIndex   = new Array(N).fill(0);
            var trailCoords = new Array(N).fill(null).map(function(){{ return []; }});
            var playing     = new Array(N).fill(true);
            var speed       = new Array(N).fill(1);

            function truckIcon(color) {{
                var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">'
                    + '<rect x="2"  y="12" width="22" height="14" rx="2" fill="' + color + '" stroke="white" stroke-width="1.5"/>'
                    + '<rect x="24" y="16" width="10" height="10" rx="2" fill="' + color + '" stroke="white" stroke-width="1.5"/>'
                    + '<circle cx="8"  cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<circle cx="20" cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<circle cx="30" cy="27" r="3" fill="white" stroke="' + color + '" stroke-width="1.5"/>'
                    + '<rect x="4" y="14" width="8" height="6" rx="1" fill="rgba(255,255,255,0.35)"/>'
                    + '</svg>';
                return L.divIcon({{
                    html: '<div style="transform:translate(-18px,-18px)">' + svg + '</div>',
                    iconSize: [36,36], iconAnchor: [18,18], className: ''
                }});
            }}

            for (var i = 0; i < N; i++) {{
                if (!routesData[i] || routesData[i].length === 0) continue;
                var start = routesData[i][0];

                markers[i] = L.marker(start, {{ icon: truckIcon(routeColors[i]) }})
                    .bindTooltip(routeNames[i], {{
                        permanent: true, direction: 'right',
                        offset: [15,-10], className: 'truck-label'
                    }})
                    .addTo({map_var});

                polylines[i] = L.polyline([], {{
                    color: routeColors[i], weight: 2.5, opacity: 0.85
                }}).addTo({map_var});

                trailCoords[i] = [start];
            }}

            function updateBar(i) {{
                if (!routesData[i] || routesData[i].length === 0) return;
                var bar = document.getElementById('bar-' + i);
                if (bar) bar.value = (stepIndex[i] / routesData[i].length) * 1000;
            }}

            window.toggleTruck = function(i) {{
                playing[i] = !playing[i];
                var btn = document.getElementById('btn-' + i);
                if (btn) btn.innerText = playing[i] ? '⏸️' : '▶️';
            }};

            window.seekTruck = function(i, delta) {{
                if (!routesData[i]) return;
                stepIndex[i] = Math.max(0, Math.min(routesData[i].length, stepIndex[i] + delta));
                trailCoords[i] = routesData[i].slice(0, stepIndex[i]);
                if (polylines[i]) polylines[i].setLatLngs(trailCoords[i]);
                if (markers[i] && trailCoords[i].length > 0)
                    markers[i].setLatLng(trailCoords[i][trailCoords[i].length - 1]);
                updateBar(i);
            }};

            window.scrubTruck = function(i, val) {{
                if (!routesData[i]) return;
                stepIndex[i] = Math.floor((val / 1000) * routesData[i].length);
                trailCoords[i] = routesData[i].slice(0, stepIndex[i]);
                if (polylines[i]) polylines[i].setLatLngs(trailCoords[i]);
                if (markers[i] && trailCoords[i].length > 0)
                    markers[i].setLatLng(trailCoords[i][trailCoords[i].length - 1]);
            }};

            window.setSpeed = function(i, val) {{
                speed[i] = parseInt(val);
                var lbl = document.getElementById('spd-label-' + i);
                if (lbl) lbl.innerText = 'x' + val;
            }};

            function animate() {{
                for (var i = 0; i < N; i++) {{
                    if (!routesData[i] || routesData[i].length === 0) continue;
                    if (!playing[i]) continue;

                    var total = routesData[i].length;
                    var idx   = stepIndex[i];

                    if (idx >= total) {{
                        stepIndex[i]   = 0;
                        trailCoords[i] = [routesData[i][0]];
                        polylines[i].setLatLngs([]);
                        markers[i].setLatLng(routesData[i][0]);
                        updateBar(i);
                        continue;
                    }}

                    var pos = routesData[i][idx];
                    markers[i].setLatLng(pos);
                    trailCoords[i].push(pos);
                    polylines[i].setLatLngs(trailCoords[i]);
                    stepIndex[i] = Math.min(idx + speed[i], total);
                    updateBar(i);
                }}
            }}

            setInterval(animate, 1000 / FPS);
        }}, 200);
    }})();
    </script>

    <style>
    .truck-label {{
        background:  transparent !important;
        border:      none        !important;
        box-shadow:  none        !important;
        font-weight: bold;
        font-size:   12px;
        color:       #111;
        text-shadow: 1px 1px 2px white, -1px -1px 2px white;
    }}
    </style>
    """

    # ── Panel de controles (una barra por camión) ─────────────────────────────
    controls_html = f"""
    <div style="
        position:fixed; bottom:0; left:0; right:0;
        z-index:9999;
        background:rgba(20,20,20,0.93);
        color:white;
        padding:8px 16px;
        font-family:'Roboto',sans-serif;
        font-size:13px;
        display:flex;
        flex-direction:column;
        gap:5px;
        box-shadow:0 -3px 12px rgba(0,0,0,0.4);">
        {control_bars}
    </div>
    """

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    html_table = f"""
    <div style="
        position:fixed; bottom:calc({len(routes)} * 52px + 20px); right:50px;
        width:370px; background:white; z-index:9999; border-radius:8px;
        box-shadow:0 0 15px rgba(0,0,0,0.2);
        font-family:'Roboto',sans-serif; font-size:13px; overflow:hidden;">

        <div style="background:#333;color:white;padding:10px;text-align:center;">
            <h4 style="margin:0;">Rutas Optimizadas (Multi-Trip)</h4>
            <small>Solución Final del Algoritmo Genético</small>
        </div>

        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f2f2f2;text-align:left;">
                    <th style="padding:8px;">Vehículo</th>
                    <th style="padding:8px;">Carga Total</th>
                    <th style="padding:8px;">Dist.</th>
                    <th style="padding:8px;">Turno</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
                <tr style="background:#e8f5e9;font-weight:bold;border-top:2px solid #ccc;">
                    <td style="padding:8px;">TOTAL FLOTA</td>
                    <td style="padding:8px;">{total_fleet_load:.2f} t</td>
                    <td style="padding:8px;">{total_fleet_dist:.2f} km</td>
                    <td style="padding:8px;">{format_time(total_fleet_time)}</td>
                </tr>
                <tr style="background:#fff3e0;font-weight:bold;border-top:1px dashed #ccc;">
                    <td style="padding:8px;">CALIFICACIÓN</td>
                    <td colspan="3" style="padding:8px;text-align:center;color:#E65100;">
                        Índice: {(total_fleet_load / total_fleet_dist if total_fleet_dist > 0 else 0):.4f} t/km
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    m.get_root().html.add_child(folium.Element(html_table))
    m.get_root().html.add_child(folium.Element(controls_html))
    m.get_root().html.add_child(folium.Element(js_code))
    folium.LayerControl().add_to(m)
    m.save(filename)
    print(f"[OK] Mapa Multi-Trip guardado: {filename}")

def export_routes_to_geopackage(vrp_system, solution, 
                                input_gpkg="cuenca_limpieza.gpkg",
                                output_gpkg="cuenca_limpieza_optimizado.gpkg"):
    """
    Exporta las rutas optimizadas a un nuevo archivo GeoPackage.
    MEJORA: Rutas siguiendo calles reales + cada ruta en su propia capa.
    
    Parámetros:
    -----------
    vrp_system : VRPSystem
        Sistema VRP con el grafo y configuración
    solution : list
        Lista de rutas optimizadas (cada ruta es una lista de nodos)
    input_gpkg : str
        Ruta al archivo GPKG de entrada
    output_gpkg : str
        Ruta al archivo GPKG de salida con las rutas optimizadas
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point, MultiLineString
    import shutil
    
    print("\n" + "="*70)
    print(f"EXPORTANDO RUTAS OPTIMIZADAS A GEOPACKAGE")
    print("="*70)
    
    try:
        # Copiar el archivo original para mantener todas las capas base
        shutil.copy2(input_gpkg, output_gpkg)
        print(f"[OK] Archivo base copiado: {input_gpkg} -> {output_gpkg}")
        
        # Obtener el grafo en coordenadas lat/lon para exportación
        G_latlon = ox.project_graph(vrp_system.city_graph, to_crs='epsg:4326')
        
        # ====================================================================
        # EXPORTAR CADA RUTA EN SU PROPIA CAPA (COMO EN EL HTML)
        # ====================================================================
        
        for route_idx, route in enumerate(solution, start=1):
            print(f"  Procesando Ruta {route_idx}/{len(solution)}...")
            
            # Calcular métricas de la ruta
            total_distance_km, total_load, total_time = calculate_route_metrics(vrp_system, route)
            
            # ================================================================
            # CONSTRUIR RUTA SIGUIENDO CALLES REALES (NO LÍNEAS RECTAS)
            # ================================================================
            
            # Construir la ruta completa con todos los segmentos
            full_route_nodes = []
            
            # 1. Garage -> Primer cliente
            path_to_first = vrp_system.get_path(vrp_system.garage, route[0])
            full_route_nodes.extend(path_to_first)
            
            # 2. Entre clientes
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                
                # Si el siguiente nodo es el landfill (descarga intermedia)
                if next_node == vrp_system.landfill:
                    path_segment = vrp_system.get_path(current_node, next_node)
                    # Evitar duplicar el nodo actual
                    full_route_nodes.extend(path_segment[1:])
                else:
                    # Camino normal entre clientes
                    path_segment = vrp_system.get_path(current_node, next_node)
                    full_route_nodes.extend(path_segment[1:])
            
            # 3. Último cliente -> Landfill
            path_to_landfill = vrp_system.get_path(route[-1], vrp_system.landfill)
            full_route_nodes.extend(path_to_landfill[1:])
            
            # 4. Landfill -> Garage (retorno)
            path_to_garage = vrp_system.get_path(vrp_system.landfill, vrp_system.garage)
            full_route_nodes.extend(path_to_garage[1:])
            
            # ================================================================
            # CONVERTIR NODOS A COORDENADAS GEOGRÁFICAS
            # ================================================================
            
            route_coords = []
            for node in full_route_nodes:
                if node in G_latlon.nodes:
                    node_data = G_latlon.nodes[node]
                    route_coords.append((node_data['x'], node_data['y']))
            
            # ================================================================
            # CREAR GEOMETRÍA DE LA RUTA (SIGUIENDO CALLES)
            # ================================================================
            
            if len(route_coords) >= 2:
                # Crear LineString que sigue las calles reales
                route_geometry = LineString(route_coords)
                
                # Crear GeoDataFrame para esta ruta
                route_data = {
                    'route_id': [route_idx],
                    'num_stops': [len(route)],
                    'distance_km': [round(total_distance_km, 2)],
                    'load_tons': [round(total_load, 2)],
                    'time_hours': [round(total_time, 2)],
                    'num_segments': [len(route_coords)],
                    'nodes': ['|'.join(map(str, route))],
                    'geometry': [route_geometry]
                }
                
                gdf_route = gpd.GeoDataFrame(route_data, crs='epsg:4326')
                
                # Guardar en capa individual
                layer_name = f'route_{route_idx}'
                gdf_route.to_file(output_gpkg, layer=layer_name, driver='GPKG')
                
                print(f"    [OK] Capa '{layer_name}' creada: {len(route_coords)} segmentos, "
                      f"{total_distance_km:.2f} km, {total_load:.2f} ton")
        
        # ====================================================================
        # CREAR CAPA RESUMEN CON TODAS LAS RUTAS (OPCIONAL)
        # ====================================================================
        
        print("\n  Creando capa resumen con todas las rutas...")
        
        all_routes_features = []
        
        for route_idx, route in enumerate(solution, start=1):
            total_distance_km, total_load, total_time = calculate_route_metrics(vrp_system, route)
            
            # Construir ruta completa (mismo proceso que arriba)
            full_route_nodes = []
            
            path_to_first = vrp_system.get_path(vrp_system.garage, route[0])
            full_route_nodes.extend(path_to_first)
            
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                path_segment = vrp_system.get_path(current_node, next_node)
                full_route_nodes.extend(path_segment[1:])
            
            path_to_landfill = vrp_system.get_path(route[-1], vrp_system.landfill)
            full_route_nodes.extend(path_to_landfill[1:])
            
            path_to_garage = vrp_system.get_path(vrp_system.landfill, vrp_system.garage)
            full_route_nodes.extend(path_to_garage[1:])
            
            route_coords = []
            for node in full_route_nodes:
                if node in G_latlon.nodes:
                    node_data = G_latlon.nodes[node]
                    route_coords.append((node_data['x'], node_data['y']))
            
            if len(route_coords) >= 2:
                route_geometry = LineString(route_coords)
                
                all_routes_features.append({
                    'route_id': route_idx,
                    'num_stops': len(route),
                    'distance_km': round(total_distance_km, 2),
                    'load_tons': round(total_load, 2),
                    'time_hours': round(total_time, 2),
                    'nodes': '|'.join(map(str, route)),
                    'geometry': route_geometry
                })
        
        # Guardar capa resumen
        if all_routes_features:
            gdf_all_routes = gpd.GeoDataFrame(all_routes_features, crs='epsg:4326')
            gdf_all_routes.to_file(output_gpkg, layer='all_routes_summary', driver='GPKG')
            print(f"  [OK] Capa 'all_routes_summary' creada: {len(solution)} rutas")
        
        # ====================================================================
        # CREAR CAPA DE PUNTOS DE PARADA (STOPS)
        # ====================================================================
        
        print("\n  Creando capa de paradas...")
        
        stop_features = []
        
        for route_idx, route in enumerate(solution, start=1):
            for stop_order, node in enumerate(route, start=1):
                if node in G_latlon.nodes:
                    node_data = G_latlon.nodes[node]
                    
                    # Obtener demanda del nodo
                    demand = vrp_system.city_graph.nodes[node].get('demand', 0.0)
                    
                    # Determinar tipo de parada
                    if node == vrp_system.garage:
                        stop_type = 'garage'
                    elif node == vrp_system.landfill:
                        stop_type = 'landfill'
                    else:
                        stop_type = 'customer'
                    
                    stop_features.append({
                        'route_id': route_idx,
                        'stop_order': stop_order,
                        'node_id': node,
                        'stop_type': stop_type,
                        'demand_tons': round(demand, 3),
                        'geometry': Point(node_data['x'], node_data['y'])
                    })
        
        # Crear GeoDataFrame con las paradas
        gdf_stops = gpd.GeoDataFrame(stop_features, crs='epsg:4326')
        
        # Guardar capa de paradas en el GPKG
        gdf_stops.to_file(output_gpkg, layer='all_stops', driver='GPKG')
        print(f"  [OK] Capa 'all_stops' agregada: {len(gdf_stops)} paradas")

        # ====================================================================
        # CREAR CAPA DE MAPA DE CALOR (HEATMAP) - DENSIDAD DE BASURA
        # ====================================================================
        
        print("\n  Creando mapa de calor de densidad...")
        
        import numpy as np
        
        # Obtener nodos con demanda
        customer_nodes = [n for n in vrp_system.customers 
                         if vrp_system.city_graph.nodes[n].get('demand', 0) > 0]
        
        heatmap_features = []
        
        if customer_nodes:
            # Calcular demandas
            demands = [vrp_system.city_graph.nodes[n].get('demand', 0) 
                      for n in customer_nodes]
            max_demand = max(demands)
            min_demand = min(demands)
            
            for node in customer_nodes:
                if node in G_latlon.nodes:
                    node_data = G_latlon.nodes[node]
                    demand = vrp_system.city_graph.nodes[node].get('demand', 0)
                    
                    # Normalizar demanda
                    if max_demand > min_demand:
                        normalized = (demand - min_demand) / (max_demand - min_demand)
                    else:
                        normalized = 1.0
                    
                    # Crear punto
                    point = Point(node_data['x'], node_data['y'])
                    
                    # Radio proporcional a la demanda (en grados)
                    base_radius = 0.0005  # ~50m
                    radius = base_radius + (normalized * base_radius * 3.0)
                    
                    # Crear círculo (buffer)
                    circle = point.buffer(radius)
                    
                    # Categoría de densidad
                    if normalized < 0.2:
                        category = 'Muy Bajo'
                    elif normalized < 0.4:
                        category = 'Bajo'
                    elif normalized < 0.6:
                        category = 'Medio'
                    elif normalized < 0.8:
                        category = 'Alto'
                    else:
                        category = 'Muy Alto'
                    
                    heatmap_features.append({
                        'node_id': node,
                        'demand_tons': round(demand, 3),
                        'normalized_demand': round(normalized, 3),
                        'density_category': category,
                        'geometry': circle
                    })
            
            # Crear GeoDataFrame del heatmap
            gdf_heatmap = gpd.GeoDataFrame(heatmap_features, crs='epsg:4326')
            gdf_heatmap.to_file(output_gpkg, layer='heatmap_density', driver='GPKG')
            print(f"  [OK] Capa 'heatmap_density' agregada: {len(gdf_heatmap)} círculos")
        
        # ====================================================================
        # CREAR CAPA DE INFORMACIÓN DEL PROYECTO
        # ====================================================================
        
        print("\n  Creando capa de información del proyecto...")
        
        try:
            # Cargar nodos del subgrafo para calcular área
            gdf_subgraph = gpd.read_file(input_gpkg, layer='subgraph_nodes')
            
            # Calcular área
            from shapely.ops import unary_union
            all_points = gdf_subgraph.geometry.unary_union
            area_polygon = all_points.convex_hull
            
            # Proyectar para calcular área en km²
            gdf_temp = gpd.GeoDataFrame({'geometry': [area_polygon]}, crs='epsg:4326')
            gdf_projected = gdf_temp.to_crs('epsg:32717')
            area_km2 = 2.42
            
            # Datos del proyecto
            poblacion = 47992  # Población estimada del centro histórico
            densidad = poblacion / area_km2 if area_km2 > 0 else 0
            
            # Calcular basura total recolectada
            total_basura = sum(
                sum(vrp_system.city_graph.nodes[node].get('demand', 0) for node in route)
                for route in solution
            )
            
            # Calcular distancia y tiempo total
            total_dist = 0
            total_time = 0
            for route in solution:
                dist, load, time = calculate_route_metrics(vrp_system, route)
                total_dist += dist
                total_time += time
            
            # Centroide para label
            centroid = area_polygon.centroid
            
            # Texto informativo
            info_text = (
                f"CENTRO HISTÓRICO DE CUENCA\\n"
                f"═══════════════════════════\\n"
                f"ÁREA Y POBLACIÓN\\n"
                f"Área: {area_km2:.2f} km²\\n"
                f"Población: {poblacion:,} hab\\n"
                f"Densidad: {densidad:.0f} hab/km²\\n"
                f"\\n"
                f"RECOLECCIÓN DE BASURA\\n"
                f"Puntos: {len(vrp_system.customers)}\\n"
                f"Basura: {total_basura:.2f} ton\\n"
                f"\\n"
                f"OPTIMIZACIÓN\\n"
                f"Rutas: {len(solution)}\\n"
                f"Distancia: {total_dist:.2f} km\\n"
                f"Tiempo: {total_time:.1f} h"
            )
            
            # Crear GeoDataFrame
            metadata_data = {
                'type': ['project_info'],
                'area_km2': [round(area_km2, 2)],
                'poblacion': [poblacion],
                'densidad_hab_km2': [round(densidad, 0)],
                'num_clientes': [len(vrp_system.customers)],
                'num_rutas': [len(solution)],
                'basura_total_ton': [round(total_basura, 2)],
                'distancia_total_km': [round(total_dist, 2)],
                'tiempo_total_h': [round(total_time, 1)],
                'label_text': [info_text],
                'geometry': [Point(centroid.x, centroid.y)]
            }
            
            gdf_metadata = gpd.GeoDataFrame(metadata_data, crs='epsg:4326')
            gdf_metadata.to_file(output_gpkg, layer='project_metadata', driver='GPKG')
            print(f"  [OK] Capa 'project_metadata' agregada")
            print(f"       Área: {area_km2:.2f} km², Densidad: {densidad:.0f} hab/km²")
            
        except Exception as e:
            print(f"  [WARNING] No se pudo crear metadata: {e}")
        
        # ====================================================================
        # RESUMEN DE EXPORTACIÓN
        # ====================================================================
        
        print(f"\n{'='*70}")
        print(f"[OK] GeoPackage exportado exitosamente: {output_gpkg}")
        print(f"{'='*70}")
        print(f"\nCAPAS DE RUTAS INDIVIDUALES:")
        for i in range(1, len(solution) + 1):
            print(f"  - 'route_{i}': Ruta {i} (siguiendo calles reales)")
        
        print(f"\nCAPAS ADICIONALES:")
        print(f"  - 'all_routes_summary': Todas las rutas juntas (resumen)")
        print(f"  - 'all_stops': Todas las paradas (puntos)")

        print(f"\nCAPAS ORIGINALES CONSERVADAS:"+f"\n  - 'all_nodes': Todos los nodos de Cuenca"+f"\n  - 'all_edges': Todas las calles de Cuenca")
        print(f"  - 'subgraph_nodes': Nodos con basura")
        print(f"  - 'subgraph_edges': Calles del área de trabajo")
        print(f"  - 'special_points': Garage y Relleno Sanitario")
        
        print("\n" + "="*70)
        print("VISUALIZACIÓN EN QGIS:")
        print("="*70)
        print("1. Abre QGIS")
        print("2. Layer > Add Layer > Add Vector Layer")
        print(f"3. Selecciona '{output_gpkg}'")
        print("\n4. CARGAR RUTAS INDIVIDUALES:")
        print("   - Carga 'route_1', 'route_2', etc.")
        print("   - Cada ruta es una capa independiente")
        print("   - Puedes activar/desactivar cada ruta individualmente")
        print("   - Las rutas SIGUEN LAS CALLES REALES (no son líneas rectas)")
        print("\n5. SIMBOLIZAR POR RUTA:")
        print("   - Click derecho en cada capa 'route_X' > Properties > Symbology")
        print("   - Asigna un color diferente a cada ruta")
        print("   - Grosor de línea recomendado: 2-3 pt")
        print("\n6. VER TODAS LAS RUTAS JUNTAS:")
        print("   - Carga 'all_routes_summary'")
        print("   - Simbolizar por 'route_id' con colores diferentes")
        print("\n7. VER PARADAS:")
        print("   - Carga 'all_stops'")
        print("   - Simbolizar por 'route_id' para ver a qué ruta pertenece cada parada")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Error al exportar rutas a GPKG: {e}")
        import traceback
        traceback.print_exc()

"""
if __name__ == "__main__":
    random.seed(42)
    vrp = VRPSystem()
    
    print(f"\nOptimizando {len(vrp.customers)} clientes con GA...")

    CONFIG['MAX_SHIFT_TIME'] = 10.0   # Amplía un poco el turno si es necesario
    N_TRUCKS = 2

    # Población mixta: 20% heurística (NN), 80% aleatoria
    nn_count = int(CONFIG['POPULATION_SIZE'] * 0.2)
    random_count = CONFIG['POPULATION_SIZE'] - nn_count
    
    print(f"Generando población inicial:")
    print(f"  - {nn_count} individuos con Nearest Neighbor (randomness=0.3)"+f"\n  - {random_count} individuos aleatorios")
    
    pop = []
    
    # Generar individuos con Nearest Neighbor
    for _ in range(nn_count):
        pop.append(create_nearest_neighbor_individual(vrp, randomness=0.17))
    
    # Generar individuos aleatorios
    for _ in range(random_count):
        pop.append(create_individual(vrp.customers))

    print("Capturando estado inicial (antes de optimizar)...")


    
    initial_chromosome = pop[0]
    _, initial_routes_structure = evaluate_chromosome(initial_chromosome, vrp)
    export_initial_solution_to_html(vrp, initial_routes_structure, filename="solucion_inicial.html")

    # AGREGAR ESTO:
    print("\n" + "="*70)
    print("ANÁLISIS DE LA SOLUCIÓN INICIAL (SIN OPTIMIZAR)")
    print("="*70)
    print_detailed_solution_analysis(vrp, initial_routes_structure, 
                                 title="SOLUCIÓN INICIAL (SIN OPTIMIZAR)")

    _, initial_routes = evaluate_chromosome(pop[0], vrp, n_trucks=N_TRUCKS)
    export_initial_solution_to_html(vrp, initial_routes, "solucion_inicial.html")
    print_detailed_solution_analysis(vrp, initial_routes, "SOLUCIÓN INICIAL")

    best_fitness = float('inf') 
    best_sol = []
    
    for gen in range(CONFIG['GENERATIONS']):
        scored = []
        for ind in pop:
            fitness, r = evaluate_chromosome(ind, vrp, n_trucks=N_TRUCKS)
            #fitness, r = evaluate_chromosome(ind, vrp)
            scored.append((fitness, ind, r))
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_sol = r
        
        scored.sort(key=lambda x: x[0])
        elite = [s[1] for s in scored[:10]]
        new_pop = elite[:]

        while len(new_pop) < CONFIG['POPULATION_SIZE']:
            p1 = random.choice(scored[:30])[1]
            p2 = random.choice(scored[:30])[1]
            child = ordered_crossover(p1, p2)
            # Aplicar mutación: 50% swap, 50% inversion (2-opt)
            if random.random() < 0.5:
                child = mutate(child)  # Swap mutation
            else:
                child = inversion_mutation(child)  # 2-Opt mutation
            
            new_pop.append(child)
        pop = new_pop
        
        if gen % 50 == 0:
            print(f"Gen {gen} | Fitness: {best_fitness:.2f} | Rutas: {len(best_sol)}")

    assert len(best_sol) <= N_TRUCKS, f"[ERROR] Se generaron {len(best_sol)} rutas, esperadas {N_TRUCKS}"

    print("\n" + "="*60)
    print(f"       RESULTADOS FINALES REALES (Fitness GA: {best_fitness:.2f})")
    print("="*60)
    
    total_real_km = 0.0
    total_real_load = 0.0
    
    print(f"{'Camión':<10} | {'Carga (t)':<12} | {'Dist. (km)':<12} | {'Tiempo':<10}")
    print("-" * 60)
    
    for i, route in enumerate(best_sol):
        r_km, r_load, r_time = calculate_route_metrics(vrp, route)
        
        total_real_km += r_km
        total_real_load += r_load
        
        hours = int(r_time)
        minutes = int((r_time - hours) * 60)
        time_str = f"{hours}h {minutes:02d}m"
        
        print(f"Ruta {i+1:<5} | {r_load:<12.2f} | {r_km:<12.2f} | {time_str}")

    print("-" * 60)
    print(f"TOTAL FLOTA: {total_real_load:.2f} ton | {total_real_km:.2f} km")
    print("="*60)
    
    print_detailed_solution_analysis(vrp, best_sol, 
                                 title="SOLUCIÓN OPTIMIZADA FINAL")
    
    # ====================================================================
    # EXPORTAR RESULTADOS
    # ====================================================================

    # 1. Exportar HTML animado de rutas
    try:
        export_routes_to_html(vrp, best_sol, filename="rutas_optimizadas.html")
        print("[OK] HTML de rutas generado: rutas_optimizadas.html")
    except Exception as e:
        print(f"[ERROR] HTML: {e}")
        import traceback
        traceback.print_exc()

    # 2. Exportar rutas a GeoPackage (NUEVO)
    try:
        export_routes_to_geopackage(
            vrp_system=vrp,
            solution=best_sol,
            input_gpkg="cuenca_limpieza.gpkg",
            output_gpkg="cuenca_limpieza_optimizado.gpkg"
        )
        print("[OK] GeoPackage optimizado generado: cuenca_limpieza_optimizado.gpkg")
    except Exception as e:
        print(f"[ERROR] GeoPackage: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("PROCESO FINALIZADO - ARCHIVOS GENERADOS:")
    print("="*70)
    print("1. rutas_optimizadas.html - Visualización web animada")
    print("2. cuenca_limpieza_optimizado.gpkg - GeoPackage con rutas optimizadas")
    print("="*70 + "\n")
"""

if __name__ == "__main__":
    random.seed(42)
    vrp = VRPSystem()
 
    print(f"\nOptimizando {len(vrp.customers)} clientes con GA (2 camiones, split variable)...")
 
    # ── Parámetros desde CONFIG (ya conectados) ──────────────────────────────
    CONFIG['MAX_SHIFT_TIME'] = 10.0
    N_TRUCKS = 2
    POP_SIZE       = CONFIG['POPULATION_SIZE']      # 60
    GENERATIONS    = CONFIG['GENERATIONS']           # 400
    MUT_PROB       = CONFIG['MUTATION_PROBABILITY']  # 0.25  ← antes ignorado
    TOURN_SIZE     = CONFIG['TOURNAMENT_SIZE']       # 5     ← antes ignorado
    ELITE_SIZE     = 10
    NN_RATIO       = 0.20                            # 20% Nearest Neighbor
 
    # ── Población inicial mixta ───────────────────────────────────────────────
    nn_count     = int(POP_SIZE * NN_RATIO)
    random_count = POP_SIZE - nn_count
 
    print(f"Generando población inicial:")
    print(f"  - {nn_count} individuos Nearest Neighbor (randomness=0.17)")
    print(f"  - {random_count} individuos aleatorios")
 
    pop = []
    for _ in range(nn_count):
        pop.append(create_nearest_neighbor_individual(vrp, randomness=0.17))
    for _ in range(random_count):
        pop.append(create_individual(vrp.customers))
 
    # ── Solución inicial (antes de optimizar) ────────────────────────────────
    print("\n" + "="*70)
    print("ANÁLISIS DE LA SOLUCIÓN INICIAL (SIN OPTIMIZAR)")
    print("="*70)
    _, initial_routes = evaluate_chromosome(pop[0], vrp, n_trucks=N_TRUCKS)
    export_initial_solution_to_html(vrp, initial_routes, "solucion_inicial.html")
    print_detailed_solution_analysis(vrp, initial_routes, "SOLUCIÓN INICIAL")
 
    # ── Bucle evolutivo ───────────────────────────────────────────────────────
    best_fitness = float('inf')
    best_sol = []
 
    for gen in range(GENERATIONS):
        # Evaluar toda la población
        scored = []
        for ind in pop:
            fitness, r = evaluate_chromosome(ind, vrp, n_trucks=N_TRUCKS)
            scored.append((fitness, ind, r))
            if fitness < best_fitness:
                best_fitness = fitness
                best_sol = r
 
        scored.sort(key=lambda x: x[0])
 
        # Elitismo: conservar los mejores sin cambios
        elite = [s[1] for s in scored[:ELITE_SIZE]]
        new_pop = elite[:]
 
        # Generar el resto mediante selección por torneo + cruce + mutación
        while len(new_pop) < POP_SIZE:
            # Selección por torneo real (usa TOURN_SIZE de CONFIG)
            p1 = tournament_selection(scored, TOURN_SIZE)
            p2 = tournament_selection(scored, TOURN_SIZE)
 
            child = ordered_crossover(p1, p2)
 
            # Mutación con probabilidad MUT_PROB (usa MUTATION_PROBABILITY de CONFIG)
            if random.random() < MUT_PROB:
                tipo = random.random()
                if tipo < 0.40:
                    child = mutate(child)            # Swap: reordena clientes
                elif tipo < 0.75:
                    child = inversion_mutation(child) # 2-Opt: invierte segmento
                else:
                    child = split_mutation(child)     # Nuevo: cambia la división
 
            new_pop.append(child)
 
        pop = new_pop
 
        if gen % 50 == 0:
            split_actual = best_sol  # ya es la estructura de rutas
            n1 = len(best_sol[0]) if len(best_sol) > 0 else 0
            n2 = len(best_sol[1]) if len(best_sol) > 1 else 0
            print(f"Gen {gen:>3} | Fitness: {best_fitness:>10.2f} | "
                  f"Clientes C1={n1}, C2={n2}")
 
    assert len(best_sol) <= N_TRUCKS, \
        f"[ERROR] Se generaron {len(best_sol)} rutas, esperadas {N_TRUCKS}"
 
    # ── Resultados finales ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"       RESULTADOS FINALES (Fitness GA: {best_fitness:.2f})")
    print("="*60)
 
    total_real_km   = 0.0
    total_real_load = 0.0
 
    print(f"{'Camión':<10} | {'Carga (t)':<12} | {'Dist. (km)':<12} | {'Tiempo':<10} | {'Clientes':<10}")
    print("-" * 65)
 
    for i, route in enumerate(best_sol):
        r_km, r_load, r_time = calculate_route_metrics(vrp, route)
        total_real_km   += r_km
        total_real_load += r_load
        hours   = int(r_time)
        minutes = int((r_time - hours) * 60)
        n_clientes = sum(1 for n in route if n != vrp.landfill)
        print(f"Ruta {i+1:<5} | {r_load:<12.2f} | {r_km:<12.2f} | "
              f"{hours}h {minutes:02d}m    | {n_clientes:<10}")
 
    print("-" * 65)
    print(f"TOTAL FLOTA: {total_real_load:.2f} ton | {total_real_km:.2f} km")
    print("="*60)
 
    print_detailed_solution_analysis(vrp, best_sol, title="SOLUCIÓN OPTIMIZADA FINAL")
 
    # ── Exportar resultados ───────────────────────────────────────────────────
    try:
        export_routes_to_html(vrp, best_sol, filename="rutas_optimizadas.html")
        print("[OK] HTML de rutas generado: rutas_optimizadas.html")
    except Exception as e:
        print(f"[ERROR] HTML: {e}")
        import traceback; traceback.print_exc()
 
    try:
        export_routes_to_geopackage(
            vrp_system=vrp,
            solution=best_sol,
            input_gpkg="cuenca_limpieza.gpkg",
            output_gpkg="cuenca_limpieza_optimizado.gpkg"
        )
        print("[OK] GeoPackage optimizado generado: cuenca_limpieza_optimizado.gpkg")
    except Exception as e:
        print(f"[ERROR] GeoPackage: {e}")
        import traceback; traceback.print_exc()
 
    print("\n" + "="*70)
    print("PROCESO FINALIZADO - ARCHIVOS GENERADOS:")
    print("="*70)
    print("1. rutas_optimizadas.html          - Visualización web animada")
    print("2. cuenca_limpieza_optimizado.gpkg - GeoPackage con rutas optimizadas")
    print("="*70 + "\n")
