import csv
import networkx as nx
import osmnx as ox
import os

class GeneticAlgorithmRoutes:
    
    
    def __init__(self, graph_file="cuenca_graph.graphml"):
        self.graph = None
        self.subgraph = None
        self.routes = []
        
        if os.path.exists(graph_file):
            print(f"Cargando grafo desde {graph_file}...")
            self.graph = ox.load_graphml(graph_file)
            print(f"[OK] Grafo cargado: {len(self.graph.nodes)} nodos\n")
        else:
            raise FileNotFoundError(f"No se encontró el archivo {graph_file}")
    
    def load_routes_from_csv(self, filename="rutas_iniciales.csv"):
        routes = []
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nodes_str = row['nodes']
                    nodes = [int(node_id) for node_id in nodes_str.split('|')]
                    routes.append(nodes)
            
            self.routes = routes
            print(f"Cargadas {len(routes)} rutas desde {filename}")
            
            lengths = [len(r) for r in routes]
            print(f"  Longitud promedio: {sum(lengths)/len(lengths):.1f} nodos")
            print(f"  Rango: {min(lengths)} - {max(lengths)} nodos\n")
            
            return routes
            
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {filename}")
            return []
        except Exception as e:
            print(f"Error al cargar rutas: {e}")
            return []
    
    def load_subgraph_nodes_from_csv(self, filename="subgraph_nodes.csv"):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                nodes = [int(row['node_id']) for row in reader]
            
            self.set_subgraph(nodes)
            print(f"Subgrafo completo cargado desde {filename}\n")
            return True
            
        except FileNotFoundError:
            print(f"No se encontró {filename}")
            print(f"   El subgrafo se creará solo con los nodos de las rutas.\n")
            return False
        except Exception as e:
            print(f"✗ Error al cargar subgrafo: {e}\n")
            return False
    
    def set_subgraph(self, node_ids):
        if self.graph is None:
            raise ValueError("Primero debes cargar el grafo")
        
        valid_nodes = [n for n in node_ids if n in self.graph.nodes]
        
        if not valid_nodes:
            raise ValueError("Ninguno de los nodos proporcionados existe en el grafo")
        
        self.subgraph = self.graph.subgraph(valid_nodes).copy()
        print(f"[OK] Subgrafo creado: {len(self.subgraph.nodes)} nodos, {len(self.subgraph.edges)} aristas")
    
    def validate_route(self, route, use_subgraph=True):
        if not route or len(route) < 2:
            return False, "Ruta vacía o con menos de 2 nodos"
        
        graph = self.subgraph if use_subgraph and self.subgraph else self.graph
        
        if graph is None:
            return False, "No hay grafo cargado"
        
        for node in route:
            if node not in graph.nodes:
                return False, f"Nodo {node} no existe en el grafo"
        
        for i in range(len(route) - 1):
            current = route[i]
            next_node = route[i + 1]
            
            if not graph.has_edge(current, next_node):
                try:
                    path = nx.shortest_path(graph, current, next_node, weight='length')
                    if len(path) > 2:  # Hay nodos intermedios faltantes
                        return False, f"Nodos {current} y {next_node} no están directamente conectados (falta(n) {len(path)-2} nodo(s) intermedio(s))"
                except nx.NetworkXNoPath:
                    return False, f"No hay camino entre nodos {current} y {next_node}"
        
        return True, "Ruta válida"
    
    def repair_route(self, route, use_subgraph=True):
        if not route or len(route) < 2:
            return None
        
        graph = self.subgraph if use_subgraph and self.subgraph else self.graph
        
        if graph is None:
            return None
        
        valid_nodes = [n for n in route if n in graph.nodes]
        if len(valid_nodes) < 2:
            return None
        
        repaired = [valid_nodes[0]]
        
        for i in range(len(valid_nodes) - 1):
            current = valid_nodes[i]
            next_node = valid_nodes[i + 1]
            
            if graph.has_edge(current, next_node):
                repaired.append(next_node)
            else:
                try:
                    path = nx.shortest_path(graph, current, next_node, weight='length')
                    repaired.extend(path[1:])
                except nx.NetworkXNoPath:
                    continue
        
        is_valid, _ = self.validate_route(repaired, use_subgraph)
        
        return repaired if is_valid else None
    
    def validate_population(self, population, repair_invalid=True, use_subgraph=True):
        valid_routes = []
        repaired_routes = []
        invalid_routes = []
        
        for route in population:
            is_valid, message = self.validate_route(route, use_subgraph)
            
            if is_valid:
                valid_routes.append(route)
            else:
                if repair_invalid:
                    repaired = self.repair_route(route, use_subgraph)
                    if repaired:
                        repaired_routes.append(repaired)
                    else:
                        invalid_routes.append(route)
                else:
                    invalid_routes.append(route)
        
        return valid_routes, repaired_routes, invalid_routes
    
    def save_routes_to_csv(self, routes, filename="rutas_resultado.csv"):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['route_id', 'length', 'nodes'])
            
            for i, route in enumerate(routes, start=1):
                nodes_str = '|'.join(map(str, route))
                writer.writerow([i, len(route), nodes_str])
        
        print(f"[OK] Rutas guardadas en: {filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ALGORITMO GENÉTICO - SISTEMA DE VALIDACIÓN DE RUTAS")
    print("="*70 + "\n")
    
    ga = GeneticAlgorithmRoutes(graph_file="cuenca_graph.graphml")
    
    rutas = ga.load_routes_from_csv("rutas_iniciales.csv")
    
    if not rutas:
        print("No se pudieron cargar las rutas. Verifica el archivo CSV.")
        exit(1)
    
    subgraph_loaded = ga.load_subgraph_nodes_from_csv("subgraph_nodes.csv")
    
    if not subgraph_loaded:
        print("Creando subgrafo con nodos de las rutas...")
        all_nodes = set()
        for route in rutas:
            all_nodes.update(route)
        ga.set_subgraph(list(all_nodes))
        print()
    
    print("="*70)
    print("VALIDANDO RUTAS CARGADAS")
    print("="*70)
    
    for i, route in enumerate(rutas[:3], start=1):
        is_valid, message = ga.validate_route(route, use_subgraph=True)
        status = "[OK] VÁLIDA" if is_valid else "✗ INVÁLIDA"
        print(f"Ruta {i}: {status} - {message}")
        print(f"  Longitud: {len(route)} nodos")
        print(f"  Primeros nodos: {route[:5]}...\n")
    
    print("="*70)
    print("EJEMPLO: SIMULACIÓN DE OPERADORES GENÉTICOS")
    print("="*70 + "\n")
    
    ruta_original = rutas[0].copy()
    ruta_mutada = ruta_original.copy()
    
    if len(ruta_mutada) > 4:
        ruta_mutada[2:5] = ruta_mutada[2:5][::-1]
    
    print("Ruta original (primeros 10 nodos):", ruta_original[:10])
    print("Ruta mutada (primeros 10 nodos):  ", ruta_mutada[:10])
    
    is_valid, message = ga.validate_route(ruta_mutada, use_subgraph=True)
    print(f"\n¿Es válida la ruta mutada? {is_valid}")
    print(f"Mensaje: {message}\n")
    
    if not is_valid:
        print("Intentando reparar la ruta...")
        ruta_reparada = ga.repair_route(ruta_mutada, use_subgraph=True)
        
        if ruta_reparada:
            print("[OK] Ruta reparada exitosamente")
            print(f"  Longitud original: {len(ruta_mutada)} nodos")
            print(f"  Longitud reparada: {len(ruta_reparada)} nodos")
            print(f"  Primeros 10 nodos reparados: {ruta_reparada[:10]}")
        else:
            print("✗ No se pudo reparar la ruta")
    
    print("\n" + "="*70)
    print("VALIDACIÓN DE POBLACIÓN COMPLETA")
    print("="*70 + "\n")
    
    valid, repaired, invalid = ga.validate_population(rutas, repair_invalid=True)
    
    print(f"Rutas válidas originalmente: {len(valid)}")
    print(f"Rutas reparadas: {len(repaired)}")
    print(f"Rutas inválidas (no reparables): {len(invalid)}")
    
    all_valid_routes = valid + repaired
    if all_valid_routes:
        ga.save_routes_to_csv(all_valid_routes, "rutas_validadas.csv")
    
    print("\n" + "="*70)
    print("✅ EJEMPLO COMPLETADO")
    print("="*70)
    print("\n📝 RESUMEN:")
    print("   - Las rutas se cargan desde 'rutas_iniciales.csv'")
    print("   - Usa validate_route() después de cada cruce/mutación")
    print("   - Usa repair_route() para arreglar rutas inválidas automáticamente")
    print("   - Usa validate_population() para validar toda una generación")
    print("\n💡 IMPORTANTE PARA EL AG:")
    print("   - Después de CRUCE: validar hijos y reparar si es necesario")
    print("   - Después de MUTACIÓN: validar individuo y reparar si es necesario")
    print("   - Antes de EVALUAR FITNESS: asegurar que toda la población sea válida")
    print("="*70 + "\n")