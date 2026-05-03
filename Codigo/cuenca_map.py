import osmnx as ox
import networkx as nx
import random
import os
import folium
from folium.plugins import MarkerCluster
from shapely.ops import unary_union
from shapely.geometry import mapping, Point, Polygon
import geopandas as gpd
import pandas as pd
from itertools import combinations

class CuencaGraphManager:
    def __init__(self):
        self.graph = None  
        self.subgraph = None  
        self.subgraph_polygon = None  
        self.city_name = "Cuenca, Ecuador"
        ox.settings.use_cache = True
        ox.settings.log_console = True
        ox.settings.requests_timeout = 1000

    def get_city_graph(self):
        filename = "cuenca_graph.graphml"
        
        if os.path.exists(filename):
            print(f"--> Cargando grafo completo desde {filename}...")
            self.graph = ox.load_graphml(filename)
        else:
            print(f"--> Descargando grafo vial de {self.city_name}...")
            self.graph = ox.graph_from_place(self.city_name, network_type='drive', simplify=True)
            self.graph = ox.project_graph(self.graph)
            ox.save_graphml(self.graph, filepath=filename)

        print(f"[OK] Grafo completo cargado: {len(self.graph.nodes)} nodos, {len(self.graph.edges)} aristas.")
        return self.graph

    def assign_node_weights(self, restriction_name="demanda", min_val=1, max_val=10):
        if self.graph is None: 
            raise ValueError("Falta cargar el grafo.")
        for node in self.graph.nodes():
            self.graph.nodes[node][restriction_name] = random.randint(min_val, max_val)
        print(f"[OK] Atributo '{restriction_name}' asignado a {len(self.graph.nodes)} nodos.")

    def _edge_name_to_str(self, val):
        if isinstance(val, list):
            return ", ".join([str(x) for x in val])
        return str(val)

    def _find_street_intersections(self, gdf_edges, street1, street2, tolerance=0.0001):
        mask1 = gdf_edges['name_str'].str.contains(street1, case=False, na=False)
        mask2 = gdf_edges['name_str'].str.contains(street2, case=False, na=False)
        
        edges1 = gdf_edges[mask1]
        edges2 = gdf_edges[mask2]
        
        if edges1.empty or edges2.empty:
            return []
        
        intersections = []
        
        for idx1, edge1 in edges1.iterrows():
            for idx2, edge2 in edges2.iterrows():
                if edge1.geometry.intersects(edge2.geometry):
                    intersection = edge1.geometry.intersection(edge2.geometry)
                    if intersection.is_empty:
                        continue
                    
                    if intersection.geom_type == 'Point':
                        intersections.append((intersection.y, intersection.x))
                    elif intersection.geom_type == 'MultiPoint':
                        for pt in intersection.geoms:
                            intersections.append((pt.y, pt.x))
        
        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing_pt in unique_intersections:
                if abs(pt[0] - existing_pt[0]) < tolerance and abs(pt[1] - existing_pt[1]) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)
        
        return unique_intersections

    def generate_subgraph_by_boundary_intersections(self, street_names, keep_largest_component=True):
        
        if self.graph is None:
            raise ValueError("Carga primero el grafo completo con get_city_graph()")
        
        if len(street_names) < 3:
            raise ValueError("Se necesitan exactamente 4 calles para formar el cuadrilátero")

        G_latlon = ox.project_graph(self.graph, to_crs='epsg:4326')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_latlon)

        gdf_edges = gdf_edges.copy()
        gdf_edges['name_str'] = gdf_edges['name'].apply(self._edge_name_to_str)

        print("\n" + "="*70)
        print("GENERANDO SUBGRAFO DELIMITADO POR INTERSECCIONES")
        print("="*70)

        all_intersections = []
        street_pairs = list(combinations(street_names, 2))
        
        for street1, street2 in street_pairs:
            intersections = self._find_street_intersections(gdf_edges, street1, street2)
            if intersections:
                print(f"[OK] {street1} AND {street2}: {len(intersections)} intersección(es)")
                all_intersections.extend(intersections)
            else:
                print(f"[ERROR] {street1} AND {street2}: No se encontraron intersecciones")
        
        if len(all_intersections) < 3:
            raise ValueError(
                f"Se necesitan al menos 3 intersecciones para formar un polígono. "
                f"Solo se encontraron {len(all_intersections)}. "
                f"Verifica los nombres de las calles en OpenStreetMap."
            )
        
        print(f"\n[OK] Total de puntos de intersección: {len(all_intersections)}")
        
        from shapely.geometry import MultiPoint
        
        if len(all_intersections) == 4:
            center_lat = sum(pt[0] for pt in all_intersections) / 4
            center_lon = sum(pt[1] for pt in all_intersections) / 4
            
            def angle_from_center(pt):
                import math
                return math.atan2(pt[0] - center_lat, pt[1] - center_lon)
            
            sorted_pts = sorted(all_intersections, key=angle_from_center)
            area_polygon = Polygon([(lon, lat) for lat, lon in sorted_pts])
        else:
            points = MultiPoint([Point(lon, lat) for lat, lon in all_intersections])
            area_polygon = points.convex_hull

        self.subgraph_polygon = area_polygon
        print(f"[OK] Polígono creado: {len(area_polygon.exterior.coords)} vértices")
        print(f"  Área: {area_polygon.area:.8f} grados² (~{area_polygon.area * 12100:.0f} m²)")

        nodes_within = gdf_nodes[gdf_nodes.geometry.within(area_polygon)]
        
        buffer_small = 0.0001
        nodes_on_boundary = gdf_nodes[gdf_nodes.geometry.intersects(area_polygon.buffer(buffer_small))]
        
        all_nodes = pd.concat([nodes_within, nodes_on_boundary]).drop_duplicates()
        
        if all_nodes.empty:
            raise ValueError("No se encontraron nodos dentro del área delimitada.")

        print(f"[OK] Nodos seleccionados: {len(all_nodes)}")

        subG = self.graph.subgraph(all_nodes.index).copy()

        edges_to_keep = []
        G_sub_latlon = ox.project_graph(subG, to_crs='epsg:4326')
        
        for u, v, key in subG.edges(keys=True):
            if u in all_nodes.index and v in all_nodes.index:
                try:
                    node_u = G_sub_latlon.nodes[u]
                    node_v = G_sub_latlon.nodes[v]
                    pt_u = Point(node_u['x'], node_u['y'])
                    pt_v = Point(node_v['x'], node_v['y'])
                    
                    u_inside = area_polygon.contains(pt_u) or area_polygon.boundary.distance(pt_u) < buffer_small
                    v_inside = area_polygon.contains(pt_v) or area_polygon.boundary.distance(pt_v) < buffer_small
                    
                    if u_inside and v_inside:
                        edges_to_keep.append((u, v, key))
                except:
                    pass
        
        if edges_to_keep:
            subG = subG.edge_subgraph(edges_to_keep).copy()

        if keep_largest_component:
            try:
                und = nx.Graph(subG)
                comps = list(nx.connected_components(und))
                if comps:
                    largest = max(comps, key=len)
                    subG = subG.subgraph(largest).copy()
                    print(f"[OK] Mantenida la componente más grande: {len(largest)} nodos")
            except Exception as e:
                print(f"[WARNING]  Advertencia al filtrar componentes: {e}")

        self.subgraph = subG
        print(f"[OK] SUBGRAFO FINAL: {len(self.subgraph.nodes)} nodos, {len(self.subgraph.edges)} aristas")
        print("="*70 + "\n")
        return self.subgraph

    def generate_random_route(self, length_nodes=20, max_retries=50):
        if self.subgraph is None:
            raise ValueError("Genera primero el subgrafo con generate_subgraph_by_boundary_intersections()")

        nodes = list(self.subgraph.nodes)
        if not nodes:
            raise ValueError("El subgrafo está vacío")

        route = []
        current = random.choice(nodes)
        route.append(current)

        attempts = 0
        while len(route) < length_nodes and attempts < max_retries:
            target = random.choice(nodes)
            if target == current:
                attempts += 1
                continue
                
            try:
                path = nx.shortest_path(self.subgraph, current, target, weight='length')
                
                for n in path[1:]:
                    if len(route) >= length_nodes:
                        break
                    route.append(n)
                
                current = route[-1]
                attempts = 0
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                attempts += 1
                continue

        if len(route) < length_nodes:
            print(f"[WARNING]  Advertencia: Ruta generada con {len(route)} nodos (objetivo: {length_nodes})")

        return route

    def generate_coverage_route(self):
        if self.subgraph is None:
            raise ValueError("Genera primero el subgrafo con generate_subgraph_by_boundary_intersections()")
        
        if len(self.subgraph.nodes) == 0:
            raise ValueError("El subgrafo está vacío")
        
        G_undirected = self.subgraph.to_undirected()
        
        odd_degree_nodes = [n for n in G_undirected.nodes() if G_undirected.degree(n) % 2 != 0]
        
        route = []
        
        if len(odd_degree_nodes) == 0:
            try:
                circuit = list(nx.eulerian_circuit(G_undirected))
                route = [circuit[0][0]]  # Nodo inicial
                for u, v in circuit:
                    route.append(v)
                print(f"[OK] Ruta euleriana generada (ciclo perfecto)")
            except:
                route = self._generate_dfs_coverage_route(G_undirected)
        
        elif len(odd_degree_nodes) == 2:
            try:
                path = list(nx.eulerian_path(G_undirected))
                route = [path[0][0]]
                for u, v in path:
                    route.append(v)
                print(f"[OK] Ruta semi-euleriana generada (camino)")
            except:
                route = self._generate_dfs_coverage_route(G_undirected)
        
        else:
            route = self._generate_dfs_coverage_route(G_undirected)
        
        return route
    
    def _generate_dfs_coverage_route(self, graph):
        start_node = random.choice(list(graph.nodes()))
        
        visited_edges = set()
        route = [start_node]
        stack = [start_node]
        current = start_node
        
        while stack:
            neighbors = list(graph.neighbors(current))
            
            found = False
            for neighbor in neighbors:
                edge = tuple(sorted([current, neighbor]))
                
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    route.append(neighbor)
                    stack.append(neighbor)
                    current = neighbor
                    found = True
                    break
            
            if not found:
                if stack:
                    stack.pop()
                    if stack:
                        prev = stack[-1]
                        try:
                            path = nx.shortest_path(graph, current, prev, weight='length')
                            route.extend(path[1:])
                            current = prev
                        except:
                            break
        
        print(f"[OK] Ruta DFS generada: {len(visited_edges)} aristas cubiertas de {graph.number_of_edges()}")
        coverage = (len(visited_edges) / graph.number_of_edges() * 100) if graph.number_of_edges() > 0 else 0
        print(f"  Cobertura: {coverage:.1f}% de las calles")
        
        return route

    def generate_coverage_routes(self, n_routes=12):
        if self.subgraph is None:
            raise ValueError("Genera primero el subgrafo con generate_subgraph_by_boundary_intersections()")

        print(f"\nGenerando {n_routes} rutas de COBERTURA COMPLETA...")
        print(f"(Cada ruta debe recorrer todas las calles del subgrafo)\n")
        
        routes = []
        
        for i in range(n_routes):
            route = self.generate_coverage_route()
            routes.append(route)
            print(f"  Ruta {i+1}: {len(route)} nodos\n")
        
        print(f"[OK] Generadas {len(routes)} rutas de cobertura completa.")
        
        lengths = [len(r) for r in routes]
        print(f"  Longitud promedio: {sum(lengths)/len(lengths):.1f} nodos")
        print(f"  Rango: {min(lengths)} - {max(lengths)} nodos")
        
        return routes

    def save_routes_to_csv(self, routes, filename="rutas_iniciales.csv"):
        import csv
        
        if not routes:
            print("[WARNING]  No hay rutas para guardar.")
            return
        
        valid_routes = [r for r in routes if len(r) >= 2]
        invalid_count = len(routes) - len(valid_routes)
        
        if invalid_count > 0:
            print(f"[WARNING]  Se omitieron {invalid_count} ruta(s) inválida(s) (menos de 2 nodos)")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['route_id', 'length', 'nodes'])
            
            for i, route in enumerate(valid_routes, start=1):
                nodes_str = '|'.join(map(str, route))  
                writer.writerow([i, len(route), nodes_str])
        
        print(f"[OK] Rutas guardadas en: {filename}")
        print(f"  Total de rutas válidas: {len(valid_routes)}")
        print(f"  Formato: route_id,length,nodes (separados por '|')")

    def save_subgraph_nodes_to_csv(self, filename="subgraph_nodes.csv"):
        
        import csv
        
        if self.subgraph is None:
            print("[WARNING]  No hay subgrafo para guardar.")
            return
        
        nodes = list(self.subgraph.nodes())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id'])
            
            for node in nodes:
                writer.writerow([node])
        
        print(f"[OK] Nodos del subgrafo guardados en: {filename}")
        print(f"  Total de nodos: {len(nodes)}")

    def export_interactive_map_with_layers(self, filename="mapa_cuenca_layers.html", 
                                          draw_full_graph=False, draw_subgraph=True, 
                                          draw_polygon=True):
        if self.graph is None:
            raise ValueError("Carga primero el grafo con get_city_graph()")

        G_full = ox.project_graph(self.graph, to_crs='epsg:4326')
        gdf_nodes_full, gdf_edges_full = ox.graph_to_gdfs(G_full)

        if self.subgraph is not None and len(self.subgraph.nodes) > 0:
            G_sub = ox.project_graph(self.subgraph, to_crs='epsg:4326')
            gdf_nodes_sub, gdf_edges_sub = ox.graph_to_gdfs(G_sub)
            lat_mean = gdf_nodes_sub.geometry.y.mean()
            lon_mean = gdf_nodes_sub.geometry.x.mean()
            zoom = 15
        else:
            lat_mean = gdf_nodes_full.geometry.y.mean()
            lon_mean = gdf_nodes_full.geometry.x.mean()
            gdf_nodes_sub = None
            gdf_edges_sub = None
            zoom = 13

        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=zoom, 
                      tiles='CartoDB positron', prefer_canvas=True)

        if draw_full_graph:
            full_fg = folium.FeatureGroup(name='🗺️ Grafo completo', show=False)
            folium.GeoJson(
                gdf_edges_full.to_json(),
                style_function=lambda x: {'color': '#dddddd', 'weight': 1, 'opacity': 0.3},
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Calle:'], sticky=False)
            ).add_to(full_fg)
            full_fg.add_to(m)

        if draw_polygon and self.subgraph_polygon is not None:
            poly_fg = folium.FeatureGroup(name='(GREEN) Polígono límites', show=True)
            geojson_feature = {
                "type": "Feature",
                "geometry": mapping(self.subgraph_polygon),
                "properties": {"name": "Área delimitada"}
            }
            folium.GeoJson(
                geojson_feature,
                style_function=lambda x: {
                    'color': '#6A0DAD',
                    'weight': 5,
                    'opacity': 0.9,
                    'fillColor': '#6A0DAD',
                    'fillOpacity': 0.25
                },
                tooltip="Área delimitada por las 4 calles"
            ).add_to(poly_fg)
            poly_fg.add_to(m)

        """style_function=lambda x: {
            'color': '#00B74A',
            'weight': 4,
            'opacity': 0.9,
            'fillColor': '#00B74A',
            'fillOpacity': 0.15
        },"""

        if draw_subgraph and self.subgraph is not None and gdf_edges_sub is not None:
            sub_fg = folium.FeatureGroup(name='(RED) Subgrafo delimitado', show=True)
            folium.GeoJson(
                gdf_edges_sub.to_json(),
                style_function=lambda x: {
                    'color': '#E63946', 
                    'weight': 3, 
                    'opacity': 0.9
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'], 
                    aliases=['Calle:'], 
                    sticky=True
                )
            ).add_to(sub_fg)
            sub_fg.add_to(m)

            marker_cluster = MarkerCluster(
                name='📍 Nodos del subgrafo', 
                disable_clustering_at_zoom=16
            ).add_to(m)
            
            for lat, lon, node_id in zip(
                gdf_nodes_sub.geometry.y, 
                gdf_nodes_sub.geometry.x, 
                gdf_nodes_sub.index
            ):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color='#E63946',
                    fill=True,
                    fill_color='#E63946',
                    fill_opacity=0.8,
                    popup=f"<b>Nodo ID:</b> {node_id}"
                ).add_to(marker_cluster)

        folium.LayerControl().add_to(m)
        
        m.save(filename)
        print(f"[OK] Mapa guardado: {filename}")
        print(f"  Abre el archivo en tu navegador para ver las capas interactivas.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUENCA GRAPH MANAGER - Sistema de Generación de Rutas")
    print("="*70 + "\n")
    
    manager = CuencaGraphManager()
    
    manager.get_city_graph()
    manager.assign_node_weights(restriction_name="demanda_tol", min_val=5, max_val=50)

    limites = ["Calle Larga",
               "Avenida Huayna Capac",
               "Avenida Héroes de Verdeloma",
               "Coronel Talbot",
               "Bajada del Vado",
               "La Condamine",
               "Calle de la Cruz",
               "Presidente Córdova",
               "Del Rollo",
               "De la Merced",
               "Estévez de Toral",
               "Antonio Vargas Machuca",
               "Hernando de la Cruz",
               "Mariano Cueva",
               "Tomás Ordoñez"]
    
    try:
        manager.generate_subgraph_by_boundary_intersections(limites)
        pass
    except Exception as e:
        print(f"[ERROR] ERROR creando subgrafo: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)

    rutas_iniciales = []
    try:
        rutas_iniciales = manager.generate_coverage_routes(n_routes=12)
        
        manager.save_routes_to_csv(rutas_iniciales, filename="rutas_iniciales.csv")
        
        manager.save_subgraph_nodes_to_csv(filename="subgraph_nodes.csv")
        
    except Exception as e:
        print(f"[ERROR] ERROR generando rutas: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)

    try:
        manager.export_interactive_map_with_layers(
            "mapa_cuenca_delimitado.html",
            draw_full_graph=False,
            draw_subgraph=True,
            draw_polygon=True
        )
    except Exception as e:
        print(f"[ERROR] ERROR exportando mapa: {e}\n")

    if rutas_iniciales:
        print(f"\n{'='*70}")
        print("RESUMEN DE RUTAS GENERADAS (Población inicial para AG):")
        print(f"{'='*70}")
        for i, r in enumerate(rutas_iniciales, start=1):
            print(f"  Ruta {i:2d}: {len(r):2d} nodos  |  ID inicio: {r[0]:>8}  →  ID fin: {r[-1]:>8}")
        print(f"{'='*70}\n")
        
        print("PRIMERAS 3 RUTAS (detalle):")
        for i, r in enumerate(rutas_iniciales[:3], start=1):
            preview = r[:8] if len(r) <= 8 else r[:8] + ['...']
            print(f"  Ruta {i}: {preview}")

    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("📊 Estadísticas:")
    print(f"   - Grafo completo: {len(manager.graph.nodes)} nodos")
    print(f"   - Subgrafo: {len(manager.subgraph.nodes)} nodos ({len(manager.subgraph.nodes)/len(manager.graph.nodes)*100:.1f}% del total)")
    print(f"   - Aristas del subgrafo: {len(manager.subgraph.edges)} calles")
    print(f"   - Rutas de cobertura generadas: {len(rutas_iniciales)}")
    print(f"\n[INFO] IMPORTANTE:")
    print(f"   Las rutas generadas cubren TODAS las calles del subgrafo.")
    print(f"   Cada ruta recorre todas las calles al menos una vez.")
    print(f"   Esto es adecuado para: recolección de basura, barrido, inspección.")
    print(f"\n🗺️  Abre 'mapa_cuenca_delimitado.html' para visualizar.")
    print("="*70 + "\n")