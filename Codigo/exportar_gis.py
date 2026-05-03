import os
import random
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from shapely.geometry import mapping
import pandas as pd
from itertools import combinations
from pyproj import Transformer

CONFIG = {
    'CITY_NAME': "Cuenca, Ecuador",
    'GRAPH_FILE': "cuenca_graph.graphml",
    'OUTPUT_GPKG': "cuenca_limpieza.gpkg",
    'CRS_PROJECTED': 'epsg:32717',
    'CRS_LATLON': 'epsg:4326',
    'TOTAL_GARBAGE': 57.59,
    
    'GARAGE_LAT': -2.876,
    'GARAGE_LON': -78.989,
    'LANDFILL_LAT': -2.9368,
    'LANDFILL_LON': -78.9226,
    
    'BOUNDARY_STREETS': [
        "Calle Larga",
        "Avenida Huayna Capac",
        "Avenida Héroes de Verdeloma",
        "Coronel Talbot",
        "Bajada del Vado",
        "La Condamine",
        "Calle de la Cruz",
        "Presidente Córdova",
        "Del Rollo",
        "De la Merced",
        "Estéved de Toral",
        "Antonio Vargas Machuca",
        "Hernando de la Cruz",
        "Mariano Cueva",
        "Tomás Ordóñez"
    ]
}

def get_or_download_graph(graph_file, city_name):
    if os.path.exists(graph_file):
        print(f"Cargando grafo desde {graph_file}...")
        G = ox.load_graphml(graph_file)
    else:
        print(f"Descargando grafo de {city_name} desde OpenStreetMap...")
        ox.settings.use_cache = True
        ox.settings.log_console = True
        ox.settings.requests_timeout = 1000
        
        G = ox.graph_from_place(city_name, network_type='drive', simplify=True)
        G = ox.project_graph(G, to_crs=CONFIG['CRS_PROJECTED'])
        ox.save_graphml(G, filepath=graph_file)
        print(f"Grafo guardado en {graph_file}")
    
    print(f"[OK] Grafo cargado: {len(G.nodes)} nodos, {len(G.edges)} aristas")
    return G

def edge_name_to_str(val):
    if isinstance(val, list):
        return ", ".join([str(x) for x in val])
    return str(val)

def find_street_intersections(gdf_edges, street1, street2, tolerance=0.0001):
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

def create_boundary_polygon(gdf_edges, street_names):
    print("\n" + "="*70)
    print("GENERANDO POLÍGONO DELIMITADOR")
    print("="*70)
    
    all_intersections = []
    street_pairs = list(combinations(street_names, 2))
    
    for street1, street2 in street_pairs:
        intersections = find_street_intersections(gdf_edges, street1, street2)
        if intersections:
            print(f"[OK] {street1} AND {street2}: {len(intersections)} intersección(es)")
            all_intersections.extend(intersections)
        else:
            print(f"[WARN] {street1} AND {street2}: No se encontraron intersecciones")
    
    if len(all_intersections) < 3:
        raise ValueError(
            f"Se necesitan al menos 3 intersecciones. Solo se encontraron {len(all_intersections)}."
        )
    
    print(f"\n[OK] Total de puntos de intersección: {len(all_intersections)}")
    
    if len(all_intersections) == 4:
        center_lat = sum(pt[0] for pt in all_intersections) / 4
        center_lon = sum(pt[1] for pt in all_intersections) / 4
        
        import math
        def angle_from_center(pt):
            return math.atan2(pt[0] - center_lat, pt[1] - center_lon)
        
        sorted_pts = sorted(all_intersections, key=angle_from_center)
        area_polygon = Polygon([(lon, lat) for lat, lon in sorted_pts])
    else:
        points = MultiPoint([Point(lon, lat) for lat, lon in all_intersections])
        area_polygon = points.convex_hull
    
    print(f"[OK] Polígono creado: {len(area_polygon.exterior.coords)} vértices")
    print(f"  Área: ~{area_polygon.area * 12100:.0f} m²")
    
    return area_polygon

def extract_subgraph(G_full, boundary_polygon, keep_largest_component=True):
    print("\n" + "="*70)
    print("EXTRAYENDO SUBGRAFO")
    print("="*70)
    
    G_latlon = ox.project_graph(G_full, to_crs='epsg:4326')
    gdf_nodes, _ = ox.graph_to_gdfs(G_latlon)
    
    # CALIBRACIÓN: Buffer intermedio (ni muy estricto ni muy permisivo)
    buffer_small = 0.0003  # Valor intermedio entre 0.0001 (muy estricto) y 0.0005 (muy permisivo)
    
    nodes_within = gdf_nodes[gdf_nodes.geometry.within(boundary_polygon)]
    nodes_on_boundary = gdf_nodes[gdf_nodes.geometry.intersects(boundary_polygon.buffer(buffer_small))]
    all_nodes = pd.concat([nodes_within, nodes_on_boundary]).drop_duplicates()
    
    if all_nodes.empty:
        raise ValueError("No se encontraron nodos dentro del área delimitada")
    
    print(f"[OK] Nodos seleccionados: {len(all_nodes)}")
    
    subG = G_full.subgraph(all_nodes.index).copy()
    
    # MANTENER filtro de aristas original (ambos nodos dentro)
    edges_to_keep = []
    G_sub_latlon = ox.project_graph(subG, to_crs='epsg:4326')
    
    for u, v, key in subG.edges(keys=True):
        if u in all_nodes.index and v in all_nodes.index:
            try:
                node_u = G_sub_latlon.nodes[u]
                node_v = G_sub_latlon.nodes[v]
                pt_u = Point(node_u['x'], node_u['y'])
                pt_v = Point(node_v['x'], node_v['y'])
                
                u_inside = boundary_polygon.contains(pt_u) or boundary_polygon.boundary.distance(pt_u) < buffer_small
                v_inside = boundary_polygon.contains(pt_v) or boundary_polygon.boundary.distance(pt_v) < buffer_small
                
                # MANTENER: Ambos nodos deben estar dentro (filtro original)
                if u_inside and v_inside:
                    edges_to_keep.append((u, v, key))
            except:
                pass
    
    if edges_to_keep:
        subG = subG.edge_subgraph(edges_to_keep).copy()
    
    # MANTENER: Filtro de componente más grande activado
    if keep_largest_component:
        try:
            und = nx.Graph(subG)
            comps = list(nx.connected_components(und))
            if comps:
                largest = max(comps, key=len)
                
                # DIAGNÓSTICO: Mostrar componentes
                if len(comps) > 1:
                    sizes = sorted([len(c) for c in comps], reverse=True)
                    print(f"[INFO] Componentes encontrados: {len(comps)}")
                    print(f"[INFO] Tamaños: {sizes[:5]}")
                
                subG = subG.subgraph(largest).copy()
                print(f"[OK] Componente más grande: {len(largest)} nodos")
        except Exception as e:
            print(f"[WARN] Advertencia al filtrar componentes: {e}")
    
    print(f"[OK] SUBGRAFO FINAL: {len(subG.nodes)} nodos, {len(subG.edges)} aristas")
    print("="*70)
    
    return subG

def find_nearest_node(graph, lat, lon):
    try:
        graph_crs = graph.graph.get('crs', 'epsg:32717')
        transformer = Transformer.from_crs("epsg:4326", graph_crs, always_xy=True)
        x_proj, y_proj = transformer.transform(lon, lat)
        nearest = ox.distance.nearest_nodes(graph, X=x_proj, Y=y_proj)
        return nearest
    except:
        nearest = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
        return nearest

def assign_garbage_demand(subgraph, total_garbage):
    print("\n" + "="*70)
    print("ASIGNANDO DEMANDA DE BASURA")
    print("="*70)
    
    nodes = list(subgraph.nodes())
    
    raw_weights = {n: random.random() for n in nodes}
    total_weight = sum(raw_weights.values())
    
    # Asignar valores redondeados a todos los nodos excepto el último
    current_sum = 0.0
    for i, node in enumerate(nodes[:-1]):  # Todos excepto el último
        val = (raw_weights[node] / total_weight) * total_garbage
        val = round(max(0.01, val), 2)
        
        subgraph.nodes[node]['demand'] = val
        current_sum += val
    
    # El último nodo recibe el residuo exacto para que la suma sea exactamente total_garbage
    last_node = nodes[-1]
    last_val = round(total_garbage - current_sum, 2)
    
    # Asegurar que el último valor no sea negativo ni menor a 0.01
    if last_val < 0.01:
        # Si el residuo es muy pequeño, ajustar el penúltimo nodo
        penultimate_node = nodes[-2]
        penultimate_val = subgraph.nodes[penultimate_node]['demand']
        adjustment = 0.01 - last_val
        
        subgraph.nodes[penultimate_node]['demand'] = round(penultimate_val - adjustment, 2)
        last_val = 0.01
    
    subgraph.nodes[last_node]['demand'] = last_val
    
    # Calcular suma final para verificación
    final_sum = sum(subgraph.nodes[n]['demand'] for n in nodes)
    
    print(f"[OK] Total basura distribuida: {final_sum:.2f} ton")
    print(f"  Objetivo: {total_garbage:.2f} ton")
    print(f"  Diferencia: {abs(final_sum - total_garbage):.10f} ton")
    print(f"  Nodos con demanda: {len(nodes)}")
    print(f"  Suma exacta alcanzada: {'✓ SÍ' if final_sum == total_garbage else '✗ NO'}")
    print("="*70)

def export_to_geopackage(graph_full, subgraph, garage_node, landfill_node, output_file):
    print("\n" + "="*70)
    print(f"EXPORTANDO A GEOPACKAGE: {output_file}")
    print("="*70)
    
    G_latlon = ox.project_graph(graph_full, to_crs=CONFIG['CRS_LATLON'])
    gdf_nodes_full, gdf_edges_full = ox.graph_to_gdfs(G_latlon)
    
    gdf_nodes_full['demand'] = 0.0
    gdf_nodes_full['node_type'] = 'other'
    
    subgraph_nodes = set(subgraph.nodes())
    for node_id in subgraph_nodes:
        if node_id in gdf_nodes_full.index:
            demand_val = subgraph.nodes[node_id].get('demand', 0.0)
            gdf_nodes_full.at[node_id, 'demand'] = demand_val
            gdf_nodes_full.at[node_id, 'node_type'] = 'customer'
    
    if garage_node in gdf_nodes_full.index:
        gdf_nodes_full.at[garage_node, 'node_type'] = 'garage'
        gdf_nodes_full.at[garage_node, 'demand'] = 0.0
    
    if landfill_node in gdf_nodes_full.index:
        gdf_nodes_full.at[landfill_node, 'node_type'] = 'landfill'
        gdf_nodes_full.at[landfill_node, 'demand'] = 0.0
    
    print(f"[OK] Exportando capas:")
    print(f"  1. 'all_nodes': Todos los nodos ({len(gdf_nodes_full)})")
    print(f"  2. 'all_edges': Todas las calles ({len(gdf_edges_full)})")
    
    gdf_nodes_full.to_file(output_file, layer='all_nodes', driver='GPKG')
    
    gdf_edges_reset = gdf_edges_full.reset_index()
    gdf_edges_reset.to_file(output_file, layer='all_edges', driver='GPKG')
    
    gdf_subgraph_nodes = gdf_nodes_full[gdf_nodes_full['node_type'] == 'customer'].copy()
    print(f"  3. 'subgraph_nodes': Nodos con basura ({len(gdf_subgraph_nodes)})")
    gdf_subgraph_nodes.to_file(output_file, layer='subgraph_nodes', driver='GPKG')
    
    G_sub_latlon = ox.project_graph(subgraph, to_crs=CONFIG['CRS_LATLON'])
    _, gdf_subgraph_edges = ox.graph_to_gdfs(G_sub_latlon)
    gdf_subgraph_edges_reset = gdf_subgraph_edges.reset_index()
    print(f"  4. 'subgraph_edges': Calles del subgrafo ({len(gdf_subgraph_edges)})")
    gdf_subgraph_edges_reset.to_file(output_file, layer='subgraph_edges', driver='GPKG')
    
    special_nodes = []
    
    if garage_node in gdf_nodes_full.index:
        garage_geom = gdf_nodes_full.loc[garage_node, 'geometry']
        special_nodes.append({
            'osmid': garage_node,
            'name': 'Garage (Inicio)',
            'type': 'garage',
            'geometry': garage_geom
        })
    
    if landfill_node in gdf_nodes_full.index:
        landfill_geom = gdf_nodes_full.loc[landfill_node, 'geometry']
        special_nodes.append({
            'osmid': landfill_node,
            'name': 'Relleno Sanitario (Descarga)',
            'type': 'landfill',
            'geometry': landfill_geom
        })
    
    if special_nodes:
        gdf_special = gpd.GeoDataFrame(special_nodes, crs=CONFIG['CRS_LATLON'])
        print(f"  5. 'special_points': Garage y Landfill (2)")
        gdf_special.to_file(output_file, layer='special_points', driver='GPKG')

    # ====================================================================
    # CREAR CAPA DE MAPA DE CALOR (HEATMAP) BASADO EN DEMANDA
    # ====================================================================
    
    print(f"  6. 'heatmap_density': Mapa de calor de densidad de basura")
    
    # Obtener solo los nodos con demanda > 0
    gdf_heatmap = gdf_nodes_full[gdf_nodes_full['demand'] > 0].copy()
    
    if not gdf_heatmap.empty:
        # Crear círculos proporcionales a la demanda para el efecto heatmap
        # El radio será proporcional a la raíz cuadrada de la demanda
        import numpy as np
        
        # Normalizar demanda entre 0 y 1
        max_demand = gdf_heatmap['demand'].max()
        min_demand = gdf_heatmap['demand'].min()
        
        if max_demand > min_demand:
            gdf_heatmap['normalized_demand'] = (
                (gdf_heatmap['demand'] - min_demand) / (max_demand - min_demand)
            )
        else:
            gdf_heatmap['normalized_demand'] = 1.0
        
        # Crear buffers (círculos) proporcionales a la demanda
        # Radio base en grados (aprox 50m = 0.0005 grados)
        base_radius = 0.0005
        max_radius_multiplier = 3.0
        
        gdf_heatmap['radius'] = (
            base_radius + 
            (gdf_heatmap['normalized_demand'] * base_radius * max_radius_multiplier)
        )
        
        # Crear geometría de círculo (buffer)
        gdf_heatmap['geometry'] = gdf_heatmap.apply(
            lambda row: row.geometry.buffer(row['radius']), 
            axis=1
        )
        
        # Clasificar en categorías para colores (bajo, medio, alto)
        gdf_heatmap['density_category'] = pd.cut(
            gdf_heatmap['demand'],
            bins=5,
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        )
        
        # Seleccionar solo columnas relevantes
        gdf_heatmap_export = gdf_heatmap[[
            'demand', 
            'normalized_demand', 
            'density_category',
            'geometry'
        ]].copy()
        
        # Guardar capa de heatmap
        gdf_heatmap_export.to_file(output_file, layer='heatmap_density', driver='GPKG')
        print(f"    [OK] {len(gdf_heatmap_export)} círculos de densidad creados")
    
    # ====================================================================
    # CREAR CAPA DE INFORMACIÓN DEL PROYECTO (METADATA)
    # ====================================================================
    
    print(f"  7. 'project_metadata': Información del área")
    
    # Calcular área del subgrafo
    gdf_subgraph_nodes = gdf_nodes_full[gdf_nodes_full['node_type'] == 'customer']
    
    if not gdf_subgraph_nodes.empty:
        from shapely.ops import unary_union
        
        # Crear convex hull del área
        all_points = gdf_subgraph_nodes.geometry.unary_union
        area_polygon = all_points.convex_hull
        
        # Calcular área en km²
        gdf_temp = gpd.GeoDataFrame({'geometry': [area_polygon]}, crs=CONFIG['CRS_LATLON'])
        gdf_projected = gdf_temp.to_crs(CONFIG['CRS_PROJECTED'])
        area_km2 = 2.42
        
        # DATOS DEL PROYECTO (según el documento)
        poblacion = 47992  # Población estimada del centro histórico
        densidad = poblacion / area_km2 if area_km2 > 0 else 0
        
        # Obtener centroide para posicionar el label
        centroid = area_polygon.centroid
        
        # Crear texto informativo
        info_text = (
            f"CENTRO HISTÓRICO DE CUENCA\\n"
            f"═══════════════════════════\\n"
            f"Área: {area_km2:.2f} km²\\n"
            f"Población: {poblacion:,} hab\\n"
            f"Densidad: {densidad:.0f} hab/km²\\n"
            f"═══════════════════════════\\n"
            f"Puntos de recolección: {len(gdf_subgraph_nodes)}\\n"
            f"Basura total: {CONFIG['TOTAL_GARBAGE']:.2f} ton\\n"
        )
        
        # Crear GeoDataFrame con metadata
        metadata_data = {
            'type': ['project_info'],
            'area_km2': [round(area_km2, 2)],
            'poblacion': [poblacion],
            'densidad_hab_km2': [round(densidad, 0)],
            'num_clientes': [len(gdf_subgraph_nodes)],
            'basura_total_ton': [CONFIG['TOTAL_GARBAGE']],
            'label_text': [info_text],
            'geometry': [Point(centroid.x, centroid.y)]
        }
        
        gdf_metadata = gpd.GeoDataFrame(metadata_data, crs=CONFIG['CRS_LATLON'])
        gdf_metadata.to_file(output_file, layer='project_metadata', driver='GPKG')
        print(f"    [OK] Metadata creada (Área: {area_km2:.2f} km², Densidad: {densidad:.0f} hab/km²)")
    
    print(f"\n[OK] GeoPackage guardado: {output_file}")
    print(f"  Total nodos con demanda: {(gdf_nodes_full['demand'] > 0).sum()}")
    print(f"  Total demanda: {gdf_nodes_full['demand'].sum():.2f} ton")
    print("="*70)

def verify_graph_reconstruction(gpkg_file):
    print("\n" + "="*70)
    print("VERIFICANDO RECONSTRUCCIÓN DEL GRAFO")
    print("="*70)
    
    try:
        gdf_nodes = gpd.read_file(gpkg_file, layer='all_nodes')
        gdf_edges = gpd.read_file(gpkg_file, layer='all_edges')
        
        gdf_nodes = gdf_nodes.set_index('osmid')
        
        if 'u' in gdf_edges.columns and 'v' in gdf_edges.columns:
            if 'key' in gdf_edges.columns:
                gdf_edges = gdf_edges.set_index(['u', 'v', 'key'])
            else:
                gdf_edges['key'] = 0
                gdf_edges = gdf_edges.set_index(['u', 'v', 'key'])
        
        G_reconstructed = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
        
        print(f"[OK] Grafo reconstruido exitosamente:")
        print(f"  Nodos: {len(G_reconstructed.nodes)}")
        print(f"  Aristas: {len(G_reconstructed.edges)}")
        
        nodes_with_demand = [n for n in G_reconstructed.nodes() 
                            if G_reconstructed.nodes[n].get('demand', 0) > 0]
        total_demand = sum(G_reconstructed.nodes[n].get('demand', 0) 
                          for n in G_reconstructed.nodes())
        
        print(f"  Nodos con demanda: {len(nodes_with_demand)}")
        print(f"  Demanda total: {total_demand:.2f} ton")
        
        garage_nodes = [n for n in G_reconstructed.nodes() 
                       if G_reconstructed.nodes[n].get('node_type') == 'garage']
        landfill_nodes = [n for n in G_reconstructed.nodes() 
                         if G_reconstructed.nodes[n].get('node_type') == 'landfill']
        
        print(f"  Garage encontrado: {len(garage_nodes) > 0} (ID: {garage_nodes[0] if garage_nodes else 'N/A'})")
        print(f"  Landfill encontrado: {len(landfill_nodes) > 0} (ID: {landfill_nodes[0] if landfill_nodes else 'N/A'})")
        print(f"\n[OK] Verificación exitosa")
        
    except Exception as e:
        print(f"[ERROR] Error al reconstruir el grafo: {e}")
        raise
    
    print("="*70)

def main():
    print("\n" + "="*70)
    print("EXPORTADOR GIS - SISTEMA VRP CUENCA")
    print("="*70)
    
    G_full = get_or_download_graph(CONFIG['GRAPH_FILE'], CONFIG['CITY_NAME'])
    
    garage_node = find_nearest_node(G_full, CONFIG['GARAGE_LAT'], CONFIG['GARAGE_LON'])
    landfill_node = find_nearest_node(G_full, CONFIG['LANDFILL_LAT'], CONFIG['LANDFILL_LON'])
    
    print(f"\n[OK] Nodos especiales identificados:")
    print(f"  Garage (Avenida del Toril): Node ID {garage_node}")
    print(f"  Landfill (Pichacay): Node ID {landfill_node}")
    
    G_latlon = ox.project_graph(G_full, to_crs='epsg:4326')
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_latlon)
    gdf_edges['name_str'] = gdf_edges['name'].apply(edge_name_to_str)
    
    boundary_polygon = create_boundary_polygon(gdf_edges, CONFIG['BOUNDARY_STREETS'])
    
    subgraph = extract_subgraph(G_full, boundary_polygon, keep_largest_component=False)
    
    random.seed(42)
    assign_garbage_demand(subgraph, CONFIG['TOTAL_GARBAGE'])
    
    export_to_geopackage(G_full, subgraph, garage_node, landfill_node, CONFIG['OUTPUT_GPKG'])
    
    verify_graph_reconstruction(CONFIG['OUTPUT_GPKG'])
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print(f"\nArchivo generado: {CONFIG['OUTPUT_GPKG']}")
    print("\nCAPAS EN EL GEOPACKAGE:")
    print("  1. 'all_nodes' - Todos los nodos de Cuenca")
    print("  2. 'all_edges' - Todas las calles de Cuenca")
    print("  3. 'subgraph_nodes' - Solo nodos con basura (zona de trabajo)")
    print("  4. 'subgraph_edges' - Solo calles de la zona de trabajo")
    print("  5. 'special_points' - Garage y Relleno Sanitario")
    print("\nEN QGIS:")
    print("  1. Abre QGIS")
    print("  2. Layer > Add Layer > Add Vector Layer")
    print("  3. Selecciona 'cuenca_limpieza.gpkg'")
    print("  4. Carga las capas: 'subgraph_nodes', 'subgraph_edges', 'special_points'")
    print("  5. Click derecho en 'subgraph_nodes' > Properties > Symbology")
    print("     - Tipo: Graduated")
    print("     - Column: demand")
    print("     - Method: Natural Breaks")
    print("     - Classify")
    print("\nINTEGRACIÓN CON algoritmo_genético.py:")
    print("  import geopandas as gpd")
    print("  import osmnx as ox")
    print("")
    print("  # Cargar grafo completo")
    print("  gdf_nodes = gpd.read_file('cuenca_limpieza.gpkg', layer='all_nodes')")
    print("  gdf_edges = gpd.read_file('cuenca_limpieza.gpkg', layer='all_edges')")
    print("  gdf_nodes = gdf_nodes.set_index('osmid')")
    print("  if 'key' not in gdf_edges.columns:")
    print("      gdf_edges['key'] = 0")
    print("  gdf_edges = gdf_edges.set_index(['u', 'v', 'key'])")
    print("  city_graph = ox.graph_from_gdfs(gdf_nodes, gdf_edges)")
    print("")
    print("  # Obtener nodos del subgrafo (clientes)")
    print("  customers = gdf_nodes[gdf_nodes['node_type'] == 'customer'].index.tolist()")
    print("")
    print("  # Obtener garage y landfill")
    print("  garage = gdf_nodes[gdf_nodes['node_type'] == 'garage'].index[0]")
    print("  landfill = gdf_nodes[gdf_nodes['node_type'] == 'landfill'].index[0]")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()