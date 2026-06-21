import rclpy
from rclpy.node import Node
import threading
import datetime
import re
from flask import Flask, render_template_string
from flask_socketio import SocketIO

from smacc2_msgs.msg import SmaccStateMachine, SmaccTransitionLogEntry

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

sm_data = {"nodes": [], "edges": [], "active_state": None}

def safe_id(raw_str):
    """ Sanitiza strings para um formato seguro de ID no Cytoscape """
    if not raw_str:
        return ""
    return re.sub(r'[^a-zA-Z0-9_]', '_', raw_str)

class SmaccVisualizerNode(Node):
    def __init__(self):
        super().__init__('smacc_web_visualizer')
        
        self.desc_sub = self.create_subscription(
            SmaccStateMachine, 
            '/SmNavTest/smacc/state_machine_description', 
            self.desc_callback, 
            10
        )
        
        self.trans_sub = self.create_subscription(
            SmaccTransitionLogEntry, 
            '/SmNavTest/smacc/transition_log', 
            self.trans_callback, 
            10
        )
        self.get_logger().info('SMACC Visualizer Node Iniciado. Acesse http://localhost:5000')

    def desc_callback(self, msg: SmaccStateMachine):
        try:
            nodes = []
            edges = []
            
            created_node_ids = set()
            
            for state in msg.states:
                state_id = safe_id(state.name)
                
                # Estado (Raiz do Agrupamento)
                nodes.append({
                    'data': {'id': state_id, 'label': state.name.split('::')[-1], 'type': 'state'}
                })
                created_node_ids.add(state_id)
                
                # Ortogonais e seus filhos (Clients/Behaviors)
                for ortho in state.orthogonals:
                    ortho_safe = safe_id(ortho.name)
                    ortho_id = f"{state_id}_{ortho_safe}"
                    
                    nodes.append({
                        'data': {'id': ortho_id, 'label': ortho.name.split('::')[-1], 'parent': state_id, 'type': 'orthogonal'}
                    })
                    created_node_ids.add(ortho_id)
                    
                    for client in ortho.client_names:
                        if not client: continue 
                        client_id = f"{state_id}_{ortho_safe}_{safe_id(client)}"
                        nodes.append({
                            'data': {'id': client_id, 'label': client.split('::')[-1], 'parent': ortho_id, 'type': 'client'}
                        })
                        created_node_ids.add(client_id)
                        
                    for cb in ortho.client_behavior_names:
                        if not cb: continue
                        cb_id = f"{state_id}_{ortho_safe}_{safe_id(cb)}"
                        nodes.append({
                            'data': {'id': cb_id, 'label': cb.split('::')[-1], 'parent': ortho_id, 'type': 'behavior'}
                        })
                        created_node_ids.add(cb_id)

                # State Reactors
                for sr in getattr(state, 'state_reactors', []):
                    sr_safe = safe_id(sr.type_name)
                    # O ID único do State Reactor dentro deste Estado
                    sr_id = f"{state_id}_{sr_safe}_{sr.index}"
                    
                    nodes.append({
                        'data': {'id': sr_id, 'label': sr.type_name.split('::')[-1], 'parent': state_id, 'type': 'reactor'}
                    })
                    created_node_ids.add(sr_id)
                    
                    # Cria as linhas pontilhadas (virtuais) que mostram quais componentes alimentam o State Reactor
                    for source_idx, es in enumerate(sr.event_sources):
                        es_ortho = safe_id(es.event_object_tag)
                        es_emitter = safe_id(es.event_source)
                        es_event = es.event_type.split('::')[-1]
                        
                        emitter_id = f"{state_id}_{es_ortho}_{es_emitter}"
                        if emitter_id in created_node_ids:
                            edges.append({
                                'data': {
                                    'id': f"virtual_{emitter_id}_to_{sr_id}_{source_idx}",
                                    'source': emitter_id,
                                    'target': sr_id,
                                    'label': es_event,
                                    'edge_type': 'virtual' 
                                }
                            })
                
                # Transições Reais (Mudanças de Estado)
                for idx, trans in enumerate(state.transitions):
                    dest_id = safe_id(trans.destiny_state_name)
                    source_id = safe_id(trans.source_state_name)
                    
                    e_source = trans.event.event_source
                    e_ortho = trans.event.event_object_tag
                    e_type = trans.event.event_type
                    
                    # 1. Tenta achar a origem exata (Client ou Behavior)
                    candidate_id = f"{source_id}_{safe_id(e_ortho)}_{safe_id(e_source)}" if (e_source and e_ortho) else None
                    
                    # 2. Verifica se a origem é, na verdade, um State Reactor que acabou de ser criado
                    sr_match = None
                    if "Sr" in e_type: 
                        sr_type_safe = safe_id(e_type.split('::')[-1]) 
                        for node_id in created_node_ids:
                            if node_id.startswith(f"{source_id}_{sr_type_safe}") and "reactor" in [n['data']['type'] for n in nodes if n['data']['id'] == node_id]:
                                sr_match = node_id
                                break

                    visual_source_id = source_id # Fallback
                    
                    if sr_match:
                        visual_source_id = sr_match
                    elif candidate_id and candidate_id in created_node_ids:
                        visual_source_id = candidate_id
                    
                    edges.append({
                        'data': {
                            'id': f"edge_{visual_source_id}_{dest_id}_{idx}",
                            'source': visual_source_id,
                            'target': dest_id,
                            'label': e_type.split("::")[-1],
                            'edge_type': 'real'
                        }
                    })
            
            sm_data['nodes'] = nodes
            sm_data['edges'] = edges
            socketio.emit('sm_update', sm_data)
            
        except Exception as e:
            self.get_logger().error(f"Erro ao processar descrição: {e}")

    def trans_callback(self, msg: SmaccTransitionLogEntry):
        try:
            dest_state_raw = msg.transition.destiny_state_name
            source_state_raw = msg.transition.source_state_name
            event_type = msg.transition.event.event_type
            e_source = msg.transition.event.event_source
            
            dest_state_id = safe_id(dest_state_raw)
            
            if hasattr(msg, 'timestamp'):
                dt = datetime.datetime.fromtimestamp(msg.timestamp.sec).strftime('%H:%M:%S')
                msec = int(msg.timestamp.nanosec / 1e6)
                time_str = f"{dt}.{msec:03d}"
            else:
                time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            
            sm_data['active_state'] = dest_state_id
            
            emit_name = e_source.split('::')[-1] if e_source else "State/Reactor"
            
            log_data = {
                'time': time_str,
                'source': f"{source_state_raw.split('::')[-1]} ({emit_name})",
                'event': event_type.split("::")[-1],
                'destiny': dest_state_raw.split('::')[-1]
            }
            
            socketio.emit('active_state_update', {'active_state': dest_state_id})
            socketio.emit('trans_log', log_data)
            
        except Exception as e:
            self.get_logger().error(f"Erro ao processar log: {e}")

# --- HTML E JAVASCRIPT ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SMACC2 Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <script src="https://unpkg.com/klayjs@0.4.1/klay.js"></script>
    <script src="https://unpkg.com/cytoscape-klay@3.1.4/cytoscape-klay.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f0f2f5;}
        header { 
            background: #2c3e50; color: white; padding: 15px; 
            display: flex; justify-content: space-between; align-items: center;
            font-size: 20px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2); z-index: 10;
        }
        .btn-export {
            background-color: #e74c3c; color: white; border: none; padding: 8px 15px;
            font-size: 14px; border-radius: 5px; cursor: pointer; font-weight: bold;
            transition: background 0.3s;
        }
        .btn-export:hover { background-color: #c0392b; }
        
        #cy-container { flex: 2; position: relative; }
        #cy { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
        #log-panel { flex: 1; background: #fff; border-top: 3px solid #bdc3c7; overflow-y: auto; display: flex; flex-direction: column;}
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;}
        th { background: #ecf0f1; position: sticky; top: 0; padding: 10px; color: #34495e; border-bottom: 2px solid #bdc3c7;}
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1c40f33; cursor: pointer; }
        .badge-event { background: #3498db; color: white; padding: 3px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;}
    </style>
</head>
<body>
    <header>
        <span>SMACC2 Dashboard (Real-Time)</span>
        <button class="btn-export" id="btn-export-pdf">Export to PDF</button>
    </header>
    
    <div id="cy-container">
        <div id="cy"></div>
    </div>
    
    <div id="log-panel">
        <table>
            <thead>
                <tr>
                    <th style="width: 150px;">Time</th>
                    <th>Source (Emitter)</th>
                    <th>Trigger Event</th>
                    <th>Destiny State</th>
                </tr>
            </thead>
            <tbody id="log-body">
            </tbody>
        </table>
    </div>

    <script>
        var socket = io();
        var cy = cytoscape({
            container: document.getElementById('cy'),
            style: [
                {
                    selector: 'node[type="state"]',
                    style: {
                        'label': 'data(label)',
                        'text-valign': 'top',
                        'text-halign': 'center',
                        'text-margin-y': -8,
                        'background-color': '#ecf0f1',
                        'border-width': 2,
                        'border-color': '#34495e',
                        'shape': 'round-rectangle',
                        'padding': '30px', 
                        'font-weight': 'bold',
                        'font-size': '16px'
                    }
                },
                {
                    selector: 'node[type="orthogonal"]',
                    style: {
                        'label': 'data(label)',
                        'text-valign': 'top',
                        'text-halign': 'center',
                        'background-color': 'rgba(26, 188, 156, 0.05)',
                        'border-width': 2,
                        'border-style': 'dashed',
                        'border-color': '#1abc9c',
                        'padding': '15px',
                        'font-size': '12px',
                        'color': '#16a085'
                    }
                },
                {
                    selector: 'node[type="client"]',
                    style: {
                        'label': 'data(label)',
                        'shape': 'rectangle',
                        'background-color': '#3498db',
                        'color': '#fff',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'padding': '10px',
                        'font-size': '11px'
                    }
                },
                {
                    selector: 'node[type="behavior"]',
                    style: {
                        'label': 'data(label)',
                        'shape': 'ellipse',
                        'background-color': '#e67e22',
                        'color': '#fff',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'padding': '8px',
                        'font-size': '11px'
                    }
                },
                /* ESTILO DO NOVO STATE REACTOR */
                {
                    selector: 'node[type="reactor"]',
                    style: {
                        'label': 'data(label)',
                        'shape': 'hexagon',
                        'background-color': '#9b59b6',
                        'color': '#fff',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'padding': '12px',
                        'font-weight': 'bold',
                        'font-size': '11px',
                        'border-width': 2,
                        'border-color': '#8e44ad'
                    }
                },
                /* ARESTAS DE TRANSIÇÃO (REAIS) */
                {
                    selector: 'edge[edge_type="real"]',
                    style: {
                        'curve-style': 'taxi',
                        'taxi-direction': 'horizontal', 
                        'taxi-turn-min-distance': 40,
                        'source-distance-from-node': 15, 
                        'target-distance-from-node': 10,
                        'width': 2.5,
                        'line-color': '#7f8c8d',
                        'target-arrow-color': '#7f8c8d',
                        'target-arrow-shape': 'triangle',
                        'label': 'data(label)',
                        'font-size': '11px',
                        'color': '#e74c3c',
                        'font-weight': 'bold',
                        'text-background-opacity': 1,
                        'text-background-color': '#ffffff',
                        'text-background-padding': '4px',
                        'text-background-shape': 'roundrectangle'
                    }
                },
                /* ARESTAS VIRTUAIS (ALIMENTANDO O REACTOR) */
                {
                    selector: 'edge[edge_type="virtual"]',
                    style: {
                        'curve-style': 'bezier',
                        'width': 1.5,
                        'line-color': '#9b59b6',
                        'line-style': 'dashed',
                        'target-arrow-color': '#9b59b6',
                        'target-arrow-shape': 'vee',
                        'label': 'data(label)',
                        'font-size': '9px',
                        'color': '#8e44ad'
                    }
                },
                {
                    selector: '.active-state',
                    style: {
                        'background-color': 'rgba(46, 204, 113, 0.2)',
                        'border-color': '#27ae60',
                        'border-width': 5,
                        'transition-property': 'background-color, border-color, border-width',
                        'transition-duration': '0.3s'
                    }
                }
            ]
        });

        socket.on('sm_update', function(data) {
            cy.elements().remove();
            cy.add(data.nodes);
            cy.add(data.edges);
            
            // Retornamos ao KLAY, alterando apenas a DIREÇÃO para criar a árvore horizontal
            cy.layout({
                name: 'klay',
                nodeDimensionsIncludeLabels: true,
                animate: false,
                fit: true,
                padding: 50,
                klay: {
                    direction: 'RIGHT', /* RIGHT = Left to Right (Árvore Horizontal) */       
                    spacing: 80,           
                    layoutHierarchy: true,    
                    edgeRouting: 'ORTHOGONAL' 
                }
            }).run();

            if (data.active_state) {
                highlightActiveState(data.active_state);
            }
        });

        socket.on('active_state_update', function(data) {
            highlightActiveState(data.active_state);
        });

        socket.on('trans_log', function(data) {
            const tbody = document.getElementById('log-body');
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td><strong>${data.time}</strong></td>
                <td>${data.source}</td>
                <td><span class="badge-event">${data.event}</span></td>
                <td><b>${data.destiny}</b></td>
            `;
            
            tbody.prepend(row);
            
            if (tbody.children.length > 100) {
                tbody.removeChild(tbody.lastChild);
            }
        });

        function highlightActiveState(stateId) {
            cy.nodes('[type="state"]').removeClass('active-state');
            var node = cy.getElementById(stateId);
            if (node.length > 0) {
                node.addClass('active-state');
            }
        }

        /* Lógica para Exportar para PDF */
        document.getElementById('btn-export-pdf').addEventListener('click', function() {
            var png64 = cy.png({ full: true, scale: 2, bg: '#f0f2f5' });
            
            const { jsPDF } = window.jspdf;
            
            var pdf = new jsPDF('l', 'mm', 'a4');
            var pdfWidth = pdf.internal.pageSize.getWidth();
            var pdfHeight = pdf.internal.pageSize.getHeight();
            
            var imgProps = new Image();
            imgProps.src = png64;
            imgProps.onload = function() {
                var imgWidth = this.width;
                var imgHeight = this.height;
                var ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
                
                var newWidth = imgWidth * ratio;
                var newHeight = imgHeight * ratio;
                var xOffset = (pdfWidth - newWidth) / 2;
                var yOffset = (pdfHeight - newHeight) / 2;
                
                pdf.addImage(png64, 'PNG', xOffset, yOffset, newWidth, newHeight);
                pdf.save('SMACC2_StateMachine_Dump.pdf');
            };
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def ros_spin_thread(node):
    rclpy.spin(node)
    node.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SmaccVisualizerNode()
    
    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
