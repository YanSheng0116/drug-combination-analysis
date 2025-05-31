# If needed, install graphviz:
# !pip install graphviz

from graphviz import Digraph

# Create a vertical (top-to-bottom) flowchart
dot = Digraph(
    comment='Preprocess Pipeline',
    format='png'
)

# Global attributes for “cute” pastel style
dot.attr(
    rankdir='TB',  # Top → Bottom
    splines='ortho',
    graph_attr='pad="0.5", bgcolor="#fdfdfd"',
    node_attr='shape=box, style="filled,rounded", color="#aaaaaa", fontname="Helvetica", fontsize="10", penwidth="1.5"',
    edge_attr='color="#cccccc", arrowsize="0.8"'
)

# Step 1: Load genes
dot.node('A', '🧬 Step 1:\nLoad 2401-Gene List\n(Input: genes_2401_df.csv)\n(Output: genes list)', fillcolor='#ffe4e1')

# Step 2: PPI → W_norm
dot.node('B', '🔗 Step 2:\nLoad PPI & Normalize\n(Input: string_network)\n(Output: W_norm 2401×2401)', fillcolor='#e6f7ff')

# Step 3: Drug–Target → df_drug
dot.node('C', '💊 Step 3:\nLoad & Transpose\nDrug–Target Matrix\n(Input: combine_drug_target_matrix.csv)\n(Output: df_drug 2401×N)', fillcolor='#fff0b3')

# Step 4: RWR → df_drug_rwr
dot.node('D', '🚀 Step 4:\nRun RWR on W_norm\n(Input: df_drug, W_norm)\n(Output: df_drug_rwr 2401×N)', fillcolor='#d9e6d0')

# Step 5a: Load dep
dot.node('E', '🧫 Step 5a:\nLoad Dependency\n(Input: new_gene_dependencies_35.csv)\n(Output: dep 2401×35)', fillcolor='#ffe6f2')

# Step 5b: Load expr
dot.node('F', '🌱 Step 5b:\nLoad Expression\n(Input: processed_expression_raw_norm.csv)\n(Output: expr 2401×35)', fillcolor='#e8e8ff')

# Step 6: Load & Filter syn
dot.node('G', '📊 Step 6:\nLoad & Filter Synergy\n(Input: synergy_score.csv)\n(Output: syn 16 369 rows)', fillcolor='#ffecd1')

# Step 7: Assemble X & y
dot.node('H', '🧩 Step 7:\nAssemble Feature Matrix X & y\n(Input: df_drug_rwr, dep, expr, syn)\n(Output: X 32 738×9608, y 32 738×1)', fillcolor='#e1ffe6')

# Step 8: Save outputs
dot.node('I', '💾 Step 8:\nSave X_reproduce.npy\n& y_reproduce.pkl', fillcolor='#f0f0f0')

# Connect nodes top-to-bottom
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# Render and display
dot.render(view=True)
dot
