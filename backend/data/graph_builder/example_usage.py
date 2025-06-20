from preprocess import load_data, clean_data, preprocess_data
from builder import build_graph

if __name__ == "__main__":
    # Example usage
    df = load_data('transactions.csv')
    df = clean_data(df)
    df = preprocess_data(df)
    G = build_graph(df)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.") 