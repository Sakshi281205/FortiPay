from preprocess import load_data, clean_data
from builder import build_graph

def main():
    df = load_data("sample_upi_data.csv")
    df = clean_data(df)
    G = build_graph(df)
    print("Graph built successfully with", len(G.nodes), "nodes.")

if __name__ == "__main__":
    main()
