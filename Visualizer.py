import matplotlib.pyplot as plt

def plot_disorder_trend(df, entity_input, disorder):
    subset = df[df["Entity"].str.lower() == entity_input.lower()]
    if subset.empty:
        print(f"No data found for '{entity_input}'")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(subset["Year"], subset[disorder], marker="o", color="blue")
    plt.title(f"{disorder.capitalize()} prevalence in {entity_input.title()}")
    plt.xlabel("Year")
    plt.ylabel("Prevalence (% of population)")
    plt.grid(True)
    plt.show()
