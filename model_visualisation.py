import torch
from torchviz import make_dot
from model import DeepNetwork, ShallowNetwork  # Importez vos modèles ici


def visualize_network():
    # Initialiser les réseaux
    deep_network = DeepNetwork()
    shallow_network = ShallowNetwork()

    x = torch.ones(1, 1)

    # Générer le graphe pour DeepNetwork
    output_deep = deep_network(x)
    dot_deep = make_dot(output_deep, params=dict(
        deep_network.named_parameters()))

    # Générer le graphe pour ShallowNetwork
    output_shallow = shallow_network(x)
    dot_shallow = make_dot(output_shallow, params=dict(
        shallow_network.named_parameters()))

    # Sauvegarder les graphes en fichiers .pdf ou .png
    dot_deep.render("deep_network", format="png")
    dot_shallow.render("shallow_network", format="png")


# Exécuter la visualisation
if __name__ == "__main__":
    visualize_network()
