import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import seaborn as sns


def test_depth_limitation(n_points=1000, n_trials=50, hidden_layer_sizes=(10,), delta=0.1):
    """
    Test des limitations de profondeur selon le Théorème 3.12 et le Lemme 4.1
    """
    np.random.seed(42)
    X = np.random.randn(n_points, 2)
    errors = []
    complexities = []

    for _ in tqdm(range(n_trials)):
        # Génération d'étiquettes aléatoires
        y = np.random.choice([-1, 1], size=n_points)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=1000,
            activation='relu'
        )

        model.fit(X, y)

        # Calcul de l'erreur et de la complexité
        predictions = model.predict(X)
        error = np.mean(predictions != y)
        errors.append(error)

        # Estimation de la complexité du modèle
        n_params = sum(l1 * l2 for l1, l2 in zip(
            (2,) + hidden_layer_sizes,
            hidden_layer_sizes + (1,)
        ))
        complexities.append(n_params)

    # Borne théorique selon le Lemme 4.1
    theoretical_bound = 0.5 * (1 - np.sqrt(
        (np.log(np.mean(complexities)) + np.log(1/delta)) / (2 * n_points)
    ))

    return {
        'errors': errors,
        'complexities': complexities,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'theoretical_bound': theoretical_bound,
        'n_layers': len(hidden_layer_sizes)
    }


def analyze_and_plot_results(architectures):
    """
    Analyse comparative des différentes architectures avec visualisations
    """
    results = {}
    for name, arch in architectures.items():
        results[name] = test_depth_limitation(hidden_layer_sizes=arch)

    # Figure 1: Comparaison des erreurs
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for name, res in results.items():
        sns.kdeplot(res['errors'],
                    label=f"{name} (profondeur={res['n_layers']})")
    plt.title('Distribution des erreurs de classification par architecture')
    plt.xlabel('Erreur de classification')
    plt.ylabel('Densité')
    plt.legend()

    # Enregistrer le graphique au format PNG
    plt.savefig('error_distribution.png')

    # Figure 2: Relation profondeur-erreur
    plt.subplot(2, 1, 2)
    depths = [res['n_layers'] for res in results.values()]
    mean_errors = [res['mean_error'] for res in results.values()]
    theo_bounds = [res['theoretical_bound'] for res in results.values()]

    plt.plot(depths, mean_errors, 'bo-', label='Erreur empirique moyenne')
    plt.plot(depths, theo_bounds, 'ro--', label='Borne théorique')
    plt.title('Impact de la profondeur sur l\'erreur de classification')
    plt.xlabel('Nombre de couches cachées')
    plt.ylabel('Erreur moyenne')
    plt.legend()

    # Enregistrer le graphique au format PNG
    plt.savefig('depth_vs_error.png')

    # Affichage des graphiques
    plt.tight_layout()
    plt.show()

    # Résultats numériques détaillés
    print("\nRésultats détaillés par architecture:")
    print("-" * 80)
    for name, res in results.items():
        print(f"\n{name} (profondeur={res['n_layers']}):")
        print(
            f"  Erreur moyenne: {res['mean_error']:.3f} ± {res['std_error']:.3f}")
        print(f"  Borne théorique: {res['theoretical_bound']:.3f}")
        print(
            f"  Nombre moyen de paramètres: {np.mean(res['complexities']):.0f}")

    return results


# Test avec différentes architectures
architectures = {
    'Réseau peu profond': (16,),
    'Réseau moyen': (8, 8),
    'Réseau profond': (4, 4, 4, 4),
    'Réseau très profond': (2, 2, 2, 2, 2, 2, 2, 2),
    'Réseau extrêmement profond': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
}

results = analyze_and_plot_results(architectures)

# Conclusions automatiques basées sur les résultats


def generate_conclusions(results):
    print("\nConclusions principales:")
    print("-" * 80)

    # Analyse de l'impact de la profondeur
    depths = [res['n_layers'] for res in results.values()]
    errors = [res['mean_error'] for res in results.values()]
    correlation = np.corrcoef(depths, errors)[0, 1]

    print(f"\n1. Relation profondeur-erreur:")
    if correlation > 0.5:
        print("   → La profondeur semble augmenter l'erreur de classification")
    elif correlation < -0.5:
        print("   → La profondeur semble ré duire l'erreur de classification")
    else:
        print("   → La relation entre profondeur et erreur n'est pas linéaire")

    # Vérification du théorème
    for name, res in results.items():
        empirical = res['mean_error']
        theoretical = res['theoretical_bound']
        print(f"\n2. Pour {name}:")
        print(
            f"   → Erreur empirique ({empirical:.3f}) vs Borne théorique ({theoretical:.3f})")
        if empirical > theoretical:
            print("   → Confirme le théorème: l'erreur est supérieure à la borne")
        else:
            print("   → Attention: l'erreur est inférieure à la borne théorique")


generate_conclusions(results)
