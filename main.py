import torch
import matplotlib.pyplot as plt
from model import DeepNetwork, ShallowNetwork
from utils import target_function, target_function_slightly_oscillatory, train_model, evaluate_model

# Set seeds for reproducibility
torch.manual_seed(42)

# Générer les données d'entraînement pour les deux fonctions cibles
x_train = torch.linspace(0, 1, 1000).unsqueeze(1)

# Initialisation des modèles, optimisateurs et fonction de perte
deep_model = DeepNetwork()
shallow_model = ShallowNetwork()

optimizer_params = {'lr': 0.01}
deep_optimizer = torch.optim.Adam(deep_model.parameters(), **optimizer_params)
shallow_optimizer = torch.optim.Adam(
    shallow_model.parameters(), **optimizer_params)
criterion = torch.nn.MSELoss()

# Entraîner et évaluer sur la fonction oscillante
y_train_oscillatory = target_function(x_train)
deep_model_oscillatory = train_model(
    deep_model, deep_optimizer, criterion, x_train, y_train_oscillatory)
shallow_model_oscillatory = train_model(
    shallow_model, shallow_optimizer, criterion, x_train, y_train_oscillatory)

x_test = torch.linspace(0, 1, 500).unsqueeze(1)
y_test_oscillatory = target_function(x_test)
deep_mse_oscillatory, deep_predictions_oscillatory = evaluate_model(
    deep_model_oscillatory, x_test, y_test_oscillatory, criterion)
shallow_mse_oscillatory, shallow_predictions_oscillatory = evaluate_model(
    shallow_model_oscillatory, x_test, y_test_oscillatory, criterion)

# Entraîner et évaluer sur la fonction quadratique
y_train_quadratic = target_function_slightly_oscillatory(x_train)
deep_model_quadratic = train_model(
    deep_model, deep_optimizer, criterion, x_train, y_train_quadratic)
shallow_model_quadratic = train_model(
    shallow_model, shallow_optimizer, criterion, x_train, y_train_quadratic)

y_test_quadratic = target_function_slightly_oscillatory(x_test)
deep_mse_quadratic, deep_predictions_quadratic = evaluate_model(
    deep_model_quadratic, x_test, y_test_quadratic, criterion)
shallow_mse_quadratic, shallow_predictions_quadratic = evaluate_model(
    shallow_model_quadratic, x_test, y_test_quadratic, criterion)

# Affichage des résultats
plt.figure(figsize=(12, 12))

# Comparaison des résultats sur la fonction oscillante
plt.subplot(2, 1, 1)
plt.plot(x_test, y_test_oscillatory,
         label='Target Function (Highly Oscillatory)', color='black')
plt.plot(x_test, deep_predictions_oscillatory,
         label=f'Deep Network (MSE: {deep_mse_oscillatory:.4f})', color='blue')
plt.plot(x_test, shallow_predictions_oscillatory,
         label=f'Shallow Network (MSE: {shallow_mse_oscillatory:.4f})', color='red')
plt.legend()
plt.title('Comparison of Deep and Shallow Networks on Highly Oscillatory Data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Comparaison des résultats sur la fonction quadratique
plt.subplot(2, 1, 2)
plt.plot(x_test, y_test_quadratic,
         label='Target Function (Slightly Oscillatory)', color='black')
plt.plot(x_test, deep_predictions_quadratic,
         label=f'Deep Network (MSE: {deep_mse_quadratic:.4f})', color='blue')
plt.plot(x_test, shallow_predictions_quadratic,
         label=f'Shallow Network (MSE: {shallow_mse_quadratic:.4f})', color='red')
plt.legend()
plt.title('Comparison of Deep and Shallow Networks on Slightly Oscillatory Data')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Afficher les MSE pour chaque fonction cible
print(
    f"Highly Oscillatory Target Function - Deep Network MSE: {deep_mse_oscillatory:.4f}")
print(
    f"Highly Oscillatory Target Function - Shallow Network MSE: {shallow_mse_oscillatory:.4f}")
print(
    f"Slightly Oscillatory Target Function - Deep Network MSE: {deep_mse_quadratic:.4f}")
print(
    f"Slightly Oscillatory Target Function - Shallow Network MSE: {shallow_mse_quadratic:.4f}")
