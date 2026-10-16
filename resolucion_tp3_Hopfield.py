# --- PROTOTIPO PRESENTADO PARA LA RESOLUCIÓN DEL TP3 - INTELIGENCIA ARTIFICIAL ---

import numpy as np


class HopfieldNetwork10x10:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.num_neurons = width * height
        self.W = np.zeros((self.num_neurons, self.num_neurons))

    def train_hebb(self, patterns):
        """Entrenamiento por Regla de Hebb (sin autoconexiones)."""
        self.W = np.zeros((self.num_neurons, self.num_neurons))
        for p in patterns:
            v = p.flatten()
            self.W += np.outer(v, v)
        np.fill_diagonal(self.W, 0)
        self.W /= self.num_neurons

    def predict(self, pattern_noisy, max_iter=5):
        """Proceso de convergencia dinámica (asíncrono/síncrono por signo)."""
        v = pattern_noisy.flatten().copy()
        for _ in range(max_iter):
            for i in range(self.num_neurons):
                raw_input = np.dot(self.W[i], v)
                v[i] = 1 if raw_input >= 0 else -1
        return v.reshape((self.height, self.width))

    def calculate_energy(self, state):
        v = state.flatten()
        return -0.5 * np.dot(v, np.dot(self.W, v))

def print_grid(matrix, title=""):
    print(f"\n--- {title} ---")
    for row in matrix:
        line = "".join([" X " if val == 1 else " . " for val in row])
        print(line)

# --- CREACIÓN DE PATRONES ---
# 1. Patrón Ideal: Aro 'C' (Cuadrado rotado 45°) + Escuadra de referencia fija en esquina inferior izquierda
pattern_ideal = np.full((10, 10), -1)

# Escuadra de referencia fija (esquina inferior izquierda [8..9, 0..1])
pattern_ideal[8, 0] = 1; pattern_ideal[9, 0] = 1; pattern_ideal[9, 1] = 1

# Geometría del Aro 'C' / Diamante centrado
pattern_ideal[2, 5] = 1
pattern_ideal[3, 4] = 1; pattern_ideal[3, 6] = 1
pattern_ideal[4, 3] = 1; pattern_ideal[4, 7] = 1
pattern_ideal[5, 4] = 1; pattern_ideal[5, 6] = 1
pattern_ideal[6, 5] = 1

# 2. Patrón Entorno con Ruido (píxeles distorsionados e incompletos)
pattern_noisy = pattern_ideal.copy()
pattern_noisy[2, 5] = -1  # Píxel faltante en el aro
pattern_noisy[0, 1] = 1   # Ruido aleatorio superior
pattern_noisy[1, 8] = 1   # Ruido aleatorio superior
pattern_noisy[7, 7] = 1   # Ruido aleatorio inferior

# --- EJECUCIÓN DEL PROTOTIPO ---
if __name__ == "__main__":
    hopfield = HopfieldNetwork10x10()
    hopfield.train_hebb([pattern_ideal])

    print_grid(pattern_ideal, "1. Patrón Referencia Guardado (Aro C + Escuadra)")
    print(f"Energía Estado Ideal: {hopfield.calculate_energy(pattern_ideal):.2f}")

    print_grid(pattern_noisy, "2. Imagen Entrada con Ruido/Incompleta")
    print(f"Energía Estado Ruidoso: {hopfield.calculate_energy(pattern_noisy):.2f}")

    # Proceso de restauración
    restored = hopfield.predict(pattern_noisy)
    print_grid(restored, "3. Imagen Restaurada por Hopfield (Estado Estable)")
    print(f"Energía Estado Final: {hopfield.calculate_energy(restored):.2f}")