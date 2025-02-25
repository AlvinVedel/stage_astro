import numpy as np
import multiprocessing as mp
import gc


## ESSAI SUR L ORTHOGONALITE DES VECTEURS CARACTERISTIQUES DE COULEUR ET LA RELATION AVEC LE REDSHIFT


def compute_target(x) :
        image = x[..., :5]
        mask = x[..., 5].astype(bool)

        indices = np.where(mask)
        pixels = image[indices]

        colors = np.zeros((4))

        colors[0] = np.mean((pixels[..., 0]-pixels[..., 1])) # u-g
        colors[1] = np.mean((pixels[..., 1] - pixels[..., 2])) # g-r
        colors[2] = np.mean((pixels[..., 3] - pixels[..., 4])) # i-z
        colors[3] = np.mean((pixels[..., 2] - pixels[..., 3])) # r-i

        return colors


from pathlib import Path

#directory = "/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec"
dir_path = Path("/lustre/fswork/projects/rech/dnz/ull82ct/astro/data/spec/")
extension = ".npz"
paths = [file for file in dir_path.rglob(f"*{extension}")]


all_x = []
all_y = []
all_z = []

for path in paths :
      

    data = np.load(path, allow_pickle=True)
    images = data["cube"]
    meta = data["info"]
    z_values = np.array([m[40] for m in meta])  



    with mp.Pool(processes=mp.cpu_count()) as pool:
        colors = pool.map(compute_target, images)


    batch = np.array(colors)

    angle_ref = np.ones((batch.shape[0], batch.shape[1]))
    angle_ref[:, -1] = 0
    direction_ref = np.zeros((batch.shape[0], batch.shape[1]))
    direction_ref[:, -1] = 1

    cosine_sim = np.sum(angle_ref * batch, axis=1) / (
        np.linalg.norm(angle_ref, axis=1) * np.linalg.norm(batch, axis=1) + 1e-8
    )

    signe_ref_sim = np.sign(np.sum(direction_ref * batch, axis=1) / (
        np.linalg.norm(direction_ref, axis=1) * np.linalg.norm(batch, axis=1) + 1e-8
    ))

    angles = np.arccos(cosine_sim) 

    x = np.cos(angles)
    y = np.sin(angles) * signe_ref_sim

    all_x.append(x)
    all_y.append(y)
    all_z.append(z_values)

    del images, angle_ref, direction_ref
    gc.collect()

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
z_values = np.concatenate(all_z, axis=0)



from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

kde = gaussian_kde(np.vstack([x, y]))
densities = kde(np.vstack([x, y]))



plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
sc = plt.scatter(x, y, c=densities, cmap='viridis', label='Représentations', s=100)  # Points colorés
    #plt.scatter(origin_x, origin_y, color='red', label='Vecteur moyen', s=100, edgecolors='black', marker='X')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Personnalisation
plt.title("Distribution des angles sur le cercle unitaire")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.axis('equal')
plt.legend()
plt.colorbar(sc, label='Densité')
plt.grid(True)

    # Afficher
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/orthog_colors.png")
plt.close()


plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', label='Cercle unitaire')  # Cercle
sc = plt.scatter(x, y, c=z_values, cmap='viridis', label='Représentations', s=100)  # Points colorés
    #plt.scatter(origin_x, origin_y, color='red', label='Vecteur moyen', s=100, edgecolors='black', marker='X')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

    # Personnalisation
plt.title("Distribution des angles sur le cercle unitaire")
plt.xlabel("cos(θ)")
plt.ylabel("sin(θ)")
plt.axis('equal')
plt.legend()
plt.colorbar(sc, label='Densité')
plt.grid(True)

    # Afficher
plt.savefig("/lustre/fswork/projects/rech/dnz/ull82ct/astro/plots/orthog_colors_Z.png")
plt.close()



   