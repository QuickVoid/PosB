# To activate the virtual environment type .venv\scripts\activate

import numpy as np
import matplotlib.pyplot as plt
from geometry import distance
from trilaterate2d_v2 import *
from scipy.spatial.distance import euclidean 
from scipy.optimize import least_squares

lx,ly = 60, 40  # afmeting van fabriekshal in lengte en breedte
Nx,Ny = 6, 4    # aantal grid-punten langs lengte- en breedte-as
Nav = 20        # aantal middelingen in verband met meetruis
sigma = 0.01    # standaard deviation of noise in meter

# baken posities
beacon_positions = np.array([
    [10, 10],  # Voorbeeld: baken op positie (x, y)
    [30, 30],  # Voeg meer bakens toe zoals [x, y] of [x, y, z] voor 3D als 2D werkt
    [50, 10]   # 3de baken meer niet nodig voor nu      
])
num_beacons = len(beacon_positions)

# baken afstand
distances_measured = np.array([
    10, 30, 50 # eerste rij van beacon_positions gekozen
])

# lijst met x-waarden en y-waarden, extra z-waarden
x_list = np.linspace(0, lx, Nx)
y_list = np.linspace(0, ly, Ny)
# z_list = np.linspace(0, lz, Nz) # voor 3D opdracht

std_noise_list = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.16, 0.32])
J_noise = np.zeros(len(std_noise_list)) # array met schattingsfout

# arrays voor opslag van resultaten gebruik np.zeros voor efficient arrays maken goed initialiseren dan vullen met berekende waardes
distances_to_beacons = np.zeros((Nx, Ny, num_beacons))  # afstanden tot elk baken (array van 4x3 x 6)
estimated_positions = np.zeros((Nx, Ny, 2))  # geschatte posities (x, y) (array van 4x2 x 3)
estimation_errors = np.zeros((Nx, Ny))  # schattingsfouten (array van 6x4)

# simulatie experimenten
for exp,std_noise in enumerate(std_noise_list): # voor verschillende hoeveelheden ruis
    J = np.zeros((Nx,Ny))
    print(f"Experiment {exp + 1} with noise {std_noise}")
    for ix,x in enumerate(x_list):  # loop alle gridpunten in x- 
        for iy,y in enumerate(y_list):  # en y-richting
            p = np.array([x,y])
            # bereken afstand tot bakens
            d = np.array([distance(b,p) for b in beacon_positions]) 

            estimation_errors_exp = []
            
            # voer Nav experimenten om schattingsfout te middelen ivm meetruis
            for i in range(Nav):
                # maak metingen door toevoegen van meetruis
                noisy_d = d + np.random.normal(0, sigma, num_beacons)
                pe = trilaterate_lstsq(beacon_positions, noisy_d)
                
                # bereken verschil in afstand tussen echte en geschatte positie
                estimated_positions[ix, iy] = pe
                estimation_errors[ix, iy] = distance(p, pe)

                # and average over Nav
                J[ix,iy] += distance(p,pe)/Nav 
            
                p = np.array([ix * (lx / (Nx - 1)), iy * (ly / (Ny - 1))])  # bereken positie van meetpunt
        
                # bereken afstand tot elk baken
                for i, beacon_pos in enumerate(beacon_positions):
                    distances_to_beacons[ix, iy, i] = distance(beacon_pos, p)

                
                estimation_errors_exp.append(estimation_error)
            estimation_errors[ix, iy] = np.mean(estimation_errors_exp)
            distances_to_beacons[ix, iy] = d
    # bepaal over alle punten op het grid de grootste schattingsfout
    J_noise[exp] = J.max()

# controleer de vorm van de arrays
# print("Afstanden tot elk baken:", distances_to_beacons.shape)
# print("Geschatte posities:", estimated_positions.shape)
# print("Schattingsfouten:", estimation_errors.shape)

# color grid representation
plt.imshow(estimation_errors, extent=(0, lx, 0, ly), origin='lower')
plt.colorbar(label='Estimation Error (m)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Estimation Error Over the Grid')
plt.savefig('estimation_error_grid.png')
plt.show()

# Plotting J_noise vs std_noise_list
plt.plot(std_noise_list, J_noise, marker='o')
plt.xlabel('Standard Deviation of Noise')
plt.ylabel('Average Estimation Error')
plt.title('Estimation Error vs Measurement Noise')
plt.savefig('error_vs_noise.png')
plt.show()

# Prepare data for 3D plot
X, Y = np.meshgrid(x_list, y_list)
Z = estimation_errors.T

# Plotting the 3D wireframe
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='blue')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Average Estimation Error (m)')
ax.set_title('3D Wireframe Plot of Average Estimation Error')

plt.savefig('3d_wireframe_plot.png')
plt.show()

# Beantwoord de volgende vragen:

# 1) Wat is de waarde van de standaard deviatie van de meetruis
#    zodat de afstand tussen de geschatte posities en echte posities
#    kleiner dan 5 cm is?
#    Antwoord: 0,02

# 2) Ga na welke afstandssensoren op de markt toegepast kunnen worden in een
#    indoor RTLS systeem met een bereik zoals nodig in deze opdracht en een
#    nauwkeurigheid gevonden bij vraag 1.
#    Antwoord: LIDAR (Light Detection and Ranging) Sensoren vb. Garmin LIDAR-Lite v3

# 3) Ga na of je een kant en klaar RTLS systeem kunt vinden waarvan de specificaties
#    in de buurt komen van wat nodig is in de toepassing van deze opdracht.
#    Antwoord: Slamtec RPLIDAR A3, nauwkeurigheid binnen 1 cm & bereik Tot 25 meter.

# 6) Ga na, of er gebieden zijn waar de schattingsfout kleiner is en waar deze groter is. 
#    Zijn de bakenposities goed gekozen?
#    Zo ja, waarom, en zo nee, hoe zou je de bakenposities willen aanpassen? Zo nodig, pas
#    de baken posities aan.
#    Antwoord: de baken posities zijn goed gekozen omdat het verschil van de schattingsfout zeer klein is met deze instellingen.
#              Ik kan de grafiek alleen niet recht krijgen zonder de schattingsfout extreem groot te maken.

# 8) Hoe nauwkeurig moeten de afstandsmetingen zijn om een gemiddelde
#    schattingsfout van 5 cm of minder te krijgen
#    Antwoord: 2cm

# 9) Zoek een RTLS systeem of afstandssensoren die voldoen aan de eis om met 5 cm
#    nauwkeurig posities te bepalen over het gebied van 40 m bij 60 m. Als je geen sys-
#    teem kunt vinden, kies dan het systeem welke het beste aan de eisen voldoet.
#    Antwoord: Velodyne Lidar, nauwkeurigheid ±3 cm & bereik tot 100 meter.
