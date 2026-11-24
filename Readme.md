<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=Monja%20Blanca%20Particles&fontSize=70&fontAlignY=35&animation=twinkling&fontColor=gradient" alt="Monja Blanca Particles"/>

<img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.github.com%2Frepos%2FGh0zd%2FMonja-Blanca-Particles&query=%24.archived&style=for-the-badge&label=Archivado&color=brightgreen" alt="Estado de archivo"/>
<img src="https://img.shields.io/github/last-commit/Gh0zd/Monja-Blanca-Particles?style=for-the-badge&label=%C3%9Altimo%20Commit" alt="Último Commit"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>

<p align="center">
  <img src="previa.gif" alt="Interfaz de Vidaa-Blocklist" width="80%" />
</p> 

<strong>Animación procedural de una flor Monja Blanca generada en tiempo real mediante partículas 3D, con iluminación, sombreado y efectos de bloom.</strong>

</div>

---

> [!NOTE]
> El proyecto utiliza una resolución interna fija (720×612) y reescala la ventana de visualización a 1080×918. El rendimiento puede variar según la CPU y la versión de OpenCV.

## Características

* **Renderizado por partículas:** Miles de partículas con superficie paramétrica tipo pétalo, muestreadas con densidad proporcional al ancho.
* **Animación por fases:** Crecimiento del tallo, apertura de la flor y desarrollo del estambre controlados mediante funciones de easing.
* **Z-buffer por capas:** Dos capas independientes (hojas/tallo y flor) con ordenamiento de profundidad y relleno morfológico de huecos.
* **Iluminación y sombreado:** Sombreado Lambertiano con luz direccional, oclusión ambiental y nervadura central en pétalos.
* **Bloom multi-nivel:** Efecto de resplandor mediante desenfoque gaussiano a múltiples escalas.
* **Cámara orbital:** Balanceo sinusoidal con inclinación elevada y zoom progresivo.

## Requisitos

* Python 3.11 o superior
* NumPy
* OpenCV (`opencv-python`)

## Instalación e inicio rápido

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Gh0zd/Monja-Blanca-Particles.git
   cd Monja-Blanca-Particles