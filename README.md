# TrackingProject

# Application de détection de gestes

Cette application détecte le geste *dab* et le signe *Jul* à partir d'une caméra en temps réel. Une musique spécifique pour chaque mouvement se lance ensuite.  
Cette application contient aussi un autre module qui permet de contrôler le volume de votre machine à l'aide de votre pouce et de votre index.

## Prérequis
- Python 3.11 ou supérieur
- Une caméra connectée et fonctionnelle
- Modules nécessaires (installés via `requirements.txt`)

## Installation
1. Clonez ou téléchargez ce projet :
   ```bash
   git clone https://github.com/baptiste-bedouret/TrackingProject.git
   cd TrackingProject

2. Créez un environnement virtuel et activez le :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Lancez le programme :
```bash
python main.py
ou 
python VolumeHandControl.py
```

Appuyer sur 'q' pour quitter l'application.




