# 0.3.3

- Remove global state.change subscription, which triggered recursive Gradio hashing on native queued/imported state.
- Keep model/mode-driven visibility and all generation behaviour.
- Reproduce the previous RecursionError and test repeated state-return callbacks after native queue creation.

# 0.3.2

- Reference context no longer collapses to one video latent block when native assembly overlap is one frame: at least 18 frames of available latent AV history are retained.
- Context selection and native LoRA/cache settings recorded in the checkpoint; no pixel re-encode added.
- CPU regression covers overlap=1 with unchanged assembly duration. Visual improvement requires a GPU comparison.

# 0.3.1

- Preserve closure-based wrappers without `__wrapped__` by following their captured native function references.
- Regression test: undecorated closure around native H3 generation, all continuation modes and existing loader/layout tests.

# Changelog

## 0.3.0

- Extension-only integration with native MiniMax H3 models; separate model declarations removed.
- Local edits to installed native generation, packing and layout; existing delegating wrappers preserved.
- Original native rendering when options are off; latent operations retain single-phase restrictions and abort handling.
- Recreate old jobs using native model entries. Existing checkpoint formats and native identities remain readable.

# Historique

## 0.2.3

- Un arrêt demandé par l'utilisateur ne déclenche plus l'erreur de checkpoint latent absent.
- La capture latente incomplète est abandonnée et aucun safetensors n'est écrit.
- Le contrôle reste actif lorsqu'un rendu signale une fin normale sans produire le checkpoint demandé.

## 0.2.2

- Nouveau nom public **H3 Latent Continue** dans la liste des plugins et le sélecteur de modèles.
- Nouveau dossier d'installation et dossier racine du ZIP : `wan2gp-h3-latent-continue`.
- Conservation des identifiants de modèles `h3_latent_*`, de la clé interne et des formats de checkpoints pour assurer la continuité avec les rendus précédents.

## 0.2.1

- Suppression complète du blocage par empreinte des sources : aucun fichier Python ni numéro de version n'est comparé avant le rendu.
- Suppression de `compatibility.json` et `source_compatibility.py`. Le diagnostic manuel ne vérifie plus que la présence de l'installation et n'est jamais appelé par le rendu.
- Intégration des changements du pipeline de Wan2GP 13.0 fourni : suivi de progression, filtres de sources Viggle vides et statuts de décodage. Progression optionnelle sur 12.72.
- Tests du formulaire/file/export sur le nouvel état `SharedState`, et maintien des trois modes de continuation et des checkpoints v1/v2.
- Contrôles de données et configurations conservés, notamment une seule phase obligatoire. Aucune garantie de compatibilité avec de futures ruptures d'interface ou de conventions des latents.

## 0.2.0

- Trois modes comparatifs : référence 0.1.7, contexte AV étendu, contexte vidéo étendu avec préfixe audio figé.
- Historique vidéo indépendant du chevauchement natif ; durée audio indépendante.
- Préfixe audio immuable, placement temporel cible, décodage conjoint et coupe à l'échantillon ; caches refusés pour ce mode.
- Checkpoints v2 avec origine audio, lecture des v1 ; réglages transmis dans le formulaire et les tâches.
- Tests de géométrie, préfixe intact, durée de sortie et continuation successive. Validation GPU/perceptuelle à réaliser.

## 0.1.7

- Correction de l'erreur `.ndim` sur la liste de blocs fournie par l'export vidéo natif.
- Lecture de huit images finales au maximum, comptage complet et conservation de la liste originale.
- Tests de blocs multiples, de nom final après mux simulé et du corps natif de l'exporteur ; 29 tests CPU.

## 0.1.6

- Instantané des options propre à chaque tâche, indépendant du champ partagé `plugin_data`.
- Raccord à `add_video_task`, diagnostic de mise en file et prise en charge de l'édition des paramètres de tâche.
- Extraction des options transmises par `**kwargs` ; refus explicite des tâches sans options récupérables.
- Régression testée en supprimant volontairement les données partagées avant la file et avant le rendu ; 27 tests CPU.

## 0.1.5

- Lecture directe des cases et du fichier latent dans les événements natifs `save_inputs` du formulaire concerné.
- Un rafraîchissement contenant des données de plugin vides ne réinitialise plus les contrôles.
- Diagnostic de liaison au démarrage et de capture des valeurs du formulaire.
- Test réel `Gradio.process_api` jusqu'à la file native puis au checkpoint CPU, avec état obsolète et vérification des tenseurs et du chemin final.

## 0.1.4

- Ajout d'un instantané des options par session et modèle et d'un diagnostic des options reçues par la tâche. Insuffisant pour le cas utilisateur, repris en 0.1.5.

## 0.1.3

- Traduction des identifiants des quatre architectures du prototype pour le prétraitement natif des LoRA.

## 0.1.2

- Normalisation des fins de ligne des sources Python dans le contrôle de compatibilité Windows.

## 0.1.1

- Ajout du chemin `profiles` obligatoire pour la découverte des modèles et du dossier correspondant.
- Normalisation des dictionnaires/objets Gradio State dans les callbacks et la préparation des tâches.
- Préservation des données des autres plugins ; valeurs vides ou invalides traitées comme options désactivées.
- Tests de découverte via le gestionnaire natif et tests des callbacks avec de vrais objets State.
- README et instructions de remplacement mis à jour.
- Aucun changement à la logique de débruitage ou au format de checkpoint (version 1).

## 0.1.0

Prototype initial de continuation latente vidéo/audio dans le formulaire de génération existant, avec sauvegarde des latents à côté de la vidéo. Rendu une phase uniquement.
